/// Idempotent schema. Safe to run on every startup. `dim` parameterizes the
/// FLOAT[N] fixed-size vector type — production uses `DEFAULT_EMBED_DIM`,
/// the eval harness passes the active model's dim.
pub fn schema_sql(dim: usize) -> String {
    format!(
        r#"
INSTALL fts; LOAD fts;

CREATE TABLE IF NOT EXISTS obs (
    id             BIGINT  PRIMARY KEY,
    ts             BIGINT  NOT NULL,
    session        VARCHAR NOT NULL,
    branch         VARCHAR,
    kind           VARCHAR NOT NULL,
    body           VARCHAR NOT NULL,
    -- Optional file reference for tool-use obs (Read / Edit / Write / Bash).
    -- Prose obs (user_prompt, assistant_message, subagent_message) leave
    -- these NULL. Enables fast filtering by path without parsing body text.
    ref_path       VARCHAR,
    ref_line_start INTEGER,
    ref_line_end   INTEGER
);

CREATE TABLE IF NOT EXISTS sessions (
    id         VARCHAR PRIMARY KEY,
    started_ts BIGINT,
    ended_ts   BIGINT,
    branch     VARCHAR,
    cwd        VARCHAR,
    summary    VARCHAR
);

-- Per-chunk embeddings. Obs bodies are split into char-windowed chunks on
-- insert (each chunk fits inside the embedder's token window), then one
-- vector is stored per chunk. Recall does cosine against every chunk and
-- group-by-obs max(score) as the vector contribution. BM25 still indexes
-- the full body in obs.body, so we get full-body lexical recall plus
-- chunk-level semantic recall in the same pipeline.
CREATE TABLE IF NOT EXISTS vec_chunks (
    obs_id    BIGINT  NOT NULL,
    chunk_idx INTEGER NOT NULL,
    emb       FLOAT[{dim}] NOT NULL,
    PRIMARY KEY (obs_id, chunk_idx)
);

-- Retained for backward-compat with existing DBs from Phase 1–3. New
-- inserts go into vec_chunks; legacy single-vector rows in `vec` are
-- ignored by recall after the migration. Drop at user discretion.
CREATE TABLE IF NOT EXISTS vec (
    id  BIGINT PRIMARY KEY,
    emb FLOAT[{dim}]
);

-- Bidirectional synonym pairs. (k8s, kubernetes) and (kubernetes, k8s) both
-- live as rows. Query-time expansion is a SQL join — no in-memory cache.
CREATE TABLE IF NOT EXISTS synonyms (
    term      VARCHAR NOT NULL,
    expansion VARCHAR NOT NULL,
    PRIMARY KEY (term, expansion)
);

-- Tracks whether the synonyms table has ever been seeded. Skipping the seed
-- on later runs means the user's `memorize syn remove` deletions stick.
CREATE TABLE IF NOT EXISTS meta (
    key   VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

-- Idempotent column additions for older DBs. DuckDB treats redundant ALTERs
-- on already-present columns as no-ops when guarded by IF NOT EXISTS.
ALTER TABLE obs ADD COLUMN IF NOT EXISTS ref_path       VARCHAR;
ALTER TABLE obs ADD COLUMN IF NOT EXISTS ref_line_start INTEGER;
ALTER TABLE obs ADD COLUMN IF NOT EXISTS ref_line_end   INTEGER;

-- ---- Code index ----
-- One row per indexed file. `mtime_ns` and `size_bytes` let us detect when
-- a file changed without re-parsing on every event.
CREATE TABLE IF NOT EXISTS files (
    path       VARCHAR PRIMARY KEY,    -- absolute, normalized
    repo_root  VARCHAR NOT NULL,
    language   VARCHAR NOT NULL,
    mtime_ns   BIGINT  NOT NULL,
    size_bytes BIGINT  NOT NULL,
    git_rev    VARCHAR
);

-- One row per AST chunk. Body hash makes re-indexing the same file a no-op
-- when content didn't change.
CREATE TABLE IF NOT EXISTS code_chunks (
    id          BIGINT  PRIMARY KEY,
    path        VARCHAR NOT NULL,
    language    VARCHAR NOT NULL,
    line_start  INTEGER NOT NULL,
    line_end    INTEGER NOT NULL,
    kind        VARCHAR NOT NULL,
    qualified   VARCHAR NOT NULL,
    body        VARCHAR NOT NULL,
    body_hash   BLOB    NOT NULL,
    -- FTS-friendly tokenization of (path + qualified). Path separators
    -- (`/_-.`) flatten to spaces, and camelCase identifiers in `qualified`
    -- split on case boundaries (`snapshotMemoGraph` → `snapshot Memo Graph`).
    -- Without this, query "memo" doesn't match qualified=`snapshotMemoGraph`
    -- because DuckDB's FTS tokenizer keeps camelCase as a single token.
    -- Populated at upsert and force-recomputed on every store init so the
    -- tokenization stays in sync with whatever rule we used last.
    path_tokens VARCHAR
);
ALTER TABLE code_chunks ADD COLUMN IF NOT EXISTS path_tokens VARCHAR;

CREATE TABLE IF NOT EXISTS vec_code (
    id  BIGINT PRIMARY KEY,
    emb FLOAT[{dim}] NOT NULL
);
"#,
        dim = dim
    )
}

/// FTS indexes — rebuilt periodically since DuckDB's FTS has no
/// incremental update. Cheap at our scale.
///
/// DuckDB FTS allows multiple columns per index but only one index per
/// table. We index body + qualified + path_tokens in a single combined
/// index so `match_bm25` scores across all three fields at once. Earlier
/// the table had two back-to-back `create_fts_index` calls with
/// `overwrite=1`; the second silently replaced the first, leaving only
/// `qualified` indexed and `body` BM25 effectively dead.
pub fn fts_index_sql() -> &'static str {
    "PRAGMA create_fts_index('obs', 'id', 'body', overwrite=1);
     PRAGMA create_fts_index('code_chunks', 'id', 'body', 'qualified', 'path_tokens', overwrite=1);"
}

/// Force-recompute `path_tokens` on store init. Cheap at our scale (~84k
/// rows on slate, single UPDATE) and guarantees the column matches whatever
/// tokenization rule the codebase last shipped — no stale tokens when we
/// change the rule. Computes path-separator splits + camelCase splits of
/// `qualified` into a single space-separated lowercased token stream.
pub fn backfill_path_tokens_sql() -> &'static str {
    "UPDATE code_chunks
        SET path_tokens = lower(
            regexp_replace(path, '[/_.\\-]', ' ', 'g')
            || ' '
            || regexp_replace(qualified, '([a-z])([A-Z])', '\\1 \\2', 'g')
        )"
}
