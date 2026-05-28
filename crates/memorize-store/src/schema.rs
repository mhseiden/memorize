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
--
-- Int8-quantized; see `migrate_vec_chunks_to_int8_sql` for the FLOAT→TINYINT
-- migration that fires on older DBs.
CREATE TABLE IF NOT EXISTS vec_chunks (
    obs_id    BIGINT  NOT NULL,
    chunk_idx INTEGER NOT NULL,
    emb_q8    TINYINT[{dim}] NOT NULL,
    PRIMARY KEY (obs_id, chunk_idx)
);
ALTER TABLE vec_chunks ADD COLUMN IF NOT EXISTS emb_q8 TINYINT[{dim}];

-- Drop the Phase 1–3 single-vector-per-obs holdover. Recall has used
-- `vec_chunks` (multi-vector + max-pool) for years; the row count on
-- live DBs is 0. Idempotent: `IF EXISTS` is a no-op on fresh installs.
DROP TABLE IF EXISTS vec;

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

-- Int8 scalar quantization. Each f32 component of the L2-normalized
-- embedder output (range [-1, 1]) maps to `round(v * 127)` as TINYINT.
-- 4× smaller on disk than FLOAT[N] and the in-memory cache the daemon
-- builds from this table runs dot products in ~3 ms over the full
-- corpus. See `migrate_vec_code_to_int8_sql` for the FLOAT→TINYINT
-- migration that fires on databases created before the change.
CREATE TABLE IF NOT EXISTS vec_code (
    id     BIGINT PRIMARY KEY,
    emb_q8 TINYINT[{dim}] NOT NULL
);
ALTER TABLE vec_code ADD COLUMN IF NOT EXISTS emb_q8 TINYINT[{dim}];
"#,
        dim = dim
    )
}

/// One-shot migration from FLOAT[N] to TINYINT[N] for the code-vector table.
/// Only fires when the legacy `emb` column is still present.
///
/// Two steps run as a single `execute_batch` so a crash in the middle
/// leaves no half-quantized rows (DuckDB groups consecutive DDL/DML in
/// a batch under one implicit transaction):
///
///   1. Backfill `emb_q8` from `emb` for any NULL rows.
///   2. Drop the legacy `emb` column.
///
/// Idempotent: after the first run, `emb` no longer exists so the
/// SET-clause's reference to `emb` would error — that's why the caller
/// must `vec_code_has_legacy_emb` first.
pub fn migrate_vec_code_to_int8_sql(dim: usize) -> String {
    format!(
        r#"
SET lambda_syntax='ENABLE_SINGLE_ARROW';
UPDATE vec_code
   SET emb_q8 = CAST(
       list_transform(emb, x -> CAST(ROUND(x * 127) AS TINYINT))
       AS TINYINT[{dim}]
   )
 WHERE emb_q8 IS NULL;
ALTER TABLE vec_code DROP COLUMN IF EXISTS emb;
"#,
        dim = dim
    )
}

/// True iff the `vec_code` table still has the legacy FLOAT[N] `emb`
/// column. After migration this returns false on subsequent startups.
pub fn vec_code_legacy_emb_probe_sql() -> &'static str {
    "SELECT count(*) > 0
       FROM duckdb_columns
      WHERE table_name = 'vec_code' AND column_name = 'emb'"
}

/// Same migration as `migrate_vec_code_to_int8_sql` but for `vec_chunks`
/// (per-chunk embeddings on the obs side). Idempotent: gated by
/// `vec_chunks_legacy_emb_probe_sql` so it only fires on DBs that still
/// have the legacy column.
pub fn migrate_vec_chunks_to_int8_sql(dim: usize) -> String {
    format!(
        r#"
SET lambda_syntax='ENABLE_SINGLE_ARROW';
UPDATE vec_chunks
   SET emb_q8 = CAST(
       list_transform(emb, x -> CAST(ROUND(x * 127) AS TINYINT))
       AS TINYINT[{dim}]
   )
 WHERE emb_q8 IS NULL;
ALTER TABLE vec_chunks DROP COLUMN IF EXISTS emb;
"#,
        dim = dim
    )
}

pub fn vec_chunks_legacy_emb_probe_sql() -> &'static str {
    "SELECT count(*) > 0
       FROM duckdb_columns
      WHERE table_name = 'vec_chunks' AND column_name = 'emb'"
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
