use crate::EMBED_DIM;

/// Idempotent schema. Safe to run on every startup.
pub fn schema_sql() -> String {
    format!(
        r#"
INSTALL fts; LOAD fts;

CREATE TABLE IF NOT EXISTS obs (
    id      BIGINT       PRIMARY KEY,
    ts      BIGINT       NOT NULL,
    session VARCHAR      NOT NULL,
    branch  VARCHAR,
    kind    VARCHAR      NOT NULL,
    body    VARCHAR      NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         VARCHAR PRIMARY KEY,
    started_ts BIGINT,
    ended_ts   BIGINT,
    branch     VARCHAR,
    cwd        VARCHAR,
    summary    VARCHAR
);

-- Embeddings sit alongside obs (1:1 by id). Kept in a separate table so the
-- text table stays narrow and recall over body alone touches less data.
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
"#,
        dim = EMBED_DIM
    )
}

/// FTS index on `obs.body`. Run after the schema is in place. We rebuild the
/// index after batched inserts because DuckDB's FTS has no incremental
/// trigger; for our workload (single-user dogfood) the rebuild is cheap.
pub fn fts_index_sql() -> &'static str {
    "PRAGMA create_fts_index('obs', 'id', 'body', overwrite=1);"
}
