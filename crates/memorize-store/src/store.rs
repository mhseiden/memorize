use crate::DEFAULT_EMBED_DIM;
use crate::fts::FtsIndex;
use crate::schema::{
    backfill_path_tokens_sql, code_chunks_has_body_probe_sql, drop_code_chunk_body_sql,
    migrate_vec_chunks_to_int8_sql, migrate_vec_code_to_int8_sql, schema_sql,
    vec_chunks_legacy_emb_probe_sql, vec_code_legacy_emb_probe_sql,
};
use crate::synonyms_seed::DEFAULT_PAIRS;
use anyhow::{Context, Result, bail};
use duckdb::{Connection, params};
use memorize_core::{Kind, NewObservation, Observation};
use std::path::Path;
use std::sync::{Mutex, OnceLock, RwLock};

/// In-memory int8 vector index. Populated lazily by callers via
/// `Store::enable_vec_cache()`; daemons enable it at startup so vector
/// search runs ~3 ms (Rust dot product) instead of ~900 ms (DuckDB
/// SQL int8 dot product over 192k rows). The eval harness leaves it
/// off — slower per query but no setup cost.
struct VecCache {
    ids: Vec<i64>,
    vecs: Vec<Vec<i8>>,
    /// id → position in the parallel arrays. Used for upsert/delete.
    by_id: std::collections::HashMap<i64, usize>,
}

/// Per-chunk obs vectors. Same data layout as VecCache, but the key is
/// non-unique: an obs has multiple chunks. Search MAX-pools by `obs_id`
/// across all rows belonging to the same obs.
struct ObsVecCache {
    obs_ids: Vec<i64>,
    vecs: Vec<Vec<i8>>,
}

/// All access goes through here. DuckDB connections are not `Sync` (single
/// writer model), so we wrap in a `Mutex`. For a single-user dogfood server
/// the contention is negligible.
pub struct Store {
    /// Read connection. Used by every `search_*` and `get_*` method.
    conn: Mutex<Connection>,
    /// Write connection. Used by every `insert_*` / `upsert_*` / `delete_*`
    /// method, plus the synonym mutators. Cloned from `conn` at open time;
    /// DuckDB MVCC keeps the two coherent. Splitting writes off the read
    /// connection means search queries don't block behind an in-flight
    /// indexer upsert.
    write_conn: Mutex<Connection>,
    embed_dim: usize,
    /// In-memory int8 vector cache for `vec_code`. `OnceLock` so it's
    /// initialized at most once per process; the inner `RwLock` permits
    /// concurrent searches alongside the indexer's upserts.
    vec_cache: OnceLock<RwLock<VecCache>>,
    /// In-memory int8 vector cache for `vec_chunks` (per-chunk obs
    /// embeddings). Same shape as `vec_cache` but the search path
    /// MAX-pools per obs_id since each obs has multiple chunks.
    obs_vec_cache: OnceLock<RwLock<ObsVecCache>>,
    /// In-process BM25 indexes (obs + code). Tantivy-backed, kept in a
    /// `RamDirectory` and rebuilt at `Store::open` from the persisted
    /// `obs.body` / `code_chunks.body` columns. The earlier DuckDB FTS path
    /// crashed when its bg rebuild worker raced the indexer's write
    /// transaction; tantivy's writer is its own concurrency boundary so
    /// route handlers and indexer mutate it directly without a worker.
    fts: FtsIndex,
}

/// BM25 + vector recall results carry the obs id, session (for diversification),
/// and the raw score. The recall pipeline composes both into a unified ranking.
#[derive(Debug, Clone)]
pub struct BM25Hit {
    pub id: i64,
    pub session: String,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct VectorHit {
    pub id: i64,
    pub session: String,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct CodeBM25Hit {
    pub id: i64,
    pub path: String,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct CodeVectorHit {
    pub id: i64,
    pub path: String,
    pub score: f64,
}

/// A chunk as handed to the indexer for insert. Carries `body` because the
/// indexer has the freshly-parsed text in hand and feeds it straight to the
/// FTS index — but the body is no longer persisted to DuckDB.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CodeChunkRow {
    pub id: i64,
    pub path: String,
    pub language: String,
    pub line_start: i32,
    pub line_end: i32,
    pub kind: String,
    pub qualified: String,
    pub body: String,
}

/// A chunk as read back from DuckDB for recall. No `body`: callers
/// reconstruct it by reading `[line_start, line_end]` from the file on disk.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CodeChunkMeta {
    pub id: i64,
    pub path: String,
    pub language: String,
    pub line_start: i32,
    pub line_end: i32,
    pub kind: String,
    pub qualified: String,
}

/// File-level metadata as stored in the `files` table — used by the indexer
/// to decide whether a notify event actually requires reparse.
#[derive(Debug, Clone)]
pub struct FileMeta {
    pub mtime_ns: i64,
    pub size_bytes: i64,
    pub git_rev: Option<String>,
}

impl Store {
    /// Open (creating if needed) a DuckDB at the given path with the default
    /// embedding dim (384, MiniLM). Production code uses this.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_dim(path, DEFAULT_EMBED_DIM)
    }

    /// Open with a specific embedding dim. Use this when the corpus was
    /// embedded by a non-default model. Schema is parameterized so
    /// `vec.emb` is `FLOAT[dim]`.
    pub fn open_with_dim<P: AsRef<Path>>(path: P, dim: usize) -> Result<Self> {
        let conn = Connection::open(path).context("open duckdb")?;
        let write_conn = conn.try_clone().context("clone write conn")?;
        let store = Store {
            conn: Mutex::new(conn),
            write_conn: Mutex::new(write_conn),
            embed_dim: dim,
            vec_cache: OnceLock::new(),
            obs_vec_cache: OnceLock::new(),
            fts: FtsIndex::new().context("create fts index")?,
        };
        store.init()?;
        store.rebuild_fts_from_db()?;
        Ok(store)
    }

    /// Open a DuckDB read-only at the given path with the default embedding
    /// dim. Used by the eval harness so it can ablate modes against a live
    /// daemon's index — DuckDB allows concurrent readers alongside one
    /// writer, so the indexer keeps running while we query. Skips the
    /// `init()` migrations because we'd lack write privileges anyway, and
    /// the live DB has already had its schema applied.
    pub fn open_read_only<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_read_only_with_dim(path, DEFAULT_EMBED_DIM)
    }

    pub fn open_read_only_with_dim<P: AsRef<Path>>(path: P, dim: usize) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            duckdb::Config::default().access_mode(duckdb::AccessMode::ReadOnly)?,
        )
        .context("open duckdb read-only")?;
        let write_conn = conn.try_clone().context("clone write conn")?;
        let store = Store {
            conn: Mutex::new(conn),
            write_conn: Mutex::new(write_conn),
            embed_dim: dim,
            vec_cache: OnceLock::new(),
            obs_vec_cache: OnceLock::new(),
            fts: FtsIndex::new().context("create fts index")?,
        };
        store.rebuild_fts_from_db()?;
        Ok(store)
    }

    /// In-memory store at the default dim — used by store tests.
    pub fn open_in_memory() -> Result<Self> {
        Self::open_in_memory_with_dim(DEFAULT_EMBED_DIM)
    }

    /// In-memory store at a specific dim. Used by the eval harness, which
    /// builds an ephemeral index per question whose vector dim depends on
    /// the active embedder.
    pub fn open_in_memory_with_dim(dim: usize) -> Result<Self> {
        let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
        let write_conn = conn.try_clone().context("clone write conn")?;
        let store = Store {
            conn: Mutex::new(conn),
            write_conn: Mutex::new(write_conn),
            embed_dim: dim,
            vec_cache: OnceLock::new(),
            obs_vec_cache: OnceLock::new(),
            fts: FtsIndex::new().context("create fts index")?,
        };
        store.init()?;
        Ok(store)
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    fn init(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(&schema_sql(self.embed_dim))
            .context("apply schema")?;

        // One-shot int8 vector migration. Only fires on DBs that still have
        // the legacy FLOAT[N] `emb` column on `vec_code`; idempotent on every
        // startup after that.
        let has_legacy_emb: bool = conn
            .query_row(vec_code_legacy_emb_probe_sql(), [], |r| r.get(0))
            .unwrap_or(false);
        if has_legacy_emb {
            eprintln!("memorize-store: migrating vec_code.emb FLOAT→TINYINT (one-time)...");
            let t = std::time::Instant::now();
            conn.execute_batch(&migrate_vec_code_to_int8_sql(self.embed_dim))
                .context("migrate vec_code to int8")?;
            eprintln!(
                "memorize-store: vec_code int8 migration done in {:.1}s",
                t.elapsed().as_secs_f64()
            );
        }

        // Same migration for vec_chunks (obs side). Small at our scale —
        // typically <1k rows — but kept symmetric with vec_code for code
        // hygiene and future-proofing.
        let has_obs_legacy: bool = conn
            .query_row(vec_chunks_legacy_emb_probe_sql(), [], |r| r.get(0))
            .unwrap_or(false);
        if has_obs_legacy {
            eprintln!("memorize-store: migrating vec_chunks.emb FLOAT→TINYINT (one-time)...");
            let t = std::time::Instant::now();
            conn.execute_batch(&migrate_vec_chunks_to_int8_sql(self.embed_dim))
                .context("migrate vec_chunks to int8")?;
            eprintln!(
                "memorize-store: vec_chunks int8 migration done in {:.1}s",
                t.elapsed().as_secs_f64()
            );
        }

        conn.execute_batch(backfill_path_tokens_sql())
            .context("backfill path_tokens")?;

        // One-time: drop the now-dead chunk `body`/`body_hash` columns. Bodies
        // are reconstructed from file line ranges (see `slice_lines`) at
        // recall/FTS-rebuild time; `body_hash` was already write-only. Gated so
        // it only fires on DBs that still have the columns.
        let has_body: bool = conn
            .query_row(code_chunks_has_body_probe_sql(), [], |r| r.get(0))
            .unwrap_or(false);
        if has_body {
            eprintln!("memorize-store: dropping code_chunks.body/body_hash (one-time)...");
            conn.execute_batch(drop_code_chunk_body_sql())
                .context("drop code_chunks body columns")?;
        }
        drop(conn);
        self.seed_synonyms_once()?;
        Ok(())
    }

    /// Stream every row of `obs` and `code_chunks` into the in-process
    /// tantivy index. Called once at `Store::open` time. The corpus is small
    /// (~200k chunks + a few thousand obs), so a cold rebuild costs roughly
    /// the same as the old DuckDB FTS rebuild — but it happens exactly once
    /// per daemon start instead of every ~5 s.
    fn rebuild_fts_from_db(&self) -> Result<()> {
        let t = std::time::Instant::now();
        let conn = self.conn.lock().unwrap();
        let mut n_obs = 0usize;
        {
            let mut stmt = conn.prepare("SELECT id, body FROM obs")?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let id: i64 = row.get(0)?;
                let body: String = row.get(1)?;
                self.fts.insert_obs(id, &body)?;
                n_obs += 1;
            }
        }
        let mut n_code = 0usize;
        {
            // The tantivy `CodeTokenizer` splits camelCase + path separators
            // itself, so we pass the raw `path` as the `path_tokens` field
            // input — the SQL-side pre-split column (`code_chunks.path_tokens`)
            // is no longer load-bearing and is left in place only to avoid a
            // schema migration in this change.
            //
            // Bodies aren't stored, so we reconstruct each chunk's text from
            // the file on disk. `ORDER BY path` groups a file's chunks
            // consecutively, so we read each file at most once and slice every
            // chunk out of that single in-memory copy. This is the startup
            // cost of dropping `body`: a DB scan plus one read per indexed
            // file. A file that changed (or vanished) since indexing yields
            // wrong/empty text here — transient, since the cold-scan that runs
            // right after reindexes changed files and corrects the FTS.
            let mut stmt = conn.prepare(
                "SELECT id, path, language, line_start, line_end, qualified
                   FROM code_chunks ORDER BY path",
            )?;
            let mut rows = stmt.query([])?;
            let mut cur_path: Option<String> = None;
            let mut cur_source: Option<String> = None;
            while let Some(row) = rows.next()? {
                let id: i64 = row.get(0)?;
                let path: String = row.get(1)?;
                let language: String = row.get(2)?;
                let line_start: i32 = row.get(3)?;
                let line_end: i32 = row.get(4)?;
                let qualified: String = row.get(5)?;
                if cur_path.as_deref() != Some(path.as_str()) {
                    cur_source = std::fs::read_to_string(&path).ok();
                    cur_path = Some(path.clone());
                }
                let body = match &cur_source {
                    Some(src) => {
                        memorize_core::slice_lines(src, line_start as u32, line_end as u32)
                    }
                    None => String::new(),
                };
                self.fts
                    .insert_code(id, &path, &language, &body, &qualified, &path)?;
                n_code += 1;
            }
        }
        drop(conn);
        self.fts.commit()?;
        eprintln!(
            "memorize-store: fts rebuilt in {:.1}s ({} obs, {} code chunks)",
            t.elapsed().as_secs_f64(),
            n_obs,
            n_code
        );
        Ok(())
    }

    fn seed_synonyms_once(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let seeded: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'synonyms_seeded'",
                [],
                |row| row.get(0),
            )
            .ok();
        if seeded.is_some() {
            return Ok(());
        }
        for (a, b) in DEFAULT_PAIRS {
            conn.execute(
                "INSERT OR IGNORE INTO synonyms(term, expansion) VALUES (?, ?), (?, ?)",
                params![a, b, b, a],
            )?;
        }
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('synonyms_seeded', '1')",
            [],
        )?;
        Ok(())
    }

    /// Convenience wrapper for the single-vector case used by tests.
    pub fn insert_obs(&self, obs: &NewObservation, ts: i64, emb: &[f32]) -> Result<i64> {
        self.insert_obs_chunked(obs, ts, std::slice::from_ref(&emb))
    }

    /// Insert with N chunk embeddings (the production path). The obs body is
    /// stored once in `obs.body` for BM25; each chunk embedding lands in
    /// `vec_chunks`. Callers (the server) are responsible for chunking the
    /// body and producing matching embeddings.
    pub fn insert_obs_chunked(
        &self,
        obs: &NewObservation,
        ts: i64,
        chunk_embs: &[&[f32]],
    ) -> Result<i64> {
        if chunk_embs.is_empty() {
            bail!("insert_obs_chunked requires at least one chunk embedding");
        }
        for (i, emb) in chunk_embs.iter().enumerate() {
            if emb.len() != self.embed_dim {
                bail!(
                    "chunk {} embedding length {} doesn't match store embed_dim={}",
                    i,
                    emb.len(),
                    self.embed_dim
                );
            }
        }
        let conn = self.write_conn.lock().unwrap();
        let id: i64 = conn.query_row(
            "INSERT INTO obs(id, ts, session, branch, kind, body,
                             ref_path, ref_line_start, ref_line_end)
             VALUES (
                 COALESCE((SELECT MAX(id) FROM obs), 0) + 1,
                 ?, ?, ?, ?, ?, ?, ?, ?
             ) RETURNING id",
            params![
                ts,
                obs.session,
                obs.branch,
                obs.kind.as_str(),
                obs.body,
                obs.ref_path,
                obs.ref_line_start,
                obs.ref_line_end,
            ],
            |row| row.get(0),
        )?;
        let dim = self.embed_dim;
        // One INSERT per chunk. Bounded by chunk count — typically 1–5.
        let mut q8_chunks: Vec<Vec<i8>> = Vec::with_capacity(chunk_embs.len());
        for (idx, emb) in chunk_embs.iter().enumerate() {
            let q8 = quantize_i8(emb);
            let literal = i8_array_literal(&q8);
            let sql = format!(
                "INSERT INTO vec_chunks(obs_id, chunk_idx, emb_q8)
                 VALUES ({id}, {idx}, {literal}::TINYINT[{dim}])"
            );
            conn.execute_batch(&sql)?;
            q8_chunks.push(q8);
        }
        drop(conn);
        // Update the in-memory obs vector cache, if enabled.
        if let Some(cache_lock) = self.obs_vec_cache.get() {
            let mut cache = cache_lock.write().unwrap();
            for q8 in q8_chunks {
                cache.push(id, q8);
            }
        }
        // Index in tantivy and commit. Tantivy's writer is its own concurrency
        // boundary, so we don't need to defer the commit to a background
        // worker — but the commit *does* invalidate readers until reload,
        // so other writers shouldn't interleave between add and commit. The
        // writer mutex inside FtsIndex handles that.
        self.fts.insert_obs(id, &obs.body)?;
        self.fts.commit()?;
        Ok(id)
    }

    /// Run BM25 over obs.body. The caller is responsible for synonym
    /// expansion; `query` here is the expanded form. The tokenizer used by
    /// the obs index (`en_stem`) does the same lowercase + Snowball stemming
    /// that DuckDB FTS did, so query and document tokens match.
    pub fn search_bm25(&self, query: &str, limit: usize) -> Result<Vec<BM25Hit>> {
        let hits = self.fts.search_obs(query, limit)?;
        if hits.is_empty() {
            return Ok(vec![]);
        }
        // Hydrate session from DuckDB. The fusion pipeline downstream
        // diversifies by session, so we can't skip it.
        let ids: Vec<i64> = hits.iter().map(|h| h.id).collect();
        let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!("SELECT id, session FROM obs WHERE id IN ({placeholders})");
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn duckdb::ToSql> =
            ids.iter().map(|id| id as &dyn duckdb::ToSql).collect();
        let sessions: std::collections::HashMap<i64, String> = stmt
            .query_map(params.as_slice(), |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .collect();
        Ok(hits
            .into_iter()
            .filter_map(|h| {
                let session = sessions.get(&h.id)?.clone();
                Some(BM25Hit {
                    id: h.id,
                    session,
                    score: h.score,
                })
            })
            .collect())
    }

    /// Cosine over per-chunk embeddings. Each chunk is scored independently;
    /// the obs takes its best-matching chunk's score (max-pool). This is the
    /// IR-literature recommended multi-vector retrieval pattern — if just one
    /// chunk of a long observation is on-topic, that chunk surfaces the obs
    /// rather than getting averaged-out across irrelevant chunks.
    pub fn search_vector(&self, query_emb: &[f32], limit: usize) -> Result<Vec<VectorHit>> {
        if query_emb.len() != self.embed_dim {
            bail!(
                "query embedding length {} doesn't match store embed_dim={}",
                query_emb.len(),
                self.embed_dim
            );
        }
        let q8 = quantize_i8(query_emb);

        // Fast path: in-memory obs cache.
        if let Some(cache_lock) = self.obs_vec_cache.get() {
            return self.search_obs_vector_cached(cache_lock, &q8, limit);
        }

        // Slow path: SQL int8 dot product, max-pooled per obs_id.
        let conn = self.conn.lock().unwrap();
        let literal = i8_array_literal(&q8);
        let dim = self.embed_dim;
        let sql = format!(
            "SELECT obs_id, session, score FROM (
                SELECT vc.obs_id AS obs_id,
                       o.session AS session,
                       MAX(
                           list_sum(
                               list_transform(
                                   list_zip(vc.emb_q8, {literal}::TINYINT[{dim}]),
                                   pair -> pair[1]::INTEGER * pair[2]::INTEGER
                               )
                           )::DOUBLE
                       ) AS score
                  FROM vec_chunks vc
                  JOIN obs o ON o.id = vc.obs_id
              GROUP BY vc.obs_id, o.session
             ) sub
            ORDER BY score DESC
            LIMIT ?"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![limit as i64], |row| {
                Ok(VectorHit {
                    id: row.get(0)?,
                    session: row.get(1)?,
                    score: row.get(2)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// In-memory obs vector search. Each obs has multiple chunks; we
    /// MAX-pool the dot product across them so the obs surfaces if any
    /// of its chunks is on-topic. Sessions are hydrated for the top-K
    /// results.
    fn search_obs_vector_cached(
        &self,
        cache_lock: &RwLock<ObsVecCache>,
        q8: &[i8],
        limit: usize,
    ) -> Result<Vec<VectorHit>> {
        let cache = cache_lock.read().unwrap();
        // MAX-pool by obs_id over the in-memory chunk vectors.
        let mut best: std::collections::HashMap<i64, i32> = std::collections::HashMap::new();
        for (oid, v) in cache.obs_ids.iter().zip(&cache.vecs) {
            let s = dot_i8(q8, v);
            let entry = best.entry(*oid).or_insert(i32::MIN);
            if s > *entry {
                *entry = s;
            }
        }
        drop(cache);
        let mut scored: Vec<(i64, i32)> = best.into_iter().collect();
        scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(limit);

        // Hydrate session for each obs_id in top-K.
        let ids: Vec<i64> = scored.iter().map(|(id, _)| *id).collect();
        let by_id_score: std::collections::HashMap<i64, i32> =
            scored.iter().map(|(id, s)| (*id, *s)).collect();
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!("SELECT id, session FROM obs WHERE id IN ({placeholders})");
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn duckdb::ToSql> =
            ids.iter().map(|id| id as &dyn duckdb::ToSql).collect();
        let mut hits: Vec<VectorHit> = stmt
            .query_map(params.as_slice(), |row| {
                let id: i64 = row.get(0)?;
                let session: String = row.get(1)?;
                Ok((id, session))
            })?
            .filter_map(|r| r.ok())
            .map(|(id, session)| VectorHit {
                id,
                session,
                score: by_id_score.get(&id).copied().unwrap_or(0) as f64,
            })
            .collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(hits)
    }

    /// Hydrate full observation records for a set of ids, preserving the
    /// input order. Used after RRF fusion to materialize the final ranked list.
    pub fn get_obs_by_ids(&self, ids: &[i64]) -> Result<Vec<Observation>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, ts, session, branch, kind, body,
                    ref_path, ref_line_start, ref_line_end
               FROM obs WHERE id IN ({placeholders})"
        );
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn duckdb::ToSql> =
            ids.iter().map(|id| id as &dyn duckdb::ToSql).collect();
        let rows = stmt
            .query_map(params.as_slice(), |row| {
                let kind: String = row.get(4)?;
                Ok(Observation {
                    id: row.get(0)?,
                    ts: row.get(1)?,
                    session: row.get(2)?,
                    branch: row.get(3)?,
                    kind: Kind::from_str(&kind),
                    body: row.get(5)?,
                    ref_path: row.get(6)?,
                    ref_line_start: row.get(7)?,
                    ref_line_end: row.get(8)?,
                })
            })?
            .collect::<Result<Vec<Observation>, _>>()?;
        // Re-sort to caller's id order — IN-clause doesn't preserve it.
        let mut by_id: std::collections::HashMap<i64, Observation> =
            rows.into_iter().map(|o| (o.id, o)).collect();
        Ok(ids.iter().filter_map(|id| by_id.remove(id)).collect())
    }

    /// SQL-side synonym expansion: takes a set of normalized query tokens,
    /// returns each token plus every recorded expansion (deduped). The result
    /// is intended to be space-joined and handed to FTS.
    pub fn expand_synonyms(&self, tokens: &[String]) -> Result<Vec<String>> {
        if tokens.is_empty() {
            return Ok(vec![]);
        }
        let placeholders: String = tokens.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT DISTINCT expansion FROM synonyms WHERE term IN ({placeholders})"
        );
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn duckdb::ToSql> =
            tokens.iter().map(|t| t as &dyn duckdb::ToSql).collect();
        let extra: Vec<String> = stmt
            .query_map(params.as_slice(), |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        let mut out: Vec<String> = tokens.to_vec();
        for e in extra {
            if !out.iter().any(|t| t.eq_ignore_ascii_case(&e)) {
                out.push(e);
            }
        }
        Ok(out)
    }

    pub fn add_synonym(&self, term: &str, expansion: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        // Bidirectional: both directions live as rows.
        conn.execute(
            "INSERT OR IGNORE INTO synonyms(term, expansion) VALUES (?, ?), (?, ?)",
            params![term, expansion, expansion, term],
        )?;
        Ok(())
    }

    pub fn remove_synonym(&self, term: &str, expansion: Option<&str>) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let deleted = match expansion {
            Some(exp) => conn.execute(
                "DELETE FROM synonyms
                  WHERE (term = ? AND expansion = ?)
                     OR (term = ? AND expansion = ?)",
                params![term, exp, exp, term],
            )?,
            None => conn.execute(
                "DELETE FROM synonyms WHERE term = ? OR expansion = ?",
                params![term, term],
            )?,
        };
        Ok(deleted)
    }

    pub fn list_synonyms(&self) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT term, expansion FROM synonyms ORDER BY term, expansion",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Compatibility shim for the eval harness, which historically called
    /// this between bulk inserts and queries. Tantivy already commits inside
    /// `insert_obs_chunked` / `upsert_code_file`, so this is now a no-op —
    /// we keep it so the harness's timing breakdown can still measure
    /// "rebuild time" (which trivially reports ~0 ms post-migration).
    pub fn rebuild_fts(&self) -> Result<()> {
        Ok(())
    }

    pub fn count_obs(&self) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row("SELECT COUNT(*) FROM obs", [], |r| r.get(0))?)
    }

    /// Drop observations older than `cutoff_ts`. Returns rows removed.
    /// Orphaned `vec_chunks` rows are cleaned in the same pass.
    pub fn evict_older_than(&self, cutoff_ts: i64) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let deleted = conn.execute("DELETE FROM obs WHERE ts < ?", params![cutoff_ts])?;
        conn.execute(
            "DELETE FROM vec_chunks WHERE obs_id NOT IN (SELECT id FROM obs)",
            [],
        )?;
        Ok(deleted)
    }

    // ---- Code index methods ----

    /// Look up current file metadata. None if we haven't indexed this path.
    pub fn get_file_meta(&self, path: &str) -> Result<Option<FileMeta>> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT mtime_ns, size_bytes, git_rev FROM files WHERE path = ?",
                params![path],
                |row| {
                    Ok(FileMeta {
                        mtime_ns: row.get(0)?,
                        size_bytes: row.get(1)?,
                        git_rev: row.get(2)?,
                    })
                },
            )
            .ok();
        Ok(row)
    }

    /// Upsert files row + replace all code_chunks for this path with the
    /// supplied set. Embeddings are inserted in lockstep via vec_code.
    /// Caller is responsible for matching chunks.len() == chunk_embs.len().
    pub fn upsert_code_file(
        &self,
        path: &str,
        repo_root: &str,
        language: &str,
        meta: &FileMeta,
        chunks: &[CodeChunkRow],
        chunk_embs: &[Vec<f32>],
    ) -> Result<()> {
        if chunks.len() != chunk_embs.len() {
            bail!(
                "chunks ({}) and chunk_embs ({}) length mismatch",
                chunks.len(),
                chunk_embs.len()
            );
        }
        for (i, emb) in chunk_embs.iter().enumerate() {
            if emb.len() != self.embed_dim {
                bail!(
                    "chunk {} embedding length {} doesn't match store embed_dim={}",
                    i,
                    emb.len(),
                    self.embed_dim
                );
            }
        }
        let conn = self.write_conn.lock().unwrap();

        // Atomic: delete stale chunks, upsert files row, insert new chunks
        // all in one transaction. Without this a crash mid-write would leave
        // the files row pointing at a fresh mtime/size while the chunk set
        // is partial — on restart `get_file_meta` would falsely report
        // "already indexed" and skip the file forever.
        conn.execute_batch("BEGIN")?;
        // We collect stale_ids + new (id, q8) pairs inside the transaction so
        // we can both apply them to DuckDB AND, on COMMIT, push the same
        // mutation into the in-memory vec cache. Without that the cache
        // would drift from the on-disk truth on every file change.
        let work: Result<(Vec<i64>, Vec<(i64, Vec<i8>)>, Vec<(i64, String, String)>)> = (|| {
            let stale_ids: Vec<i64> = {
                let mut stmt =
                    conn.prepare("SELECT id FROM code_chunks WHERE path = ?")?;
                let rows = stmt.query_map(params![path], |row| row.get::<_, i64>(0))?;
                rows.collect::<std::result::Result<Vec<_>, _>>()?
            };
            if !stale_ids.is_empty() {
                let placeholders: String =
                    stale_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let del_chunks =
                    format!("DELETE FROM code_chunks WHERE id IN ({placeholders})");
                let del_vec =
                    format!("DELETE FROM vec_code WHERE id IN ({placeholders})");
                let params_vec: Vec<&dyn duckdb::ToSql> =
                    stale_ids.iter().map(|id| id as &dyn duckdb::ToSql).collect();
                conn.execute(&del_chunks, params_vec.as_slice())?;
                conn.execute(&del_vec, params_vec.as_slice())?;
            }

            conn.execute(
                "INSERT INTO files(path, repo_root, language, mtime_ns, size_bytes, git_rev)
                      VALUES (?, ?, ?, ?, ?, ?)
                 ON CONFLICT(path) DO UPDATE SET
                    repo_root = EXCLUDED.repo_root,
                    language = EXCLUDED.language,
                    mtime_ns = EXCLUDED.mtime_ns,
                    size_bytes = EXCLUDED.size_bytes,
                    git_rev = EXCLUDED.git_rev",
                params![
                    path,
                    repo_root,
                    language,
                    meta.mtime_ns,
                    meta.size_bytes,
                    meta.git_rev
                ],
            )?;

            let base: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(id), 0) FROM code_chunks",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            let dim = self.embed_dim;
            let mut new_pairs: Vec<(i64, Vec<i8>)> = Vec::with_capacity(chunks.len());
            // Capture what we need to feed tantivy after COMMIT. We don't
            // touch the index until the DuckDB tx is durable so the two
            // can't diverge if commit fails.
            let mut fts_inserts: Vec<(i64, String, String)> = Vec::with_capacity(chunks.len());
            for (i, c) in chunks.iter().enumerate() {
                let id = base + 1 + i as i64;
                conn.execute(
                    "INSERT INTO code_chunks(id, path, language, line_start, line_end,
                                             kind, qualified, path_tokens)
                     VALUES (?, ?, ?, ?, ?, ?, ?,
                             lower(
                                 regexp_replace(?, '[/_.\\-]', ' ', 'g')
                                 || ' '
                                 || regexp_replace(?, '([a-z])([A-Z])', '\\1 \\2', 'g')
                             ))",
                    params![
                        id,
                        path,
                        c.language,
                        c.line_start,
                        c.line_end,
                        c.kind,
                        c.qualified,
                        path,
                        c.qualified,
                    ],
                )?;
                let q8 = quantize_i8(&chunk_embs[i]);
                let literal = i8_array_literal(&q8);
                let insert_vec = format!(
                    "INSERT INTO vec_code(id, emb_q8) VALUES ({id}, {literal}::TINYINT[{dim}])"
                );
                conn.execute_batch(&insert_vec)?;
                new_pairs.push((id, q8));
                fts_inserts.push((id, c.body.clone(), c.qualified.clone()));
            }
            Ok((stale_ids, new_pairs, fts_inserts))
        })();

        match work {
            Ok((stale_ids, new_pairs, fts_inserts)) => {
                conn.execute_batch("COMMIT")?;
                drop(conn);
                // Apply the same mutations to the in-memory cache, if enabled.
                if let Some(cache_lock) = self.vec_cache.get() {
                    let mut cache = cache_lock.write().unwrap();
                    for id in &stale_ids {
                        cache.remove(*id);
                    }
                    for (id, q8) in new_pairs {
                        cache.insert(id, q8);
                    }
                }
                // Sync tantivy: drop stale chunk ids, add new ones, commit
                // once at the end so we don't reload the searcher between
                // every chunk.
                for id in &stale_ids {
                    self.fts.delete_code(*id)?;
                }
                for (id, body, qualified) in fts_inserts {
                    self.fts
                        .insert_code(id, path, language, &body, &qualified, path)?;
                }
                self.fts.commit()?;
                Ok(())
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    /// Read the embed-model identity persisted in the `meta` table. Returns
    /// `None` if no value has been stamped yet (fresh DB, or pre-tagging
    /// upgrade). Callers compare this against the binary's current
    /// `memorize_embed::model_tag()` and refuse to start if they disagree
    /// while vectors are present.
    pub fn stored_model_tag(&self) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let row: std::result::Result<String, duckdb::Error> = conn.query_row(
            "SELECT value FROM meta WHERE key = 'embed_model_tag'",
            [],
            |r| r.get(0),
        );
        match row {
            Ok(s) => Ok(Some(s)),
            Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Persist the embed-model identity into the `meta` table. Overwrites
    /// any existing value — caller is responsible for asserting that the
    /// vector tables are empty when changing models.
    pub fn set_model_tag(&self, tag: &str) -> Result<()> {
        let conn = self.write_conn.lock().unwrap();
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('embed_model_tag', ?)
             ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
            params![tag],
        )?;
        Ok(())
    }

    /// Drop every code-index table (files, code_chunks, vec_code) and the
    /// obs-side vector table (vec_chunks). Used by `memorize reindex` before
    /// switching embed models. Leaves `obs` and `sessions` rows intact —
    /// session memory text is preserved even though its vectors are gone,
    /// and the watcher hooks will re-embed new captures with the new model.
    pub fn wipe_code_index(&self) -> Result<()> {
        let conn = self.write_conn.lock().unwrap();
        conn.execute_batch(
            "DELETE FROM vec_code;
             DELETE FROM code_chunks;
             DELETE FROM files;
             DELETE FROM vec_chunks;",
        )?;
        drop(conn);
        if let Some(cache_lock) = self.vec_cache.get() {
            let mut cache = cache_lock.write().unwrap();
            cache.ids.clear();
            cache.vecs.clear();
            cache.by_id.clear();
        }
        if let Some(cache_lock) = self.obs_vec_cache.get() {
            let mut cache = cache_lock.write().unwrap();
            cache.obs_ids.clear();
            cache.vecs.clear();
        }
        self.fts.clear_code()?;
        self.fts.commit()?;
        Ok(())
    }

    /// Drop all chunks + vec rows for a deleted file.
    pub fn delete_code_file(&self, path: &str) -> Result<()> {
        let conn = self.write_conn.lock().unwrap();
        let stale_ids: Vec<i64> = {
            let mut stmt =
                conn.prepare("SELECT id FROM code_chunks WHERE path = ?")?;
            let rows = stmt.query_map(params![path], |row| row.get::<_, i64>(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        if !stale_ids.is_empty() {
            let placeholders: String =
                stale_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let del_chunks = format!("DELETE FROM code_chunks WHERE id IN ({placeholders})");
            let del_vec = format!("DELETE FROM vec_code WHERE id IN ({placeholders})");
            let params_vec: Vec<&dyn duckdb::ToSql> =
                stale_ids.iter().map(|id| id as &dyn duckdb::ToSql).collect();
            conn.execute(&del_chunks, params_vec.as_slice())?;
            conn.execute(&del_vec, params_vec.as_slice())?;
        }
        conn.execute("DELETE FROM files WHERE path = ?", params![path])?;
        drop(conn);
        // Mirror the removal in the in-memory cache.
        if let Some(cache_lock) = self.vec_cache.get() {
            let mut cache = cache_lock.write().unwrap();
            for id in &stale_ids {
                cache.remove(*id);
            }
        }
        if !stale_ids.is_empty() {
            for id in &stale_ids {
                self.fts.delete_code(*id)?;
            }
            self.fts.commit()?;
        }
        Ok(())
    }

    /// Stream every (id, int8 embedding) tuple from `vec_code` to `callback`.
    /// Used by both the in-memory cache loader and by the eval's int8 PoC.
    ///
    /// We use `array_to_string` because DuckDB's Rust binding doesn't
    /// implement `FromSql for Vec<i8>` (or Vec<f32>) directly. The string
    /// path is fine for one-time bulk reads: 192k × 1.5 KB = ~290 MB
    /// transferred + parsed in ~5 s.
    pub fn for_each_code_vector(
        &self,
        mut callback: impl FnMut(i64, &[i8]) -> Result<()>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, array_to_string(emb_q8, ',') AS emb_str FROM vec_code",
        )?;
        let mut rows = stmt.query([])?;
        let mut buf: Vec<i8> = Vec::with_capacity(self.embed_dim);
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let s: String = row.get(1)?;
            buf.clear();
            for tok in s.split(',') {
                let v: i8 = tok
                    .parse()
                    .with_context(|| format!("parse emb_q8 component '{tok}'"))?;
                buf.push(v);
            }
            callback(id, &buf)?;
        }
        Ok(())
    }

    /// Sample N chunks at random, returning `(path, qualified)`. Used by the
    /// eval harness to build a synthetic query bank from the live index.
    /// Filters to rows with a non-trivial `qualified` so the symbol token
    /// is usable as a query.
    pub fn sample_code_chunks(&self, n: usize) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock().unwrap();
        // SAMPLE wants a constant literal, not a parameter. Validate the
        // count so we don't get a SQL injection vector here.
        let n = n.min(1_000_000);
        let sql = format!(
            "SELECT path, qualified
               FROM code_chunks
              WHERE qualified IS NOT NULL
                AND length(qualified) >= 6
                AND qualified NOT LIKE '%#part-%'
              USING SAMPLE {n} ROWS"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn count_code_chunks(&self) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row("SELECT COUNT(*) FROM code_chunks", [], |r| r.get(0))?)
    }

    /// BM25 over `code_chunks.body` + `qualified` + path tokens. The tantivy
    /// `CodeTokenizer` does the camelCase + path-separator split on both the
    /// indexed text and the query string, so `irMemo/memo.ts` matches a doc
    /// at `packages/.../irMemo/memo.ts` without any caller-side preprocessing.
    pub fn search_code_bm25(
        &self,
        query: &str,
        limit: usize,
        language: Option<&str>,
        path_prefix: Option<&str>,
    ) -> Result<Vec<CodeBM25Hit>> {
        let hits = self.fts.search_code(query, limit, language, path_prefix)?;
        Ok(hits
            .into_iter()
            .map(|h| CodeBM25Hit {
                id: h.id,
                path: h.path,
                score: h.score,
            })
            .collect())
    }

    /// Vector search over `vec_code.emb_q8`. If the in-memory cache is
    /// enabled (`enable_vec_cache`), the dot product runs in Rust over
    /// the packed int8 index (~3 ms for 192k vectors). Otherwise we fall
    /// back to a SQL int8 dot product via `list_reduce(list_zip(...))`
    /// (~900 ms at the same scale — correct, but only suitable for the
    /// eval harness and one-off tools).
    pub fn search_code_vector(
        &self,
        query_emb: &[f32],
        limit: usize,
        language: Option<&str>,
        path_prefix: Option<&str>,
    ) -> Result<Vec<CodeVectorHit>> {
        if query_emb.len() != self.embed_dim {
            bail!(
                "query embedding length {} doesn't match store embed_dim={}",
                query_emb.len(),
                self.embed_dim
            );
        }
        let q8 = quantize_i8(query_emb);

        // Fast path: in-memory cache.
        if let Some(cache_lock) = self.vec_cache.get() {
            return self.search_code_vector_cached(cache_lock, &q8, limit, language, path_prefix);
        }

        // Slow path: SQL int8 dot product over the live table. Each row
        // computes `sum(emb_q8[i] * query[i])` via list_zip + list_transform
        // + list_sum.
        let conn = self.conn.lock().unwrap();
        let literal = i8_array_literal(&q8);
        let dim = self.embed_dim;
        let mut sql = format!(
            "SELECT cc.id, cc.path,
                    list_sum(
                        list_transform(
                            list_zip(vc.emb_q8, {literal}::TINYINT[{dim}]),
                            pair -> pair[1]::INTEGER * pair[2]::INTEGER
                        )
                    )::DOUBLE AS s
               FROM vec_code vc
               JOIN code_chunks cc ON cc.id = vc.id
              WHERE 1=1"
        );
        let mut params_vec: Vec<Box<dyn duckdb::ToSql>> = vec![];
        if let Some(lang) = language {
            sql.push_str(" AND cc.language = ?");
            params_vec.push(Box::new(lang.to_string()));
        }
        if let Some(prefix) = path_prefix {
            sql.push_str(" AND cc.path LIKE ?");
            params_vec.push(Box::new(format!("{prefix}%")));
        }
        sql.push_str(" ORDER BY s DESC LIMIT ?");
        params_vec.push(Box::new(limit as i64));
        let param_refs: Vec<&dyn duckdb::ToSql> =
            params_vec.iter().map(|b| b.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok(CodeVectorHit {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    score: row.get(2)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// In-memory cached vector search. The cache holds (id, [i8; 384]) for
    /// every chunk; dot product is a hot loop the optimizer autovectorizes.
    /// Language and path filters apply after the top-K scan (cheap because
    /// we hydrate paths for the top results only).
    fn search_code_vector_cached(
        &self,
        cache_lock: &RwLock<VecCache>,
        q8: &[i8],
        limit: usize,
        language: Option<&str>,
        path_prefix: Option<&str>,
    ) -> Result<Vec<CodeVectorHit>> {
        let cache = cache_lock.read().unwrap();
        // Score everything, take top `limit * 5` candidates (room for filters).
        let pool_size = (limit * 5).max(limit);
        let mut scored: Vec<(i64, i32)> = cache
            .ids
            .iter()
            .zip(&cache.vecs)
            .map(|(id, v)| (*id, dot_i8(q8, v)))
            .collect();
        // Partial-sort: full sort_unstable for simplicity at our scale; could
        // switch to select_nth_unstable if profiling demands.
        scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(pool_size.min(scored.len()));
        drop(cache);

        // Hydrate path + language for the candidates, apply filters, take top N.
        let ids: Vec<i64> = scored.iter().map(|(id, _)| *id).collect();
        let by_id_score: std::collections::HashMap<i64, i32> =
            scored.iter().map(|(id, s)| (*id, *s)).collect();
        let rows = self.get_code_chunks_by_ids(&ids)?;
        let mut hits: Vec<CodeVectorHit> = rows
            .into_iter()
            .filter(|r| language.map(|l| r.language == l).unwrap_or(true))
            .filter(|r| path_prefix.map(|p| r.path.starts_with(p)).unwrap_or(true))
            .map(|r| {
                let s = by_id_score.get(&r.id).copied().unwrap_or(0);
                CodeVectorHit {
                    id: r.id,
                    path: r.path,
                    score: s as f64,
                }
            })
            .collect();
        // Re-sort by score after the get_code_chunks_by_ids hydration shuffles order.
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(limit);
        Ok(hits)
    }

    /// Build the in-memory int8 vector caches (code-side `vec_code` and
    /// obs-side `vec_chunks`). Daemons call this once at startup after
    /// `Store::open`. Idempotent — second call is a no-op. ~5 s for 192k
    /// code vectors on the live DB; the obs cache is small in comparison.
    pub fn enable_vec_cache(&self) -> Result<()> {
        if self.vec_cache.get().is_none() {
            let t = std::time::Instant::now();
            let mut ids: Vec<i64> = Vec::new();
            let mut vecs: Vec<Vec<i8>> = Vec::new();
            self.for_each_code_vector(|id, q8| {
                ids.push(id);
                vecs.push(q8.to_vec());
                Ok(())
            })?;
            let mut by_id: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
            for (pos, id) in ids.iter().enumerate() {
                by_id.insert(*id, pos);
            }
            let n = ids.len();
            let cache = VecCache { ids, vecs, by_id };
            let _ = self.vec_cache.set(RwLock::new(cache));
            eprintln!(
                "memorize-store: vec_cache (code) loaded ({} vectors in {:.1}s)",
                n,
                t.elapsed().as_secs_f64()
            );
        }
        if self.obs_vec_cache.get().is_none() {
            let t = std::time::Instant::now();
            let mut obs_ids: Vec<i64> = Vec::new();
            let mut vecs: Vec<Vec<i8>> = Vec::new();
            self.for_each_obs_vector(|oid, q8| {
                obs_ids.push(oid);
                vecs.push(q8.to_vec());
                Ok(())
            })?;
            let n = obs_ids.len();
            let cache = ObsVecCache { obs_ids, vecs };
            let _ = self.obs_vec_cache.set(RwLock::new(cache));
            eprintln!(
                "memorize-store: obs_vec_cache loaded ({} vectors in {:.1}s)",
                n,
                t.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }

    /// Stream every (obs_id, int8 embedding) tuple from `vec_chunks` to
    /// `callback`. Symmetric with `for_each_code_vector`.
    pub fn for_each_obs_vector(
        &self,
        mut callback: impl FnMut(i64, &[i8]) -> Result<()>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT obs_id, array_to_string(emb_q8, ',') AS emb_str FROM vec_chunks",
        )?;
        let mut rows = stmt.query([])?;
        let mut buf: Vec<i8> = Vec::with_capacity(self.embed_dim);
        while let Some(row) = rows.next()? {
            let obs_id: i64 = row.get(0)?;
            let s: String = row.get(1)?;
            buf.clear();
            for tok in s.split(',') {
                let v: i8 = tok
                    .parse()
                    .with_context(|| format!("parse emb_q8 component '{tok}'"))?;
                buf.push(v);
            }
            callback(obs_id, &buf)?;
        }
        Ok(())
    }

    pub fn get_code_chunks_by_ids(&self, ids: &[i64]) -> Result<Vec<CodeChunkMeta>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, path, language, line_start, line_end, kind, qualified
               FROM code_chunks WHERE id IN ({placeholders})"
        );
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn duckdb::ToSql> =
            ids.iter().map(|id| id as &dyn duckdb::ToSql).collect();
        let mut by_id: std::collections::HashMap<i64, CodeChunkMeta> = stmt
            .query_map(params.as_slice(), |row| {
                Ok(CodeChunkMeta {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    language: row.get(2)?,
                    line_start: row.get(3)?,
                    line_end: row.get(4)?,
                    kind: row.get(5)?,
                    qualified: row.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|c| (c.id, c))
            .collect();
        Ok(ids.iter().filter_map(|id| by_id.remove(id)).collect())
    }

    // ---- end code index methods ----

    /// Session register / upsert. Called from SessionStart hook.
    pub fn upsert_session(
        &self,
        id: &str,
        started_ts: i64,
        branch: Option<&str>,
        cwd: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO sessions(id, started_ts, branch, cwd)
                  VALUES (?, ?, ?, ?)
             ON CONFLICT(id) DO UPDATE
                SET branch = COALESCE(EXCLUDED.branch, sessions.branch),
                    cwd = COALESCE(EXCLUDED.cwd, sessions.cwd)",
            params![id, started_ts, branch, cwd],
        )?;
        Ok(())
    }
}

/// `[0.12, -0.34, ...]` — SQL literal injection of our own embeddings.
/// Safe because the input is `&[f32]`, not anything user-controlled.
/// `{:e}` always emits a decimal point or exponent, so DuckDB parses each as
/// a float rather than an int.
#[allow(dead_code)]
fn float_array_literal(v: &[f32]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(v.len() * 14);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "{:e}", x);
    }
    s.push(']');
    s
}

/// Format a TINYINT array as a DuckDB list literal `[1,-2,3,...]`. Used
/// for inline SQL binding of int8 vectors at insert/search time.
fn i8_array_literal(v: &[i8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(v.len() * 5);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "{x}");
    }
    s.push(']');
    s
}

/// Quantize a unit-norm f32 embedding to i8. Each component in `[-1, 1]`
/// maps to `round(v * 127)`, saturated to `[-127, 127]` so the result
/// fits in `i8`.
pub fn quantize_i8(emb: &[f32]) -> Vec<i8> {
    emb.iter()
        .map(|&v| (v * 127.0).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// i8 dot product. Autovectorizes on NEON/AVX2 — ~30 ns per 384-d call.
#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    // Defensive: handle dimension mismatch by zipping shortest. The
    // production path always matches `embed_dim` on both sides.
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as i32) * (*y as i32))
        .sum()
}

impl VecCache {
    fn insert(&mut self, id: i64, vec: Vec<i8>) {
        if let Some(&pos) = self.by_id.get(&id) {
            self.vecs[pos] = vec;
            return;
        }
        let pos = self.ids.len();
        self.ids.push(id);
        self.vecs.push(vec);
        self.by_id.insert(id, pos);
    }

    fn remove(&mut self, id: i64) {
        // Swap-remove keeps the parallel arrays compact. Update the
        // displaced element's by_id entry.
        if let Some(pos) = self.by_id.remove(&id) {
            let last = self.ids.len() - 1;
            self.ids.swap_remove(pos);
            self.vecs.swap_remove(pos);
            if pos != last {
                let moved_id = self.ids[pos];
                self.by_id.insert(moved_id, pos);
            }
        }
    }
}

impl ObsVecCache {
    /// Append one chunk vector for the given obs. No by_id index — obs
    /// have multiple chunks and we always scan the full vector at search
    /// time anyway. Removal isn't currently used (obs eviction is rare).
    fn push(&mut self, obs_id: i64, vec: Vec<i8>) {
        self.obs_ids.push(obs_id);
        self.vecs.push(vec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memorize_core::Kind;

    fn dummy_emb(scale: f32) -> Vec<f32> {
        (0..DEFAULT_EMBED_DIM).map(|i| (i as f32 / DEFAULT_EMBED_DIM as f32) * scale).collect()
    }

    #[test]
    fn open_and_seed() {
        let s = Store::open_in_memory().unwrap();
        assert_eq!(s.count_obs().unwrap(), 0);
        let syns = s.list_synonyms().unwrap();
        assert!(syns.iter().any(|(t, e)| t == "k8s" && e == "kubernetes"));
        // Both directions seeded.
        assert!(syns.iter().any(|(t, e)| t == "kubernetes" && e == "k8s"));
    }

    #[test]
    fn insert_and_count() {
        let s = Store::open_in_memory().unwrap();
        let obs = NewObservation {
            session: "s1".to_string(),
            kind: Kind::Manual,
            body: "learned about kubernetes pod scheduling".to_string(),
            branch: Some("main".to_string()),
            ..Default::default()
        };
        let id = s.insert_obs(&obs, 1000, &dummy_emb(1.0)).unwrap();
        assert_eq!(id, 1);
        assert_eq!(s.count_obs().unwrap(), 1);
    }

    #[test]
    fn bm25_returns_hits() {
        let s = Store::open_in_memory().unwrap();
        let obs = NewObservation {
            session: "s1".to_string(),
            kind: Kind::Manual,
            body: "learned about kubernetes pod scheduling".to_string(),
            branch: None,
            ..Default::default()
        };
        s.insert_obs(&obs, 1000, &dummy_emb(1.0)).unwrap();
        s.rebuild_fts().unwrap();
        let hits = s.search_bm25("kubernetes", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 1);
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn vector_returns_hits() {
        let s = Store::open_in_memory().unwrap();
        let obs = NewObservation {
            session: "s1".to_string(),
            kind: Kind::Manual,
            body: "anything".to_string(),
            branch: None,
            ..Default::default()
        };
        s.insert_obs(&obs, 1000, &dummy_emb(1.0)).unwrap();
        let hits = s.search_vector(&dummy_emb(1.0), 10).unwrap();
        assert_eq!(hits.len(), 1);
        // Self-similarity. We now store int8-quantized vectors and search
        // returns the raw dot product (not normalized cosine). The exact
        // value depends on the dummy embedding shape; what matters is that
        // it's positive and larger than any non-self score would be.
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn synonyms_expand() {
        let s = Store::open_in_memory().unwrap();
        let expanded = s.expand_synonyms(&["k8s".to_string()]).unwrap();
        assert!(expanded.iter().any(|t| t == "kubernetes"));
        assert!(expanded.iter().any(|t| t == "k8s"));
    }

    #[test]
    fn synonyms_user_edits() {
        let s = Store::open_in_memory().unwrap();
        s.add_synonym("foo", "bar").unwrap();
        let expanded = s.expand_synonyms(&["foo".to_string()]).unwrap();
        assert!(expanded.iter().any(|t| t == "bar"));
        let removed = s.remove_synonym("foo", Some("bar")).unwrap();
        assert!(removed >= 1);
        let expanded = s.expand_synonyms(&["foo".to_string()]).unwrap();
        assert!(!expanded.iter().any(|t| t == "bar"));
    }

    #[test]
    fn eviction() {
        let s = Store::open_in_memory().unwrap();
        let mk = |body: &str| NewObservation {
            session: "s1".to_string(),
            kind: Kind::Manual,
            body: body.to_string(),
            branch: None,
            ..Default::default()
        };
        s.insert_obs(&mk("old"), 100, &dummy_emb(1.0)).unwrap();
        s.insert_obs(&mk("new"), 1000, &dummy_emb(1.0)).unwrap();
        let removed = s.evict_older_than(500).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(s.count_obs().unwrap(), 1);
    }

    #[test]
    fn hydrate_preserves_order() {
        let s = Store::open_in_memory().unwrap();
        let mk = |body: &str| NewObservation {
            session: "s1".to_string(),
            kind: Kind::Manual,
            body: body.to_string(),
            branch: None,
            ..Default::default()
        };
        let a = s.insert_obs(&mk("alpha"), 100, &dummy_emb(1.0)).unwrap();
        let b = s.insert_obs(&mk("beta"), 200, &dummy_emb(1.0)).unwrap();
        let c = s.insert_obs(&mk("gamma"), 300, &dummy_emb(1.0)).unwrap();
        let got = s.get_obs_by_ids(&[c, a, b]).unwrap();
        assert_eq!(got.iter().map(|o| o.id).collect::<Vec<_>>(), vec![c, a, b]);
    }
}
