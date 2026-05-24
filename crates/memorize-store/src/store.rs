use crate::DEFAULT_EMBED_DIM;
use crate::schema::{backfill_path_tokens_sql, fts_index_sql, schema_sql};
use crate::synonyms_seed::DEFAULT_PAIRS;
use anyhow::{Context, Result, bail};
use duckdb::{Connection, params};
use memorize_core::{Kind, NewObservation, Observation};
use std::path::Path;
use std::sync::Mutex;

/// All access goes through here. DuckDB connections are not `Sync` (single
/// writer model), so we wrap in a `Mutex`. For a single-user dogfood server
/// the contention is negligible.
pub struct Store {
    conn: Mutex<Connection>,
    embed_dim: usize,
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
        let store = Store { conn: Mutex::new(conn), embed_dim: dim };
        store.init()?;
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
        let store = Store { conn: Mutex::new(conn), embed_dim: dim };
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
        conn.execute_batch(backfill_path_tokens_sql())
            .context("backfill path_tokens")?;
        conn.execute_batch(fts_index_sql()).context("create fts index")?;
        drop(conn);
        self.seed_synonyms_once()?;
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
        let conn = self.conn.lock().unwrap();
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
        for (idx, emb) in chunk_embs.iter().enumerate() {
            let literal = float_array_literal(emb);
            let sql = format!(
                "INSERT INTO vec_chunks(obs_id, chunk_idx, emb)
                 VALUES ({id}, {idx}, {literal}::FLOAT[{dim}])"
            );
            conn.execute_batch(&sql)?;
        }
        Ok(id)
    }

    /// Run BM25 over obs.body. The caller is responsible for synonym expansion;
    /// `query` here is the expanded form. DuckDB's FTS `match_bm25` accepts the
    /// query as a string with implicit OR semantics, so an expansion of
    /// "k8s kubernetes kube" will match a document containing any term.
    pub fn search_bm25(&self, query: &str, limit: usize) -> Result<Vec<BM25Hit>> {
        let conn = self.conn.lock().unwrap();
        let sql = "
            SELECT obs.id, obs.session, fts_main_obs.match_bm25(obs.id, ?) AS score
              FROM obs
             WHERE score IS NOT NULL
          ORDER BY score DESC
             LIMIT ?
        ";
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt
            .query_map(params![query, limit as i64], |row| {
                Ok(BM25Hit {
                    id: row.get(0)?,
                    session: row.get(1)?,
                    score: row.get(2)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
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
        let conn = self.conn.lock().unwrap();
        let literal = float_array_literal(query_emb);
        let dim = self.embed_dim;
        let sql = format!(
            "SELECT obs_id, session, score FROM (
                SELECT vc.obs_id AS obs_id,
                       o.session AS session,
                       MAX(array_cosine_similarity(vc.emb, {literal}::FLOAT[{dim}])) AS score
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

    /// FTS in DuckDB needs a rebuild to pick up new rows; we batch inserts and
    /// rebuild on demand. Cheap at our scale.
    pub fn rebuild_fts(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(fts_index_sql())?;
        Ok(())
    }

    pub fn count_obs(&self) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row("SELECT COUNT(*) FROM obs", [], |r| r.get(0))?)
    }

    /// Drop observations older than `cutoff_ts`. Returns rows removed.
    /// Orphaned vector rows (both legacy `vec` and `vec_chunks`) are cleaned
    /// in the same pass.
    pub fn evict_older_than(&self, cutoff_ts: i64) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let deleted = conn.execute("DELETE FROM obs WHERE ts < ?", params![cutoff_ts])?;
        conn.execute("DELETE FROM vec WHERE id NOT IN (SELECT id FROM obs)", [])?;
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
        let conn = self.conn.lock().unwrap();

        // Atomic: delete stale chunks, upsert files row, insert new chunks
        // all in one transaction. Without this a crash mid-write would leave
        // the files row pointing at a fresh mtime/size while the chunk set
        // is partial — on restart `get_file_meta` would falsely report
        // "already indexed" and skip the file forever.
        conn.execute_batch("BEGIN")?;
        let res: Result<()> = (|| {
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
            for (i, c) in chunks.iter().enumerate() {
                let id = base + 1 + i as i64;
                conn.execute(
                    "INSERT INTO code_chunks(id, path, language, line_start, line_end,
                                             kind, qualified, body, body_hash, path_tokens)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
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
                        c.body,
                        hash_bytes(c.body.as_bytes()),
                        path,
                        c.qualified,
                    ],
                )?;
                let literal = float_array_literal(&chunk_embs[i]);
                let insert_vec = format!(
                    "INSERT INTO vec_code(id, emb) VALUES ({id}, {literal}::FLOAT[{dim}])"
                );
                conn.execute_batch(&insert_vec)?;
            }
            Ok(())
        })();

        match res {
            Ok(()) => {
                conn.execute_batch("COMMIT")?;
                Ok(())
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    /// Drop all chunks + vec rows for a deleted file.
    pub fn delete_code_file(&self, path: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
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
        Ok(())
    }

    pub fn count_code_chunks(&self) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row("SELECT COUNT(*) FROM code_chunks", [], |r| r.get(0))?)
    }

    /// BM25 over code_chunks.body. Optional language and path-prefix filters.
    pub fn search_code_bm25(
        &self,
        query: &str,
        limit: usize,
        language: Option<&str>,
        path_prefix: Option<&str>,
    ) -> Result<Vec<CodeBM25Hit>> {
        let conn = self.conn.lock().unwrap();
        // The transforms inside `match_bm25` mirror the same regex pair used
        // for `path_tokens` at index time (see `backfill_path_tokens_sql`):
        // first split camelCase boundaries (`snapshotMemoGraph` →
        // `snapshot Memo Graph`), then split path-style separators
        // (`packages/.../irMemo/memo.ts` → `packages ... irMemo memo ts`).
        // Without this, a query like `irMemo/memo.ts` would tokenize as a
        // single FTS token and miss everything we indexed. Co-located with
        // the search SQL so changing one rule keeps both sides in lockstep —
        // no Rust-side mirror to drift.
        let mut sql = String::from(
            "SELECT cc.id, cc.path,
                    fts_main_code_chunks.match_bm25(
                        cc.id,
                        regexp_replace(
                            regexp_replace(?, '([a-z])([A-Z])', '\\1 \\2', 'g'),
                            '[/_.\\-]',
                            ' ',
                            'g'
                        )
                    ) AS score
               FROM code_chunks cc
              WHERE score IS NOT NULL",
        );
        let mut params_vec: Vec<Box<dyn duckdb::ToSql>> = vec![Box::new(query.to_string())];
        if let Some(lang) = language {
            sql.push_str(" AND cc.language = ?");
            params_vec.push(Box::new(lang.to_string()));
        }
        if let Some(prefix) = path_prefix {
            sql.push_str(" AND cc.path LIKE ?");
            params_vec.push(Box::new(format!("{prefix}%")));
        }
        sql.push_str(" ORDER BY score DESC LIMIT ?");
        params_vec.push(Box::new(limit as i64));
        let param_refs: Vec<&dyn duckdb::ToSql> =
            params_vec.iter().map(|b| b.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok(CodeBM25Hit {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    score: row.get(2)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Vector cosine over vec_code.emb with optional filters.
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
        let conn = self.conn.lock().unwrap();
        let literal = float_array_literal(query_emb);
        let dim = self.embed_dim;
        let mut sql = format!(
            "SELECT cc.id, cc.path,
                    array_cosine_similarity(vc.emb, {literal}::FLOAT[{dim}]) AS s
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

    pub fn get_code_chunks_by_ids(&self, ids: &[i64]) -> Result<Vec<CodeChunkRow>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, path, language, line_start, line_end, kind, qualified, body
               FROM code_chunks WHERE id IN ({placeholders})"
        );
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn duckdb::ToSql> =
            ids.iter().map(|id| id as &dyn duckdb::ToSql).collect();
        let mut by_id: std::collections::HashMap<i64, CodeChunkRow> = stmt
            .query_map(params.as_slice(), |row| {
                Ok(CodeChunkRow {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    language: row.get(2)?,
                    line_start: row.get(3)?,
                    line_end: row.get(4)?,
                    kind: row.get(5)?,
                    qualified: row.get(6)?,
                    body: row.get(7)?,
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

/// SHA-256 of arbitrary bytes. Used to detect when a code chunk's body is
/// unchanged between re-indexes so we can skip embedding work.
fn hash_bytes(bytes: &[u8]) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().to_vec()
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
        // Self-cosine should be ~1.0.
        assert!((hits[0].score - 1.0).abs() < 1e-4);
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
