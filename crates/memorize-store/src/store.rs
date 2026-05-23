use crate::EMBED_DIM;
use crate::schema::{fts_index_sql, schema_sql};
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

impl Store {
    /// Open (creating if needed) a DuckDB at the given path. Applies schema +
    /// FTS index + first-run synonym seed.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path).context("open duckdb")?;
        let store = Store { conn: Mutex::new(conn) };
        store.init()?;
        Ok(store)
    }

    /// In-memory DB for tests. Schema applied; synonyms seeded.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("open in-memory duckdb")?;
        let store = Store { conn: Mutex::new(conn) };
        store.init()?;
        Ok(store)
    }

    fn init(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(&schema_sql()).context("apply schema")?;
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

    /// Returns the assigned (id, ts).
    pub fn insert_obs(&self, obs: &NewObservation, ts: i64, emb: &[f32]) -> Result<i64> {
        if emb.len() != EMBED_DIM {
            bail!(
                "embedding length {} doesn't match EMBED_DIM={}",
                emb.len(),
                EMBED_DIM
            );
        }
        let conn = self.conn.lock().unwrap();
        let id: i64 = conn.query_row(
            "INSERT INTO obs(id, ts, session, branch, kind, body)
             VALUES (
                 COALESCE((SELECT MAX(id) FROM obs), 0) + 1,
                 ?, ?, ?, ?, ?
             ) RETURNING id",
            params![ts, obs.session, obs.branch, obs.kind.as_str(), obs.body],
            |row| row.get(0),
        )?;
        // Embedding insert via SQL literal — embeddings are our own f32s, no
        // injection surface, and parameter-binding fixed-size arrays in
        // duckdb-rs is awkward.
        let literal = float_array_literal(emb);
        let insert_vec = format!(
            "INSERT INTO vec(id, emb) VALUES ({id}, {literal}::FLOAT[{EMBED_DIM}])"
        );
        conn.execute_batch(&insert_vec)?;
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

    /// Cosine over vec.emb. We never read embeddings back into Rust — DuckDB
    /// runs the comparison and returns a scalar score per row.
    pub fn search_vector(&self, query_emb: &[f32], limit: usize) -> Result<Vec<VectorHit>> {
        if query_emb.len() != EMBED_DIM {
            bail!(
                "query embedding length {} doesn't match EMBED_DIM={}",
                query_emb.len(),
                EMBED_DIM
            );
        }
        let conn = self.conn.lock().unwrap();
        let literal = float_array_literal(query_emb);
        let sql = format!(
            "SELECT v.id, o.session,
                    array_cosine_similarity(v.emb, {literal}::FLOAT[{EMBED_DIM}]) AS s
               FROM vec v JOIN obs o ON o.id = v.id
              ORDER BY s DESC
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
            "SELECT id, ts, session, branch, kind, body
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
    /// Vector rows are cleaned in the same pass.
    pub fn evict_older_than(&self, cutoff_ts: i64) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let deleted = conn.execute("DELETE FROM obs WHERE ts < ?", params![cutoff_ts])?;
        conn.execute(
            "DELETE FROM vec WHERE id NOT IN (SELECT id FROM obs)",
            [],
        )?;
        Ok(deleted)
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use memorize_core::Kind;

    fn dummy_emb(scale: f32) -> Vec<f32> {
        (0..EMBED_DIM).map(|i| (i as f32 / EMBED_DIM as f32) * scale).collect()
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
        };
        let a = s.insert_obs(&mk("alpha"), 100, &dummy_emb(1.0)).unwrap();
        let b = s.insert_obs(&mk("beta"), 200, &dummy_emb(1.0)).unwrap();
        let c = s.insert_obs(&mk("gamma"), 300, &dummy_emb(1.0)).unwrap();
        let got = s.get_obs_by_ids(&[c, a, b]).unwrap();
        assert_eq!(got.iter().map(|o| o.id).collect::<Vec<_>>(), vec![c, a, b]);
    }
}
