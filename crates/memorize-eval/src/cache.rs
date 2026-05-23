//! On-disk embedding cache for the eval harness.
//!
//! Each ablation re-run (different `--mode`, `--rrf-k`, etc.) processes the
//! same ~24k haystack sessions. Without a cache that's ~9 minutes of ONNX
//! per run; with a cache, the first run pays the cost and subsequent runs
//! load embeddings from a local DuckDB and skip ~89% of the pipeline.
//!
//! Layout: `~/.memorize/eval-cache.duckdb`, single table keyed by SHA-256 hex
//! of the text body. Embeddings stored as raw little-endian f32 BLOBs
//! (1536 bytes per 384-d vector). Native-endian would be fine for personal
//! use, but explicit byte order means the cache survives a hypothetical move
//! across architectures.

use anyhow::{Context, Result, bail};
use duckdb::{Connection, params};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct Cache {
    conn: Mutex<Connection>,
}

impl Cache {
    pub fn open() -> Result<Self> {
        let path = cache_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("create cache dir")?;
        }
        let conn = Connection::open(&path)
            .with_context(|| format!("open cache duckdb at {}", path.display()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS embed_cache (
                 key        VARCHAR PRIMARY KEY,
                 emb        BLOB NOT NULL,
                 created_ts BIGINT NOT NULL
             );",
        )
        .context("create embed_cache schema")?;
        Ok(Cache { conn: Mutex::new(conn) })
    }

    /// Batch lookup: returns a map from `keys[i]` to embedding for keys present
    /// in the cache. Misses are simply absent from the returned map.
    pub fn get_many(&self, keys: &[String]) -> Result<HashMap<String, Vec<f32>>> {
        if keys.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders: String = keys.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!("SELECT key, emb FROM embed_cache WHERE key IN ({placeholders})");
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn duckdb::ToSql> =
            keys.iter().map(|k| k as &dyn duckdb::ToSql).collect();
        let rows = stmt.query_map(params.as_slice(), |row| {
            let key: String = row.get(0)?;
            let bytes: Vec<u8> = row.get(1)?;
            Ok((key, bytes))
        })?;
        let mut out = HashMap::with_capacity(keys.len());
        for r in rows {
            let (k, bytes) = r?;
            out.insert(k, bytes_to_f32_vec(&bytes)?);
        }
        Ok(out)
    }

    /// Batch insert. Skips duplicates that arrived from a parallel writer
    /// (we don't have one yet, but cheap to be safe).
    pub fn put_many(&self, entries: &[(String, &[f32])]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().unwrap();
        // One prepared statement, multiple executes — keeps the transaction
        // implicit and the syntax simple.
        let mut stmt = conn.prepare(
            "INSERT OR IGNORE INTO embed_cache(key, emb, created_ts) VALUES (?, ?, ?)",
        )?;
        for (key, emb) in entries {
            let bytes = f32_vec_to_bytes(emb);
            stmt.execute(params![key, &bytes, now])?;
        }
        Ok(())
    }

    /// Total stored entries. Useful in startup logs.
    pub fn len(&self) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row("SELECT COUNT(*) FROM embed_cache", [], |r| r.get(0))?)
    }
}

/// Cache key for a text input. Includes `memorize_embed::MODEL_TAG` so
/// swapping the active embedding model cleanly invalidates older entries —
/// they simply no longer match.
pub fn hash_text(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(memorize_embed::model_tag().as_bytes());
    h.update(b":");
    h.update(text.as_bytes());
    let bytes: [u8; 32] = h.finalize().into();
    let mut s = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn cache_path() -> PathBuf {
    if let Ok(p) = std::env::var("MEMORIZE_EVAL_CACHE") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".memorize").join("eval-cache.duckdb")
}

/// Convert `&[f32]` to a `Vec<u8>` of length `4 * f32s.len()`, little-endian.
fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Inverse of `f32_vec_to_bytes`. Errors if the byte length isn't a multiple of 4.
fn bytes_to_f32_vec(b: &[u8]) -> Result<Vec<f32>> {
    if b.len() % 4 != 0 {
        bail!("cache blob length {} is not a multiple of 4", b.len());
    }
    let mut out = Vec::with_capacity(b.len() / 4);
    for chunk in b.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_bytes() {
        let v: Vec<f32> = vec![0.0, 1.0, -1.0, 3.14159, f32::INFINITY, -0.5];
        let bytes = f32_vec_to_bytes(&v);
        assert_eq!(bytes.len(), v.len() * 4);
        let back = bytes_to_f32_vec(&bytes).unwrap();
        assert_eq!(back, v);
    }

    #[test]
    fn bytes_must_be_aligned() {
        let bad = vec![0u8; 7];
        assert!(bytes_to_f32_vec(&bad).is_err());
    }

    #[test]
    fn hash_is_deterministic_and_distinct() {
        assert_eq!(hash_text("hello"), hash_text("hello"));
        assert_ne!(hash_text("hello"), hash_text("Hello"));
    }

    #[test]
    fn hash_namespaces_by_model_tag() {
        // Hash must depend on memorize_embed::MODEL_TAG so swapping models
        // partitions the cache. Sentinel: a strategyless SHA256 of the same
        // input must NOT collide with our cache key.
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"hello");
        let raw: [u8; 32] = h.finalize().into();
        let raw_hex: String = raw.iter().map(|b| format!("{b:02x}")).collect();
        assert_ne!(hash_text("hello"), raw_hex);
    }
}
