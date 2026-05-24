//! Code-chunk recall: hybrid BM25 + vector with path-aware scoring.
//!
//! Same shape as `pipeline::recall_with_config` but for `code_chunks` rows
//! instead of `obs`. Lives here (not in `memorize-server`) so the eval
//! harness can drive it directly without going through HTTP — the daemon
//! holds the DuckDB write lock, but read-only opens are allowed, so an
//! eval process can ablate `Mode` while the indexer keeps running.
//!
//! Behaviour the route was doing inline that's now configurable:
//!
//!   - **Mode** — Hybrid / Bm25Only / VectorOnly. The route always passes
//!     Hybrid; the eval harness varies it to measure each stream's lift.
//!   - **Path-segment exact-match boost** — if a query token appears as a
//!     literal path segment (split on `/_-.`), the chunk's RRF score is
//!     bumped by `path_boost` per match. Magnitude defaults to ~0.02 per
//!     match, equal to one stream's RRF score at rank 1.
//!   - **Per-file diversification** — cap N chunks per file in the final
//!     ranking. Without it, a chatty test file with `memo` repeated dozens
//!     of times per chunk would take 7+ of the top 10 slots and squeeze
//!     out every other relevant file.

use crate::Mode;
use anyhow::Result;
use memorize_store::Store;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

/// All code-recall knobs. Defaults match what the live `/code/search`
/// route runs.
#[derive(Debug, Clone)]
pub struct CodeRecallConfig {
    pub mode: Mode,
    /// RRF constant. 60 matches the obs-side recall and agentmemory.
    pub rrf_k: f64,
    /// Per-stream top-K before fusion.
    pub per_stream_top_k: usize,
    /// Bonus added to a chunk's RRF score for each query token that appears
    /// as a literal path segment. 0.02 ≈ one stream's rank-1 RRF score, so
    /// two path matches lift roughly as much as ranking #1 in both streams.
    /// Set to 0.0 to disable the boost entirely.
    pub path_boost: f64,
    /// Cap chunks per file in the final ranking. `None` disables.
    pub max_per_file: Option<usize>,
    /// Optional language filter applied to both streams.
    pub language: Option<String>,
    /// Optional path prefix filter applied to both streams.
    pub path_prefix: Option<String>,
}

impl Default for CodeRecallConfig {
    fn default() -> Self {
        Self {
            mode: Mode::Hybrid,
            rrf_k: 60.0,
            per_stream_top_k: 50,
            path_boost: 0.02,
            max_per_file: Some(2),
            language: None,
            path_prefix: None,
        }
    }
}

/// One returned chunk + its fused score.
#[derive(Debug, Serialize)]
pub struct CodeRecalled {
    pub id: i64,
    pub path: String,
    pub language: String,
    pub line_start: i32,
    pub line_end: i32,
    pub kind: String,
    pub qualified: String,
    pub body: String,
    pub score: f64,
}

/// End-to-end code recall.
pub fn recall_code(
    store: &Store,
    query: &str,
    query_emb: &[f32],
    limit: usize,
    config: &CodeRecallConfig,
) -> Result<Vec<CodeRecalled>> {
    if limit == 0 || query.trim().is_empty() {
        return Ok(vec![]);
    }
    let fused = fused_search(store, query, query_emb, config);
    let picked = diversify(&fused, limit, config.max_per_file);
    hydrate(store, &picked)
}

/// Run both streams, fuse via RRF, apply path-segment boost. Returns
/// (id, path, score) sorted desc by score.
pub fn fused_search(
    store: &Store,
    q: &str,
    q_emb: &[f32],
    config: &CodeRecallConfig,
) -> Vec<(i64, String, f64)> {
    let lang = config.language.as_deref();
    let prefix = config.path_prefix.as_deref();

    let bm25 = match config.mode {
        Mode::VectorOnly => vec![],
        Mode::Hybrid | Mode::Bm25Only => store
            .search_code_bm25(q, config.per_stream_top_k, lang, prefix)
            .unwrap_or_default(),
    };
    let vec_hits = match config.mode {
        Mode::Bm25Only => vec![],
        Mode::Hybrid | Mode::VectorOnly => store
            .search_code_vector(q_emb, config.per_stream_top_k, lang, prefix)
            .unwrap_or_default(),
    };

    let k = config.rrf_k;
    let mut acc: HashMap<i64, (f64, String)> = HashMap::new();
    for (rank, hit) in bm25.iter().enumerate() {
        let entry = acc
            .entry(hit.id)
            .or_insert_with(|| (0.0, hit.path.clone()));
        entry.0 += 1.0 / (k + (rank + 1) as f64);
    }
    for (rank, hit) in vec_hits.iter().enumerate() {
        let entry = acc
            .entry(hit.id)
            .or_insert_with(|| (0.0, hit.path.clone()));
        entry.0 += 1.0 / (k + (rank + 1) as f64);
    }

    if config.path_boost > 0.0 {
        let q_tokens = tokenize_for_path_boost(q);
        if !q_tokens.is_empty() {
            for (_id, (score, path)) in acc.iter_mut() {
                let matches = count_path_segment_matches(path, &q_tokens);
                *score += config.path_boost * matches as f64;
            }
        }
    }

    let mut fused: Vec<(i64, String, f64)> = acc
        .into_iter()
        .map(|(id, (score, path))| (id, path, score))
        .collect();
    fused.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

/// Apply per-file cap, backfilling from the unrestricted pool if the cap
/// leaves us short of `limit`.
fn diversify(
    fused: &[(i64, String, f64)],
    limit: usize,
    max_per_file: Option<usize>,
) -> Vec<(i64, f64)> {
    let cap = match max_per_file {
        Some(c) if c > 0 => c,
        _ => return fused.iter().take(limit).map(|(id, _, s)| (*id, *s)).collect(),
    };
    let mut per_file: HashMap<String, usize> = HashMap::new();
    let mut picked: Vec<(i64, f64)> = Vec::with_capacity(limit);
    for (id, path, score) in fused {
        let n = per_file.entry(path.clone()).or_insert(0);
        if *n >= cap {
            continue;
        }
        *n += 1;
        picked.push((*id, *score));
        if picked.len() >= limit {
            return picked;
        }
    }
    // Backfill if the cap left us short.
    let already: HashSet<i64> = picked.iter().map(|(id, _)| *id).collect();
    for (id, _path, score) in fused {
        if picked.len() >= limit {
            break;
        }
        if !already.contains(id) {
            picked.push((*id, *score));
        }
    }
    picked
}

fn hydrate(store: &Store, picked: &[(i64, f64)]) -> Result<Vec<CodeRecalled>> {
    let ids: Vec<i64> = picked.iter().map(|(id, _)| *id).collect();
    let rows = store.get_code_chunks_by_ids(&ids)?;
    let scores: Vec<f64> = picked.iter().map(|(_, s)| *s).collect();
    Ok(rows
        .into_iter()
        .zip(scores)
        .map(|(r, score)| CodeRecalled {
            id: r.id,
            path: r.path,
            language: r.language,
            line_start: r.line_start,
            line_end: r.line_end,
            kind: r.kind,
            qualified: r.qualified,
            body: r.body,
            score,
        })
        .collect())
}

/// Lowercase + split the query on whitespace and `/_-.` so each token
/// can be compared against path segments. Returns empty for whitespace-
/// only input.
fn tokenize_for_path_boost(q: &str) -> Vec<String> {
    q.split(|c: char| c.is_whitespace() || matches!(c, '/' | '_' | '-' | '.'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Count how many query tokens appear as exact path segments (case-
/// insensitive). `packages/entities/src/irMemo/memo.ts` yields the
/// segment set {packages, entities, src, irmemo, memo, ts}. Each query
/// token is counted at most once.
fn count_path_segment_matches(path: &str, q_tokens: &[String]) -> usize {
    let segs: HashSet<String> = path
        .split(|c: char| matches!(c, '/' | '_' | '-' | '.'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect();
    q_tokens.iter().filter(|t| segs.contains(*t)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_for_path_boost_basic() {
        let toks = tokenize_for_path_boost("IR memo memoization cache");
        assert_eq!(toks, vec!["ir", "memo", "memoization", "cache"]);
    }

    #[test]
    fn tokenize_for_path_boost_path_style() {
        let toks = tokenize_for_path_boost("irMemo/memo.ts");
        assert_eq!(toks, vec!["irmemo", "memo", "ts"]);
    }

    #[test]
    fn count_path_segment_matches_hits() {
        let q = tokenize_for_path_boost("ir memo");
        let n =
            count_path_segment_matches("/Users/max/src/slate/packages/entities/src/irMemo/memo.ts", &q);
        // "ir" doesn't appear as its own segment; "memo" appears (memo.ts).
        // "irmemo" is also a segment but doesn't match the token "ir" exactly.
        assert!(n >= 1);
    }

    #[test]
    fn diversify_caps_per_file() {
        let fused = vec![
            (1, "a.ts".to_string(), 0.5),
            (2, "a.ts".to_string(), 0.4),
            (3, "a.ts".to_string(), 0.3),
            (4, "b.ts".to_string(), 0.25),
            (5, "c.ts".to_string(), 0.2),
        ];
        let picked = diversify(&fused, 4, Some(2));
        // 2x a.ts allowed, then b.ts, then c.ts.
        assert_eq!(picked.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![1, 2, 4, 5]);
    }

    #[test]
    fn diversify_backfills_when_short() {
        // limit higher than distinct files after the cap → backfill kicks in.
        let fused = vec![
            (1, "a.ts".to_string(), 0.5),
            (2, "a.ts".to_string(), 0.4),
            (3, "a.ts".to_string(), 0.3),
        ];
        let picked = diversify(&fused, 3, Some(2));
        // First two via cap, third via backfill from same file.
        assert_eq!(picked.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![1, 2, 3]);
    }
}
