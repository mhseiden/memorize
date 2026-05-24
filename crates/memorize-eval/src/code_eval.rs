//! Code-recall A/B harness. Opens the live `~/.memorize/db.duckdb`
//! read-only, runs a query bank under each Mode (Hybrid / Bm25Only /
//! VectorOnly), and reports R@K + MRR.
//!
//! DuckDB allows concurrent readers alongside the daemon writer, so the
//! eval runs without disturbing the indexer.
//!
//! Two query sources:
//!   - Hand-curated TOML bank (`bench/code-queries.toml`) — biased toward
//!     queries we'd remember to write.
//!   - Synthetic queries auto-generated from the index — for each random
//!     chunk, use its `qualified` field as the "query" and expect that
//!     chunk's file in top-K. Catches recall blind spots a hand-curated
//!     bank would miss.

use anyhow::{Context, Result};
use memorize_recall::{CodeRecallConfig, Mode as RecallMode, recall_code};
use memorize_store::Store;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Deserialize)]
pub struct QueryBank {
    pub queries: Vec<BankQuery>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BankQuery {
    pub query: String,
    /// One or more paths (or path substrings) that any acceptable answer
    /// must match. A hit at rank R counts as a recall@K success if any
    /// `expect` substring is in its path.
    pub expect: Vec<String>,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub path_prefix: Option<String>,
}

#[derive(Debug, Clone)]
struct SyntheticQuery {
    query: String,
    expected_path: String,
}

#[derive(Debug, Serialize)]
struct PerQueryResult {
    query: String,
    category: String,
    expect: Vec<String>,
    mode: String,
    rank_of_first_correct: Option<usize>,
    top_paths: Vec<String>,
    latency_ms: f64,
}

#[derive(Debug, Default, Serialize, Clone)]
struct ModeStats {
    n_queries: usize,
    n_recall_at_5: usize,
    n_recall_at_10: usize,
    n_recall_at_20: usize,
    sum_reciprocal_rank: f64,
    sum_latency_ms: f64,
}

impl ModeStats {
    fn r_at(&self, _k: usize, n_hit: usize) -> f64 {
        if self.n_queries == 0 {
            0.0
        } else {
            n_hit as f64 / self.n_queries as f64
        }
    }
    fn mrr(&self) -> f64 {
        if self.n_queries == 0 {
            0.0
        } else {
            self.sum_reciprocal_rank / self.n_queries as f64
        }
    }
    fn mean_latency_ms(&self) -> f64 {
        if self.n_queries == 0 {
            0.0
        } else {
            self.sum_latency_ms / self.n_queries as f64
        }
    }
}

/// Which scoring backend a given eval iteration uses.
#[derive(Copy, Clone)]
enum EvalMode {
    Recall(RecallMode),
    Int8Only,
    Int8Hybrid,
}

pub struct CodeEvalOpts {
    pub db_path: PathBuf,
    pub bank_path: PathBuf,
    pub limit: usize,
    pub synthetic_count: usize,
    pub out_dir: PathBuf,
    /// If true, also evaluate int8-quantized vector modes (int8-only and
    /// int8-hybrid). One-time startup cost to stream all f32 vectors from
    /// the snapshot and pack them into 384-byte i8 arrays (~5-10s for
    /// 192k vectors), then ~75 MB resident.
    pub include_int8: bool,
}

/// 384-d int8-quantized vector. Each f32 component (already L2-normalized
/// to [-1, 1] by the embedder) maps to an i8 via `round(v * 127)`. Storage
/// is 384 bytes per vector vs 1536 bytes raw — 4× reduction. Cosine
/// similarity becomes dot product divided by 127² (we skip the divide for
/// ranking purposes since it's monotonic).
type Q8Vec = [i8; 384];

fn quantize_i8(emb: &[f32]) -> Q8Vec {
    let mut out: Q8Vec = [0i8; 384];
    for (i, &v) in emb.iter().take(384).enumerate() {
        // Saturating round; embedder produces values in [-1, 1] but a tiny
        // numerical drift outside the unit ball could push past i8 range.
        let scaled = (v * 127.0).round().clamp(-127.0, 127.0);
        out[i] = scaled as i8;
    }
    out
}

/// Dot product of two i8 vectors as i32. Autovectorizes on Apple Silicon
/// (NEON) and x86 (AVX2) — measured ~30 ns per call for 384-d on M4 Pro.
#[inline]
fn dot_i8(a: &Q8Vec, b: &Q8Vec) -> i32 {
    let mut acc: i32 = 0;
    for i in 0..384 {
        acc += (a[i] as i32) * (b[i] as i32);
    }
    acc
}

/// In-memory int8-vector index. ~73 MB for 192k vectors (192k × 384 bytes).
struct Int8Index {
    ids: Vec<i64>,
    vecs: Vec<Q8Vec>,
}

impl Int8Index {
    fn load(store: &Store) -> Result<Self> {
        let mut ids: Vec<i64> = Vec::new();
        let mut vecs: Vec<Q8Vec> = Vec::new();
        // The store now returns int8 vectors directly (production-side
        // quantization). The PoC's local quantize_i8 is still here for
        // the query path — we quantize the query embedding the same way.
        store.for_each_code_vector(|id, q8| {
            ids.push(id);
            let mut arr: Q8Vec = [0i8; 384];
            for (i, &v) in q8.iter().take(384).enumerate() {
                arr[i] = v;
            }
            vecs.push(arr);
            Ok(())
        })?;
        Ok(Int8Index { ids, vecs })
    }

    /// Return top-N (id, dot_product) pairs, highest dot first.
    fn search(&self, q: &Q8Vec, n: usize) -> Vec<(i64, i32)> {
        let mut scored: Vec<(i64, i32)> = self
            .ids
            .iter()
            .zip(&self.vecs)
            .map(|(id, v)| (*id, dot_i8(q, v)))
            .collect();
        // Highest dot product first.
        scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(n);
        scored
    }
}

pub fn run(opts: CodeEvalOpts) -> Result<()> {
    let bank = load_bank(&opts.bank_path)?;
    eprintln!(
        "loaded {} hand-curated queries from {}",
        bank.queries.len(),
        opts.bank_path.display()
    );

    let store = Store::open_read_only(&opts.db_path)
        .with_context(|| format!("open {} read-only", opts.db_path.display()))?;

    let synth = if opts.synthetic_count > 0 {
        let s = sample_synthetic(&store, opts.synthetic_count)?;
        eprintln!(
            "sampled {} synthetic queries from the index",
            s.len()
        );
        s
    } else {
        vec![]
    };

    let mut modes: Vec<(&str, EvalMode)> = vec![
        ("hybrid", EvalMode::Recall(RecallMode::Hybrid)),
        ("bm25", EvalMode::Recall(RecallMode::Bm25Only)),
        ("vector", EvalMode::Recall(RecallMode::VectorOnly)),
    ];

    let int8_index: Option<Int8Index> = if opts.include_int8 {
        let t = Instant::now();
        eprintln!("loading int8 index from snapshot (one-time)...");
        let idx = Int8Index::load(&store)?;
        eprintln!(
            "  {} vectors quantized to i8 in {:.1}s ({:.1} MB in memory)",
            idx.vecs.len(),
            t.elapsed().as_secs_f64(),
            (idx.vecs.len() * std::mem::size_of::<Q8Vec>()) as f64 / 1e6,
        );
        modes.push(("int8-only", EvalMode::Int8Only));
        modes.push(("int8-hybrid", EvalMode::Int8Hybrid));
        Some(idx)
    } else {
        None
    };

    let mut per_query: Vec<PerQueryResult> = Vec::new();
    let mut stats_curated: std::collections::HashMap<String, ModeStats> = Default::default();
    let mut stats_synth: std::collections::HashMap<String, ModeStats> = Default::default();
    let mut stats_overall: std::collections::HashMap<String, ModeStats> = Default::default();

    for (mode_name, mode) in &modes {
        for q in &bank.queries {
            let res = run_one(
                &store,
                int8_index.as_ref(),
                &q.query,
                q.path_prefix.clone(),
                opts.limit,
                *mode,
            )?;
            let rank = first_correct_rank(&res.paths, &q.expect);
            let row = PerQueryResult {
                query: q.query.clone(),
                category: q.category.clone(),
                expect: q.expect.clone(),
                mode: mode_name.to_string(),
                rank_of_first_correct: rank,
                top_paths: res.paths.clone(),
                latency_ms: res.latency_ms,
            };
            accumulate(&mut stats_curated, mode_name, rank, res.latency_ms);
            accumulate(&mut stats_overall, mode_name, rank, res.latency_ms);
            per_query.push(row);
        }

        for q in &synth {
            let expect = vec![q.expected_path.clone()];
            let res = run_one(
                &store,
                int8_index.as_ref(),
                &q.query,
                None,
                opts.limit,
                *mode,
            )?;
            let rank = first_correct_rank(&res.paths, &expect);
            let row = PerQueryResult {
                query: q.query.clone(),
                category: "synthetic".to_string(),
                expect,
                mode: mode_name.to_string(),
                rank_of_first_correct: rank,
                top_paths: res.paths.clone(),
                latency_ms: res.latency_ms,
            };
            accumulate(&mut stats_synth, mode_name, rank, res.latency_ms);
            accumulate(&mut stats_overall, mode_name, rank, res.latency_ms);
            per_query.push(row);
        }
    }


    std::fs::create_dir_all(&opts.out_dir).ok();
    let report = render_report(&stats_curated, &stats_synth, &stats_overall);
    eprintln!("\n{report}");

    let md_path = opts.out_dir.join("code-eval.md");
    std::fs::write(&md_path, &report)?;
    let json_path = opts.out_dir.join("code-eval.json");
    let json = serde_json::to_string_pretty(&per_query)?;
    std::fs::write(&json_path, json)?;
    eprintln!("wrote {} and {}", md_path.display(), json_path.display());
    Ok(())
}

struct RunOnce {
    paths: Vec<String>,
    latency_ms: f64,
}

fn run_one(
    store: &Store,
    int8_index: Option<&Int8Index>,
    query: &str,
    path_prefix: Option<String>,
    limit: usize,
    mode: EvalMode,
) -> Result<RunOnce> {
    let q_emb = memorize_embed::embed(query).context("embed eval query")?;
    let t = Instant::now();
    let paths: Vec<String> = match mode {
        EvalMode::Recall(rmode) => {
            let cfg = CodeRecallConfig {
                mode: rmode,
                path_prefix,
                ..CodeRecallConfig::default()
            };
            recall_code(store, query, &q_emb, limit, &cfg)?
                .into_iter()
                .map(|r| r.path)
                .collect()
        }
        EvalMode::Int8Only => int8_only_paths(
            store,
            int8_index.expect("int8 index must be loaded for Int8Only"),
            &q_emb,
            limit,
            path_prefix.as_deref(),
        )?,
        EvalMode::Int8Hybrid => int8_hybrid_paths(
            store,
            int8_index.expect("int8 index must be loaded for Int8Hybrid"),
            query,
            &q_emb,
            limit,
            path_prefix.as_deref(),
        )?,
    };
    let latency_ms = t.elapsed().as_secs_f64() * 1000.0;
    Ok(RunOnce {
        paths,
        latency_ms,
    })
}

/// Int8-only ranking. Quantize the query, dot-product over the in-memory
/// int8 index, hydrate paths, apply path-prefix filter + per-file diversification.
fn int8_only_paths(
    store: &Store,
    idx: &Int8Index,
    q_emb: &[f32],
    limit: usize,
    path_prefix: Option<&str>,
) -> Result<Vec<String>> {
    let q_q8 = quantize_i8(q_emb);
    // Pull a generous pool — we'll trim after diversification.
    let pool = idx.search(&q_q8, 200);
    let ids: Vec<i64> = pool.iter().map(|(id, _)| *id).collect();
    let rows = store.get_code_chunks_by_ids(&ids)?;
    let by_id: std::collections::HashMap<i64, String> =
        rows.into_iter().map(|r| (r.id, r.path)).collect();
    let paths: Vec<String> = pool
        .into_iter()
        .filter_map(|(id, _)| by_id.get(&id).cloned())
        .filter(|p| path_prefix.map(|pp| p.starts_with(pp)).unwrap_or(true))
        .collect();
    Ok(diversify_paths(paths, 2, limit))
}

/// BM25 + int8 cosine, RRF-fused. Mirrors the production fused_search
/// shape (per_stream=50, rrf_k=60, path_boost=0.02, per-file cap=2) with
/// the int8 index swapped in for the vector stream.
fn int8_hybrid_paths(
    store: &Store,
    idx: &Int8Index,
    query: &str,
    q_emb: &[f32],
    limit: usize,
    path_prefix: Option<&str>,
) -> Result<Vec<String>> {
    const PER_STREAM: usize = 50;
    const RRF_K: f64 = 60.0;
    const PATH_BOOST: f64 = 0.02;

    let bm25 = store
        .search_code_bm25(query, PER_STREAM, None, path_prefix)
        .unwrap_or_default();
    let q_q8 = quantize_i8(q_emb);
    let q8_hits = idx.search(&q_q8, PER_STREAM);

    let mut acc: std::collections::HashMap<i64, (f64, Option<String>)> =
        std::collections::HashMap::new();
    for (rank, hit) in bm25.iter().enumerate() {
        let entry = acc
            .entry(hit.id)
            .or_insert_with(|| (0.0, Some(hit.path.clone())));
        entry.0 += 1.0 / (RRF_K + (rank + 1) as f64);
    }
    for (rank, (id, _)) in q8_hits.iter().enumerate() {
        let entry = acc.entry(*id).or_insert_with(|| (0.0, None));
        entry.0 += 1.0 / (RRF_K + (rank + 1) as f64);
    }

    // Hydrate paths for q8-only hits.
    let need_path: Vec<i64> = acc
        .iter()
        .filter_map(|(id, (_, p))| if p.is_none() { Some(*id) } else { None })
        .collect();
    if !need_path.is_empty() {
        let rows = store.get_code_chunks_by_ids(&need_path)?;
        for r in rows {
            if let Some(slot) = acc.get_mut(&r.id) {
                slot.1 = Some(r.path);
            }
        }
    }

    // Path-segment exact-match boost — same rule as production.
    let q_tokens = tokenize_for_path_boost(query);
    if !q_tokens.is_empty() {
        for (_id, (score, p)) in acc.iter_mut() {
            if let Some(path) = p {
                let n = count_path_segment_matches(path, &q_tokens);
                *score += PATH_BOOST * n as f64;
            }
        }
    }

    let mut fused: Vec<(String, f64)> = acc
        .into_iter()
        .filter_map(|(_, (s, p))| p.map(|path| (path, s)))
        .filter(|(p, _)| path_prefix.map(|pp| p.starts_with(pp)).unwrap_or(true))
        .collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let paths: Vec<String> = fused.into_iter().map(|(p, _)| p).collect();
    Ok(diversify_paths(paths, 2, limit))
}

fn diversify_paths(paths: Vec<String>, cap: usize, limit: usize) -> Vec<String> {
    let mut per_file: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut out: Vec<String> = Vec::with_capacity(limit);
    for p in &paths {
        let n = per_file.entry(p.clone()).or_insert(0);
        if *n >= cap {
            continue;
        }
        *n += 1;
        out.push(p.clone());
        if out.len() >= limit {
            return out;
        }
    }
    if out.len() < limit {
        let already: HashSet<String> = out.iter().cloned().collect();
        for p in paths {
            if out.len() >= limit {
                break;
            }
            if !already.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

fn tokenize_for_path_boost(q: &str) -> Vec<String> {
    q.split(|c: char| c.is_whitespace() || matches!(c, '/' | '_' | '-' | '.'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn count_path_segment_matches(path: &str, q_tokens: &[String]) -> usize {
    let segs: HashSet<String> = path
        .split(|c: char| matches!(c, '/' | '_' | '-' | '.'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect();
    q_tokens.iter().filter(|t| segs.contains(*t)).count()
}

/// First (1-based) rank where any expected substring is contained in a path.
fn first_correct_rank(paths: &[String], expect: &[String]) -> Option<usize> {
    for (i, p) in paths.iter().enumerate() {
        if expect.iter().any(|e| p.contains(e)) {
            return Some(i + 1);
        }
    }
    None
}

fn accumulate(
    bucket: &mut std::collections::HashMap<String, ModeStats>,
    mode: &str,
    rank: Option<usize>,
    latency_ms: f64,
) {
    let s = bucket.entry(mode.to_string()).or_default();
    s.n_queries += 1;
    s.sum_latency_ms += latency_ms;
    if let Some(r) = rank {
        if r <= 5 {
            s.n_recall_at_5 += 1;
        }
        if r <= 10 {
            s.n_recall_at_10 += 1;
        }
        if r <= 20 {
            s.n_recall_at_20 += 1;
        }
        s.sum_reciprocal_rank += 1.0 / r as f64;
    }
}

fn load_bank(path: &Path) -> Result<QueryBank> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&raw).context("parse query bank toml")
}

/// Sample N chunks from the index, build a synthetic query from each. We
/// prefer chunks whose `qualified` field is non-empty — that's the cleanest
/// label to use as a "natural" symbol-style query.
fn sample_synthetic(store: &Store, n: usize) -> Result<Vec<SyntheticQuery>> {
    let rows = store.sample_code_chunks(n)?;
    let mut out: Vec<SyntheticQuery> = Vec::with_capacity(n);
    let mut seen: HashSet<String> = HashSet::new();
    for (path, qualified) in rows {
        // Pick the first comma-separated token if there are several.
        let symbol = qualified.split(',').next().unwrap_or("").trim();
        if symbol.is_empty() || symbol.len() < 4 || !seen.insert(symbol.to_string()) {
            continue;
        }
        out.push(SyntheticQuery {
            query: symbol.to_string(),
            expected_path: path,
        });
    }
    Ok(out)
}

fn render_report(
    curated: &std::collections::HashMap<String, ModeStats>,
    synth: &std::collections::HashMap<String, ModeStats>,
    overall: &std::collections::HashMap<String, ModeStats>,
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# memorize-eval — code recall A/B");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "Each query runs under several modes. The headline number is recall@5 — does any expected path appear in the top 5 hits. Modes that weren't enabled in this run are omitted.\n"
    );

    let modes = ["hybrid", "bm25", "vector", "int8-only", "int8-hybrid"];
    let mut section = |label: &str, src: &std::collections::HashMap<String, ModeStats>| {
        let _ = writeln!(s, "## {label}\n");
        let _ = writeln!(
            s,
            "| mode | n | R@5 | R@10 | R@20 | MRR | mean latency (ms) |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|---|");
        for m in &modes {
            let st = match src.get(*m) {
                Some(s) if s.n_queries > 0 => s.clone(),
                _ => continue,
            };
            let _ = writeln!(
                s,
                "| {} | {} | {:.1}% | {:.1}% | {:.1}% | {:.3} | {:.1} |",
                m,
                st.n_queries,
                st.r_at(5, st.n_recall_at_5) * 100.0,
                st.r_at(10, st.n_recall_at_10) * 100.0,
                st.r_at(20, st.n_recall_at_20) * 100.0,
                st.mrr(),
                st.mean_latency_ms()
            );
        }
        let _ = writeln!(s);
    };

    if !overall.is_empty() {
        section("Overall", overall);
    }
    if !curated.is_empty() {
        section("Hand-curated queries", curated);
    }
    if !synth.is_empty() {
        section("Synthetic queries (qualified → path)", synth);
    }

    s
}
