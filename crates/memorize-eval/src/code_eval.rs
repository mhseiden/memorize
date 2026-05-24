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

pub struct CodeEvalOpts {
    pub db_path: PathBuf,
    pub bank_path: PathBuf,
    pub limit: usize,
    pub synthetic_count: usize,
    pub out_dir: PathBuf,
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

    let modes = [
        ("hybrid", RecallMode::Hybrid),
        ("bm25", RecallMode::Bm25Only),
        ("vector", RecallMode::VectorOnly),
    ];

    let mut per_query: Vec<PerQueryResult> = Vec::new();
    let mut stats_curated: std::collections::HashMap<String, ModeStats> = Default::default();
    let mut stats_synth: std::collections::HashMap<String, ModeStats> = Default::default();
    let mut stats_overall: std::collections::HashMap<String, ModeStats> = Default::default();

    for (mode_name, mode) in &modes {
        for q in &bank.queries {
            let res = run_one(&store, &q.query, q.path_prefix.clone(), opts.limit, *mode)?;
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
            let res = run_one(&store, &q.query, None, opts.limit, *mode)?;
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
    query: &str,
    path_prefix: Option<String>,
    limit: usize,
    mode: RecallMode,
) -> Result<RunOnce> {
    let cfg = CodeRecallConfig {
        mode,
        path_prefix,
        ..CodeRecallConfig::default()
    };
    let q_emb = memorize_embed::embed(query).context("embed eval query")?;
    let t = Instant::now();
    let res = recall_code(store, query, &q_emb, limit, &cfg)?;
    let latency_ms = t.elapsed().as_secs_f64() * 1000.0;
    Ok(RunOnce {
        paths: res.into_iter().map(|r| r.path).collect(),
        latency_ms,
    })
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
        "Each query runs under three modes (Hybrid / BM25-only / Vector-only). The headline number is recall@5 — does any expected path appear in the top 5 hits.\n"
    );

    let modes = ["hybrid", "bm25", "vector"];
    let mut section = |label: &str, src: &std::collections::HashMap<String, ModeStats>| {
        let _ = writeln!(s, "## {label}\n");
        let _ = writeln!(
            s,
            "| mode | n | R@5 | R@10 | R@20 | MRR | mean latency (ms) |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|---|");
        for m in &modes {
            let st = src.get(*m).cloned().unwrap_or_default();
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
