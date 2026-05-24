//! Reproducible quality measurement for memorize against LongMemEval-S.
//!
//! Two subcommands:
//!   `memorize-eval fetch`   — download the 265MB dataset from HuggingFace
//!   `memorize-eval run ...` — run the harness with the given ablation knobs

mod cache;
mod code_eval;
mod dataset;
mod harness;
mod metrics;
mod report;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use memorize_recall::{Mode as RecallMode, RecallConfig};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "memorize-eval", about = "LongMemEval-S harness for memorize")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download LongMemEval-S (~265MB) from HuggingFace.
    Fetch,
    /// Run the harness, write artifacts under --out.
    Run {
        #[arg(long, value_enum, default_value_t = ModeArg::Hybrid)]
        mode: ModeArg,
        #[arg(long, default_value_t = 60.0)]
        rrf_k: f64,
        #[arg(long, default_value_t = 50)]
        top_k: usize,
        #[arg(long, default_value_t = true, value_parser = parse_on_off)]
        diversify: bool,
        #[arg(long, default_value_t = 3)]
        diversify_cap: usize,
        #[arg(long, default_value_t = true, value_parser = parse_on_off)]
        synonyms: bool,
        /// Optional: sample N questions instead of all 500.
        #[arg(long)]
        limit: Option<usize>,
        /// Output directory for report.md / report.json / summary.csv.
        #[arg(long, default_value = "out")]
        out: PathBuf,
        /// Disable the on-disk embedding cache (useful for cold-cache timing baselines).
        #[arg(long)]
        no_cache: bool,
    },
    /// Run the code-recall A/B harness against the live `~/.memorize/db.duckdb`
    /// (opened read-only — the daemon can keep indexing). Sweeps Hybrid /
    /// Bm25Only / VectorOnly and reports R@K + MRR.
    CodeEval {
        /// Path to the DuckDB file. Defaults to ~/.memorize/db.duckdb.
        #[arg(long)]
        db: Option<PathBuf>,
        /// Hand-curated query bank (TOML).
        #[arg(long, default_value = "bench/code-queries.toml")]
        bank: PathBuf,
        /// Top-K returned per query.
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// How many synthetic queries to sample from the live index.
        /// 0 to disable.
        #[arg(long, default_value_t = 200)]
        synthetic: usize,
        /// Output directory for code-eval.md / code-eval.json.
        #[arg(long, default_value = "out")]
        out: PathBuf,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ModeArg {
    Hybrid,
    Bm25Only,
    VectorOnly,
}

impl From<ModeArg> for RecallMode {
    fn from(a: ModeArg) -> Self {
        match a {
            ModeArg::Hybrid => RecallMode::Hybrid,
            ModeArg::Bm25Only => RecallMode::Bm25Only,
            ModeArg::VectorOnly => RecallMode::VectorOnly,
        }
    }
}

fn parse_on_off(s: &str) -> std::result::Result<bool, String> {
    match s.to_lowercase().as_str() {
        "on" | "true" | "1" | "yes" => Ok(true),
        "off" | "false" | "0" | "no" => Ok(false),
        other => Err(format!("expected on|off, got {other}")),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Fetch => dataset::fetch(),
        Cmd::Run {
            mode,
            rrf_k,
            top_k,
            diversify,
            diversify_cap,
            synonyms,
            limit,
            out,
            no_cache,
        } => {
            let cfg = RecallConfig {
                mode: mode.into(),
                rrf_k,
                per_stream_top_k: top_k,
                diversify_cap: diversify.then_some(diversify_cap),
                use_synonyms: synonyms,
            };
            harness::run(cfg, limit, &out, !no_cache)
        }
        Cmd::CodeEval { db, bank, limit, synthetic, out } => {
            let db_path = db.unwrap_or_else(|| {
                let home = std::env::var("HOME").expect("HOME unset");
                PathBuf::from(home).join(".memorize/db.duckdb")
            });
            code_eval::run(code_eval::CodeEvalOpts {
                db_path,
                bank_path: bank,
                limit,
                synthetic_count: synthetic,
                out_dir: out,
            })
        }
    }
}
