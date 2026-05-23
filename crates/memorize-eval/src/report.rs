//! Three output formats: report.md (human), report.json (machine), summary.csv (tracking).

use crate::metrics::{Aggregate, PerQuestion, PhaseSummary};
use anyhow::{Context, Result};
use chrono::Utc;
use memorize_recall::{Mode, RecallConfig};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

pub fn write_all(
    out_dir: &Path,
    cfg: &RecallConfig,
    total_questions: usize,
    elapsed: Duration,
    overall: &Aggregate,
    per_type: &[(String, Aggregate)],
    phase_summary: &[PhaseSummary],
    records: &[PerQuestion],
) -> Result<()> {
    fs::create_dir_all(out_dir).context("create out dir")?;
    write_markdown(out_dir, cfg, total_questions, elapsed, overall, per_type, phase_summary)?;
    write_jsonl(out_dir, records)?;
    append_summary_csv(out_dir, cfg, total_questions, elapsed, overall, phase_summary)?;
    Ok(())
}

fn write_markdown(
    out_dir: &Path,
    cfg: &RecallConfig,
    total: usize,
    elapsed: Duration,
    overall: &Aggregate,
    per_type: &[(String, Aggregate)],
    phase_summary: &[PhaseSummary],
) -> Result<()> {
    let path = out_dir.join("report.md");
    let mut f = fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;

    let mode = mode_str(cfg.mode);
    let div = match cfg.diversify_cap {
        Some(n) => format!("on (cap={n})"),
        None => "off".into(),
    };
    let syn = if cfg.use_synonyms { "on" } else { "off" };
    let sha = git_sha().unwrap_or_else(|| "unknown".into());
    let ts = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

    writeln!(f, "# LongMemEval-S — memorize {}", env!("CARGO_PKG_VERSION"))?;
    writeln!(f)?;
    writeln!(
        f,
        "**model:** `{}` ({}d) &nbsp;&nbsp;**config:** mode=`{mode}` rrf_k={} top_k={} diversify={div} synonyms={syn}",
        memorize_embed::model_tag(),
        memorize_embed::embedding_dim(),
        cfg.rrf_k,
        cfg.per_stream_top_k
    )?;
    writeln!(f, "**git:** `{sha}` &nbsp;&nbsp;**run:** {ts} &nbsp;&nbsp;**questions:** {total} &nbsp;&nbsp;**elapsed:** {:.1}s", elapsed.as_secs_f64())?;
    writeln!(f)?;
    writeln!(f, "## Overall")?;
    writeln!(f)?;
    writeln!(f, "| R@5 | R@10 | R@20 | NDCG@10 | MRR | P@5 |")?;
    writeln!(f, "|---|---|---|---|---|---|")?;
    writeln!(
        f,
        "| **{:.3}** | **{:.3}** | **{:.3}** | {:.3} | {:.3} | {:.3} |",
        overall.recall_at_5,
        overall.recall_at_10,
        overall.recall_at_20,
        overall.ndcg_at_10,
        overall.mrr,
        overall.precision_at_5
    )?;
    writeln!(f)?;
    writeln!(f, "## By question type")?;
    writeln!(f)?;
    writeln!(f, "| type | count | R@5 | R@10 | NDCG@10 | MRR |")?;
    writeln!(f, "|---|---|---|---|---|---|")?;
    for (ty, agg) in per_type {
        writeln!(
            f,
            "| `{ty}` | {} | {:.3} | {:.3} | {:.3} | {:.3} |",
            agg.count, agg.recall_at_5, agg.recall_at_10, agg.ndcg_at_10, agg.mrr
        )?;
    }

    writeln!(f)?;
    writeln!(f, "## Phase timings (per question, milliseconds)")?;
    writeln!(f)?;
    writeln!(f, "| phase | total_ms | mean | p50 | p95 | max | share |")?;
    writeln!(f, "|---|---|---|---|---|---|---|")?;
    for p in phase_summary {
        writeln!(
            f,
            "| `{}` | {} | {:.1} | {} | {} | {} | {:.1}% |",
            p.phase, p.total_ms, p.mean_ms, p.p50_ms, p.p95_ms, p.max_ms, p.share_pct
        )?;
    }
    Ok(())
}

fn write_jsonl(out_dir: &Path, records: &[PerQuestion]) -> Result<()> {
    let path = out_dir.join("report.json");
    let mut f = fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
    for r in records {
        let line = serde_json::to_string(r)?;
        writeln!(f, "{line}")?;
    }
    Ok(())
}

fn append_summary_csv(
    out_dir: &Path,
    cfg: &RecallConfig,
    total: usize,
    elapsed: Duration,
    overall: &Aggregate,
    phase_summary: &[PhaseSummary],
) -> Result<()> {
    let path = out_dir.join("summary.csv");
    let exists = path.exists();
    let mut f = fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&path)
        .with_context(|| format!("open {}", path.display()))?;
    if !exists {
        writeln!(
            f,
            "ts,git_sha,model,model_dim,mode,rrf_k,top_k,diversify_cap,synonyms,limit,r5,r10,r20,ndcg10,mrr,p5,elapsed_s,embed_haystack_share,recall_share"
        )?;
    }
    // Two phase shares we care about most when tuning: embed (the bulk in the
    // baseline) and recall (the production hot path).
    let share = |name: &str| -> f64 {
        phase_summary
            .iter()
            .find(|p| p.phase == name)
            .map(|p| p.share_pct)
            .unwrap_or(0.0)
    };
    let embed_share = share("embed_haystack");
    let recall_share = share("recall");
    let div = cfg
        .diversify_cap
        .map(|n| n.to_string())
        .unwrap_or_else(|| "off".into());
    writeln!(
        f,
        "{ts},{sha},{model},{model_dim},{mode},{rrf_k},{top_k},{div},{syn},{total},{r5:.4},{r10:.4},{r20:.4},{ndcg10:.4},{mrr:.4},{p5:.4},{elapsed:.1},{embed_share:.1},{recall_share:.1}",
        model = memorize_embed::model_tag(),
        model_dim = memorize_embed::embedding_dim(),
        ts = Utc::now().format("%Y-%m-%dT%H:%M:%SZ"),
        sha = git_sha().unwrap_or_else(|| "unknown".into()),
        mode = mode_str(cfg.mode),
        rrf_k = cfg.rrf_k,
        top_k = cfg.per_stream_top_k,
        syn = if cfg.use_synonyms { "on" } else { "off" },
        r5 = overall.recall_at_5,
        r10 = overall.recall_at_10,
        r20 = overall.recall_at_20,
        ndcg10 = overall.ndcg_at_10,
        mrr = overall.mrr,
        p5 = overall.precision_at_5,
        elapsed = elapsed.as_secs_f64(),
    )?;
    Ok(())
}

fn mode_str(m: Mode) -> &'static str {
    match m {
        Mode::Hybrid => "hybrid",
        Mode::Bm25Only => "bm25-only",
        Mode::VectorOnly => "vector-only",
    }
}

fn git_sha() -> Option<String> {
    let out = Command::new("git")
        .arg("rev-parse")
        .arg("--short")
        .arg("HEAD")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    Some(s.trim().to_string())
}
