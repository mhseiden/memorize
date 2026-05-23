//! Per-question loop: build an ephemeral index from a question's haystack,
//! recall against the question, score gold-session ranks.

use crate::cache::{Cache, hash_text};
use crate::dataset::{self, Question};
use crate::metrics::{PerQuestion, PhaseTimings, aggregate, by_type, summarize_phases};
use crate::report;
use anyhow::{Context, Result};
use memorize_core::{Kind, NewObservation};
use memorize_recall::{RecallConfig, recall_with_config};
use memorize_store::Store;
use std::path::Path;
use std::time::Instant;

pub fn run(cfg: RecallConfig, limit: Option<usize>, out_dir: &Path, use_cache: bool) -> Result<()> {
    eprintln!("loading dataset…");
    let questions = dataset::load()?;
    let total = limit.unwrap_or(questions.len()).min(questions.len());
    eprintln!("dataset: {} questions; running {}", questions.len(), total);
    eprintln!(
        "config: model={} ({}d)  mode={:?} k={} top_k={} diversify={:?} syn={} cache={}",
        memorize_embed::model_tag(),
        memorize_embed::embedding_dim(),
        cfg.mode,
        cfg.rrf_k,
        cfg.per_stream_top_k,
        cfg.diversify_cap,
        cfg.use_synonyms,
        use_cache
    );

    let cache = if use_cache {
        let c = Cache::open().context("open embedding cache")?;
        eprintln!("cache: {} entries preloaded", c.len()?);
        Some(c)
    } else {
        None
    };

    let start = Instant::now();
    let mut records: Vec<PerQuestion> = Vec::with_capacity(total);
    let mut total_hits = 0usize;
    let mut total_misses = 0usize;
    for (i, q) in questions.iter().take(total).enumerate() {
        let rec = run_one(q, &cfg, cache.as_ref())
            .with_context(|| format!("question {} ({})", i, q.question_id))?;
        total_hits += rec.timings.cache_hits;
        total_misses += rec.timings.cache_misses;
        records.push(rec);
        if (i + 1) % 25 == 0 || i + 1 == total {
            let pct = ((i + 1) as f64 / total as f64) * 100.0;
            let r5 = records.iter().map(|r| crate::metrics::recall_any_at(r, 5)).sum::<f64>() / records.len() as f64;
            let hit_rate = if total_hits + total_misses > 0 {
                (total_hits as f64 / (total_hits + total_misses) as f64) * 100.0
            } else {
                0.0
            };
            eprintln!(
                "  {:>3}/{} ({:>5.1}%) R@5={:.3} cache_hit={:>5.1}% elapsed={:.0}s",
                i + 1, total, pct, r5, hit_rate, start.elapsed().as_secs_f64()
            );
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "\ncache: {} hits, {} misses (overall {:.1}% hit rate)",
        total_hits,
        total_misses,
        if total_hits + total_misses > 0 {
            (total_hits as f64 / (total_hits + total_misses) as f64) * 100.0
        } else {
            0.0
        }
    );
    let overall = aggregate(&records);
    let per_type = by_type(&records);

    eprintln!("\n=== overall ===");
    eprintln!(
        "R@5={:.4} R@10={:.4} R@20={:.4} NDCG@10={:.4} MRR={:.4} P@5={:.4}",
        overall.recall_at_5,
        overall.recall_at_10,
        overall.recall_at_20,
        overall.ndcg_at_10,
        overall.mrr,
        overall.precision_at_5
    );
    eprintln!("elapsed {:.1}s", elapsed.as_secs_f64());

    let phase_summary = summarize_phases(&records);
    eprintln!("\n=== phase timings ===");
    eprintln!(
        "{:<16} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "phase", "total_ms", "mean_ms", "p50", "p95", "max", "share"
    );
    for p in &phase_summary {
        eprintln!(
            "{:<16} {:>10} {:>10.1} {:>10} {:>10} {:>10} {:>7.1}%",
            p.phase, p.total_ms, p.mean_ms, p.p50_ms, p.p95_ms, p.max_ms, p.share_pct
        );
    }

    report::write_all(
        out_dir,
        &cfg,
        total,
        elapsed,
        &overall,
        &per_type,
        &phase_summary,
        &records,
    )?;
    eprintln!("\nartifacts written to {}", out_dir.display());
    Ok(())
}

fn run_one(q: &Question, cfg: &RecallConfig, cache: Option<&Cache>) -> Result<PerQuestion> {
    let started = Instant::now();
    let mut timings = PhaseTimings::default();

    // 1. Open store. Dim matches the active embedding model.
    let t = Instant::now();
    let dim = memorize_embed::embedding_dim();
    let store = Store::open_in_memory_with_dim(dim).context("open in-memory store")?;
    if !cfg.use_synonyms {
        let _ = clear_synonyms(&store);
    }
    timings.open_store_ms = t.elapsed().as_millis();

    // 2. Embed all sessions — cached lookups first, then a single batched
    //    ONNX call for the misses.
    let t = Instant::now();
    let bodies: Vec<String> = (0..q.haystack_sessions.len())
        .map(|i| q.session_body(i))
        .collect();
    let embeddings = embed_with_cache(&bodies, cache, &mut timings)
        .context("embed haystack")?;
    timings.embed_haystack_ms = t.elapsed().as_millis();

    // 3. Insert obs + vec rows.
    let t = Instant::now();
    let now = chrono::Utc::now().timestamp();
    for (i, body) in bodies.iter().enumerate() {
        let obs = NewObservation {
            session: q.haystack_session_ids[i].clone(),
            kind: Kind::Other,
            body: body.clone(),
            branch: None,
            ..Default::default()
        };
        store
            .insert_obs(&obs, now, &embeddings[i])
            .with_context(|| format!("insert session #{i}"))?;
    }
    timings.insert_ms = t.elapsed().as_millis();

    // 4. Rebuild FTS (cheap at our scale but worth measuring).
    let t = Instant::now();
    store.rebuild_fts().context("rebuild fts")?;
    timings.rebuild_fts_ms = t.elapsed().as_millis();

    // 5. Embed the query (single text → one ONNX call).
    let t = Instant::now();
    let q_emb = memorize_embed::embed(&q.question).context("embed question")?;
    timings.embed_query_ms = t.elapsed().as_millis();

    // 6. Recall pipeline.
    let t = Instant::now();
    let results = recall_with_config(&store, &q.question, &q_emb, 50, cfg)
        .context("recall")?;
    timings.recall_ms = t.elapsed().as_millis();

    // 7. Score: compute 1-based ranks of gold sessions.
    let t = Instant::now();
    let gold = &q.answer_session_ids;
    let mut gold_ranks: Vec<usize> = Vec::with_capacity(gold.len());
    for (rank0, r) in results.iter().enumerate() {
        if gold.contains(&r.obs.session) {
            gold_ranks.push(rank0 + 1);
        }
    }
    timings.score_ms = t.elapsed().as_millis();

    Ok(PerQuestion {
        question_id: q.question_id.clone(),
        question_type: q.question_type.clone(),
        gold_total: gold.len(),
        gold_ranks,
        latency_ms: started.elapsed().as_millis(),
        timings,
    })
}

/// Return embeddings for `bodies` in input order, populating cache misses
/// from a single batched ONNX call. Updates `cache_hits` / `cache_misses` on
/// the supplied timings.
fn embed_with_cache(
    bodies: &[String],
    cache: Option<&Cache>,
    timings: &mut PhaseTimings,
) -> Result<Vec<Vec<f32>>> {
    let keys: Vec<String> = bodies.iter().map(|b| hash_text(b)).collect();
    let cached = match cache {
        Some(c) => c.get_many(&keys)?,
        None => std::collections::HashMap::new(),
    };

    // Indices that aren't in the cache and need a fresh embedding.
    let mut miss_indices: Vec<usize> = Vec::new();
    let mut miss_texts: Vec<&str> = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        if !cached.contains_key(key) {
            miss_indices.push(i);
            miss_texts.push(bodies[i].as_str());
        }
    }
    timings.cache_hits = bodies.len() - miss_indices.len();
    timings.cache_misses = miss_indices.len();

    let new_embs = if miss_texts.is_empty() {
        Vec::new()
    } else {
        memorize_embed::embed_batch(&miss_texts).context("embed cache misses")?
    };

    // Write new embeddings into the cache before assembling the output —
    // cheap (under 1ms for ~48 inserts) and means a crash mid-question
    // doesn't waste the work.
    if let Some(c) = cache {
        let to_put: Vec<(String, &[f32])> = miss_indices
            .iter()
            .zip(new_embs.iter())
            .map(|(idx, emb)| (keys[*idx].clone(), emb.as_slice()))
            .collect();
        c.put_many(&to_put)?;
    }

    // Assemble in original order.
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(bodies.len());
    let mut new_iter = new_embs.into_iter();
    for (i, key) in keys.iter().enumerate() {
        if let Some(v) = cached.get(key) {
            out.push(v.clone());
        } else {
            let _ = i; // index aligns by construction
            out.push(new_iter.next().expect("miss count and new_embs length agree"));
        }
    }
    Ok(out)
}

fn clear_synonyms(store: &Store) -> Result<()> {
    // memorize-store doesn't (and shouldn't) expose raw SQL, so we satisfy
    // the ablation by removing each seeded pair. Cheap (~50 rows).
    let pairs = store.list_synonyms()?;
    for (term, expansion) in pairs {
        let _ = store.remove_synonym(&term, Some(&expansion));
    }
    Ok(())
}

