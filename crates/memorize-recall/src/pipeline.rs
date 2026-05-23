use crate::diversify::diversify_by_session;
use crate::expand::build_fts_query;
use crate::rrf::{Fused, fuse};
use crate::{Mode, RecallConfig};
use anyhow::Result;
use memorize_core::Observation;
use memorize_store::Store;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Recalled {
    pub obs: Observation,
    /// Final fused-and-diversified score. Useful for debugging recall quality.
    pub score: f64,
}

/// End-to-end recall with default config — production callers want this.
pub fn recall(
    store: &Store,
    query: &str,
    query_emb: &[f32],
    limit: usize,
) -> Result<Vec<Recalled>> {
    recall_with_config(store, query, query_emb, limit, &RecallConfig::default())
}

/// End-to-end recall with explicit config. Eval harness uses this to ablate.
pub fn recall_with_config(
    store: &Store,
    query: &str,
    query_emb: &[f32],
    limit: usize,
    cfg: &RecallConfig,
) -> Result<Vec<Recalled>> {
    if limit == 0 {
        return Ok(vec![]);
    }

    // BM25 query — with or without synonym expansion.
    let bm25 = match cfg.mode {
        Mode::VectorOnly => vec![],
        Mode::Hybrid | Mode::Bm25Only => {
            let fts_query = if cfg.use_synonyms {
                build_fts_query(query, store)?
            } else {
                query.to_string()
            };
            store.search_bm25(&fts_query, cfg.per_stream_top_k)?
        }
    };

    let vec = match cfg.mode {
        Mode::Bm25Only => vec![],
        Mode::Hybrid | Mode::VectorOnly => store.search_vector(query_emb, cfg.per_stream_top_k)?,
    };

    let fused: Vec<Fused> = fuse(&bm25, &vec, cfg.rrf_k);

    let final_ranked = match cfg.diversify_cap {
        Some(cap) => diversify_by_session(fused, limit, cap),
        None => fused.into_iter().take(limit).collect(),
    };

    let ids: Vec<i64> = final_ranked.iter().map(|f| f.id).collect();
    let observations = store.get_obs_by_ids(&ids)?;
    let scores: Vec<f64> = final_ranked.iter().map(|f| f.score).collect();

    Ok(observations
        .into_iter()
        .zip(scores)
        .map(|(obs, score)| Recalled { obs, score })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use memorize_core::{Kind, NewObservation};
    use memorize_store::DEFAULT_EMBED_DIM as EMBED_DIM;

    /// Synthetic fixed-pattern embeddings so cosine ranking is predictable
    /// without invoking fastembed in tests.
    fn synth_emb(seed: u32) -> Vec<f32> {
        let mut v = vec![0.0f32; EMBED_DIM];
        for (i, x) in v.iter_mut().enumerate() {
            *x = (((seed as usize + i) * 2654435761) % 1000) as f32 / 1000.0;
        }
        let norm: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    fn ins(store: &Store, session: &str, body: &str, ts: i64, seed: u32) -> i64 {
        let obs = NewObservation {
            session: session.into(),
            kind: Kind::Manual,
            body: body.into(),
            branch: None,
            ..Default::default()
        };
        store.insert_obs(&obs, ts, &synth_emb(seed)).unwrap()
    }

    #[test]
    fn synonym_expansion_finds_kubernetes_via_k8s_query() {
        let s = Store::open_in_memory().unwrap();
        ins(&s, "s1", "learned about kubernetes pod scheduling today", 100, 1);
        ins(&s, "s2", "the weather is nice today", 200, 2);
        s.rebuild_fts().unwrap();

        let q = "k8s scheduling";
        let q_emb = synth_emb(99);
        let out = recall(&s, q, &q_emb, 5).unwrap();
        assert!(!out.is_empty());
        assert_eq!(out[0].obs.id, 1);
    }

    #[test]
    fn diversification_caps_per_session() {
        let s = Store::open_in_memory().unwrap();
        for i in 0..5 {
            ins(&s, "s1", &format!("alpha doc number {}", i), 100 + i as i64, 10 + i as u32);
        }
        ins(&s, "s2", "alpha distinct one", 200, 100);
        ins(&s, "s2", "alpha distinct two", 201, 101);
        s.rebuild_fts().unwrap();

        let q_emb = synth_emb(99);
        let out = recall(&s, "alpha", &q_emb, 7).unwrap();
        let s1_in_top3 = out.iter().take(3).filter(|r| r.obs.session == "s1").count();
        assert!(s1_in_top3 <= 3);
    }

    #[test]
    fn empty_limit_returns_empty() {
        let s = Store::open_in_memory().unwrap();
        let out = recall(&s, "anything", &synth_emb(1), 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn empty_corpus_returns_empty() {
        let s = Store::open_in_memory().unwrap();
        let out = recall(&s, "anything", &synth_emb(1), 5).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn bm25_only_mode_skips_vector_stream() {
        // Insert two docs: one with the keyword, one without. Vector embeddings
        // are random — in vector-only mode either could win. In bm25-only mode,
        // the keyword doc must come first.
        let s = Store::open_in_memory().unwrap();
        ins(&s, "s1", "no relevant text here at all", 100, 1);
        ins(&s, "s2", "definitive keyword: bananaphone", 200, 2);
        s.rebuild_fts().unwrap();

        let cfg = RecallConfig { mode: Mode::Bm25Only, ..Default::default() };
        let out = recall_with_config(&s, "bananaphone", &synth_emb(99), 5, &cfg).unwrap();
        assert_eq!(out[0].obs.id, 2);
    }

    #[test]
    fn synonyms_off_means_no_expansion() {
        let s = Store::open_in_memory().unwrap();
        ins(&s, "s1", "learned about kubernetes pod scheduling today", 100, 1);
        ins(&s, "s2", "totally unrelated content", 200, 2);
        s.rebuild_fts().unwrap();

        // With synonyms off, "k8s" doesn't expand to "kubernetes" — BM25 finds nothing.
        // The vector stream might still rank doc 1 first, but the BM25 contribution disappears.
        let cfg = RecallConfig {
            mode: Mode::Bm25Only,
            use_synonyms: false,
            ..Default::default()
        };
        let q_emb = synth_emb(99);
        let out = recall_with_config(&s, "k8s", &q_emb, 5, &cfg).unwrap();
        // BM25 over "k8s" alone against bodies that contain "kubernetes" — zero hits.
        assert!(out.is_empty());
    }
}
