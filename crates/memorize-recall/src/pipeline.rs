use crate::diversify::diversify_by_session;
use crate::expand::build_fts_query;
use crate::rrf::{Fused, fuse};
use crate::PER_STREAM_TOP_K;
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

/// End-to-end recall. Caller owns the store + the query embedding (so we don't
/// take a dependency on `memorize-embed` here — easier to test, and the
/// caller already has the embedder warmed up in production).
pub fn recall(
    store: &Store,
    query: &str,
    query_emb: &[f32],
    limit: usize,
) -> Result<Vec<Recalled>> {
    if limit == 0 {
        return Ok(vec![]);
    }
    let fts_query = build_fts_query(query, store)?;
    let bm25 = store.search_bm25(&fts_query, PER_STREAM_TOP_K)?;
    let vec = store.search_vector(query_emb, PER_STREAM_TOP_K)?;

    let fused: Vec<Fused> = fuse(&bm25, &vec);
    let diversified = diversify_by_session(fused, limit);

    let ids: Vec<i64> = diversified.iter().map(|f| f.id).collect();
    let observations = store.get_obs_by_ids(&ids)?;
    let scores: Vec<f64> = diversified.iter().map(|f| f.score).collect();

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
    use memorize_store::EMBED_DIM;

    /// Synthetic fixed-pattern embeddings so cosine ranking is predictable
    /// without invoking fastembed in tests.
    fn synth_emb(seed: u32) -> Vec<f32> {
        let mut v = vec![0.0f32; EMBED_DIM];
        for (i, x) in v.iter_mut().enumerate() {
            *x = (((seed as usize + i) * 2654435761) % 1000) as f32 / 1000.0;
        }
        // L2 normalize so cosine = dot.
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
        };
        store.insert_obs(&obs, ts, &synth_emb(seed)).unwrap()
    }

    #[test]
    fn synonym_expansion_finds_kubernetes_via_k8s_query() {
        let s = Store::open_in_memory().unwrap();
        // No occurrence of "k8s" in the doc — only "kubernetes". The query
        // says "k8s scheduling". Synonym expansion must bridge the gap.
        ins(&s, "s1", "learned about kubernetes pod scheduling today", 100, 1);
        ins(&s, "s2", "the weather is nice today", 200, 2);
        s.rebuild_fts().unwrap();

        let q = "k8s scheduling";
        let q_emb = synth_emb(99);
        let out = recall(&s, q, &q_emb, 5).unwrap();
        assert!(!out.is_empty());
        // The kubernetes doc should be ranked first thanks to BM25 hitting on
        // both "kubernetes" (via synonym) and "scheduling".
        assert_eq!(out[0].obs.id, 1);
    }

    #[test]
    fn diversification_caps_per_session() {
        let s = Store::open_in_memory().unwrap();
        // Five docs in s1 + two in s2, all matching "alpha".
        for i in 0..5 {
            ins(&s, "s1", &format!("alpha doc number {}", i), 100 + i as i64, 10 + i as u32);
        }
        ins(&s, "s2", "alpha distinct one", 200, 100);
        ins(&s, "s2", "alpha distinct two", 201, 101);
        s.rebuild_fts().unwrap();

        let q_emb = synth_emb(99);
        let out = recall(&s, "alpha", &q_emb, 7).unwrap();
        // All 7 returned, but the first chunk should respect the per-session cap.
        let first_three_sessions: Vec<&str> =
            out.iter().take(3).map(|r| r.obs.session.as_str()).collect();
        // No session appears more than 3 times in the first 3.
        let s1_in_top3 = first_three_sessions.iter().filter(|s| **s == "s1").count();
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
}
