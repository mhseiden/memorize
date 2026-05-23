use crate::RRF_K;
use memorize_store::{BM25Hit, VectorHit};
use std::collections::HashMap;

/// Fused candidate: id, session (carried for diversification), accumulated RRF score.
#[derive(Debug, Clone)]
pub struct Fused {
    pub id: i64,
    pub session: String,
    pub score: f64,
}

/// Reciprocal Rank Fusion over two ranked streams. Score for each doc is the
/// sum of `1 / (k + rank)` across streams it appears in. RRF is rank-based, so
/// no score normalization needed.
///
/// Both input vectors should be sorted by their native score, descending.
pub fn fuse(bm25: &[BM25Hit], vec: &[VectorHit]) -> Vec<Fused> {
    let mut acc: HashMap<i64, (String, f64)> = HashMap::new();

    for (rank, hit) in bm25.iter().enumerate() {
        let r = (rank + 1) as f64;
        let entry = acc
            .entry(hit.id)
            .or_insert_with(|| (hit.session.clone(), 0.0));
        entry.1 += 1.0 / (RRF_K + r);
    }
    for (rank, hit) in vec.iter().enumerate() {
        let r = (rank + 1) as f64;
        let entry = acc
            .entry(hit.id)
            .or_insert_with(|| (hit.session.clone(), 0.0));
        entry.1 += 1.0 / (RRF_K + r);
    }

    let mut out: Vec<Fused> = acc
        .into_iter()
        .map(|(id, (session, score))| Fused { id, session, score })
        .collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(id: i64, sess: &str, score: f64) -> BM25Hit {
        BM25Hit { id, session: sess.into(), score }
    }
    fn v(id: i64, sess: &str, score: f64) -> VectorHit {
        VectorHit { id, session: sess.into(), score }
    }

    #[test]
    fn fuse_promotes_dual_stream_hit() {
        // Doc 1 is rank-2 in BM25 and rank-1 in vector; doc 2 is rank-1 in
        // BM25 only. The dual-stream doc should fuse higher.
        let bm25 = vec![b(2, "s1", 1.0), b(1, "s1", 0.5)];
        let vec = vec![v(1, "s1", 0.9), v(3, "s1", 0.5)];
        let fused = fuse(&bm25, &vec);
        assert_eq!(fused[0].id, 1);
    }

    #[test]
    fn fuse_handles_empty_streams() {
        let fused = fuse(&[], &[]);
        assert!(fused.is_empty());
    }

    #[test]
    fn fuse_single_stream_preserves_order() {
        let bm25 = vec![b(3, "s1", 1.0), b(1, "s1", 0.5), b(2, "s1", 0.1)];
        let fused = fuse(&bm25, &[]);
        assert_eq!(fused.iter().map(|f| f.id).collect::<Vec<_>>(), vec![3, 1, 2]);
    }
}
