//! Retrieval metrics over per-question gold-rank records.

use serde::Serialize;

/// One question's outcome. `gold_ranks` is the 1-based positions in the
/// returned ranked list at which each gold session appeared. A gold session
/// that didn't appear contributes nothing.
#[derive(Debug, Clone, Serialize)]
pub struct PerQuestion {
    pub question_id: String,
    pub question_type: String,
    pub gold_total: usize,
    pub gold_ranks: Vec<usize>,
    pub latency_ms: u128,
    pub timings: PhaseTimings,
}

/// Wall-clock per phase, in milliseconds. Captured per question so we can
/// see where the harness spends its time and decide where to optimize.
/// `cache_hits` and `cache_misses` are counts (not ms) — number of haystack
/// sessions served from the embedding cache vs. freshly embedded.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PhaseTimings {
    pub open_store_ms: u128,
    pub embed_haystack_ms: u128,
    pub insert_ms: u128,
    pub rebuild_fts_ms: u128,
    pub embed_query_ms: u128,
    pub recall_ms: u128,
    pub score_ms: u128,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl PhaseTimings {
    pub fn total_ms(&self) -> u128 {
        self.open_store_ms
            + self.embed_haystack_ms
            + self.insert_ms
            + self.rebuild_fts_ms
            + self.embed_query_ms
            + self.recall_ms
            + self.score_ms
    }
}

/// Aggregated phase timings across a slice of PerQuestion records.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseSummary {
    pub phase: &'static str,
    pub total_ms: u128,
    pub mean_ms: f64,
    pub p50_ms: u128,
    pub p95_ms: u128,
    pub max_ms: u128,
    pub share_pct: f64,
}

pub fn summarize_phases(records: &[PerQuestion]) -> Vec<PhaseSummary> {
    let phases: &[(&'static str, fn(&PhaseTimings) -> u128)] = &[
        ("open_store", |t| t.open_store_ms),
        ("embed_haystack", |t| t.embed_haystack_ms),
        ("insert", |t| t.insert_ms),
        ("rebuild_fts", |t| t.rebuild_fts_ms),
        ("embed_query", |t| t.embed_query_ms),
        ("recall", |t| t.recall_ms),
        ("score", |t| t.score_ms),
    ];

    // Grand total across all phases × all questions.
    let grand_total: u128 = records.iter().map(|r| r.timings.total_ms()).sum::<u128>().max(1);

    phases
        .iter()
        .map(|(name, extract)| {
            let mut vals: Vec<u128> = records.iter().map(|r| extract(&r.timings)).collect();
            vals.sort_unstable();
            let n = vals.len().max(1);
            let total: u128 = vals.iter().sum();
            PhaseSummary {
                phase: name,
                total_ms: total,
                mean_ms: total as f64 / n as f64,
                p50_ms: percentile(&vals, 50),
                p95_ms: percentile(&vals, 95),
                max_ms: *vals.last().unwrap_or(&0),
                share_pct: (total as f64 / grand_total as f64) * 100.0,
            }
        })
        .collect()
}

fn percentile(sorted: &[u128], p: usize) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    // Nearest-rank, conservative for our use case.
    let idx = ((sorted.len() as f64 - 1.0) * (p as f64 / 100.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Aggregated metrics for a slice of PerQuestion records.
#[derive(Debug, Clone, Serialize)]
pub struct Aggregate {
    pub count: usize,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub recall_at_20: f64,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub precision_at_5: f64,
}

pub fn aggregate(records: &[PerQuestion]) -> Aggregate {
    let n = records.len() as f64;
    if records.is_empty() {
        return Aggregate {
            count: 0,
            recall_at_5: 0.0,
            recall_at_10: 0.0,
            recall_at_20: 0.0,
            ndcg_at_10: 0.0,
            mrr: 0.0,
            precision_at_5: 0.0,
        };
    }
    Aggregate {
        count: records.len(),
        recall_at_5: records.iter().map(|r| recall_any_at(r, 5)).sum::<f64>() / n,
        recall_at_10: records.iter().map(|r| recall_any_at(r, 10)).sum::<f64>() / n,
        recall_at_20: records.iter().map(|r| recall_any_at(r, 20)).sum::<f64>() / n,
        ndcg_at_10: records.iter().map(|r| ndcg_at(r, 10)).sum::<f64>() / n,
        mrr: records.iter().map(|r| reciprocal_rank(r)).sum::<f64>() / n,
        precision_at_5: records.iter().map(|r| precision_at(r, 5)).sum::<f64>() / n,
    }
}

/// 1 if any gold rank ≤ k, else 0. Matches LongMemEval's `recall_any@K`.
pub fn recall_any_at(r: &PerQuestion, k: usize) -> f64 {
    if r.gold_ranks.iter().any(|&rank| rank <= k) { 1.0 } else { 0.0 }
}

/// NDCG@k with binary relevance. DCG = sum(1 / log2(rank+1)) for gold ranks ≤ k.
/// IDCG = same formula assuming the top-min(|gold|, k) positions are all gold.
pub fn ndcg_at(r: &PerQuestion, k: usize) -> f64 {
    let dcg: f64 = r
        .gold_ranks
        .iter()
        .filter(|&&rank| rank <= k)
        .map(|&rank| 1.0 / ((rank as f64 + 1.0).log2()))
        .sum();
    let ideal_hits = r.gold_total.min(k);
    if ideal_hits == 0 {
        return 0.0;
    }
    let idcg: f64 = (1..=ideal_hits)
        .map(|i| 1.0 / ((i as f64 + 1.0).log2()))
        .sum();
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

/// Reciprocal of the first gold rank. 0 if no gold appeared.
pub fn reciprocal_rank(r: &PerQuestion) -> f64 {
    r.gold_ranks
        .iter()
        .copied()
        .min()
        .map(|rank| 1.0 / rank as f64)
        .unwrap_or(0.0)
}

/// Fraction of top-k positions that were gold.
pub fn precision_at(r: &PerQuestion, k: usize) -> f64 {
    let hits = r.gold_ranks.iter().filter(|&&rank| rank <= k).count();
    hits as f64 / k as f64
}

/// Group records by `question_type` and aggregate each bucket.
pub fn by_type(records: &[PerQuestion]) -> Vec<(String, Aggregate)> {
    use std::collections::BTreeMap;
    let mut buckets: BTreeMap<String, Vec<PerQuestion>> = BTreeMap::new();
    for r in records {
        buckets.entry(r.question_type.clone()).or_default().push(r.clone());
    }
    buckets
        .into_iter()
        .map(|(k, v)| (k, aggregate(&v)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pq(ranks: Vec<usize>, gold_total: usize) -> PerQuestion {
        PerQuestion {
            question_id: "q".into(),
            question_type: "test".into(),
            gold_total,
            gold_ranks: ranks,
            latency_ms: 0,
            timings: PhaseTimings::default(),
        }
    }

    #[test]
    fn recall_any_basic() {
        assert_eq!(recall_any_at(&pq(vec![3], 1), 5), 1.0);
        assert_eq!(recall_any_at(&pq(vec![6], 1), 5), 0.0);
        assert_eq!(recall_any_at(&pq(vec![], 1), 5), 0.0);
    }

    #[test]
    fn mrr_picks_first_gold() {
        // Two golds at positions 4 and 1. MRR uses the better one.
        assert!((reciprocal_rank(&pq(vec![4, 1], 2)) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn precision_at_5_counts_gold_in_topk() {
        // 2 golds in top-5 → 2/5 = 0.4
        assert!((precision_at(&pq(vec![1, 3, 9], 3), 5) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn ndcg_perfect_when_gold_at_top() {
        // 2 golds, both in top-2 → NDCG = 1.0
        let r = pq(vec![1, 2], 2);
        assert!((ndcg_at(&r, 10) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ndcg_decays_with_rank() {
        let top = pq(vec![1], 1);
        let mid = pq(vec![5], 1);
        assert!(ndcg_at(&top, 10) > ndcg_at(&mid, 10));
    }

    #[test]
    fn aggregate_empty() {
        let agg = aggregate(&[]);
        assert_eq!(agg.count, 0);
        assert_eq!(agg.recall_at_5, 0.0);
    }
}
