use crate::{MAX_PER_SESSION, rrf::Fused};
use std::collections::HashMap;

/// Cap consecutive sessions in the result list. Mirrors agentmemory's
/// `diversifyBySession` (hardcoded max 3). If the cap leaves us short of
/// `limit`, fill the rest unconstrained from the same input order so we
/// don't truncate the result set artificially.
pub fn diversify_by_session(fused: Vec<Fused>, limit: usize) -> Vec<Fused> {
    let mut per_session: HashMap<String, usize> = HashMap::new();
    let mut kept: Vec<Fused> = Vec::with_capacity(limit);
    let mut deferred: Vec<Fused> = Vec::new();

    for f in fused {
        let count = per_session.entry(f.session.clone()).or_insert(0);
        if *count < MAX_PER_SESSION {
            *count += 1;
            kept.push(f);
            if kept.len() == limit {
                return kept;
            }
        } else {
            deferred.push(f);
        }
    }

    // Fall back to unconstrained fill if diversification thinned the list.
    for f in deferred {
        if kept.len() == limit {
            break;
        }
        kept.push(f);
    }

    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(id: i64, sess: &str, score: f64) -> Fused {
        Fused { id, session: sess.into(), score }
    }

    #[test]
    fn caps_per_session() {
        // 5 hits from session A, 2 from B. Cap is 3 per session; expect 3 A
        // first (input order), then 2 B, then 2 leftover A from the deferred
        // fill.
        let fused = vec![
            f(1, "A", 1.0), f(2, "A", 0.9), f(3, "A", 0.8),
            f(4, "A", 0.7), f(5, "A", 0.6),
            f(6, "B", 0.5), f(7, "B", 0.4),
        ];
        let out = diversify_by_session(fused, 10);
        let from_a = out.iter().filter(|f| f.session == "A").count();
        let from_b = out.iter().filter(|f| f.session == "B").count();
        assert_eq!(from_a, 5);
        assert_eq!(from_b, 2);
        // First three positions are A (cap), next two are B, then the deferred A's.
        assert_eq!(
            out.iter().map(|f| f.session.as_str()).collect::<Vec<_>>(),
            vec!["A", "A", "A", "B", "B", "A", "A"]
        );
    }

    #[test]
    fn early_termination_on_limit() {
        let fused: Vec<_> = (0..10).map(|i| f(i, "A", 1.0 - i as f64 * 0.1)).collect();
        let out = diversify_by_session(fused, 2);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn empty_input_empty_output() {
        let out = diversify_by_session(vec![], 5);
        assert!(out.is_empty());
    }
}
