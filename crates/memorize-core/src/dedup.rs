use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Mutex;

/// 5-minute SHA-256 dedup window. Identical (session, kind, body[..500])
/// payloads within the window are dropped silently — agent retries and hook
/// echoes are common, and we'd rather lose a real duplicate than store five.
const WINDOW_SECS: i64 = 300;
const HASH_PREFIX_BYTES: usize = 500;

pub struct Dedup {
    seen: Mutex<HashMap<[u8; 32], i64>>,
}

impl Dedup {
    pub fn new() -> Self {
        Self { seen: Mutex::new(HashMap::new()) }
    }

    /// Returns `true` if this is fresh (caller should persist); `false` if it
    /// duplicates a recent entry (caller should drop). Time is passed in
    /// rather than read from a clock so tests stay deterministic.
    pub fn check_and_insert(
        &self,
        session: &str,
        kind: &str,
        body: &str,
        now_secs: i64,
    ) -> bool {
        let hash = compute_hash(session, kind, body);
        let mut seen = self.seen.lock().expect("dedup mutex poisoned");
        prune(&mut seen, now_secs);
        if let Some(&prev_ts) = seen.get(&hash) {
            if now_secs - prev_ts <= WINDOW_SECS {
                return false;
            }
        }
        seen.insert(hash, now_secs);
        true
    }
}

impl Default for Dedup {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_hash(session: &str, kind: &str, body: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(session.as_bytes());
    h.update([0]);
    h.update(kind.as_bytes());
    h.update([0]);
    let prefix_end = body.len().min(HASH_PREFIX_BYTES);
    h.update(&body.as_bytes()[..prefix_end]);
    h.finalize().into()
}

fn prune(map: &mut HashMap<[u8; 32], i64>, now_secs: i64) {
    map.retain(|_, ts| now_secs - *ts <= WINDOW_SECS);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_insert_is_fresh() {
        let d = Dedup::new();
        assert!(d.check_and_insert("s1", "tool_use", "Bash echo hi", 100));
    }

    #[test]
    fn second_identical_within_window_dropped() {
        let d = Dedup::new();
        assert!(d.check_and_insert("s1", "tool_use", "Bash echo hi", 100));
        assert!(!d.check_and_insert("s1", "tool_use", "Bash echo hi", 200));
    }

    #[test]
    fn same_payload_past_window_is_fresh_again() {
        let d = Dedup::new();
        assert!(d.check_and_insert("s1", "tool_use", "Bash echo hi", 100));
        assert!(d.check_and_insert("s1", "tool_use", "Bash echo hi", 100 + WINDOW_SECS + 1));
    }

    #[test]
    fn different_session_not_dropped() {
        let d = Dedup::new();
        assert!(d.check_and_insert("s1", "tool_use", "Bash echo hi", 100));
        assert!(d.check_and_insert("s2", "tool_use", "Bash echo hi", 101));
    }

    #[test]
    fn body_prefix_collision_dropped() {
        // Two bodies identical in the first 500 bytes but different after — by
        // design these dedup. Keeps tool outputs that only differ in a trailing
        // timestamp from re-storing.
        let d = Dedup::new();
        let prefix = "x".repeat(HASH_PREFIX_BYTES);
        let a = format!("{prefix}aaa");
        let b = format!("{prefix}bbb");
        assert!(d.check_and_insert("s1", "tool_use", &a, 100));
        assert!(!d.check_and_insert("s1", "tool_use", &b, 110));
    }
}
