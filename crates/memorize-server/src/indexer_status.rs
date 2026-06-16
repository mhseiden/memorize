//! Live status of the background indexer thread. Read by `/status` to give
//! clients a real-time view of what the indexer is doing.
//!
//! Updated under a Mutex from the indexer thread. Reads are cheap snapshots.

use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexerPhase {
    Disabled,
    /// Cold-scan in progress — walking the configured roots.
    Scanning,
    /// Cold-scan done; receiving file events via notify-debouncer.
    Watching,
    /// Watcher has been quiet for a while; thread is waiting on the channel.
    Idle,
    /// Indexer failed to start or crashed.
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexerSnapshot {
    pub phase: IndexerPhase,
    pub current_root: Option<String>,
    pub roots: Vec<String>,
    /// Files touched (parsed + embedded + upserted) during the lifetime of
    /// this serve process.
    pub files_indexed: u64,
    /// Files visited but skipped (mtime+size match → already current).
    pub files_skipped: u64,
    /// Files visited but excluded (matched exclude pattern, too big, etc.).
    pub files_excluded: u64,
    /// Total chunks emitted across the lifetime of this serve.
    pub chunks_written: u64,
    /// Errors during indexing — kept small (last 8) for diagnostic surface.
    pub recent_errors: Vec<String>,
    /// Last file the watcher touched (any kind of event).
    pub last_event_path: Option<String>,
    /// Last event timestamp, unix seconds.
    pub last_event_ts: Option<i64>,
    /// When cold-scan completed, unix seconds. None if still scanning.
    pub cold_scan_completed_ts: Option<i64>,
}

impl IndexerSnapshot {
    pub fn initial(roots: Vec<String>, enabled: bool) -> Self {
        Self {
            phase: if enabled {
                IndexerPhase::Scanning
            } else {
                IndexerPhase::Disabled
            },
            current_root: None,
            roots,
            files_indexed: 0,
            files_skipped: 0,
            files_excluded: 0,
            chunks_written: 0,
            recent_errors: Vec::new(),
            last_event_path: None,
            last_event_ts: None,
            cold_scan_completed_ts: None,
        }
    }
}

/// One churning path and how many events it drew within the window.
#[derive(Debug, Clone, Serialize)]
pub struct PathCount {
    pub path: String,
    pub count: u64,
}

/// Rolling view of recent watcher events — the surface for root-causing churn.
#[derive(Debug, Clone, Serialize)]
pub struct ChurnSummary {
    pub window_secs: u64,
    pub events: u64,
    pub events_per_min: f64,
    /// Hottest paths within the window, descending.
    pub top_paths: Vec<PathCount>,
    /// Events grouped by configured root (longest-prefix match), else "other".
    pub by_root: HashMap<String, u64>,
    /// Events grouped by coarse notify kind (Create/Modify/Remove/…).
    pub by_kind: HashMap<String, u64>,
}

/// Bounded rolling buffer of `(when, path, kind)` over the configured window.
#[derive(Debug)]
struct ChurnTracker {
    window: Duration,
    max_len: usize,
    roots: Vec<String>,
    top_n: usize,
    events: VecDeque<(Instant, String, String)>,
}

impl ChurnTracker {
    const MAX_LEN: usize = 5000;
    const TOP_N: usize = 20;

    fn new(window_secs: u64, roots: Vec<String>) -> Self {
        Self {
            window: Duration::from_secs(window_secs),
            max_len: Self::MAX_LEN,
            roots,
            top_n: Self::TOP_N,
            events: VecDeque::new(),
        }
    }

    fn evict(&mut self, now: Instant) {
        while let Some((ts, _, _)) = self.events.front() {
            if now.duration_since(*ts) > self.window || self.events.len() > self.max_len {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    fn record(&mut self, now: Instant, path: String, kind: String) {
        self.events.push_back((now, path, kind));
        self.evict(now);
    }

    fn summary(&mut self, now: Instant) -> ChurnSummary {
        self.evict(now);
        let mut paths: HashMap<&str, u64> = HashMap::new();
        let mut by_root: HashMap<String, u64> = HashMap::new();
        let mut by_kind: HashMap<String, u64> = HashMap::new();
        for (_, path, kind) in &self.events {
            *paths.entry(path.as_str()).or_default() += 1;
            *by_kind.entry(kind.clone()).or_default() += 1;
            let root = self
                .roots
                .iter()
                .filter(|r| path.starts_with(r.as_str()))
                .max_by_key(|r| r.len())
                .cloned()
                .unwrap_or_else(|| "other".to_string());
            *by_root.entry(root).or_default() += 1;
        }
        let mut top: Vec<PathCount> = paths
            .into_iter()
            .map(|(p, c)| PathCount { path: p.to_string(), count: c })
            .collect();
        top.sort_by(|a, b| b.count.cmp(&a.count).then(a.path.cmp(&b.path)));
        top.truncate(self.top_n);
        let events = self.events.len() as u64;
        let window_secs = self.window.as_secs();
        let events_per_min = if window_secs == 0 {
            0.0
        } else {
            events as f64 * 60.0 / window_secs as f64
        };
        ChurnSummary {
            window_secs,
            events,
            events_per_min,
            top_paths: top,
            by_root,
            by_kind,
        }
    }
}

/// Shared handle. Cloning is cheap (Arc) — every component that touches the
/// status grabs a reference.
#[derive(Debug, Clone)]
pub struct IndexerStatus {
    inner: std::sync::Arc<Mutex<IndexerSnapshot>>,
    churn: std::sync::Arc<Mutex<ChurnTracker>>,
    started_at: Instant,
}

const MAX_RECENT_ERRORS: usize = 8;

impl IndexerStatus {
    pub fn new(initial: IndexerSnapshot, churn_window_secs: u64) -> Self {
        let churn = ChurnTracker::new(churn_window_secs, initial.roots.clone());
        Self {
            inner: std::sync::Arc::new(Mutex::new(initial)),
            churn: std::sync::Arc::new(Mutex::new(churn)),
            started_at: Instant::now(),
        }
    }

    /// Rolling churn view for `/status` and the heartbeat.
    pub fn churn_summary(&self) -> ChurnSummary {
        self.churn
            .lock()
            .expect("churn mutex")
            .summary(Instant::now())
    }

    pub fn snapshot(&self) -> IndexerSnapshot {
        self.inner.lock().expect("indexer-status mutex").clone()
    }

    pub fn server_uptime_secs(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    pub fn update<F: FnOnce(&mut IndexerSnapshot)>(&self, f: F) {
        let mut s = self.inner.lock().expect("indexer-status mutex");
        f(&mut s);
    }

    pub fn set_phase(&self, phase: IndexerPhase) {
        self.update(|s| s.phase = phase);
    }

    pub fn set_current_root(&self, root: Option<String>) {
        self.update(|s| s.current_root = root);
    }

    pub fn record_file_indexed(&self, chunks: usize) {
        self.update(|s| {
            s.files_indexed += 1;
            s.chunks_written += chunks as u64;
        });
    }

    pub fn record_file_skipped(&self) {
        self.update(|s| s.files_skipped += 1);
    }

    pub fn record_file_excluded(&self) {
        self.update(|s| s.files_excluded += 1);
    }

    pub fn record_error(&self, msg: String) {
        self.update(|s| {
            s.recent_errors.push(msg);
            if s.recent_errors.len() > MAX_RECENT_ERRORS {
                let n = s.recent_errors.len() - MAX_RECENT_ERRORS;
                s.recent_errors.drain(..n);
            }
        });
    }

    pub fn record_event(&self, path: String, kind: String) {
        let ts = chrono::Utc::now().timestamp();
        self.churn
            .lock()
            .expect("churn mutex")
            .record(Instant::now(), path.clone(), kind);
        self.update(|s| {
            s.last_event_path = Some(path);
            s.last_event_ts = Some(ts);
        });
    }

    pub fn mark_cold_scan_complete(&self) {
        let ts = chrono::Utc::now().timestamp();
        self.update(|s| {
            s.cold_scan_completed_ts = Some(ts);
            s.phase = IndexerPhase::Watching;
            s.current_root = None;
        });
    }
}
