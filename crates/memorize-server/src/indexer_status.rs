//! Live status of the background indexer thread. Read by `/status` to give
//! clients a real-time view of what the indexer is doing.
//!
//! Updated under a Mutex from the indexer thread. Reads are cheap snapshots.

use serde::Serialize;
use std::sync::Mutex;
use std::time::Instant;

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

/// Shared handle. Cloning is cheap (Arc) — every component that touches the
/// status grabs a reference.
#[derive(Debug, Clone)]
pub struct IndexerStatus {
    inner: std::sync::Arc<Mutex<IndexerSnapshot>>,
    started_at: Instant,
}

const MAX_RECENT_ERRORS: usize = 8;

impl IndexerStatus {
    pub fn new(initial: IndexerSnapshot) -> Self {
        Self {
            inner: std::sync::Arc::new(Mutex::new(initial)),
            started_at: Instant::now(),
        }
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

    pub fn record_event(&self, path: String) {
        let ts = chrono::Utc::now().timestamp();
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
