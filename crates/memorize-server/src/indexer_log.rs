//! JSONL log at `~/.memorize/indexer.log`. Append-only, single writer
//! (the indexer thread), single-generation size rotation, no buffering. One
//! line per milestone or error event.

use chrono::Utc;
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct IndexerLog {
    path: Option<PathBuf>,
    /// Rotate when the file reaches this size. 0 disables rotation.
    max_bytes: u64,
    file: Mutex<Option<File>>,
}

impl IndexerLog {
    pub fn open(max_bytes: u64) -> Self {
        let path = log_path().ok();
        let file = path.as_ref().and_then(|p| open_append(p));
        IndexerLog {
            path,
            max_bytes,
            file: Mutex::new(file),
        }
    }

    pub fn write<T: Serialize>(&self, event: &T) {
        let mut guard = match self.file.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        self.rotate_if_needed(&mut guard);
        let f = match guard.as_mut() {
            Some(f) => f,
            None => return,
        };
        if let Ok(mut line) = serde_json::to_string(event) {
            line.push('\n');
            // POSIX append: one write_all per line is atomic up to PIPE_BUF.
            let _ = f.write_all(line.as_bytes());
        }
    }

    /// Single-generation size rotation: once `indexer.log` reaches the cap,
    /// move it to `indexer.log.1` (overwriting any prior one) and reopen a
    /// fresh file. Keeps the activity log bounded without losing the most
    /// recent window.
    fn rotate_if_needed(&self, guard: &mut Option<File>) {
        if self.max_bytes == 0 {
            return;
        }
        let path = match &self.path {
            Some(p) => p,
            None => return,
        };
        let too_big = guard
            .as_ref()
            .and_then(|f| f.metadata().ok())
            .map(|m| m.len() >= self.max_bytes)
            .unwrap_or(false);
        if !too_big {
            return;
        }
        let mut rotated = path.clone().into_os_string();
        rotated.push(".1");
        // Drop the current handle before renaming, then reopen fresh. On any
        // rename failure, just reopen the original so we never drop events.
        *guard = None;
        let _ = std::fs::rename(path, PathBuf::from(rotated));
        *guard = open_append(path);
    }
}

fn open_append(path: &PathBuf) -> Option<File> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .ok()
}

/// `~/.memorize/indexer.log` by default; `MEMORIZE_INDEXER_LOG` overrides.
fn log_path() -> std::io::Result<PathBuf> {
    if let Ok(p) = std::env::var("MEMORIZE_INDEXER_LOG") {
        return Ok(PathBuf::from(p));
    }
    let home = std::env::var("HOME")
        .map_err(|_| std::io::Error::other("HOME unset"))?;
    Ok(PathBuf::from(home).join(".memorize").join("indexer.log"))
}

// ---- structured event shapes ----
//
// Defined as one outer enum so all events share the same `event` tag and the
// log stays jq-friendly: `jq 'select(.event=="file_indexed")'`.

#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum LogEvent {
    Started {
        ts: String,
        roots: Vec<String>,
        respect_gitignore: bool,
    },
    Disabled {
        ts: String,
    },
    ColdScanRootStart {
        ts: String,
        root: String,
    },
    ColdScanRootDone {
        ts: String,
        root: String,
        files_indexed: u64,
        files_skipped: u64,
        files_excluded: u64,
        elapsed_ms: u128,
    },
    ColdScanComplete {
        ts: String,
        total_files_indexed: u64,
        total_files_skipped: u64,
        total_chunks_in_index: i64,
    },
    FileIndexed {
        ts: String,
        path: String,
        language: String,
        chunks: usize,
        bytes: u64,
    },
    FileSkippedUnchanged {
        ts: String,
        path: String,
    },
    FileSkippedTooBig {
        ts: String,
        path: String,
        bytes: u64,
    },
    FileRemoved {
        ts: String,
        path: String,
    },
    WatcherEvent {
        ts: String,
        path: String,
        kind: String,
    },
    Error {
        ts: String,
        at: &'static str,
        path: Option<String>,
        msg: String,
    },
}

/// Helper: current timestamp formatted as RFC3339.
pub fn now() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}
