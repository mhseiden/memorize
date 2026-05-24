//! JSONL log at `~/.memorize/indexer.log`. Append-only, single writer
//! (the indexer thread), no rotation, no buffering. One line per
//! milestone or error event.

use chrono::Utc;
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct IndexerLog {
    file: Mutex<Option<File>>,
}

impl IndexerLog {
    pub fn open() -> Self {
        let file = log_path()
            .ok()
            .and_then(|p| {
                if let Some(parent) = p.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(&p)
                    .ok()
            });
        IndexerLog {
            file: Mutex::new(file),
        }
    }

    pub fn write<T: Serialize>(&self, event: &T) {
        let mut guard = match self.file.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
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
