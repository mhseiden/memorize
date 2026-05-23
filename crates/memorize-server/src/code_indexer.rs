//! Code indexer + file watcher.
//!
//! Cold-start: walk configured roots, index any file we haven't seen or
//! whose mtime/size has changed. Then start a debounced `notify` watcher
//! that re-indexes files on save and removes them on delete.
//!
//! This runs on its own background thread spawned at `serve` startup.
//! Failures are best-effort — logged to stderr (if MEMORIZE_VERBOSE is on)
//! and otherwise swallowed. The HTTP server keeps serving regardless.

use crate::state::ServerState;
use anyhow::{Context, Result};
use memorize_code::{CodeChunk, language_for_path};
use memorize_store::{CodeChunkRow, FileMeta};
use notify::RecursiveMode;
use notify_debouncer_full::DebouncedEvent;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

/// Static excludes — directories/files we never index. fnmatch-ish prefixes.
const ALWAYS_EXCLUDE: &[&str] = &[
    "/target/",
    "/node_modules/",
    "/.git/",
    "/dist/",
    "/build/",
    "/.next/",
    "/.cache/",
    "/__pycache__/",
];

/// Roots watched / scanned at startup. Configurable later; defaults match
/// where this user's repos live.
pub fn default_roots() -> Vec<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    if let Ok(raw) = std::env::var("MEMORIZE_CODE_ROOTS") {
        raw.split(':')
            .filter(|s| !s.is_empty())
            .map(|s| expand_tilde(s, &home))
            .collect()
    } else {
        [
            "~/Vibes/memorize",
            "~/Repos",
            "~/src",
        ]
        .iter()
        .map(|s| expand_tilde(s, &home))
        .filter(|p| p.exists())
        .collect()
    }
}

fn expand_tilde(s: &str, home: &str) -> PathBuf {
    if let Some(rest) = s.strip_prefix("~/") {
        PathBuf::from(home).join(rest)
    } else {
        PathBuf::from(s)
    }
}

/// Entry point. Spawns a background thread; returns immediately.
pub fn spawn(state: Arc<ServerState>) {
    std::thread::spawn(move || {
        let roots = default_roots();
        if roots.is_empty() {
            log("code-indexer: no roots configured, skipping");
            return;
        }
        log(&format!("code-indexer: roots = {:?}", roots));

        // Cold-start scan: index anything new or changed.
        for root in &roots {
            if let Err(e) = scan_root(&state, root) {
                log(&format!("code-indexer scan {}: {e}", root.display()));
            }
        }
        log(&format!(
            "code-indexer: cold scan complete; {} chunks in index",
            state.store.count_code_chunks().unwrap_or(0)
        ));

        // Watcher: react to subsequent changes.
        if let Err(e) = watch(&state, &roots) {
            log(&format!("code-indexer watcher: {e}"));
        }
    });
}

fn scan_root(state: &ServerState, root: &Path) -> Result<()> {
    walk_dir(state, root, root)
}

fn walk_dir(state: &ServerState, root: &Path, dir: &Path) -> Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()), // permission denied / vanished dir
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if is_excluded(&path) {
            continue;
        }
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if ft.is_dir() {
            // Don't follow symlinks — keep the scan bounded.
            let _ = walk_dir(state, root, &path);
        } else if ft.is_file() {
            if language_for_path(&path).is_some() {
                if let Err(e) = index_file(state, root, &path) {
                    log(&format!("index {}: {e}", path.display()));
                }
            }
        }
    }
    Ok(())
}

fn is_excluded(path: &Path) -> bool {
    let s = path.to_string_lossy();
    ALWAYS_EXCLUDE.iter().any(|pat| s.contains(pat))
}

/// Index one file. Skipped if (path, mtime, size) match what's already in
/// `files` — that's our cheap dedup check.
fn index_file(state: &ServerState, root: &Path, path: &Path) -> Result<()> {
    let path_str = path.to_string_lossy().to_string();
    let meta_fs = std::fs::metadata(path).with_context(|| format!("stat {path_str}"))?;
    if meta_fs.len() > 1_000_000 {
        // 1MB cap — pathological generated files don't belong in semantic search.
        return Ok(());
    }
    let mtime_ns = meta_fs
        .modified()
        .ok()
        .and_then(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .ok()
                .map(|d| d.as_nanos() as i64)
        })
        .unwrap_or(0);
    let size_bytes = meta_fs.len() as i64;

    if let Ok(Some(prev)) = state.store.get_file_meta(&path_str) {
        if prev.mtime_ns == mtime_ns && prev.size_bytes == size_bytes {
            return Ok(()); // unchanged, skip
        }
    }

    let source = std::fs::read_to_string(path).with_context(|| format!("read {path_str}"))?;
    let language = match language_for_path(path) {
        Some(l) => l,
        None => return Ok(()),
    };

    let chunks: Vec<CodeChunk> = memorize_code::chunk_source(&source, language)?;
    if chunks.is_empty() {
        return Ok(());
    }

    // Batch-embed chunk bodies in one ONNX call.
    let bodies: Vec<&str> = chunks.iter().map(|c| c.body.as_str()).collect();
    let embs = memorize_embed::embed_batch(&bodies).context("embed chunks")?;

    let rows: Vec<CodeChunkRow> = chunks
        .into_iter()
        .map(|c| CodeChunkRow {
            id: 0, // assigned in upsert_code_file
            path: path_str.clone(),
            language: c.language,
            line_start: c.line_start as i32,
            line_end: c.line_end as i32,
            kind: c.kind,
            qualified: c.qualified,
            body: c.body,
        })
        .collect();

    let meta = FileMeta {
        mtime_ns,
        size_bytes,
        git_rev: None,
    };
    state
        .store
        .upsert_code_file(
            &path_str,
            &root.to_string_lossy(),
            language,
            &meta,
            &rows,
            &embs,
        )
        .context("upsert code file")?;
    Ok(())
}

fn watch(state: &ServerState, roots: &[PathBuf]) -> Result<()> {
    let (tx, rx) = mpsc::channel();
    let mut debouncer = notify_debouncer_full::new_debouncer(
        Duration::from_millis(250),
        None,
        move |res: notify_debouncer_full::DebounceEventResult| {
            let _ = tx.send(res);
        },
    )?;
    for root in roots {
        if let Err(e) = debouncer.watch(root, RecursiveMode::Recursive) {
            log(&format!("watch {}: {e}", root.display()));
        }
    }
    // Hold the debouncer; dropping it would stop the watcher.
    let _keep_alive = debouncer;

    let mut last_fts_rebuild = std::time::Instant::now();
    let fts_rebuild_interval = Duration::from_secs(5);

    while let Ok(res) = rx.recv() {
        match res {
            Ok(events) => {
                for ev in events {
                    handle_event(state, roots, &ev);
                }
                // Periodically rebuild FTS so newly-indexed code is searchable.
                if last_fts_rebuild.elapsed() > fts_rebuild_interval {
                    let _ = state.store.rebuild_fts();
                    last_fts_rebuild = std::time::Instant::now();
                }
            }
            Err(errs) => {
                for e in errs {
                    log(&format!("watcher error: {e}"));
                }
            }
        }
    }
    Ok(())
}

fn handle_event(state: &ServerState, roots: &[PathBuf], event: &DebouncedEvent) {
    use notify::EventKind;
    for path in &event.paths {
        if is_excluded(path) {
            continue;
        }
        match event.kind {
            EventKind::Remove(_) => {
                let _ = state
                    .store
                    .delete_code_file(&path.to_string_lossy());
            }
            EventKind::Create(_) | EventKind::Modify(_) => {
                if !path.is_file() {
                    continue;
                }
                if language_for_path(path).is_none() {
                    continue;
                }
                // Find the root this path is under so we can record repo_root.
                let root = roots
                    .iter()
                    .find(|r| path.starts_with(r))
                    .cloned()
                    .unwrap_or_else(|| path.parent().unwrap_or(path).to_path_buf());
                if let Err(e) = index_file(state, &root, path) {
                    log(&format!("re-index {}: {e}", path.display()));
                }
            }
            _ => {}
        }
    }
}

fn log(msg: &str) {
    if std::env::var("MEMORIZE_VERBOSE").is_ok() {
        eprintln!("[memorize] {msg}");
    }
}
