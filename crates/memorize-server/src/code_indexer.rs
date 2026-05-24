//! Code indexer + file watcher.
//!
//! Lifecycle:
//!  1. Spawn at server startup (if enabled in config).
//!  2. **Start the watcher first**, so file changes during cold-scan still
//!     queue and get picked up. Debouncer holds events in its channel.
//!  3. Cold-scan each configured root in turn — stat every file, re-parse
//!     only those whose `(mtime_ns, size_bytes)` differ from what's in
//!     the `files` table.
//!  4. Enter the steady-state watcher loop, draining the debounced event
//!     channel (any events that arrived during scan land here too).
//!
//! All milestone events go to two places:
//!  - `~/.memorize/indexer.log` (structured JSONL, append-only)
//!  - `ServerState::indexer_status` (in-memory snapshot for `/status`)

use crate::indexer_log::{IndexerLog, LogEvent, now};
use crate::indexer_status::IndexerPhase;
use crate::state::ServerState;
use anyhow::{Context, Result};
use memorize_code::{CodeChunk, language_for_path};
use memorize_store::{CodeChunkRow, FileMeta};
use notify::RecursiveMode;
use notify_debouncer_full::DebouncedEvent;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub fn spawn(state: Arc<ServerState>) {
    if !state.config.code_index.enabled {
        let log = IndexerLog::open();
        log.write(&LogEvent::Disabled { ts: now() });
        state.indexer_status.set_phase(IndexerPhase::Disabled);
        return;
    }

    std::thread::spawn(move || {
        let log = IndexerLog::open();
        let roots = resolved_roots(&state.config.code_index.roots);
        log.write(&LogEvent::Started {
            ts: now(),
            roots: roots.iter().map(|p| p.display().to_string()).collect(),
            respect_gitignore: state.config.code_index.respect_gitignore,
        });
        if roots.is_empty() {
            state.indexer_status.set_phase(IndexerPhase::Idle);
            return;
        }

        // Set up the watcher BEFORE cold-scan so file events that arrive
        // during the scan still queue and get drained later. Debouncer holds
        // them in its mpsc channel; the watcher loop drains after scan.
        let (tx, rx) = match start_watcher(&state, &roots) {
            Ok(pair) => pair,
            Err(e) => {
                log.write(&LogEvent::Error {
                    ts: now(),
                    at: "start_watcher",
                    path: None,
                    msg: e.to_string(),
                });
                state.indexer_status.set_phase(IndexerPhase::Error);
                return;
            }
        };

        // Cold-scan.
        for root in &roots {
            state.indexer_status.set_current_root(Some(root.display().to_string()));
            log.write(&LogEvent::ColdScanRootStart {
                ts: now(),
                root: root.display().to_string(),
            });
            let started = Instant::now();
            let baseline = state.indexer_status.snapshot();
            if let Err(e) = scan_root(&state, root, &log) {
                log.write(&LogEvent::Error {
                    ts: now(),
                    at: "scan_root",
                    path: Some(root.display().to_string()),
                    msg: e.to_string(),
                });
                state
                    .indexer_status
                    .record_error(format!("scan {}: {e}", root.display()));
            }
            let after = state.indexer_status.snapshot();
            log.write(&LogEvent::ColdScanRootDone {
                ts: now(),
                root: root.display().to_string(),
                files_indexed: after.files_indexed - baseline.files_indexed,
                files_skipped: after.files_skipped - baseline.files_skipped,
                files_excluded: after.files_excluded - baseline.files_excluded,
                elapsed_ms: started.elapsed().as_millis(),
            });
        }

        state.indexer_status.mark_cold_scan_complete();
        let final_snap = state.indexer_status.snapshot();
        log.write(&LogEvent::ColdScanComplete {
            ts: now(),
            total_files_indexed: final_snap.files_indexed,
            total_files_skipped: final_snap.files_skipped,
            total_chunks_in_index: state.store.count_code_chunks().unwrap_or(0),
        });

        // Drain watcher events forever. `tx` lives because the debouncer
        // (held inside `start_watcher`'s returned guard) keeps it alive.
        drain_watcher(&state, &roots, &log, &rx, tx);
    });
}

/// Re-split chunks so none exceed the embedder's max-seq window. Same logic
/// as the bench helper — see crates/memorize-cli/src/bench.rs.
fn enforce_token_cap(chunks: Vec<CodeChunk>) -> anyhow::Result<Vec<CodeChunk>> {
    let cap = memorize_embed::max_seq_tokens().saturating_sub(8).max(1);
    let mut out: Vec<CodeChunk> = Vec::with_capacity(chunks.len());
    for c in chunks {
        let pieces = memorize_embed::split_to_token_cap(&c.body, cap)?;
        if pieces.len() == 1 {
            out.push(c);
            continue;
        }
        let total = pieces.len();
        for (i, body) in pieces.into_iter().enumerate() {
            let qualified = if c.qualified.is_empty() {
                format!("part-{}/{total}", i + 1)
            } else {
                format!("{}#part-{}/{total}", c.qualified, i + 1)
            };
            out.push(CodeChunk {
                language: c.language.clone(),
                line_start: c.line_start,
                line_end: c.line_end,
                kind: c.kind.clone(),
                qualified,
                body,
            });
        }
    }
    Ok(out)
}

fn resolved_roots(raw: &[String]) -> Vec<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    raw.iter()
        .map(|s| {
            if let Some(rest) = s.strip_prefix("~/") {
                PathBuf::from(&home).join(rest)
            } else {
                PathBuf::from(s)
            }
        })
        .filter(|p| p.exists())
        .collect()
}

fn scan_root(state: &ServerState, root: &Path, log: &IndexerLog) -> Result<()> {
    if state.config.code_index.respect_gitignore {
        scan_with_ignore(state, root, log)
    } else {
        walk_dir(state, root, root, log)
    }
}

fn scan_with_ignore(state: &ServerState, root: &Path, log: &IndexerLog) -> Result<()> {
    let walker = ignore::WalkBuilder::new(root)
        .standard_filters(true)
        .follow_links(false)
        .build();
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            visit_file(state, root, path, log);
        }
    }
    Ok(())
}

fn walk_dir(state: &ServerState, root: &Path, dir: &Path, log: &IndexerLog) -> Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if is_excluded(&path, &state.config.code_index.excludes) {
            state.indexer_status.record_file_excluded();
            continue;
        }
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if ft.is_dir() {
            let _ = walk_dir(state, root, &path, log);
        } else if ft.is_file() {
            visit_file(state, root, &path, log);
        }
    }
    Ok(())
}

/// One source of truth for per-file decisions used by both scanning paths
/// and the watcher event handler.
fn visit_file(state: &ServerState, root: &Path, path: &Path, log: &IndexerLog) {
    if is_excluded(path, &state.config.code_index.excludes) {
        state.indexer_status.record_file_excluded();
        return;
    }
    if !language_allowed(path, &state.config.code_index.languages) {
        return;
    }
    match index_file(state, root, path, log) {
        Ok(IndexOutcome::Indexed { chunks }) => {
            state.indexer_status.record_file_indexed(chunks);
        }
        Ok(IndexOutcome::Unchanged) => {
            state.indexer_status.record_file_skipped();
        }
        Ok(IndexOutcome::TooBig) => {
            // Already logged inside index_file.
        }
        Err(e) => {
            let msg = e.to_string();
            log.write(&LogEvent::Error {
                ts: now(),
                at: "index_file",
                path: Some(path.display().to_string()),
                msg: msg.clone(),
            });
            state
                .indexer_status
                .record_error(format!("{}: {msg}", path.display()));
        }
    }
}

fn is_excluded(path: &Path, excludes: &[String]) -> bool {
    let s = path.to_string_lossy();
    excludes.iter().any(|pat| s.contains(pat.as_str()))
}

fn language_allowed(path: &Path, allow: &[String]) -> bool {
    let lang = match language_for_path(path) {
        Some(l) => l,
        None => return false,
    };
    if allow.is_empty() {
        return true;
    }
    let allowed: HashSet<&str> = allow.iter().map(|s| s.as_str()).collect();
    allowed.contains(lang)
}

enum IndexOutcome {
    Indexed { chunks: usize },
    Unchanged,
    TooBig,
}

fn index_file(
    state: &ServerState,
    root: &Path,
    path: &Path,
    log: &IndexerLog,
) -> Result<IndexOutcome> {
    let cfg = &state.config.code_index;
    let path_str = path.to_string_lossy().to_string();
    let meta_fs = std::fs::metadata(path).with_context(|| format!("stat {path_str}"))?;
    if meta_fs.len() > cfg.max_file_bytes {
        log.write(&LogEvent::FileSkippedTooBig {
            ts: now(),
            path: path_str.clone(),
            bytes: meta_fs.len(),
        });
        state.indexer_status.record_file_excluded();
        return Ok(IndexOutcome::TooBig);
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
            // Quiet on the skipped-unchanged path — would explode log size
            // on cold-scans. Status counter still bumps.
            return Ok(IndexOutcome::Unchanged);
        }
    }

    let source = std::fs::read_to_string(path).with_context(|| format!("read {path_str}"))?;
    let language = match language_for_path(path) {
        Some(l) => l,
        None => return Ok(IndexOutcome::Unchanged),
    };

    let chunks: Vec<CodeChunk> = memorize_code::chunk_source(&source, language)?;
    if chunks.is_empty() {
        return Ok(IndexOutcome::Unchanged);
    }
    let chunks = enforce_token_cap(chunks).context("token-cap split")?;

    let bodies: Vec<&str> = chunks.iter().map(|c| c.body.as_str()).collect();
    let embs = memorize_embed::embed_batch(&bodies).context("embed chunks")?;

    let rows: Vec<CodeChunkRow> = chunks
        .into_iter()
        .map(|c| CodeChunkRow {
            id: 0,
            path: path_str.clone(),
            language: c.language,
            line_start: c.line_start as i32,
            line_end: c.line_end as i32,
            kind: c.kind,
            qualified: c.qualified,
            body: c.body,
        })
        .collect();

    let chunk_count = rows.len();
    let meta = FileMeta {
        mtime_ns,
        size_bytes,
        git_rev: None,
    };
    state
        .store
        .upsert_code_file(&path_str, &root.to_string_lossy(), language, &meta, &rows, &embs)
        .context("upsert code file")?;

    log.write(&LogEvent::FileIndexed {
        ts: now(),
        path: path_str,
        language: language.to_string(),
        chunks: chunk_count,
        bytes: meta_fs.len(),
    });

    Ok(IndexOutcome::Indexed {
        chunks: chunk_count,
    })
}

/// Returns the receiver and a sender that keeps the debouncer alive (since
/// the debouncer owns the actual sender we move into the closure, and we
/// pin it via the static held by the spawning thread).
fn start_watcher(
    state: &ServerState,
    roots: &[PathBuf],
) -> Result<(WatcherHandle, mpsc::Receiver<notify_debouncer_full::DebounceEventResult>)> {
    let cfg = &state.config.code_index;
    let (tx, rx) = mpsc::channel();
    let mut debouncer = notify_debouncer_full::new_debouncer(
        Duration::from_millis(cfg.debounce_ms),
        None,
        move |res: notify_debouncer_full::DebounceEventResult| {
            let _ = tx.send(res);
        },
    )?;
    for root in roots {
        if let Err(e) = debouncer.watch(root, RecursiveMode::Recursive) {
            // Don't fail the whole indexer on a single root failure.
            state
                .indexer_status
                .record_error(format!("watch {}: {e}", root.display()));
        }
    }
    Ok((WatcherHandle { _debouncer: debouncer }, rx))
}

/// Holds the debouncer to keep the watcher alive for the lifetime of the
/// indexer thread. Dropping this stops the watcher.
struct WatcherHandle {
    _debouncer: notify_debouncer_full::Debouncer<
        notify::RecommendedWatcher,
        notify_debouncer_full::RecommendedCache,
    >,
}

fn drain_watcher(
    state: &ServerState,
    roots: &[PathBuf],
    log: &IndexerLog,
    rx: &mpsc::Receiver<notify_debouncer_full::DebounceEventResult>,
    _handle: WatcherHandle,
) {
    let cfg = &state.config.code_index;
    let rebuild_interval = Duration::from_secs(cfg.fts_rebuild_interval_secs.max(1));
    let mut last_fts_rebuild = Instant::now();

    // Use recv_timeout so we can transition phase to Idle when quiet.
    loop {
        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(Ok(events)) => {
                state.indexer_status.set_phase(IndexerPhase::Watching);
                for ev in events {
                    handle_event(state, roots, log, &ev);
                }
                if last_fts_rebuild.elapsed() > rebuild_interval {
                    let _ = state.store.rebuild_fts();
                    last_fts_rebuild = Instant::now();
                }
            }
            Ok(Err(errs)) => {
                for e in errs {
                    log.write(&LogEvent::Error {
                        ts: now(),
                        at: "watcher",
                        path: None,
                        msg: e.to_string(),
                    });
                    state
                        .indexer_status
                        .record_error(format!("watcher: {e}"));
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Idle — no events recently.
                state.indexer_status.set_phase(IndexerPhase::Idle);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn handle_event(
    state: &ServerState,
    roots: &[PathBuf],
    log: &IndexerLog,
    event: &DebouncedEvent,
) {
    use notify::EventKind;
    let cfg = &state.config.code_index;
    for path in &event.paths {
        state
            .indexer_status
            .record_event(path.display().to_string());
        log.write(&LogEvent::WatcherEvent {
            ts: now(),
            path: path.display().to_string(),
            kind: format!("{:?}", event.kind),
        });
        if is_excluded(path, &cfg.excludes) {
            continue;
        }
        match event.kind {
            EventKind::Remove(_) => {
                let _ = state.store.delete_code_file(&path.to_string_lossy());
                log.write(&LogEvent::FileRemoved {
                    ts: now(),
                    path: path.display().to_string(),
                });
            }
            EventKind::Create(_) | EventKind::Modify(_) => {
                if !path.is_file() {
                    continue;
                }
                let root = roots
                    .iter()
                    .find(|r| path.starts_with(r))
                    .cloned()
                    .unwrap_or_else(|| path.parent().unwrap_or(path).to_path_buf());
                visit_file(state, &root, path, log);
            }
            _ => {}
        }
    }
}
