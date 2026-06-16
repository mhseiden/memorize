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
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// De-duplicating queue of paths that recall found stale and wants re-indexed
/// out of band. Recall returns the on-disk text at the old line offsets (which
/// may be wrong for a changed file) and enqueues the path here; this queue
/// feeds a worker that re-parses the file so the next recall is correct.
///
/// Backed by a `HashSet` so a burst of stale chunks for the same file collapses
/// into a single reindex.
#[derive(Default)]
pub struct ReindexQueue {
    inner: Mutex<HashSet<PathBuf>>,
    ready: Condvar,
}

impl ReindexQueue {
    /// Enqueue a path for reindex. Non-blocking; duplicates collapse.
    pub fn request(&self, path: PathBuf) {
        let mut q = self.inner.lock().unwrap();
        if q.insert(path) {
            self.ready.notify_one();
        }
    }

    /// Block until at least one path is queued, then take all of them.
    fn drain_blocking(&self) -> Vec<PathBuf> {
        let mut q = self.inner.lock().unwrap();
        while q.is_empty() {
            q = self.ready.wait(q).unwrap();
        }
        q.drain().collect()
    }
}

pub fn spawn(state: Arc<ServerState>) {
    if !state.config.code_index.enabled {
        let log = IndexerLog::open(state.config.code_index.max_indexer_log_bytes);
        log.write(&LogEvent::Disabled { ts: now() });
        state.indexer_status.set_phase(IndexerPhase::Disabled);
        return;
    }

    std::thread::spawn(move || {
        let log = IndexerLog::open(state.config.code_index.max_indexer_log_bytes);
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

        // Out-of-band reindex worker: drains paths that recall flagged stale.
        // Its own thread because `drain_watcher` below never returns.
        {
            let state = state.clone();
            let roots = roots.clone();
            std::thread::spawn(move || reindex_worker(&state, &roots));
        }

        // Drain watcher events forever. `tx` lives because the debouncer
        // (held inside `start_watcher`'s returned guard) keeps it alive.
        drain_watcher(&state, &roots, &log, &rx, tx);
    });
}

/// Service `ServerState::reindex_queue` forever: re-parse each stale path so
/// the next recall sees correct chunks. A path whose file no longer exists is
/// dropped from the index instead of reparsed.
fn reindex_worker(state: &ServerState, roots: &[PathBuf]) {
    let log = IndexerLog::open(state.config.code_index.max_indexer_log_bytes);
    loop {
        for path in state.reindex_queue.drain_blocking() {
            if !path.exists() {
                let _ = state.store.delete_code_file(&path.to_string_lossy());
                continue;
            }
            let Some(root) = root_for_path(roots, &path) else {
                continue;
            };
            if let Err(e) = index_file(state, root, &path, &log) {
                log.write(&LogEvent::Error {
                    ts: now(),
                    at: "reindex",
                    path: Some(path.display().to_string()),
                    msg: e.to_string(),
                });
                state
                    .indexer_status
                    .record_error(format!("reindex {}: {e}", path.display()));
            }
        }
    }
}

/// Pick the indexed root that contains `path` — the longest matching prefix,
/// so nested roots resolve to the most specific one.
fn root_for_path<'a>(roots: &'a [PathBuf], path: &Path) -> Option<&'a PathBuf> {
    roots
        .iter()
        .filter(|r| path.starts_with(r))
        .max_by_key(|r| r.components().count())
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

/// Coarse event class for the churn `by_kind` histogram.
fn coarse_kind(kind: &notify::EventKind) -> &'static str {
    use notify::EventKind;
    match kind {
        EventKind::Create(_) => "create",
        EventKind::Modify(_) => "modify",
        EventKind::Remove(_) => "remove",
        EventKind::Access(_) => "access",
        _ => "other",
    }
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
    // Use recv_timeout so we can transition phase to Idle when quiet.
    // FTS is updated synchronously inside each `upsert_code_file` /
    // `delete_code_file`, so no periodic rebuild is needed here.
    let mut last_beat = Instant::now();
    let mut last_counts = {
        let s = state.indexer_status.snapshot();
        (s.files_indexed, s.files_skipped, s.files_excluded)
    };
    loop {
        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(Ok(events)) => {
                state.indexer_status.set_phase(IndexerPhase::Watching);
                for ev in events {
                    handle_event(state, roots, log, &ev);
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
        maybe_heartbeat(state, &mut last_beat, &mut last_counts);
    }
}

/// Emit a periodic operator-facing summary to the human log (stderr →
/// server.log). Silent during quiet periods so the log stays useful.
fn maybe_heartbeat(state: &ServerState, last_beat: &mut Instant, last: &mut (u64, u64, u64)) {
    const HEARTBEAT: Duration = Duration::from_secs(60);
    if last_beat.elapsed() < HEARTBEAT {
        return;
    }
    *last_beat = Instant::now();
    let snap = state.indexer_status.snapshot();
    let cur = (snap.files_indexed, snap.files_skipped, snap.files_excluded);
    let (di, ds, de) = (
        cur.0.saturating_sub(last.0),
        cur.1.saturating_sub(last.1),
        cur.2.saturating_sub(last.2),
    );
    *last = cur;
    let churn = state.indexer_status.churn_summary();
    if churn.events == 0 && di == 0 && ds == 0 && de == 0 {
        return; // nothing happened — don't spam the log
    }
    let top = churn
        .top_paths
        .first()
        .map(|p| format!("{} ({}x)", p.path, p.count))
        .unwrap_or_else(|| "-".to_string());
    tracing::info!(
        window_events = churn.events,
        events_per_min = format!("{:.1}", churn.events_per_min),
        indexed = di,
        skipped = ds,
        excluded = de,
        top_churn = %top,
        "indexer heartbeat"
    );
}

fn handle_event(
    state: &ServerState,
    roots: &[PathBuf],
    log: &IndexerLog,
    event: &DebouncedEvent,
) {
    use notify::EventKind;
    let cfg = &state.config.code_index;
    let kind = coarse_kind(&event.kind);
    for path in &event.paths {
        // Exclude first: `.git/`, `node_modules/`, etc. generate enormous event
        // volume (lock files, refs, index). Filtering before we record keeps
        // them out of the churn surface and the indexer.log entirely.
        if is_excluded(path, &cfg.excludes) {
            continue;
        }
        state
            .indexer_status
            .record_event(path.display().to_string(), kind.to_string());
        log.write(&LogEvent::WatcherEvent {
            ts: now(),
            path: path.display().to_string(),
            kind: format!("{:?}", event.kind),
        });
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

#[cfg(test)]
mod tests {
    use super::*;
    use notify::event::ModifyKind;
    use notify::{Event, EventKind};

    fn modify_event(path: &str) -> DebouncedEvent {
        let ev = Event::new(EventKind::Modify(ModifyKind::Any)).add_path(PathBuf::from(path));
        DebouncedEvent::new(ev, Instant::now())
    }

    /// Regression: `.git/` (and other excluded) events must be filtered BEFORE
    /// they touch the churn surface or the indexer log. Before the fix the
    /// exclude check ran after `record_event` + `log.write`, so git lock-file
    /// churn flooded both.
    #[test]
    fn excluded_paths_never_enter_churn_or_log() {
        // Redirect the indexer log so the test never writes the real one.
        let tmp = std::env::temp_dir().join(format!("memorize-test-{}.log", std::process::id()));
        // SAFETY: single test, restored implicitly at process exit; we only
        // need the path redirected for this thread's IndexerLog::open below.
        unsafe { std::env::set_var("MEMORIZE_INDEXER_LOG", &tmp) };
        let _ = std::fs::remove_file(&tmp);

        let state = ServerState::in_memory(8192).expect("in-memory state");
        let roots = vec![PathBuf::from("/repo")];
        let log = IndexerLog::open(0);

        // Excluded: must leave churn empty and write nothing.
        handle_event(&state, &roots, &log, &modify_event("/repo/.git/index.lock"));
        assert_eq!(
            state.indexer_status.churn_summary().events,
            0,
            "excluded .git event leaked into churn"
        );

        // Non-excluded source path is recorded once (indexing itself may fail
        // because the file doesn't exist, but record happens first).
        handle_event(&state, &roots, &log, &modify_event("/repo/src/main.rs"));
        let churn = state.indexer_status.churn_summary();
        assert_eq!(churn.events, 1, "source event should be recorded exactly once");
        assert_eq!(churn.top_paths[0].path, "/repo/src/main.rs");

        let logged = std::fs::read_to_string(&tmp).unwrap_or_default();
        assert!(
            !logged.contains(".git/index.lock"),
            "excluded path must not appear in the indexer log"
        );
        let _ = std::fs::remove_file(&tmp);
    }
}
