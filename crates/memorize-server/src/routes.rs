use crate::state::ServerState;
use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use memorize_core::{Kind, NewObservation, chunk_for_embedding, truncate_body};
use memorize_recall::recall;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tiny_http::{Method, Request, Response, Server};

const VERBOSE_ENV: &str = "MEMORIZE_VERBOSE";

#[derive(Debug, Deserialize)]
struct CaptureReq {
    session: String,
    kind: String,
    body: String,
    #[serde(default)]
    branch: Option<String>,
    #[serde(default)]
    ref_path: Option<String>,
    #[serde(default)]
    ref_line_start: Option<i32>,
    #[serde(default)]
    ref_line_end: Option<i32>,
}

#[derive(Debug, Serialize)]
struct CaptureResp {
    id: Option<i64>,
    stored: bool,
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RecallReq {
    query: String,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum SynReq {
    Add { term: String, expansion: String },
    Remove { term: String, #[serde(default)] expansion: Option<String> },
}

#[derive(Debug, Deserialize)]
struct CodeSearchReq {
    query: String,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    language: Option<String>,
    /// Agent-controlled explicit narrowing. Overrides cwd-based auto-scope.
    #[serde(default)]
    path: Option<String>,
    /// "auto" (default) | "current" | "all". Agent's policy for cwd-based scoping.
    #[serde(default)]
    scope: Option<String>,
    /// Caller's working directory at MCP-shim startup. Not part of the
    /// agent-facing tool schema — the shim sets it as transport metadata.
    /// Used to derive an auto-scope when `path` is absent and `scope` isn't
    /// "all".
    #[serde(default)]
    cwd: Option<String>,
}

/// Bind and serve. Blocks the calling thread; spawn before calling if you
/// want a background server.
pub fn serve(state: ServerState, bind: &str) -> Result<()> {
    let server = Server::http(bind).map_err(|e| anyhow!("bind {bind}: {e}"))?;
    log(&format!("memorize: listening on {bind}"));

    // Eviction on startup. Cheap; one DELETE.
    let ttl_secs = std::env::var("MEMORIZE_TTL_DAYS")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(90)
        * 86_400;
    let cutoff = Utc::now().timestamp() - ttl_secs;
    match state.store.evict_older_than(cutoff) {
        Ok(n) if n > 0 => log(&format!("evicted {n} obs older than {ttl_secs}s")),
        Ok(_) => {}
        Err(e) => log(&format!("eviction error: {e}")),
    }

    let state = Arc::new(state);

    // Background code indexer: cold-scan + watcher. The thread checks its
    // own `config.code_index.enabled` and exits cleanly if disabled.
    crate::code_indexer::spawn(state.clone());

    for req in server.incoming_requests() {
        let st = state.clone();
        // Single-threaded server for simplicity. Hooks fire one at a time per
        // Claude Code session, so contention is minimal.
        if let Err(e) = handle(&st, req) {
            log(&format!("handler error: {e}"));
        }
    }
    Ok(())
}

fn handle(state: &ServerState, mut req: Request) -> Result<()> {
    let method = req.method().clone();
    let url = req.url().to_string();
    match (&method, url.as_str()) {
        (Method::Get, "/health") => respond_json(req, 200, &serde_json::json!({"ok": true})),
        (Method::Get, "/status") => handle_status(state, req),
        (Method::Get, u) if u.starts_with("/context") => {
            // Context = recall over the query string args, formatted as markdown.
            let q = extract_query_param(u, "query").unwrap_or_default();
            let budget = extract_query_param(u, "budget")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(state.token_budget);
            handle_context(state, req, &q, budget)
        }
        (Method::Post, "/capture") => {
            let body = read_body(&mut req)?;
            let parsed: CaptureReq = serde_json::from_str(&body).context("parse /capture body")?;
            handle_capture(state, req, parsed)
        }
        (Method::Post, "/recall") => {
            let body = read_body(&mut req)?;
            let parsed: RecallReq = serde_json::from_str(&body).context("parse /recall body")?;
            handle_recall(state, req, parsed)
        }
        (Method::Get, "/syn") => handle_syn_list(state, req),
        (Method::Post, "/syn") => {
            let body = read_body(&mut req)?;
            let parsed: SynReq = serde_json::from_str(&body).context("parse /syn body")?;
            handle_syn(state, req, parsed)
        }
        (Method::Post, "/code/search") => {
            let body = read_body(&mut req)?;
            let parsed: CodeSearchReq =
                serde_json::from_str(&body).context("parse /code/search body")?;
            handle_code_search(state, req, parsed)
        }
        _ => respond_text(req, 404, "not found"),
    }
}

fn handle_capture(state: &ServerState, req: Request, payload: CaptureReq) -> Result<()> {
    let kind = Kind::from_str(&payload.kind);
    let now = Utc::now().timestamp();

    // 1. Privacy filter.
    let body = state.privacy.redact(&payload.body);

    // 2. Truncate noisy tool dumps.
    let body = truncate_body(&body);

    // 3. Dedup.
    if !state
        .dedup
        .check_and_insert(&payload.session, payload.kind.as_str(), &body, now)
    {
        return respond_json(
            req,
            200,
            &CaptureResp {
                id: None,
                stored: false,
                reason: Some("duplicate".to_string()),
            },
        );
    }

    // 4. Chunk + embed. Each chunk fits inside the embedder's context window;
    //    one vector per chunk lands in vec_chunks, BM25 still indexes the
    //    full body for full-length lexical recall.
    let chunks = chunk_for_embedding(&body);
    let chunk_embs = memorize_embed::embed_batch(&chunks).context("embed chunks")?;
    let chunk_emb_refs: Vec<&[f32]> = chunk_embs.iter().map(|v| v.as_slice()).collect();

    // 5. Persist.
    let obs = NewObservation {
        session: payload.session,
        kind,
        body,
        branch: payload.branch,
        ref_path: payload.ref_path,
        ref_line_start: payload.ref_line_start,
        ref_line_end: payload.ref_line_end,
    };
    let id = state
        .store
        .insert_obs_chunked(&obs, now, &chunk_emb_refs)
        .context("insert obs")?;

    respond_json(
        req,
        200,
        &CaptureResp {
            id: Some(id),
            stored: true,
            reason: None,
        },
    )
}

fn handle_recall(state: &ServerState, req: Request, payload: RecallReq) -> Result<()> {
    // FTS index needs a rebuild after recent inserts before BM25 sees them.
    // Cheap at our scale; we just do it on every recall.
    let _ = state.store.rebuild_fts();
    let emb = memorize_embed::embed(&payload.query).context("embed query")?;
    let results = recall(
        &state.store,
        &payload.query,
        &emb,
        payload.limit.unwrap_or(10),
    )?;
    respond_json(req, 200, &results)
}

fn handle_context(state: &ServerState, req: Request, query: &str, budget: usize) -> Result<()> {
    if query.is_empty() {
        return respond_text(req, 200, "");
    }
    let _ = state.store.rebuild_fts();
    let emb = memorize_embed::embed(query).context("embed context query")?;
    let results = recall(&state.store, query, &emb, 20)?;

    // Render markdown, trimmed to budget. Rough char-count heuristic: 4 chars/token.
    let max_chars = budget.saturating_mul(4);
    let mut md = String::new();
    md.push_str("# memorize: prior context\n\n");
    for r in &results {
        let line = format!(
            "- **[{kind}]** _{session}_: {body}\n",
            kind = r.obs.kind.as_str(),
            session = r.obs.session,
            body = first_line(&r.obs.body, 200),
        );
        if md.len() + line.len() > max_chars {
            break;
        }
        md.push_str(&line);
    }
    respond_text(req, 200, &md)
}

fn handle_status(state: &ServerState, req: Request) -> Result<()> {
    let indexer = state.indexer_status.snapshot();
    let obs_count = state.store.count_obs().unwrap_or(-1);
    let code_chunks = state.store.count_code_chunks().unwrap_or(-1);
    let payload = serde_json::json!({
        "ok": true,
        "uptime_secs": state.indexer_status.server_uptime_secs(),
        "obs_count": obs_count,
        "code_chunks_count": code_chunks,
        "config": {
            "code_index": {
                "enabled": state.config.code_index.enabled,
                "roots": state.config.code_index.roots,
                "languages": state.config.code_index.languages,
                "respect_gitignore": state.config.code_index.respect_gitignore,
                "max_file_bytes": state.config.code_index.max_file_bytes,
            }
        },
        "indexer": indexer,
    });
    respond_json(req, 200, &payload)
}

fn handle_code_search(state: &ServerState, req: Request, payload: CodeSearchReq) -> Result<()> {
    let _ = state.store.rebuild_fts();
    let q = payload.query.trim();
    let scope_arg = payload.scope.as_deref().unwrap_or("current");

    // Resolve the effective path filter + any user-facing warnings.
    // Explicit `path` overrides cwd-derived scope; `scope=all` ignores cwd.
    let (effective_prefix, warnings) = resolve_scope(
        state,
        payload.cwd.as_deref(),
        payload.path.as_deref(),
        scope_arg,
    );

    if q.is_empty() {
        return respond_json(
            req,
            200,
            &serde_json::json!({"results": Vec::<serde_json::Value>::new(), "warnings": warnings}),
        );
    }
    let limit = payload.limit.unwrap_or(10);
    let per_stream = 50usize;
    let lang = payload.language.as_deref();
    let prefix = effective_prefix.as_deref();

    let q_emb = memorize_embed::embed(q).context("embed code query")?;
    let fused = fused_code_search(state, q, &q_emb, per_stream, lang, prefix);

    // Per-file diversification: cap at MAX_PER_FILE chunks from any single
    // file. Without this, large files dominate the top — memo.test.ts taking
    // 7 slots squeezes out 5 other files. agentmemory does the same trick at
    // session granularity; the file-level analogue is what users want for
    // code search.
    const MAX_PER_FILE: usize = 2;
    let mut per_file: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut picked: Vec<(i64, f64)> = Vec::with_capacity(limit);
    for (id, path, score) in &fused {
        let n = per_file.entry(path.clone()).or_insert(0);
        if *n >= MAX_PER_FILE {
            continue;
        }
        *n += 1;
        picked.push((*id, *score));
        if picked.len() >= limit {
            break;
        }
    }
    // If diversification left us short of `limit` (small index, narrow query),
    // backfill with the next-best chunks ignoring the per-file cap.
    if picked.len() < limit {
        let already: std::collections::HashSet<i64> =
            picked.iter().map(|(id, _)| *id).collect();
        for (id, _path, score) in &fused {
            if picked.len() >= limit {
                break;
            }
            if !already.contains(id) {
                picked.push((*id, *score));
            }
        }
    }

    let ids: Vec<i64> = picked.iter().map(|(id, _)| *id).collect();
    let rows = state.store.get_code_chunks_by_ids(&ids)?;
    let scores: Vec<f64> = picked.iter().map(|(_, s)| *s).collect();
    let results: Vec<serde_json::Value> = rows
        .into_iter()
        .zip(scores)
        .map(|(r, s)| {
            serde_json::json!({
                "id": r.id,
                "path": r.path,
                "language": r.language,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "kind": r.kind,
                "qualified": r.qualified,
                "body": r.body,
                "score": s,
            })
        })
        .collect();
    respond_json(
        req,
        200,
        &serde_json::json!({"results": results, "warnings": warnings}),
    )
}

/// Sentinel prefix that no real `code_chunks.path` will match. Used when we
/// need to deliberately return zero results (cwd or path outside all roots).
const NEVER_MATCHES: &str = "/dev/null/memorize-never-matches/";

/// Resolve the effective path-prefix filter for a code_recall call, plus any
/// warnings to surface to the agent.
///
/// Rules:
///  - Explicit `path` wins, but it must be under *some* indexed root —
///    otherwise we deliberately return zero results.
///  - `scope=all` ignores `cwd` and searches everything.
///  - Otherwise (`scope=current`, the default): scope to the configured
///    root containing `cwd`. If `cwd` is missing or outside all roots,
///    return zero results.
///
/// "Zero results" is implemented by setting the prefix to a sentinel path
/// nothing in the index will ever start with. The caller doesn't need a
/// special branch.
fn resolve_scope(
    state: &ServerState,
    cwd: Option<&str>,
    explicit_path: Option<&str>,
    scope: &str,
) -> (Option<String>, Vec<String>) {
    let roots = resolved_root_paths(state);

    // Explicit path. Must be under some indexed root, else return nothing.
    if let Some(p) = explicit_path {
        let p_path = std::path::Path::new(p);
        let under_a_root = roots.iter().any(|r| p_path.starts_with(r));
        if under_a_root {
            return (Some(p.to_string()), Vec::new());
        }
        let roots_str: Vec<String> = roots.iter().map(|r| r.display().to_string()).collect();
        return (
            Some(NEVER_MATCHES.to_string()),
            vec![format!(
                "path ({p}) is not under any indexed root ({}). Returning no results. Narrow to a path inside an indexed root or add it to `code_index.roots` in ~/.memorize/config.toml.",
                roots_str.join(", ")
            )],
        );
    }

    if scope == "all" {
        // Agent explicitly asked for cross-root. No cwd check.
        return (None, Vec::new());
    }

    // scope == "current" (the default): require cwd to be under some root.
    let cwd_root = cwd
        .filter(|c| !c.is_empty())
        .and_then(|c| {
            let cwd_path = std::path::Path::new(c);
            roots.iter().find(|r| cwd_path.starts_with(r)).cloned()
        });
    match cwd_root {
        Some(r) => (Some(r.display().to_string()), Vec::new()),
        None => {
            let cwd_disp = cwd.unwrap_or("(no cwd)");
            let roots_str: Vec<String> = roots.iter().map(|r| r.display().to_string()).collect();
            let msg = format!(
                "cwd ({cwd_disp}) is not under any indexed root ({}). Returning no results. Add this path to `code_index.roots` in ~/.memorize/config.toml, or pass `scope=\"all\"` to search across everything.",
                roots_str.join(", ")
            );
            (Some(NEVER_MATCHES.to_string()), vec![msg])
        }
    }
}

/// Resolve configured roots into absolute `PathBuf`s with `~/` expansion.
/// Missing paths are dropped — same logic the indexer uses.
fn resolved_root_paths(state: &ServerState) -> Vec<std::path::PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    state
        .config
        .code_index
        .roots
        .iter()
        .map(|s| {
            if let Some(rest) = s.strip_prefix("~/") {
                std::path::PathBuf::from(&home).join(rest)
            } else {
                std::path::PathBuf::from(s)
            }
        })
        .filter(|p| p.exists())
        .collect()
}

/// Run a single hybrid-fused code search. Pulled out so the auto-scope
/// fallback path can call it twice without duplicating the fusion logic.
fn fused_code_search(
    state: &ServerState,
    q: &str,
    q_emb: &[f32],
    per_stream: usize,
    lang: Option<&str>,
    prefix: Option<&str>,
) -> Vec<(i64, String, f64)> {
    let bm25 = state
        .store
        .search_code_bm25(q, per_stream, lang, prefix)
        .unwrap_or_default();
    let vec_hits = state
        .store
        .search_code_vector(q_emb, per_stream, lang, prefix)
        .unwrap_or_default();

    use std::collections::HashMap;
    let k = 60.0f64;
    // Track score + path per id (path is identical across streams so first
    // seen wins).
    let mut acc: HashMap<i64, (f64, String)> = HashMap::new();
    for (rank, hit) in bm25.iter().enumerate() {
        let entry = acc
            .entry(hit.id)
            .or_insert_with(|| (0.0, hit.path.clone()));
        entry.0 += 1.0 / (k + (rank + 1) as f64);
    }
    for (rank, hit) in vec_hits.iter().enumerate() {
        let entry = acc
            .entry(hit.id)
            .or_insert_with(|| (0.0, hit.path.clone()));
        entry.0 += 1.0 / (k + (rank + 1) as f64);
    }

    // Path-segment exact-match boost. If a query token appears as a literal
    // path segment (split on /_-.), the chunk gets a bonus per match. This
    // mirrors what humans usually expect from code search — typing "memo"
    // should surface files under .../irMemo/memo.ts before chat-heavy chunks
    // that just happen to mention "memo" a lot. We pick the boost magnitude
    // (~0.02 per match) to be roughly equal to one stream's RRF score at
    // rank 1; two path-segment matches ≈ ranking #1 in both streams.
    let q_tokens = tokenize_for_path_boost(q);
    if !q_tokens.is_empty() {
        for (_id, (score, path)) in acc.iter_mut() {
            let matches = count_path_segment_matches(path, &q_tokens);
            *score += 0.02 * matches as f64;
        }
    }

    let mut fused: Vec<(i64, String, f64)> = acc
        .into_iter()
        .map(|(id, (score, path))| (id, path, score))
        .collect();
    fused.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

/// Lowercase + split the query on whitespace and `/_-.` so each token can be
/// compared against path segments. Returns empty for whitespace-only input.
fn tokenize_for_path_boost(q: &str) -> Vec<String> {
    q.split(|c: char| c.is_whitespace() || matches!(c, '/' | '_' | '-' | '.'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Count how many query tokens appear as exact (case-insensitive) path
/// segments. Segments are the slash-separated components AND their
/// underscore/dot/dash sub-pieces, so `packages/entities/src/irMemo/memo.ts`
/// yields {packages, entities, src, irmemo, memo, ts}. Each match counts at
/// most once per token to keep the boost bounded.
fn count_path_segment_matches(path: &str, q_tokens: &[String]) -> usize {
    use std::collections::HashSet;
    let segs: HashSet<String> = path
        .split(|c: char| matches!(c, '/' | '_' | '-' | '.'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect();
    q_tokens.iter().filter(|t| segs.contains(*t)).count()
}

fn handle_syn(state: &ServerState, req: Request, payload: SynReq) -> Result<()> {
    match payload {
        SynReq::Add { term, expansion } => {
            state.store.add_synonym(&term, &expansion)?;
            respond_json(req, 200, &serde_json::json!({"ok": true}))
        }
        SynReq::Remove { term, expansion } => {
            let n = state
                .store
                .remove_synonym(&term, expansion.as_deref())?;
            respond_json(req, 200, &serde_json::json!({"ok": true, "deleted": n}))
        }
    }
}

fn handle_syn_list(state: &ServerState, req: Request) -> Result<()> {
    let pairs = state.store.list_synonyms()?;
    respond_json(req, 200, &pairs)
}

fn first_line(s: &str, max: usize) -> String {
    let line = s.lines().next().unwrap_or("");
    if line.len() <= max {
        return line.to_string();
    }
    let mut cut = max;
    while !line.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &line[..cut])
}

fn read_body(req: &mut Request) -> Result<String> {
    let mut buf = String::new();
    req.as_reader().read_to_string(&mut buf)?;
    Ok(buf)
}

fn extract_query_param<'a>(url: &'a str, key: &str) -> Option<String> {
    let q = url.split_once('?')?.1;
    for pair in q.split('&') {
        let (k, v) = pair.split_once('=')?;
        if k == key {
            return Some(percent_decode(v));
        }
    }
    None
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) =
                (hex_nibble(bytes[i + 1]), hex_nibble(bytes[i + 2]))
            {
                out.push((h << 4) | l);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn respond_json<T: Serialize>(req: Request, status: u16, body: &T) -> Result<()> {
    let s = serde_json::to_string(body)?;
    let resp = Response::from_string(s)
        .with_status_code(status)
        .with_header("Content-Type: application/json".parse::<tiny_http::Header>().unwrap());
    req.respond(resp).ok();
    Ok(())
}

fn respond_text(req: Request, status: u16, body: &str) -> Result<()> {
    let resp = Response::from_string(body).with_status_code(status);
    req.respond(resp).ok();
    Ok(())
}

fn log(msg: &str) {
    if std::env::var(VERBOSE_ENV).is_ok() {
        eprintln!("[memorize] {msg}");
    }
}
