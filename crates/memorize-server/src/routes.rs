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
    #[serde(default)]
    path_prefix: Option<String>,
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

    // Background code indexer: cold-scan + watcher. Skipped if disabled.
    if std::env::var("MEMORIZE_CODE_INDEX").as_deref() != Ok("0") {
        crate::code_indexer::spawn(state.clone());
    }

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

fn handle_code_search(state: &ServerState, req: Request, payload: CodeSearchReq) -> Result<()> {
    // FTS index needs a rebuild if recent code chunks were written; cheap.
    let _ = state.store.rebuild_fts();
    let q = payload.query.trim();
    if q.is_empty() {
        return respond_json(req, 200, &Vec::<serde_json::Value>::new());
    }
    let limit = payload.limit.unwrap_or(10);
    let per_stream = 50usize;
    let lang = payload.language.as_deref();
    let prefix = payload.path_prefix.as_deref();

    // BM25 + vector streams.
    let bm25 = state
        .store
        .search_code_bm25(q, per_stream, lang, prefix)
        .unwrap_or_default();
    let q_emb = memorize_embed::embed(q).context("embed code query")?;
    let vec_hits = state
        .store
        .search_code_vector(&q_emb, per_stream, lang, prefix)
        .unwrap_or_default();

    // RRF fusion (k=60), no diversification (each chunk is its own entity;
    // multiple chunks from one file is fine — the model wants to see them).
    use std::collections::HashMap;
    let k = 60.0f64;
    let mut acc: HashMap<i64, f64> = HashMap::new();
    for (rank, hit) in bm25.iter().enumerate() {
        *acc.entry(hit.id).or_default() += 1.0 / (k + (rank + 1) as f64);
    }
    for (rank, hit) in vec_hits.iter().enumerate() {
        *acc.entry(hit.id).or_default() += 1.0 / (k + (rank + 1) as f64);
    }
    let mut fused: Vec<(i64, f64)> = acc.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(limit);

    let ids: Vec<i64> = fused.iter().map(|(id, _)| *id).collect();
    let rows = state.store.get_code_chunks_by_ids(&ids)?;
    let scores: Vec<f64> = fused.iter().map(|(_, s)| *s).collect();
    let out: Vec<serde_json::Value> = rows
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
    respond_json(req, 200, &out)
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
