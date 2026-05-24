//! Minimal MCP stdio server for memorize.
//!
//! Speaks the MCP JSON-RPC subset that Claude Code uses: `initialize`,
//! `notifications/initialized`, `tools/list`, `tools/call`. Two tools expose
//! the existing HTTP server's recall and capture endpoints. Hand-rolled —
//! the surface is small enough that an SDK dependency would cost more
//! compile time than it saves.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};

/// We advertise this protocol version on initialize. Claude Code sends its
/// own version and accepts any version the server declares it supports.
const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "memorize";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Deserialize)]
struct Request {
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct Response {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i32,
    message: String,
}

/// Run the stdio loop against an already-running memorize HTTP server.
/// `http_url` is the base — e.g. `http://127.0.0.1:3111`.
pub fn run_stdio(http_url: &str) -> Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();

    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).context("read stdin")?;
        if n == 0 {
            return Ok(()); // EOF
        }
        if line.trim().is_empty() {
            continue;
        }

        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => continue, // ignore malformed lines
        };
        if req.jsonrpc != "2.0" {
            continue;
        }

        // Notifications have no id and expect no response.
        if req.id.is_none() {
            if req.method == "notifications/initialized" {
                // Client signals it's ready; nothing to do.
            }
            continue;
        }

        let response = dispatch(&req, http_url);
        let serialized = serde_json::to_string(&response)?;
        writeln!(writer, "{serialized}")?;
        writer.flush()?;
    }
}

fn dispatch(req: &Request, http_url: &str) -> Response {
    let id = req.id.clone().unwrap_or(Value::Null);
    let result_or_err = match req.method.as_str() {
        "initialize" => Ok(handle_initialize()),
        "tools/list" => Ok(handle_tools_list()),
        "tools/call" => handle_tools_call(&req.params, http_url),
        // Common no-op pings we should answer without error.
        "ping" => Ok(json!({})),
        other => Err(RpcError {
            code: -32601,
            message: format!("method not found: {other}"),
        }),
    };
    match result_or_err {
        Ok(result) => Response {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        },
        Err(err) => Response {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(err),
        },
    }
}

fn handle_initialize() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        }
    })
}

fn handle_tools_list() -> Value {
    // Two tools, separate by intent. Agent chooses based on the query
    // shape: prose questions → session_recall; code/symbol queries →
    // code_recall.
    json!({
        "tools": [
            {
                "name": "session_recall",
                "description": "Search past session memory (user prompts, assistant messages, subagent results, and compact tool references) for relevant context. Use when the user's question references prior work, decisions, or things \"we\" did — before exploring from scratch.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (keywords, file names, concepts)"
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results to return (default 10)"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "code_recall",
                "description": "Search the indexed local codebase for semantically relevant functions, classes, or code blocks. AST-chunked via tree-sitter, hybrid BM25 + vector. Returns {path, line_start, line_end, qualified, body}.\n\nResults are scoped to the indexed root containing your current working directory. If that's not the repo you want to search, pass `scope=\"all\"` (cross-root) or `path` (a specific subtree). If cwd is outside every indexed root, no results are returned — pass `scope=\"all\"` explicitly to broaden.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (function names, type names, concepts)"
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results to return (default 10)"
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by language: rust | typescript | javascript | python | go | bash"
                        },
                        "path": {
                            "type": "string",
                            "description": "Narrow to files under this path prefix. Must be under an indexed root, or no results are returned."
                        },
                        "scope": {
                            "type": "string",
                            "description": "current (default) — scope to the indexed root containing cwd; outside-root → empty. all — search every indexed root, ignoring cwd."
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    })
}

fn handle_tools_call(params: &Value, http_url: &str) -> Result<Value, RpcError> {
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RpcError {
            code: -32602,
            message: "missing tool name".into(),
        })?;
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

    match name {
        "session_recall" => call_recall(&arguments, http_url),
        "code_recall" => call_code_recall(&arguments, http_url),
        // Back-compat: older agents may still send memory_recall.
        "memory_recall" => call_recall(&arguments, http_url),
        other => Err(RpcError {
            code: -32602,
            message: format!("unknown tool: {other}"),
        }),
    }
}

fn call_recall(args: &Value, http_url: &str) -> Result<Value, RpcError> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RpcError {
            code: -32602,
            message: "memory_recall: missing `query`".into(),
        })?;
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(10);

    let body = json!({"query": query, "limit": limit}).to_string();
    match http_post(&format!("{http_url}/recall"), &body) {
        Ok(resp) => Ok(tool_result_text(&format_recall(&resp), false)),
        Err(e) => {
            let msg = e.to_string();
            if classify_unreachable(&msg) {
                // isError=true so the model sees the situation as an actionable
                // tool failure rather than a protocol error.
                Ok(tool_result_text(&unreachable_message(http_url), true))
            } else {
                Ok(tool_result_text(&format!("session_recall failed: {msg}"), true))
            }
        }
    }
}

fn call_code_recall(args: &Value, http_url: &str) -> Result<Value, RpcError> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RpcError {
            code: -32602,
            message: "code_recall: missing `query`".into(),
        })?;
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10);
    let mut body = json!({"query": query, "limit": limit});
    if let Some(lang) = args.get("language").and_then(|v| v.as_str()) {
        body["language"] = json!(lang);
    }
    if let Some(p) = args.get("path").and_then(|v| v.as_str()) {
        body["path"] = json!(p);
    }
    if let Some(sc) = args.get("scope").and_then(|v| v.as_str()) {
        body["scope"] = json!(sc);
    }
    // Spawn-time cwd is transport metadata — not exposed in the tool schema.
    // The server uses it to derive an auto-scope when neither `path` nor
    // `scope=all` is set. Claude Code launches the shim from the session's
    // project root.
    if let Ok(cwd) = std::env::current_dir() {
        body["cwd"] = json!(cwd.to_string_lossy());
    }
    match http_post(&format!("{http_url}/code/search"), &body.to_string()) {
        Ok(resp) => Ok(tool_result_text(&format_code_recall(&resp), false)),
        Err(e) => {
            let msg = e.to_string();
            if classify_unreachable(&msg) {
                Ok(tool_result_text(&unreachable_message(http_url), true))
            } else {
                Ok(tool_result_text(&format!("code_recall failed: {msg}"), true))
            }
        }
    }
}

/// Render the code/search JSON response as compact markdown for the model.
/// Server returns `{results: [...], warnings: [...]}`. Warnings (e.g. "your
/// cwd is outside the indexed roots") are prepended so the model sees the
/// situation before the results.
fn format_code_recall(raw: &str) -> String {
    let parsed: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return raw.to_string(),
    };
    let results = parsed.get("results").and_then(|v| v.as_array());
    let warnings = parsed.get("warnings").and_then(|v| v.as_array());
    // Back-compat: bare array still accepted.
    let arr = results.cloned().or_else(|| parsed.as_array().cloned());

    let mut out = String::new();
    if let Some(ws) = warnings {
        for w in ws {
            if let Some(s) = w.as_str() {
                out.push_str(&format!("⚠ {s}\n"));
            }
        }
        if !ws.is_empty() && results.is_some() {
            out.push('\n');
        }
    }

    let arr = match arr {
        Some(a) => a,
        None => {
            // Couldn't make sense of the shape — surface the raw text.
            out.push_str(raw);
            return out;
        }
    };
    if arr.is_empty() {
        if out.is_empty() {
            return "(no code matches)".to_string();
        }
        out.push_str("(no code matches)\n");
        return out;
    }
    for hit in &arr {
        let path = hit.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        let line_start = hit.get("line_start").and_then(|v| v.as_i64()).unwrap_or(0);
        let line_end = hit.get("line_end").and_then(|v| v.as_i64()).unwrap_or(0);
        let qualified = hit.get("qualified").and_then(|v| v.as_str()).unwrap_or("");
        let body = hit.get("body").and_then(|v| v.as_str()).unwrap_or("");
        let score = hit.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let qual = if qualified.is_empty() {
            String::new()
        } else {
            format!(" {qualified}")
        };
        out.push_str(&format!(
            "─── {path}:{line_start}-{line_end}{qual} (rrf {score:.3}) ───\n{body}\n\n"
        ));
    }
    out
}

/// MCP tool result body: `{content: [{type:"text", text:"..."}], isError: bool}`.
fn tool_result_text(text: &str, is_error: bool) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "isError": is_error
    })
}

/// Re-shape the recall JSON into a compact, model-friendly markdown listing.
fn format_recall(raw: &str) -> String {
    let parsed: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return raw.to_string(),
    };
    let arr = match parsed.as_array() {
        Some(a) => a,
        None => return raw.to_string(),
    };
    if arr.is_empty() {
        return "(no relevant memory)".to_string();
    }
    let mut out = String::new();
    for hit in arr {
        let obs = hit.get("obs").unwrap_or(&Value::Null);
        let kind = obs.get("kind").and_then(|v| v.as_str()).unwrap_or("?");
        let session = obs.get("session").and_then(|v| v.as_str()).unwrap_or("?");
        let body = obs.get("body").and_then(|v| v.as_str()).unwrap_or("");
        let score = hit.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let body_short = trim_oneline(body, 240);
        out.push_str(&format!(
            "- [{kind} | sess {sess_short} | rrf {score:.3}] {body_short}\n",
            sess_short = &session.chars().take(8).collect::<String>(),
        ));
    }
    out
}

fn trim_oneline(s: &str, max: usize) -> String {
    let first = s.lines().next().unwrap_or("").trim();
    if first.len() <= max {
        first.to_string()
    } else {
        let mut cut = max;
        while !first.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}…", &first[..cut])
    }
}

fn http_post(url: &str, body: &str) -> Result<String> {
    let resp = ureq::post(url)
        .set("Content-Type", "application/json")
        .send_string(body)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(resp.into_string()?)
}

/// Distinguish "server isn't running" from other HTTP errors. ureq surfaces
/// connection-refused as `Status(_, _)` for HTTP errors or `Transport(_)` for
/// network-level failures. We only want to upgrade the message when the
/// daemon itself isn't reachable — auth/4xx/5xx from a running server still
/// flow through as generic errors.
fn classify_unreachable(err_msg: &str) -> bool {
    let s = err_msg.to_lowercase();
    s.contains("connection refused")
        || s.contains("connection reset")
        || s.contains("connect error")
        || s.contains("tcp connect error")
        || s.contains("connection closed")
        || s.contains("transport error")
        || s.contains("os error 61") // ECONNREFUSED on macOS
        || s.contains("dns")
        || s.contains("timed out")
}

fn unreachable_message(http_url: &str) -> String {
    format!(
        "`memorize serve` is not running at {http_url}. \
         Memory recall is unavailable until it's started. \
         Start it with `memorize serve` (foreground) or load the launchd \
         service: `launchctl load ~/Library/LaunchAgents/com.mhseiden.memorize.plist`."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_response_shape() {
        let r = handle_initialize();
        assert_eq!(r["serverInfo"]["name"], "memorize");
        assert!(r["protocolVersion"].is_string());
        assert!(r["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_has_session_and_code() {
        let r = handle_tools_list();
        let tools = r["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"session_recall"));
        assert!(names.contains(&"code_recall"));
    }

    #[test]
    fn format_recall_compacts() {
        let raw = serde_json::json!([
            {
                "obs": {
                    "id": 1,
                    "ts": 100,
                    "session": "abc12345-rest",
                    "branch": null,
                    "kind": "user_prompt",
                    "body": "first line\nsecond line"
                },
                "score": 0.123
            },
            {
                "obs": {
                    "id": 2,
                    "ts": 101,
                    "session": "def67890-rest",
                    "branch": null,
                    "kind": "tool_use",
                    "body": "tool output"
                },
                "score": 0.05
            }
        ])
        .to_string();
        let out = format_recall(&raw);
        // Compacted to one line per hit, kind/session/score visible, body truncated.
        assert!(out.contains("user_prompt"));
        assert!(out.contains("abc12345"));
        assert!(out.contains("first line"));
        // Second line should NOT appear (we keep only the first line).
        assert!(!out.contains("second line"));
    }

    #[test]
    fn format_recall_empty() {
        assert_eq!(format_recall("[]"), "(no relevant memory)");
    }

    #[test]
    fn dispatch_unknown_method() {
        let req = Request {
            jsonrpc: "2.0".into(),
            id: Some(json!(1)),
            method: "nonsense".into(),
            params: json!({}),
        };
        let resp = dispatch(&req, "http://127.0.0.1:9");
        assert!(resp.error.is_some());
    }
}
