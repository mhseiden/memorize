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
    // Tool + parameter descriptions adopted verbatim from upstream agentmemory
    // (src/mcp/tools-registry.ts). Their wording, our (minimal) schemas — we
    // don't expose params we can't honor (no format/token_budget/type/etc.
    // in v1).
    json!({
        "tools": [
            {
                "name": "memory_recall",
                "description": "Search past session observations for relevant context. Use when you need to recall what happened in previous sessions, find past decisions, or look up how a file was modified before.",
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
                "name": "memory_save",
                "description": "Explicitly save an important insight, decision, or pattern to long-term memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The insight or decision to remember"
                        }
                    },
                    "required": ["content"]
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
        "memory_recall" => call_recall(&arguments, http_url),
        "memory_save" => call_save(&arguments, http_url),
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
    let resp = http_post(&format!("{http_url}/recall"), &body).map_err(|e| RpcError {
        code: -32000,
        message: format!("server unreachable: {e}"),
    })?;
    Ok(tool_result_text(&format_recall(&resp), false))
}

fn call_save(args: &Value, http_url: &str) -> Result<Value, RpcError> {
    let content = args
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RpcError {
            code: -32602,
            message: "memory_save: missing `content`".into(),
        })?;
    let session = format!("mcp-{}", now_secs());
    let payload = json!({
        "session": session,
        "kind": "manual",
        "body": content,
    });
    let resp = http_post(&format!("{http_url}/capture"), &payload.to_string())
        .map_err(|e| RpcError {
            code: -32000,
            message: format!("server unreachable: {e}"),
        })?;
    Ok(tool_result_text(&resp, false))
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

fn now_secs() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
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
    fn tools_list_has_two() {
        let r = handle_tools_list();
        let tools = r["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"memory_recall"));
        assert!(names.contains(&"memory_save"));
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
