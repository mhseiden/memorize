//! Claude Code hook event parsing + dispatch to the memorize server.
//!
//! Capture policy:
//!
//! - **Prose** events (user_prompt, assistant_message, subagent_message)
//!   store full body. These are the actual signal — what the user asked
//!   and what the model concluded.
//! - **Tool events** (Read/Edit/Write/Bash/etc.) store **compact reference
//!   one-liners**, never the tool's input or output content. The body
//!   field becomes something like `Read(src/foo.rs:10-200)` or
//!   `Bash($ git status, exit=0, 0.4s)`. The reference plus the file at
//!   its current state (via code_recall or a fresh Read) is enough to
//!   reconstruct context without persisting file contents into the
//!   session index.

use crate::client;
use crate::config::Config;
use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::Value;
use std::io::Read;
use std::process::Command;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Common {
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    cwd: Option<String>,
}

pub fn run(cfg: &Config, hook: &str) -> Result<()> {
    let mut raw = String::new();
    std::io::stdin().read_to_string(&mut raw)?;
    if raw.trim().is_empty() {
        return Ok(());
    }

    let value: Value = serde_json::from_str(&raw).context("hook stdin must be JSON")?;
    let common: Common = serde_json::from_value(value.clone()).unwrap_or(Common {
        session_id: None,
        cwd: None,
    });
    let session = common.session_id.clone().unwrap_or_else(|| "unknown".into());
    let branch = common.cwd.as_deref().and_then(git_branch);

    match hook {
        "session-start" => {
            // Inject prior context to stdout (Claude Code prepends it).
            // We don't store an obs for the session-start event itself any
            // more — it was noise.
            let query = common
                .cwd
                .as_deref()
                .map(|c| {
                    std::path::Path::new(c)
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or(c)
                        .to_string()
                })
                .unwrap_or_else(|| "recent".to_string());
            let url = format!(
                "/context?query={}&budget={}",
                percent_encode(&query),
                cfg.token_budget
            );
            if let Ok(md) = client::get(cfg, &url) {
                if !md.trim().is_empty() {
                    println!("{md}");
                }
            }
        }
        "session-end" | "stop" => {
            // Stop hook carries `last_assistant_message` — that's the
            // single most valuable obs in the conversation.
            let msg = value
                .get("last_assistant_message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if !msg.is_empty() {
                post_prose(cfg, &session, "assistant_message", msg, &branch)?;
            }
        }
        "user-prompt-submit" => {
            let prompt = value
                .get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if !prompt.is_empty() {
                post_prose(cfg, &session, "user_prompt", prompt, &branch)?;
            }
        }
        "post-tool-use" => {
            // Compact reference obs only. Never the body of the tool I/O.
            let tool_name = value
                .get("tool_name")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown");
            let tool_input = value.get("tool_input").cloned().unwrap_or(Value::Null);
            let tool_response = value.get("tool_response").cloned().unwrap_or(Value::Null);
            let is_error = tool_response
                .get("is_error")
                .and_then(|b| b.as_bool())
                .unwrap_or(false);
            if let Some(payload) = build_tool_ref(tool_name, &tool_input, &tool_response, is_error)
            {
                post_ref(cfg, &session, &payload, &branch)?;
            }
            // Tools we don't store (Grep/Glob/etc.) hit None and just exit.
        }
        "subagent-stop" => {
            // Subagent's final assistant-style message is the research artifact.
            let msg = value
                .get("last_assistant_message")
                .and_then(|v| v.as_str())
                .or_else(|| value.get("content").and_then(|v| v.as_str()))
                .unwrap_or("")
                .trim();
            if !msg.is_empty() {
                post_prose(cfg, &session, "subagent_message", msg, &branch)?;
            }
        }
        other => return Err(anyhow!("unknown hook: {other}")),
    }
    Ok(())
}

/// Built record for a tool-use obs: body + optional file reference.
struct ToolRef {
    kind: &'static str,
    body: String,
    ref_path: Option<String>,
    ref_line_start: Option<i32>,
    ref_line_end: Option<i32>,
}

fn build_tool_ref(
    name: &str,
    input: &Value,
    response: &Value,
    is_error: bool,
) -> Option<ToolRef> {
    let kind = if is_error { "tool_failure" } else { "tool_use" };
    match name {
        "Read" => {
            let path = input.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
            let offset = input
                .get("offset")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32);
            let limit = input
                .get("limit")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32);
            let (line_start, line_end) = match (offset, limit) {
                (Some(o), Some(l)) => (Some(o), Some(o + l - 1)),
                (Some(o), None) => (Some(o), None),
                _ => (None, None),
            };
            let range = match (line_start, line_end) {
                (Some(s), Some(e)) => format!(":{s}-{e}"),
                (Some(s), None) => format!(":{s}-"),
                _ => String::new(),
            };
            Some(ToolRef {
                kind,
                body: format!("Read({path}{range})"),
                ref_path: Some(path.to_string()).filter(|s| !s.is_empty()),
                ref_line_start: line_start,
                ref_line_end: line_end,
            })
        }
        "Edit" => {
            let path = input.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
            // Edit doesn't carry a line range in the payload — we'd have to
            // diff old vs new. For now record just the path; intent can be
            // inferred from surrounding prose obs.
            Some(ToolRef {
                kind,
                body: format!("Edit({path})"),
                ref_path: Some(path.to_string()).filter(|s| !s.is_empty()),
                ref_line_start: None,
                ref_line_end: None,
            })
        }
        "Write" => {
            let path = input.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
            let content_lines = input
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.lines().count() as i32);
            let len_str = content_lines
                .map(|n| format!(", {n} lines"))
                .unwrap_or_default();
            Some(ToolRef {
                kind,
                body: format!("Write({path}{len_str})"),
                ref_path: Some(path.to_string()).filter(|s| !s.is_empty()),
                ref_line_start: None,
                ref_line_end: content_lines,
            })
        }
        "Bash" => {
            let cmd = input
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let cmd_short = truncate_chars(cmd, 200);
            let exit = response
                .get("interrupted")
                .and_then(|v| v.as_bool())
                .map(|b| if b { "interrupted" } else { "exit=?" }.to_string())
                .or_else(|| {
                    response
                        .get("exit_code")
                        .and_then(|v| v.as_i64())
                        .map(|n| format!("exit={n}"))
                })
                .unwrap_or_else(|| "exit=?".into());
            Some(ToolRef {
                kind,
                body: format!("Bash($ {cmd_short}, {exit})"),
                ref_path: None,
                ref_line_start: None,
                ref_line_end: None,
            })
        }
        "WebFetch" => {
            let url = input.get("url").and_then(|v| v.as_str()).unwrap_or("");
            Some(ToolRef {
                kind,
                body: format!("WebFetch({url})"),
                ref_path: None,
                ref_line_start: None,
                ref_line_end: None,
            })
        }
        "Task" => {
            let stype = input
                .get("subagent_type")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let prompt = input
                .get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let pshort = truncate_chars(prompt, 100);
            Some(ToolRef {
                kind,
                body: format!("Task({stype}, \"{pshort}\")"),
                ref_path: None,
                ref_line_start: None,
                ref_line_end: None,
            })
        }
        // Grep, Glob, NotebookEdit, etc. — explicitly skipped.
        _ => None,
    }
}

fn post_prose(
    cfg: &Config,
    session: &str,
    kind: &str,
    body: &str,
    branch: &Option<String>,
) -> Result<()> {
    let payload = serde_json::json!({
        "session": session,
        "kind": kind,
        "body": body,
        "branch": branch,
    });
    let _ = client::post_json(cfg, "/capture", &payload.to_string());
    Ok(())
}

fn post_ref(
    cfg: &Config,
    session: &str,
    r: &ToolRef,
    branch: &Option<String>,
) -> Result<()> {
    let payload = serde_json::json!({
        "session": session,
        "kind": r.kind,
        "body": r.body,
        "branch": branch,
        "ref_path": r.ref_path,
        "ref_line_start": r.ref_line_start,
        "ref_line_end": r.ref_line_end,
    });
    let _ = client::post_json(cfg, "/capture", &payload.to_string());
    Ok(())
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut cut = max;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &s[..cut])
}

fn git_branch(cwd: &str) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .arg("rev-parse")
        .arg("--abbrev-ref")
        .arg("HEAD")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.as_bytes() {
        match *b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*b as char);
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}
