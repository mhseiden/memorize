//! Claude Code hook event parsing + dispatch to the memorize server.
//!
//! Claude Code fires hooks with JSON on stdin. The shapes differ per event;
//! we map each to a canonical `{session, kind, body, branch?}` and POST.
//! Some hooks (SessionStart) additionally write markdown back to stdout, which
//! Claude Code injects into the next session as system context.

use crate::client;
use crate::config::Config;
use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
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

    // No payload? That happens occasionally when the user invokes manually.
    if raw.trim().is_empty() {
        return Ok(());
    }

    let value: serde_json::Value =
        serde_json::from_str(&raw).context("hook stdin must be JSON")?;
    let common: Common = serde_json::from_value(value.clone()).unwrap_or(Common {
        session_id: None,
        cwd: None,
    });

    let session = common
        .session_id
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let branch = common.cwd.as_deref().and_then(git_branch);

    match hook {
        "session-start" => {
            // Record the session and emit injected context.
            post_capture(cfg, &session, "session_start", "session started", &branch)?;
            // Build a recall query from the cwd path and recent activity.
            let context_query = common
                .cwd
                .as_deref()
                .map(|c| {
                    let leaf = std::path::Path::new(c)
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or(c);
                    leaf.to_string()
                })
                .unwrap_or_else(|| "recent".to_string());
            // Pull /context markdown and print to stdout — Claude Code injects it.
            let url = format!(
                "/context?query={}&budget={}",
                percent_encode(&context_query),
                cfg.token_budget
            );
            if let Ok(md) = client::get(cfg, &url) {
                if !md.trim().is_empty() {
                    println!("{md}");
                }
            }
        }
        "session-end" | "stop" => {
            post_capture(cfg, &session, "session_stop", &short_summary(&value), &branch)?;
        }
        "user-prompt-submit" => {
            let prompt = value
                .get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if !prompt.trim().is_empty() {
                post_capture(cfg, &session, "user_prompt", &prompt, &branch)?;
            }
        }
        "post-tool-use" => {
            let tool_name = value
                .get("tool_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let tool_input = compact_json(value.get("tool_input"));
            let tool_resp = compact_json(value.get("tool_response"));
            let is_error = value
                .get("tool_response")
                .and_then(|r| r.get("is_error"))
                .and_then(|b| b.as_bool())
                .unwrap_or(false);
            let body = format!("{tool_name}\nINPUT:\n{tool_input}\nOUTPUT:\n{tool_resp}");
            let kind = if is_error { "tool_failure" } else { "tool_use" };
            post_capture(cfg, &session, kind, &body, &branch)?;
        }
        "subagent-stop" => {
            // SubagentStop carries the subagent's final message. Useful for
            // remembering Explore agent findings.
            let body = short_summary(&value);
            post_capture(cfg, &session, "subagent_stop", &body, &branch)?;
        }
        other => return Err(anyhow!("unknown hook: {other}")),
    }
    Ok(())
}

fn post_capture(
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
    // Best-effort. Hook failures must never wedge Claude Code.
    let _ = client::post_json(cfg, "/capture", &payload.to_string());
    Ok(())
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

fn compact_json(v: Option<&serde_json::Value>) -> String {
    match v {
        None => String::new(),
        Some(s) if s.is_string() => s.as_str().unwrap_or("").to_string(),
        Some(other) => other.to_string(),
    }
}

fn short_summary(value: &serde_json::Value) -> String {
    // Generic "last meaningful field" extractor for hook payloads that don't
    // have a single canonical body field. We just stringify the entire JSON;
    // dedup + truncation in the server keeps it bounded.
    value.to_string()
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.as_bytes() {
        match *b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*b as char);
            }
            _ => {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}
