//! Wire memorize into Claude Code: 6 hook stubs in `~/.claude/hooks/` and a
//! merged update to `~/.claude/settings.json` that appends our hooks to
//! whatever's already configured (so e.g. `notify-on-stop.sh` is preserved).

use crate::config::Config;
use anyhow::{Context, Result};
use std::path::PathBuf;

/// Map of (Claude-Code event name) → (filename, --hook arg).
const HOOKS: &[(&str, &str, &str)] = &[
    ("SessionStart", "memorize-session-start.sh", "session-start"),
    ("SessionEnd", "memorize-session-end.sh", "session-end"),
    ("UserPromptSubmit", "memorize-user-prompt-submit.sh", "user-prompt-submit"),
    ("PostToolUse", "memorize-post-tool-use.sh", "post-tool-use"),
    ("Stop", "memorize-stop.sh", "stop"),
    ("SubagentStop", "memorize-subagent-stop.sh", "subagent-stop"),
];

const CLAUDE_MD_START: &str = "<!-- memorize:recall-hint:start -->";
const CLAUDE_MD_END: &str = "<!-- memorize:recall-hint:end -->";
// Legacy single-marker from earlier installs; we treat its presence as "stale,
// rewrite from scratch" so users don't get pinned to outdated wording.
const CLAUDE_MD_LEGACY_MARKER: &str = "<!-- memorize:recall-hint -->";

const CLAUDE_MD_HINT: &str = r#"

<!-- memorize:recall-hint:start -->
## Memory tools

You have two MCP recall tools backed by the local memorize daemon:

- `session_recall(query, limit?)` — searches prior session conversation memory (user prompts, your past assistant messages, subagent results, and compact references to files touched in prior sessions). Reach for this whenever the user's question references prior work, decisions, or things "we" did.
- `code_recall(query, limit?, language?, path_prefix?)` — searches a local code index that's kept fresh on file save. AST-chunked via tree-sitter, returns function/class-level snippets with `{path, line_start, line_end}`. Reach for this when you need to find where something is defined or how a concept is implemented across the codebase, before resorting to `Grep`/`Glob` from scratch.

The session index records *intent and conclusions* — not file contents. Tool calls store compact references like `Read(src/foo.rs:10-50)`; dereference them via `code_recall` or your own `Read` tool when you need the current code.
<!-- memorize:recall-hint:end -->
"#;

pub fn run(cfg: &Config, dry_run: bool) -> Result<()> {
    let home = std::env::var("HOME").context("HOME unset")?;
    let claude_dir = PathBuf::from(&home).join(".claude");
    let hooks_dir = claude_dir.join("hooks");
    let settings_path = claude_dir.join("settings.json");
    let claude_md = claude_dir.join("CLAUDE.md");

    let bin = which_memorize(cfg)?;
    eprintln!("memorize binary: {}", bin.display());

    // 0. Drop a default config file on first install so users have something
    //    to edit. Idempotent — existing configs are never touched.
    if !dry_run {
        match memorize_server::config::write_default_if_missing() {
            Ok(true) => eprintln!(
                "wrote default config to {}",
                memorize_server::config::config_path().display()
            ),
            Ok(false) => {} // already exists; quiet
            Err(e) => eprintln!("warning: couldn't write default config: {e}"),
        }
    }

    // Surface the launchd hint when the agent isn't installed yet. We don't
    // auto-install it (loading a launchd service is more invasive than the
    // hook stubs we always write).
    let launchd_plist = PathBuf::from(&home)
        .join("Library/LaunchAgents/com.mhseiden.memorize.plist");
    if !launchd_plist.exists() {
        eprintln!();
        eprintln!("tip: run `memorize install-launchd` to keep the daemon");
        eprintln!("     running across reboots and crashes (recommended).");
    }

    // 1. Write hook stub scripts.
    if !dry_run {
        std::fs::create_dir_all(&hooks_dir).context("create ~/.claude/hooks")?;
    }
    for (_event, filename, hook_arg) in HOOKS {
        let path = hooks_dir.join(filename);
        let script = format!(
            "#!/bin/sh\nexec {} capture --hook {}\n",
            bin.display(),
            hook_arg
        );
        if dry_run {
            eprintln!("[dry-run] would write {}", path.display());
        } else {
            std::fs::write(&path, &script).with_context(|| format!("write {}", path.display()))?;
            // Mark executable.
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = std::fs::metadata(&path)?.permissions();
                perms.set_mode(0o755);
                std::fs::set_permissions(&path, perms)?;
            }
            eprintln!("wrote {}", path.display());
        }
    }

    // 2. Merge into settings.json.
    let mut settings: serde_json::Value = if settings_path.exists() {
        let raw = std::fs::read_to_string(&settings_path)?;
        serde_json::from_str(&raw).unwrap_or_else(|_| serde_json::json!({}))
    } else {
        serde_json::json!({})
    };
    let hooks_obj = settings
        .as_object_mut()
        .context("settings.json root is not an object")?
        .entry("hooks")
        .or_insert_with(|| serde_json::json!({}));
    let hooks_obj = hooks_obj
        .as_object_mut()
        .context("settings.hooks is not an object")?;

    for (event, filename, _) in HOOKS {
        let arr = hooks_obj
            .entry(event.to_string())
            .or_insert_with(|| serde_json::json!([]))
            .as_array_mut()
            .context("hook event entry must be an array")?;

        let command = format!("~/.claude/hooks/{}", filename);
        let already_present = arr.iter().any(|group| {
            group
                .get("hooks")
                .and_then(|h| h.as_array())
                .map(|inner| {
                    inner
                        .iter()
                        .any(|h| h.get("command").and_then(|c| c.as_str()) == Some(command.as_str()))
                })
                .unwrap_or(false)
        });
        if already_present {
            eprintln!("{event}: memorize hook already wired, skipping");
            continue;
        }
        arr.push(serde_json::json!({
            "hooks": [{
                "type": "command",
                "command": command,
                "async": true,
                "timeout": 10
            }]
        }));
        eprintln!("{event}: queued append of ~/.claude/hooks/{filename}");
    }

    if dry_run {
        eprintln!("[dry-run] would write {}", settings_path.display());
        eprintln!("preview:\n{}", serde_json::to_string_pretty(&settings)?);
    } else {
        let pretty = serde_json::to_string_pretty(&settings)?;
        std::fs::write(&settings_path, pretty)
            .with_context(|| format!("write {}", settings_path.display()))?;
        eprintln!("updated {}", settings_path.display());
    }

    // 3. Wire the MCP server entry into ~/.claude.json (separate from
    //    ~/.claude/settings.json — Claude Code reads mcpServers from the
    //    home-level file, hooks from the settings.json one).
    let claude_json = PathBuf::from(&home).join(".claude.json");
    let mut root: serde_json::Value = if claude_json.exists() {
        let raw = std::fs::read_to_string(&claude_json)?;
        serde_json::from_str(&raw).unwrap_or_else(|_| serde_json::json!({}))
    } else {
        serde_json::json!({})
    };
    let servers = root
        .as_object_mut()
        .context("~/.claude.json root is not an object")?
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .context(".mcpServers is not an object")?;
    let mcp_entry = serde_json::json!({
        "type": "stdio",
        "command": bin.display().to_string(),
        "args": ["mcp"],
        "env": {}
    });
    let already_correct = servers
        .get("memorize")
        .and_then(|v| v.get("command"))
        .and_then(|c| c.as_str())
        == Some(bin.to_str().unwrap_or(""));
    if already_correct {
        eprintln!("MCP entry already correct in ~/.claude.json, skipping");
    } else if dry_run {
        eprintln!("[dry-run] would set mcpServers.memorize in {}", claude_json.display());
    } else {
        servers.insert("memorize".to_string(), mcp_entry);
        // Back up before overwriting — ~/.claude.json holds session UI
        // state we don't want to nuke if our write fails.
        let backup = claude_json.with_extension("json.bak.memorize");
        if claude_json.exists() {
            std::fs::copy(&claude_json, &backup).ok();
        }
        let pretty = serde_json::to_string_pretty(&root)?;
        std::fs::write(&claude_json, pretty)
            .with_context(|| format!("write {}", claude_json.display()))?;
        eprintln!("wired MCP server into {}", claude_json.display());
        eprintln!("  → restart Claude Code (or /mcp inside a session) to pick it up");
    }

    // 4. Update CLAUDE.md hint. Replace existing block (start/end markers, or
    //    legacy single-marker block) so wording can evolve across releases.
    let claude_md_existing = std::fs::read_to_string(&claude_md).unwrap_or_default();
    let stripped = strip_existing_hint(&claude_md_existing);
    let needs_write = stripped.trim_end() != claude_md_existing.trim_end()
        || !claude_md_existing.contains(CLAUDE_MD_START);
    if !needs_write {
        eprintln!("CLAUDE.md hint already up to date");
    } else if dry_run {
        eprintln!("[dry-run] would refresh CLAUDE.md hint");
    } else {
        let mut updated = stripped.trim_end().to_string();
        updated.push_str(CLAUDE_MD_HINT);
        std::fs::write(&claude_md, updated)
            .with_context(|| format!("write {}", claude_md.display()))?;
        eprintln!("refreshed memory-tool hint in {}", claude_md.display());
    }

    Ok(())
}

/// Remove any prior memorize hint block — whether it's bracketed by the
/// start/end markers or the older single-marker form (which has no end, so
/// we cut from the marker to the next blank-then-`##` heading or EOF).
fn strip_existing_hint(content: &str) -> String {
    // Modern form: start..=end inclusive.
    if let (Some(start), Some(end_marker_pos)) =
        (content.find(CLAUDE_MD_START), content.find(CLAUDE_MD_END))
    {
        let end = end_marker_pos + CLAUDE_MD_END.len();
        let mut out = String::with_capacity(content.len());
        out.push_str(&content[..start]);
        out.push_str(&content[end..]);
        return out;
    }
    // Legacy form: from the legacy marker to EOF (the original installer
    // always appended at the bottom).
    if let Some(start) = content.find(CLAUDE_MD_LEGACY_MARKER) {
        return content[..start].to_string();
    }
    content.to_string()
}

fn which_memorize(_cfg: &Config) -> Result<PathBuf> {
    // Prefer ~/.local/bin/memorize if it exists; otherwise the absolute
    // path of the current binary (handy when iterating with `cargo run`).
    let home = std::env::var("HOME").context("HOME unset")?;
    let local_bin = PathBuf::from(&home).join(".local/bin/memorize");
    if local_bin.exists() {
        return Ok(local_bin);
    }
    Ok(std::env::current_exe()?)
}
