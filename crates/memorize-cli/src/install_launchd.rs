//! `memorize install-launchd` — install + load a launchd user-agent that
//! keeps `memorize serve` running across reboots and crashes.
//!
//! The plist lives at `~/Library/LaunchAgents/com.mhseiden.memorize.plist`
//! so the user can edit / unload it via standard launchctl tooling. We use
//! `KeepAlive` (auto-restart on crash) plus `RunAtLoad` (start now and on
//! every login).

use anyhow::{Context, Result};
use std::path::PathBuf;

const LABEL: &str = "com.mhseiden.memorize";

pub fn run(dry_run: bool) -> Result<()> {
    let home = std::env::var("HOME").context("HOME unset")?;
    let bin = which_memorize(&home)?;
    let plist_path = PathBuf::from(&home)
        .join("Library/LaunchAgents")
        .join(format!("{LABEL}.plist"));
    let log_dir = PathBuf::from(&home).join(".memorize");
    std::fs::create_dir_all(&log_dir).ok();
    let stdout_log = log_dir.join("server.log");
    let stderr_log = log_dir.join("server.log");

    let contents = plist_xml(&bin, &stdout_log, &stderr_log);

    if dry_run {
        eprintln!("[dry-run] plist target: {}", plist_path.display());
        eprintln!("--- contents ---");
        eprintln!("{contents}");
        eprintln!("--- would run: launchctl unload {plist} ; launchctl load -w {plist}",
                  plist = plist_path.display());
        return Ok(());
    }

    if let Some(parent) = plist_path.parent() {
        std::fs::create_dir_all(parent).context("create LaunchAgents dir")?;
    }
    std::fs::write(&plist_path, contents).context("write plist")?;
    eprintln!("wrote {}", plist_path.display());

    // Unload-then-load so re-running after a binary change picks up the new
    // ProgramArguments. The first unload may fail (not loaded yet) — fine.
    let _ = std::process::Command::new("launchctl")
        .args(["unload", plist_path.to_str().unwrap()])
        .output();
    let out = std::process::Command::new("launchctl")
        .args(["load", "-w", plist_path.to_str().unwrap()])
        .output()
        .context("invoke launchctl load")?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("launchctl load failed: {}", err.trim());
    }
    eprintln!("loaded launchd service {LABEL}");
    eprintln!("  → `memorize serve` will now restart on crash and at login");
    eprintln!("  → unload with: launchctl unload {}", plist_path.display());
    eprintln!("  → tail logs:   tail -f {}", stdout_log.display());
    Ok(())
}

pub fn uninstall(dry_run: bool) -> Result<()> {
    let home = std::env::var("HOME").context("HOME unset")?;
    let plist_path = PathBuf::from(&home)
        .join("Library/LaunchAgents")
        .join(format!("{LABEL}.plist"));
    if !plist_path.exists() {
        eprintln!("(launchd service not installed; nothing to uninstall)");
        return Ok(());
    }
    if dry_run {
        eprintln!("[dry-run] would unload + remove {}", plist_path.display());
        return Ok(());
    }
    let _ = std::process::Command::new("launchctl")
        .args(["unload", plist_path.to_str().unwrap()])
        .output();
    std::fs::remove_file(&plist_path).ok();
    eprintln!("uninstalled launchd service {LABEL}");
    Ok(())
}

fn which_memorize(home: &str) -> Result<PathBuf> {
    let candidate = PathBuf::from(home).join(".local/bin/memorize");
    if candidate.exists() {
        return Ok(candidate);
    }
    // Fall back to current_exe — useful during development before the user has
    // copied a release build into ~/.local/bin.
    Ok(std::env::current_exe()?)
}

fn plist_xml(bin: &std::path::Path, stdout: &std::path::Path, stderr: &std::path::Path) -> String {
    // Hand-written so we can include explanatory comments. Standard launchd
    // user-agent shape; PATH includes /opt/homebrew and /usr/local/bin so
    // child processes (tree-sitter parse helpers, etc.) find any system deps.
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{bin}</string>
        <string>serve</string>
    </array>

    <!-- Start at login and after each `launchctl load`. -->
    <key>RunAtLoad</key>
    <true/>

    <!-- Restart on crash, but back off if it crashes immediately. -->
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>5</integer>

    <!-- Logs (merged stdout/stderr — same file). Rotate manually. -->
    <key>StandardOutPath</key>
    <string>{stdout}</string>
    <key>StandardErrorPath</key>
    <string>{stderr}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <!-- Surface indexer milestones via the structured indexer.log; this
             keeps the merged server.log focused on HTTP traffic. -->
        <key>MEMORIZE_VERBOSE</key>
        <string>1</string>
        <!-- tracing log level (server.log). memorize at info; silence tantivy's
             chatty segment-merge logs so the heartbeat + status stay readable. -->
        <key>MEMORIZE_LOG</key>
        <string>info,tantivy=warn</string>
    </dict>

    <!-- Run as the logged-in user; no special privileges. -->
    <key>ProcessType</key>
    <string>Interactive</string>
</dict>
</plist>
"#,
        label = LABEL,
        bin = bin.display(),
        stdout = stdout.display(),
        stderr = stderr.display(),
    )
}
