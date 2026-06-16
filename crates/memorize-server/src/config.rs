//! User-facing config loaded from `~/.memorize/config.toml` at startup.
//!
//! Precedence: env var > config file > built-in default. Env vars are kept
//! as overrides because they're useful for one-shot debug runs.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default)]
pub struct UserConfig {
    pub code_index: CodeIndexConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct CodeIndexConfig {
    /// Repo roots to scan + watch. `~/` is expanded. Missing paths are
    /// silently dropped from the active set.
    pub roots: Vec<String>,
    /// Path substrings to skip. Matches with `path.contains(pattern)` so a
    /// pattern like `/target/` excludes anything under any `target/` dir.
    pub excludes: Vec<String>,
    /// Subset of supported languages to actually index. Empty = all of them
    /// (the default).
    pub languages: Vec<String>,
    /// Largest file we'll consider for the index. Anything bigger gets
    /// skipped silently — generated / minified / vendored content doesn't
    /// belong in semantic search.
    pub max_file_bytes: u64,
    /// If true, honor any `.gitignore` / `.ignore` / global gitignore files
    /// found while walking. Requires the `ignore` crate.
    pub respect_gitignore: bool,
    /// Debounce for the file watcher. 250ms covers typical IDE save bursts.
    pub debounce_ms: u64,
    /// Master switch — false disables the whole indexer thread.
    pub enabled: bool,
    /// Rolling window (seconds) for the `/status` churn summary + heartbeat.
    pub churn_window_secs: u64,
    /// Size cap for `~/.memorize/indexer.log`; rotated to `.1` when exceeded.
    pub max_indexer_log_bytes: u64,
}

impl Default for CodeIndexConfig {
    fn default() -> Self {
        Self {
            roots: default_roots(),
            excludes: default_excludes(),
            languages: vec![],
            max_file_bytes: 1_048_576,
            respect_gitignore: true,
            debounce_ms: 250,
            enabled: true,
            churn_window_secs: 600,
            max_indexer_log_bytes: 16_777_216,
        }
    }
}

fn default_roots() -> Vec<String> {
    ["~/Vibes/memorize", "~/Repos", "~/src"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn default_excludes() -> Vec<String> {
    [
        "/target/",
        "/node_modules/",
        "/.git/",
        "/dist/",
        "/build/",
        "/.next/",
        "/.cache/",
        "/__pycache__/",
        "/.venv/",
        "/venv/",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Default location: `~/.memorize/config.toml`. Override with
/// `MEMORIZE_CONFIG`.
pub fn config_path() -> PathBuf {
    if let Ok(p) = std::env::var("MEMORIZE_CONFIG") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".memorize").join("config.toml")
}

/// Load config from disk if present, else return defaults. Bad TOML is a hard
/// error so the user sees the problem — silently falling back would mask
/// typos.
pub fn load() -> anyhow::Result<UserConfig> {
    let path = config_path();
    if !path.exists() {
        return Ok(UserConfig::default());
    }
    let raw = std::fs::read_to_string(&path)?;
    let cfg: UserConfig = toml::from_str(&raw)?;
    Ok(cfg)
}

/// Apply env-var overrides to a loaded config. Env wins over both file and
/// default.
pub fn apply_env_overrides(cfg: &mut UserConfig) {
    if let Ok(raw) = std::env::var("MEMORIZE_CODE_ROOTS") {
        cfg.code_index.roots = raw
            .split(':')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
    }
    if std::env::var("MEMORIZE_CODE_INDEX").as_deref() == Ok("0") {
        cfg.code_index.enabled = false;
    }
}

/// Write a default config file with all-defaults + helpful comments. Used
/// by `install-hooks` so the user has something to edit. Skipped if the
/// file already exists.
pub fn write_default_if_missing() -> anyhow::Result<bool> {
    let path = config_path();
    if path.exists() {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, default_config_toml())?;
    Ok(true)
}

fn default_config_toml() -> String {
    // Hand-written so the file ships with explanatory comments. Keys must
    // match the `Deserialize` shape above.
    r#"# memorize configuration. Reload by restarting `memorize serve`.

[code_index]
# Repo roots to scan and watch. `~/` is expanded. Missing paths are skipped.
# Env override: MEMORIZE_CODE_ROOTS (colon-separated).
roots = [
    "~/Vibes/memorize",
    "~/Repos",
    "~/src",
]

# Substring excludes. A path is skipped if any pattern is found in it.
excludes = [
    "/target/",
    "/node_modules/",
    "/.git/",
    "/dist/",
    "/build/",
    "/.next/",
    "/.cache/",
    "/__pycache__/",
    "/.venv/",
    "/venv/",
]

# Subset of supported languages to index. Leave empty for all.
# Supported: rust, typescript, javascript, python, go, bash.
languages = []

# Largest file we'll index (bytes). Above this, files are silently skipped.
max_file_bytes = 1048576

# Honor .gitignore / .ignore files encountered during the walk.
respect_gitignore = true

# Debounce window for the file watcher (ms).
debounce_ms = 250

# Master switch. false disables the indexer thread entirely.
# Env override: MEMORIZE_CODE_INDEX=0
enabled = true

# Rolling window (seconds) for the /status churn summary and the log heartbeat.
churn_window_secs = 600

# Size cap for ~/.memorize/indexer.log. When exceeded it is rotated to
# indexer.log.1 (single generation) so the activity log stays bounded.
max_indexer_log_bytes = 16777216
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_round_trips() {
        let toml_str = default_config_toml();
        let parsed: UserConfig = toml::from_str(&toml_str).unwrap();
        assert!(parsed.code_index.enabled);
        assert!(parsed.code_index.respect_gitignore);
        assert!(!parsed.code_index.excludes.is_empty());
    }

    #[test]
    fn partial_toml_fills_with_defaults() {
        // User specifies only roots; everything else should default.
        let raw = r#"
[code_index]
roots = ["~/Code"]
"#;
        let parsed: UserConfig = toml::from_str(raw).unwrap();
        assert_eq!(parsed.code_index.roots, vec!["~/Code"]);
        assert!(parsed.code_index.enabled);
        assert_eq!(parsed.code_index.max_file_bytes, 1_048_576);
    }

    #[test]
    fn env_overrides_apply() {
        let mut cfg = UserConfig::default();
        // Save current env so we don't pollute the process.
        let prev_roots = std::env::var("MEMORIZE_CODE_ROOTS").ok();
        let prev_disable = std::env::var("MEMORIZE_CODE_INDEX").ok();
        // SAFETY: tests are single-threaded by default for this module.
        unsafe {
            std::env::set_var("MEMORIZE_CODE_ROOTS", "/tmp/a:/tmp/b");
            std::env::set_var("MEMORIZE_CODE_INDEX", "0");
        }
        apply_env_overrides(&mut cfg);
        assert_eq!(cfg.code_index.roots, vec!["/tmp/a", "/tmp/b"]);
        assert!(!cfg.code_index.enabled);
        // Restore.
        unsafe {
            match prev_roots {
                Some(v) => std::env::set_var("MEMORIZE_CODE_ROOTS", v),
                None => std::env::remove_var("MEMORIZE_CODE_ROOTS"),
            }
            match prev_disable {
                Some(v) => std::env::set_var("MEMORIZE_CODE_INDEX", v),
                None => std::env::remove_var("MEMORIZE_CODE_INDEX"),
            }
        }
    }
}
