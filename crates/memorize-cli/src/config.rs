//! CLI-side config. Env-var driven for Phase 1; Phase 3 adds TOML.

use std::path::PathBuf;

pub struct Config {
    pub port: u16,
    pub token_budget: usize,
    db_path_override: Option<PathBuf>,
}

impl Config {
    pub fn load() -> Self {
        let port = std::env::var("MEMORIZE_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3111);
        let token_budget = std::env::var("MEMORIZE_TOKEN_BUDGET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2000);
        let db_path_override = std::env::var("MEMORIZE_DB_PATH").ok().map(PathBuf::from);
        Self { port, token_budget, db_path_override }
    }

    pub fn data_dir(&self) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home).join(".memorize")
    }

    pub fn db_path(&self) -> PathBuf {
        self.db_path_override
            .clone()
            .unwrap_or_else(|| self.data_dir().join("db.duckdb"))
    }
}
