//! Tiny HTTP client for talking to a locally-running `memorize serve`.

use crate::config::Config;
use anyhow::{Context, Result, anyhow};

fn url(cfg: &Config, path: &str) -> String {
    format!("http://127.0.0.1:{}{}", cfg.port, path)
}

pub fn post_json(cfg: &Config, path: &str, body: &str) -> Result<String> {
    let resp = ureq::post(&url(cfg, path))
        .set("Content-Type", "application/json")
        .send_string(body)
        .map_err(|e| anyhow!("POST {path}: {e}"))?;
    resp.into_string().context("read response body")
}

pub fn get(cfg: &Config, path: &str) -> Result<String> {
    let resp = ureq::get(&url(cfg, path))
        .call()
        .map_err(|e| anyhow!("GET {path}: {e}"))?;
    resp.into_string().context("read response body")
}
