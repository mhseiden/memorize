//! LongMemEval-S dataset acquisition and parsing.
//!
//! Source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
//!
//! ~265MB JSON, 500 questions. Each question carries ~48 "haystack" sessions
//! and a set of gold session ids that contain the answer.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const HF_URL: &str =
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json";

/// SHA-256 from the HuggingFace LFS pointer. If the upstream file ever
/// changes we want to fail loudly rather than silently measure against a
/// different dataset.
const EXPECTED_SHA256: &str =
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442";
const EXPECTED_BYTES: u64 = 277_383_467;

pub fn data_path() -> PathBuf {
    // Anchored to the crate root so the binary works regardless of cwd.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join("data").join("longmemeval_s.json")
}

pub fn fetch() -> Result<()> {
    let path = data_path();
    if path.exists() {
        eprintln!("dataset present: {} ({} bytes)", path.display(), std::fs::metadata(&path)?.len());
        verify_sha(&path)?;
        eprintln!("SHA-256 OK");
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("create data dir")?;
    }

    eprintln!("downloading {EXPECTED_BYTES} bytes from {HF_URL}");
    let resp = ureq::get(HF_URL)
        .call()
        .map_err(|e| anyhow::anyhow!("HTTP: {e}"))?;
    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(&path).context("create output file")?;

    let mut buf = vec![0u8; 1 << 20]; // 1MB
    let mut total = 0u64;
    let mut last_logged = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        total += n as u64;
        // Progress every ~16MB so the log isn't a wall of noise.
        if total - last_logged >= 16 * (1 << 20) {
            let pct = (total as f64 / EXPECTED_BYTES as f64) * 100.0;
            eprintln!("  {total:>11} / {EXPECTED_BYTES} bytes ({pct:.1}%)");
            last_logged = total;
        }
    }
    eprintln!("download complete: {total} bytes");

    if total != EXPECTED_BYTES {
        bail!("byte count mismatch: got {total}, expected {EXPECTED_BYTES}");
    }
    verify_sha(&path)?;
    eprintln!("SHA-256 OK");
    Ok(())
}

fn verify_sha(path: &Path) -> Result<()> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let got = hex_encode(&hasher.finalize());
    if got != EXPECTED_SHA256 {
        bail!(
            "SHA-256 mismatch:\n  got      {got}\n  expected {EXPECTED_SHA256}\n  → upstream dataset may have changed; delete and re-fetch, or update EXPECTED_SHA256"
        );
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = std::fmt::Write::write_fmt(&mut s, format_args!("{b:02x}"));
    }
    s
}

// ---- Parsing -----------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct Question {
    pub question_id: String,
    pub question: String,
    pub question_type: String,
    pub haystack_sessions: Vec<Vec<Turn>>,
    pub haystack_session_ids: Vec<String>,
    pub answer_session_ids: Vec<String>,
}

impl Question {
    /// Concatenate a session's turns into the text we index. Each turn gets a
    /// role-prefixed line so BM25 can match on role-specific phrasing.
    pub fn session_body(&self, idx: usize) -> String {
        let session = &self.haystack_sessions[idx];
        let mut out = String::with_capacity(session.len() * 256);
        for t in session {
            out.push_str(&t.role);
            out.push_str(": ");
            out.push_str(&t.content);
            out.push('\n');
        }
        out
    }
}

pub fn load() -> Result<Vec<Question>> {
    let path = data_path();
    if !path.exists() {
        bail!(
            "dataset not found at {}. Run `memorize-eval fetch` first.",
            path.display()
        );
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    let questions: Vec<Question> =
        serde_json::from_str(&raw).context("parse LongMemEval-S JSON")?;

    // Sanity-check the first record so we fail loudly if the upstream
    // schema ever shifts.
    if questions.is_empty() {
        bail!("dataset parsed but contained zero questions");
    }
    let q0 = &questions[0];
    if q0.haystack_sessions.len() != q0.haystack_session_ids.len() {
        bail!(
            "schema mismatch in q0: {} haystack_sessions but {} haystack_session_ids",
            q0.haystack_sessions.len(),
            q0.haystack_session_ids.len()
        );
    }
    if q0.answer_session_ids.is_empty() {
        bail!("schema mismatch in q0: no answer_session_ids");
    }
    Ok(questions)
}
