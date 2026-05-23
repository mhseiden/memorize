//! MiniLM-L6-v2 (384d) embeddings via `fastembed`. Same model agentmemory
//! uses on LongMemEval-S — comparisons stay apples-to-apples.
//!
//! The model is lazy-initialized on first call (a one-time ONNX download to
//! `~/.memorize/models/`) and reused for the process lifetime. We hold the
//! singleton behind a `Mutex` because `TextEmbedding` itself is `Send` but
//! not `Sync` — and we only ever embed one thing at a time on the hot path.

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static EMBEDDER: OnceLock<Mutex<TextEmbedding>> = OnceLock::new();

/// Where ONNX model files live. Default: `~/.memorize/models`. Override with
/// `MEMORIZE_MODEL_DIR` if you want to share a cache between projects.
fn cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("MEMORIZE_MODEL_DIR") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".memorize").join("models")
}

fn ensure_init() -> Result<&'static Mutex<TextEmbedding>> {
    if let Some(e) = EMBEDDER.get() {
        return Ok(e);
    }
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).context("create model cache dir")?;
    let options = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
        .with_cache_dir(dir)
        .with_show_download_progress(false);
    let model = TextEmbedding::try_new(options).context("init AllMiniLML6V2")?;
    let _ = EMBEDDER.set(Mutex::new(model));
    Ok(EMBEDDER.get().expect("just set"))
}

/// Embed a single string. 384 floats, L2-normalized (MiniLM default pooling).
pub fn embed(text: &str) -> Result<Vec<f32>> {
    let m = ensure_init()?;
    let lock = m.lock().expect("embedder mutex poisoned");
    let mut out = lock.embed(vec![text], None).context("embed")?;
    Ok(out.pop().expect("embed returned empty Vec"))
}

/// Batch embedding. Slightly more efficient than calling `embed` in a loop
/// because tokenization+inference are batched.
pub fn embed_batch(texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    let m = ensure_init()?;
    let lock = m.lock().expect("embedder mutex poisoned");
    lock.embed(texts.to_vec(), None).context("embed batch")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Marked #[ignore] because it downloads ~90MB of ONNX on first run. Run
    // explicitly with `cargo test -p memorize-embed -- --ignored`.
    #[test]
    #[ignore]
    fn embed_returns_384() {
        let v = embed("hello world").unwrap();
        assert_eq!(v.len(), 384);
    }

    #[test]
    #[ignore]
    fn semantically_similar_have_high_cosine() {
        let a = embed("kubernetes pod scheduling").unwrap();
        let b = embed("k8s container orchestration").unwrap();
        let c = embed("the weather is nice today").unwrap();
        let cos = |x: &[f32], y: &[f32]| -> f32 {
            let dot: f32 = x.iter().zip(y).map(|(a, b)| a * b).sum();
            let na: f32 = x.iter().map(|a| a * a).sum::<f32>().sqrt();
            let nb: f32 = y.iter().map(|a| a * a).sum::<f32>().sqrt();
            dot / (na * nb)
        };
        assert!(cos(&a, &b) > cos(&a, &c));
    }
}
