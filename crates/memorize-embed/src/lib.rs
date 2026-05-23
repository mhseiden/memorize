//! Text embeddings via `fastembed` 5.x.
//!
//! Model selection (default: `AllMiniLML6V2`):
//!
//! ```text
//! MEMORIZE_EMBED_MODEL = minilm | gte-large | jina-code
//! ```
//!
//! Other knobs:
//! - `MEMORIZE_MODEL_DIR`: override the ONNX cache directory (default `~/.memorize/models`).
//!
//! CoreML execution provider on macOS is **opt-in** via `MEMORIZE_EMBED_COREML=1`.
//! Measured results on macOS 15 / M4 Pro / ort 2.0-rc.12:
//!
//!  - **MiniLM**: compiles cleanly under MLProgram, but inference is ~10×
//!    slower than CPU EP (11s/q vs 1.2s/q). Cause is well-documented: ORT's
//!    CoreML EP fragments transformer graphs into many CPU/ANE regions,
//!    and the per-batch memory marshaling + dynamic-shape reshape overhead
//!    dominates at our small batch size and model size.
//!  - **GTE-Large**: fails to compile (CoreML rejects unsupported ops with
//!    error -7).
//!  - **Jina-Code**: untested, same architecture family as the other two.
//!
//! Configured for opt-in use anyway with:
//!  - MLProgram backend (legacy NeuralNetwork crashes on macOS 14+).
//!  - `with_model_cache_dir` so successive process spawns reuse the compile.
//!  - `SpecializationStrategy::FastPrediction` for inference latency over
//!    specialization time.
//!  - `MEMORIZE_EMBED_COREML_UNITS={all,ane,gpu,cpu}` to vary hardware target.
//!  - `MEMORIZE_EMBED_COREML_PROFILE=1` to log per-op dispatch — the ORT-
//!    recommended diagnostic when perf is unexpected.

use anyhow::{Context, Result, bail};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone)]
struct Choice {
    model: EmbeddingModel,
    tag: &'static str,
    dim: usize,
}

const MINILM: Choice = Choice {
    model: EmbeddingModel::AllMiniLML6V2,
    tag: "all-minilm-l6-v2",
    dim: 384,
};
const GTE_LARGE: Choice = Choice {
    model: EmbeddingModel::GTELargeENV15,
    tag: "gte-large-en-v1.5",
    dim: 1024,
};
const JINA_CODE: Choice = Choice {
    model: EmbeddingModel::JinaEmbeddingsV2BaseCode,
    tag: "jina-embeddings-v2-base-code",
    dim: 768,
};

static CHOICE: OnceLock<Choice> = OnceLock::new();
static EMBEDDER: OnceLock<Mutex<TextEmbedding>> = OnceLock::new();

fn resolve_choice() -> Choice {
    match std::env::var("MEMORIZE_EMBED_MODEL")
        .unwrap_or_else(|_| "minilm".to_string())
        .to_lowercase()
        .as_str()
    {
        "minilm" | "all-minilm-l6-v2" | "" => MINILM,
        "gte-large" | "gte-large-en-v1.5" => GTE_LARGE,
        "jina-code" | "jina-embeddings-v2-base-code" => JINA_CODE,
        other => panic!(
            "MEMORIZE_EMBED_MODEL={other:?} is not recognized. Use minilm | gte-large | jina-code."
        ),
    }
}

fn cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("MEMORIZE_MODEL_DIR") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".memorize").join("models")
}

fn current() -> &'static Choice {
    CHOICE.get_or_init(resolve_choice)
}

pub fn model_tag() -> &'static str {
    current().tag
}

pub fn embedding_dim() -> usize {
    current().dim
}

fn ensure_init() -> Result<&'static Mutex<TextEmbedding>> {
    if let Some(e) = EMBEDDER.get() {
        return Ok(e);
    }
    let choice = current();
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).context("create model cache dir")?;

    let mut options = TextInitOptions::new(choice.model.clone())
        .with_cache_dir(dir.clone())
        .with_show_download_progress(false);

    // CoreML EP — opt-in. MLProgram backend, per-model on-disk compile
    // cache, FastPrediction specialization. See module docs for the
    // measured tradeoffs.
    #[cfg(target_os = "macos")]
    if std::env::var("MEMORIZE_EMBED_COREML").as_deref() == Ok("1") {
        use ort::ep::coreml::{CoreML, ComputeUnits, ModelFormat, SpecializationStrategy};
        let cache = dir.join("coreml-cache");
        std::fs::create_dir_all(&cache).ok();
        let profile =
            std::env::var("MEMORIZE_EMBED_COREML_PROFILE").as_deref() == Ok("1");
        let compute_units = match std::env::var("MEMORIZE_EMBED_COREML_UNITS")
            .as_deref()
            .unwrap_or("all")
        {
            "ane" | "cpu_ane" => ComputeUnits::CPUAndNeuralEngine,
            "gpu" | "cpu_gpu" => ComputeUnits::CPUAndGPU,
            "cpu" => ComputeUnits::CPUOnly,
            _ => ComputeUnits::All,
        };
        options = options.with_execution_providers(vec![
            CoreML::default()
                .with_model_format(ModelFormat::MLProgram)
                .with_model_cache_dir(cache.display().to_string())
                .with_specialization_strategy(SpecializationStrategy::FastPrediction)
                .with_compute_units(compute_units)
                .with_profile_compute_plan(profile)
                .build(),
        ]);
    }

    let model = TextEmbedding::try_new(options)
        .with_context(|| format!("init embedder for {}", choice.tag))?;
    let _ = EMBEDDER.set(Mutex::new(model));
    Ok(EMBEDDER.get().expect("just set"))
}

/// Embed a single string. Inputs exceeding the model's max-token window are
/// silently truncated by the tokenizer (model-dependent: MiniLM = 512,
/// GTE-Large and Jina-Code = 8192).
pub fn embed(text: &str) -> Result<Vec<f32>> {
    let m = ensure_init()?;
    let mut lock = m.lock().expect("embedder mutex poisoned");
    let mut out = lock.embed(vec![text], None).context("embed")?;
    let v = out.pop().expect("embed returned empty Vec");
    let expected = current().dim;
    if v.len() != expected {
        bail!(
            "embedding dim {} doesn't match expected {} for {}",
            v.len(),
            expected,
            current().tag
        );
    }
    Ok(v)
}

/// Batch embedding. Single ONNX inference for all inputs.
pub fn embed_batch(texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    let m = ensure_init()?;
    let mut lock = m.lock().expect("embedder mutex poisoned");
    let out = lock.embed(texts.to_vec(), None).context("embed batch")?;
    let expected = current().dim;
    if let Some(v) = out.first() {
        if v.len() != expected {
            bail!(
                "embedding dim {} doesn't match expected {} for {}",
                v.len(),
                expected,
                current().tag
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_choice_is_minilm() {
        if std::env::var("MEMORIZE_EMBED_MODEL").is_err() {
            assert_eq!(MINILM.tag, "all-minilm-l6-v2");
            assert_eq!(MINILM.dim, 384);
        }
    }

    #[test]
    #[ignore]
    fn embed_returns_expected_dim() {
        let v = embed("hello world").unwrap();
        assert_eq!(v.len(), embedding_dim());
    }
}
