//! Text embeddings via `fastembed` 5.x.
//!
//! Model selection (default: `AllMiniLML6V2` — fp32):
//!
//! ```text
//! MEMORIZE_EMBED_MODEL = minilm | minilm-q | gte-large | jina-code
//! ```
//!
//! `minilm-q` is the INT8 quantized variant. The cited 2–4× CPU speedup is
//! x86+VNNI only — measured on M4 Pro / ORT 2.0-rc.12, INT8 was ~15% **slower**
//! than fp32 because ORT's ARM kernels fall back to emulated INT8. Stay on fp32
//! unless running on x86 with VNNI.
//!
//! See `README.md` in this crate for the backend-selection history (Candle
//! Metal / Accelerate, INT8 quantization, CoreML EP) and why fastembed/ORT
//! CPU is the local ceiling on Apple Silicon.
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
use tokenizers::Tokenizer;

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
const MINILM_Q: Choice = Choice {
    model: EmbeddingModel::AllMiniLML6V2Q,
    tag: "all-minilm-l6-v2-q",
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
/// Tokenizer used for token-cap accounting. Loaded independently of the
/// (heavier) `TextEmbedding` model; callers that just need to size inputs
/// don't pay the ONNX initialization cost.
static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

fn resolve_choice() -> Choice {
    match std::env::var("MEMORIZE_EMBED_MODEL")
        .unwrap_or_else(|_| "minilm".to_string())
        .to_lowercase()
        .as_str()
    {
        "minilm" | "all-minilm-l6-v2" | "" => MINILM,
        "minilm-q" | "all-minilm-l6-v2-q" => MINILM_Q,
        "gte-large" | "gte-large-en-v1.5" => GTE_LARGE,
        "jina-code" | "jina-embeddings-v2-base-code" => JINA_CODE,
        other => panic!(
            "MEMORIZE_EMBED_MODEL={other:?} is not recognized. Use minilm-q | minilm | gte-large | jina-code."
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

/// Max sequence length accepted by the current embedding model. Anything
/// longer is silently truncated inside the tokenizer/model — callers can
/// avoid that by re-splitting with [`split_to_token_cap`] first.
pub fn max_seq_tokens() -> usize {
    // MiniLM and Jina-code both ship with 512 in the model config. GTE-Large
    // is 512 too despite the model architecture supporting more.
    match current().tag {
        "all-minilm-l6-v2" | "all-minilm-l6-v2-q" => 512,
        "gte-large-en-v1.5" => 512,
        "jina-embeddings-v2-base-code" => 8192,
        _ => 512,
    }
}

fn ensure_tokenizer() -> Result<&'static Tokenizer> {
    if let Some(t) = TOKENIZER.get() {
        return Ok(t);
    }
    // Find tokenizer.json under the current model's snapshot dir. Cache
    // layout matches what fastembed downloads to ~/.memorize/models.
    let dir = cache_dir();
    let repo_dir_name = match current().tag {
        "all-minilm-l6-v2" => "models--Qdrant--all-MiniLM-L6-v2-onnx",
        "all-minilm-l6-v2-q" => "models--Xenova--all-MiniLM-L6-v2",
        "gte-large-en-v1.5" => "models--Alibaba-NLP--gte-large-en-v1.5",
        "jina-embeddings-v2-base-code" => "models--jinaai--jina-embeddings-v2-base-code",
        other => bail!("no tokenizer mapping for model {other}"),
    };
    let snapshots = dir.join(repo_dir_name).join("snapshots");
    if !snapshots.exists() {
        // Force a download by initializing the model — first call populates
        // the snapshot dir.
        let _ = ensure_init()?;
    }
    let snap = std::fs::read_dir(&snapshots)
        .with_context(|| format!("read snapshots dir {}", snapshots.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("tokenizer.json").exists())
        .with_context(|| format!("no tokenizer.json under {}", snapshots.display()))?;
    let mut tok = Tokenizer::from_file(snap.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    // The vendored tokenizer.json sometimes hardcodes truncation (MiniLM ships
    // max_length=128). Disable so callers see the actual token count — they
    // do their own splitting based on the model's real max-seq window.
    tok.with_truncation(None)
        .map_err(|e| anyhow::anyhow!("disable truncation: {e}"))?;
    let _ = TOKENIZER.set(tok);
    Ok(TOKENIZER.get().expect("just set"))
}

/// Token count for a single string, using the current model's tokenizer.
pub fn count_tokens(text: &str) -> Result<usize> {
    let tok = ensure_tokenizer()?;
    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().len())
}

/// Split a string into pieces whose token count is <= `cap`. Splits at line
/// boundaries when possible — single lines longer than `cap` tokens get
/// further split at byte boundaries proportional to their share of tokens
/// (rare in source code; only triggered by minified single-line files).
///
/// Returns the original string unchanged when it already fits.
pub fn split_to_token_cap(text: &str, cap: usize) -> Result<Vec<String>> {
    if cap == 0 {
        bail!("cap must be > 0");
    }
    if count_tokens(text)? <= cap {
        return Ok(vec![text.to_string()]);
    }
    let mut pieces: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut buf_tokens = 0usize;
    for line in text.split('\n') {
        // Lazily push the joining newline so the count matches what the
        // model would see if we joined pieces back together.
        let candidate = if buf.is_empty() {
            line.to_string()
        } else {
            format!("{buf}\n{line}")
        };
        let n = count_tokens(&candidate)?;
        if n <= cap {
            buf = candidate;
            buf_tokens = n;
            continue;
        }
        // Adding this line overflows. If `buf` is non-empty, flush it and
        // start a new buffer with `line`.
        if !buf.is_empty() {
            pieces.push(std::mem::take(&mut buf));
            buf_tokens = 0;
        }
        // The line on its own might still exceed `cap` (single huge minified
        // line). Byte-split proportionally as a last resort.
        let line_tokens = count_tokens(line)?;
        if line_tokens > cap {
            for piece in byte_split_to_token_cap(line, cap)? {
                pieces.push(piece);
            }
        } else {
            buf = line.to_string();
            buf_tokens = line_tokens;
        }
    }
    if !buf.is_empty() {
        pieces.push(buf);
    }
    let _ = buf_tokens; // last value isn't read past loop exit
    Ok(pieces)
}

/// Byte-level fallback: split `line` into roughly equal pieces such that each
/// is under `cap` tokens. Used only when a single line is too long.
fn byte_split_to_token_cap(line: &str, cap: usize) -> Result<Vec<String>> {
    let total_tokens = count_tokens(line)?;
    if total_tokens <= cap {
        return Ok(vec![line.to_string()]);
    }
    let pieces_needed = total_tokens.div_ceil(cap);
    let bytes = line.as_bytes();
    let target_bytes_per_piece = bytes.len() / pieces_needed;
    let mut out: Vec<String> = Vec::with_capacity(pieces_needed);
    let mut cur = 0usize;
    while cur < bytes.len() {
        let mut end = (cur + target_bytes_per_piece).min(bytes.len());
        // Don't split mid-codepoint.
        while end < bytes.len() && (bytes[end] & 0xC0) == 0x80 {
            end += 1;
        }
        // Verify under cap; if not, halve.
        loop {
            let slice = &line[cur..end];
            let n = count_tokens(slice)?;
            if n <= cap || end - cur <= 16 {
                out.push(slice.to_string());
                cur = end;
                break;
            }
            end = cur + (end - cur) / 2;
            while end < bytes.len() && (bytes[end] & 0xC0) == 0x80 {
                end += 1;
            }
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
