//! Text embeddings via llama.cpp (`llama-cpp-2`). MiniLM on Metal on macOS.
//!
//! Model: `sentence-transformers/all-MiniLM-L6-v2` packaged as GGUF (downloaded
//! from `leliuga/all-MiniLM-L6-v2-GGUF` on first call, cached under
//! `~/.cache/huggingface/hub`). Override the GGUF source via env:
//!
//! ```text
//! MEMORIZE_EMBED_GGUF_REPO = "leliuga/all-MiniLM-L6-v2-GGUF"
//! MEMORIZE_EMBED_GGUF_FILE = "all-MiniLM-L6-v2.F32.gguf"
//! ```
//!
//! # Batching: just pass everything, llama.cpp schedules
//!
//! Callers do **not** need to chunk inputs into smaller batches before
//! calling `embed_batch`. Pass all N chunks of a file in one call. Inside
//! `decode()`, llama.cpp splits the work into ubatches of size `n_ubatch`
//! and runs them sequentially on the Metal command queue — that's the
//! scheduler we want, not one we should reimplement. Manually loop-calling
//! `embed()` per chunk costs an extra Rust↔Metal dispatch per item with no
//! upside.
//!
//! There IS a throughput-vs-batch-size curve (peak at batch=4 for MiniLM on
//! M-series, monotonic decline past batch=16 because padding waste eats
//! GPU utilization). But that's a question of which call-site batch size
//! is optimal, not whether we should fan out inside this crate. The
//! production indexer batches per-file (median ~6 chunks, naturally in the
//! peak zone); cross-file batching has been measured and doesn't help. See
//! `README.md` for the curve.
//!
//! See `README.md` for the full backend-selection history (Candle, MLX,
//! INT8 quantization, CoreML EP) and why llama.cpp+Metal won.

use anyhow::{Context, Result, anyhow, bail};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use std::cell::RefCell;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

const REPO_GGUF: &str = "leliuga/all-MiniLM-L6-v2-GGUF";
const FILE_GGUF: &str = "all-MiniLM-L6-v2.F32.gguf";
const REPO_TOKENIZER: &str = "sentence-transformers/all-MiniLM-L6-v2";

// llama.cpp requires backend init exactly once per process.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
// Model lives for the process; LlamaContext borrows from it. By holding it in
// a OnceLock we get a 'static reference safe to share across threads
// (LlamaModel is Send+Sync).
static MODEL: OnceLock<LlamaModel> = OnceLock::new();
// LlamaContext is !Send (holds a NonNull). Stash one per thread instead of a
// global Mutex; the indexer is single-threaded and the HTTP server gets one
// context per worker, which is what we want anyway.
thread_local! {
    static CTX: RefCell<Option<LlamaContext<'static>>> = const { RefCell::new(None) };
}
// HF-shaped tokenizer for the chunk-cap splitter. Much faster per-call than
// model.str_to_token; both share the MiniLM WordPiece vocab so counts agree.
static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

fn repo_gguf() -> String {
    std::env::var("MEMORIZE_EMBED_GGUF_REPO").unwrap_or_else(|_| REPO_GGUF.to_string())
}

fn file_gguf() -> String {
    std::env::var("MEMORIZE_EMBED_GGUF_FILE").unwrap_or_else(|_| FILE_GGUF.to_string())
}

fn hf_api() -> Result<hf_hub::api::sync::Api> {
    hf_hub::api::sync::ApiBuilder::new()
        .build()
        .context("hf-hub init")
}

fn ensure_backend() -> Result<&'static LlamaBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let mut b = LlamaBackend::init().context("init llama backend")?;
    // Silence llama.cpp's stderr chatter (per-call lines like "decode: cannot
    // decode batches with this context (calling encode() instead)", model
    // load dumps, etc.). Set `MEMORIZE_EMBED_VERBOSE=1` to keep them.
    if std::env::var("MEMORIZE_EMBED_VERBOSE").ok().as_deref() != Some("1") {
        b.void_logs();
    }
    let _ = BACKEND.set(b);
    Ok(BACKEND.get().expect("just set"))
}

fn ensure_model() -> Result<&'static LlamaModel> {
    if let Some(m) = MODEL.get() {
        return Ok(m);
    }
    let backend = ensure_backend()?;
    let api = hf_api()?;
    let model_path = api
        .model(repo_gguf())
        .get(&file_gguf())
        .with_context(|| format!("download GGUF {}", file_gguf()))?;
    // 1000 = "all layers" — MiniLM has 6 transformer blocks, well under.
    let params = LlamaModelParams::default().with_n_gpu_layers(1000);
    let model = LlamaModel::load_from_file(backend, &model_path, &params)
        .with_context(|| format!("load GGUF {}", model_path.display()))?;
    let _ = MODEL.set(model);
    Ok(MODEL.get().expect("just set"))
}

/// llama.cpp hard cap on n_seq_max is 256. We don't need this many sequences
/// per `decode()` call in practice; partitioning happens above this anyway.
const N_SEQ_MAX: u32 = 256;

/// Total tokens we'll feed llama.cpp in a single `decode()` call. Sized to
/// fit the Metal compute buffer comfortably on M-series — larger values
/// caused `kIOGPUCommandBufferCallbackErrorOutOfMemory` mid-scan, which
/// poisons the context. `embed_batch` partitions the caller's input into
/// sub-batches that each fit here, so callers still pass everything in one
/// call.
///
/// Override via `MEMORIZE_EMBED_N_BATCH` for benchmarking only.
fn n_batch_cap() -> u32 {
    std::env::var("MEMORIZE_EMBED_N_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192)
}

fn build_context() -> Result<LlamaContext<'static>> {
    let backend = ensure_backend()?;
    let model = ensure_model()?;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    let n_batch = n_batch_cap();
    let params = LlamaContextParams::default()
        .with_embeddings(true)
        .with_n_threads_batch(threads)
        .with_n_seq_max(N_SEQ_MAX)
        .with_n_batch(n_batch)
        .with_n_ubatch(n_batch)
        // Mean-pool matches sentence-transformers on top of MiniLM. Models that
        // ship their own pooling metadata (some BGE variants) get overridden;
        // models that ship pooling=-1 (Jina-code) get this default. A log
        // line from llama.cpp confirms what was set.
        .with_pooling_type(LlamaPoolingType::Mean);
    model.new_context(backend, params).context("new context")
}

fn ensure_tokenizer() -> Result<&'static Tokenizer> {
    if let Some(t) = TOKENIZER.get() {
        return Ok(t);
    }
    let api = hf_api()?;
    let tok_path = api
        .model(REPO_TOKENIZER.to_string())
        .get("tokenizer.json")
        .with_context(|| format!("download tokenizer.json from {REPO_TOKENIZER}"))?;
    let mut tok =
        Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("load tokenizer: {e}"))?;
    // MiniLM ships tokenizer.json with max_length=128; disable so callers see
    // true counts and apply their own cap.
    tok.with_truncation(None)
        .map_err(|e| anyhow!("disable truncation: {e}"))?;
    let _ = TOKENIZER.set(tok);
    Ok(TOKENIZER.get().expect("just set"))
}

pub fn model_tag() -> String {
    // Derive from the GGUF filename so logs/reports identify whichever model
    // is actually loaded (handy when MEMORIZE_EMBED_GGUF_FILE is set).
    let f = file_gguf();
    f.strip_suffix(".gguf").unwrap_or(&f).to_lowercase()
}

pub fn embedding_dim() -> usize {
    // Read from the loaded model so downstream code stays correct when the
    // GGUF source is overridden (e.g. BGE-M3 at 1024-d vs MiniLM at 384-d).
    ensure_model().map(|m| m.n_embd() as usize).unwrap_or(384)
}

pub fn max_seq_tokens() -> usize {
    ensure_model().map(|m| m.n_ctx_train() as usize).unwrap_or(512)
}

pub fn count_tokens(text: &str) -> Result<usize> {
    let tok = ensure_tokenizer()?;
    let enc = tok.encode(text, false).map_err(|e| anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().len())
}

pub fn embed(text: &str) -> Result<Vec<f32>> {
    let mut v = embed_batch(&[text])?;
    Ok(v.pop().expect("embed_batch returned empty for 1 input"))
}

pub fn embed_batch(texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    let model = ensure_model()?;
    let seq_cap = model.n_ctx_train() as usize;
    let n_batch = n_batch_cap() as usize;

    // Tokenize each input. AddBos::Always inserts whatever leading special
    // token the model expects ([CLS] for BERT-family, <|begin_of_text|> /
    // similar for decoder-style embedders).
    let mut tokenized: Vec<Vec<llama_cpp_2::token::LlamaToken>> = Vec::with_capacity(texts.len());
    for t in texts {
        let mut toks = model
            .str_to_token(t, AddBos::Always)
            .with_context(|| format!("tokenize ({} chars)", t.len()))?;
        // Defensive cap against the model's positional-embedding window in
        // case the caller skipped split_to_token_cap().
        if toks.len() > seq_cap {
            toks.truncate(seq_cap);
        }
        tokenized.push(toks);
    }

    // Partition into sub-batches that each fit n_batch tokens AND N_SEQ_MAX
    // sequences. The encoder asserts n_ubatch >= n_tokens, so we have to
    // split here rather than relying on llama.cpp to schedule across ubatches.
    let mut sub_ranges: Vec<std::ops::Range<usize>> = Vec::new();
    let mut start = 0usize;
    let mut cur_tokens = 0usize;
    for i in 0..tokenized.len() {
        let this_tokens = tokenized[i].len();
        let in_progress = i - start;
        // Flush when adding this sequence would overflow EITHER the token
        // budget or the seq-max budget. Always include at least one sequence
        // per sub-batch — a single chunk over n_batch was already truncated
        // to seq_cap above.
        if in_progress > 0
            && (cur_tokens + this_tokens > n_batch || in_progress + 1 > N_SEQ_MAX as usize)
        {
            sub_ranges.push(start..i);
            start = i;
            cur_tokens = 0;
        }
        cur_tokens += this_tokens;
    }
    sub_ranges.push(start..tokenized.len());

    let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    CTX.with(|cell| -> Result<()> {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(build_context()?);
        }

        for range in &sub_ranges {
            let chunk_tokens: usize = tokenized[range.clone()].iter().map(|v| v.len()).sum();
            let mut batch = LlamaBatch::new(chunk_tokens.max(1), range.len() as i32);
            for (local_idx, toks) in tokenized[range.clone()].iter().enumerate() {
                let seq_id: i32 = local_idx.try_into().context("seq_id out of range")?;
                batch.add_sequence(toks, seq_id, false).context("add_sequence")?;
            }

            // Try once; on error, drop the context (llama.cpp reports "in
            // error state from a previous command buffer failure - recreate
            // the backend to recover" after a GPU command buffer failure —
            // typically OOM. Without dropping, every subsequent call would
            // inherit the poisoned state.) and retry once. If it still fails,
            // propagate.
            let first_try =
                embed_into(slot.as_mut().expect("just set"), &mut batch, &mut out, range.len());
            if let Err(e1) = first_try {
                eprintln!(
                    "memorize-embed: context error on sub-batch ({range:?}, {chunk_tokens} tokens): {e1}; rebuilding and retrying"
                );
                *slot = None;
                *slot = Some(build_context().context("rebuild context after error")?);
                let ctx = slot.as_mut().expect("just set");
                embed_into(ctx, &mut batch, &mut out, range.len())
                    .with_context(|| format!("retry after error: {e1}"))?;
            }
        }
        Ok(())
    })?;
    Ok(out)
}

fn embed_into(
    ctx: &mut LlamaContext<'static>,
    batch: &mut LlamaBatch,
    out: &mut Vec<Vec<f32>>,
    n_seqs: usize,
) -> Result<()> {
    ctx.clear_kv_cache();
    // decode() works for both encoder-only (BERT/MiniLM/BGE — llama.cpp
    // internally redirects to encode() with a log line) and decoder-style
    // embedding models (Jina-code based on Qwen2, e5-mistral, etc.).
    // Calling encode() on a decoder model segfaults.
    ctx.decode(batch).context("decode")?;

    let n_seq_i32: i32 = n_seqs.try_into().context("n_seqs out of range")?;
    for i in 0..n_seq_i32 {
        let emb = ctx
            .embeddings_seq_ith(i)
            .with_context(|| format!("embeddings_seq_ith({i})"))?;
        let mut v = emb.to_vec();
        l2_normalize_in_place(&mut v);
        out.push(v);
    }
    Ok(())
}

fn l2_normalize_in_place(v: &mut [f32]) {
    let mag2: f32 = v.iter().map(|x| x * x).sum();
    let mag = mag2.sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= mag;
    }
}

/// Split a string into pieces with at most `cap` tokens each. Splits at line
/// boundaries when possible; single lines exceeding `cap` fall back to
/// byte-proportional splitting (rare in source code; only minified files).
pub fn split_to_token_cap(text: &str, cap: usize) -> Result<Vec<String>> {
    if cap == 0 {
        bail!("cap must be > 0");
    }
    if count_tokens(text)? <= cap {
        return Ok(vec![text.to_string()]);
    }
    let mut pieces: Vec<String> = Vec::new();
    let mut buf = String::new();
    for line in text.split('\n') {
        let candidate = if buf.is_empty() {
            line.to_string()
        } else {
            format!("{buf}\n{line}")
        };
        let n = count_tokens(&candidate)?;
        if n <= cap {
            buf = candidate;
            continue;
        }
        if !buf.is_empty() {
            pieces.push(std::mem::take(&mut buf));
        }
        let line_tokens = count_tokens(line)?;
        if line_tokens > cap {
            for piece in byte_split_to_token_cap(line, cap)? {
                pieces.push(piece);
            }
        } else {
            buf = line.to_string();
        }
    }
    if !buf.is_empty() {
        pieces.push(buf);
    }
    Ok(pieces)
}

fn byte_split_to_token_cap(line: &str, cap: usize) -> Result<Vec<String>> {
    let total = count_tokens(line)?;
    if total <= cap {
        return Ok(vec![line.to_string()]);
    }
    let pieces_needed = total.div_ceil(cap);
    let bytes = line.as_bytes();
    let target = bytes.len() / pieces_needed;
    let mut out: Vec<String> = Vec::with_capacity(pieces_needed);
    let mut cur = 0usize;
    while cur < bytes.len() {
        let mut end = (cur + target).min(bytes.len());
        while end < bytes.len() && (bytes[end] & 0xC0) == 0x80 {
            end += 1;
        }
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

    // All tests below load the model + GGUF, so they're gated on the network
    // (HF download on first run) and on a working Metal device. Run manually
    // with `cargo test -p memorize-embed --release -- --ignored`.

    #[test]
    #[ignore]
    fn embed_returns_expected_dim() {
        let v = embed("hello world").unwrap();
        assert_eq!(v.len(), embedding_dim());
        assert_eq!(embedding_dim(), 384, "default model should be MiniLM-L6 (384-d)");
    }

    #[test]
    #[ignore]
    fn embed_is_l2_normalized() {
        let v = embed("hello world").unwrap();
        let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-3, "L2 norm should be ~1.0, got {mag}");
    }

    #[test]
    #[ignore]
    fn embed_batch_matches_individual() {
        // Batched and unbatched embeddings should be byte-identical (modulo
        // float order). This guards against accidental cross-talk between
        // sequences in the batched path.
        let texts = ["alpha", "beta sequence", "gamma ray"];
        let batched = embed_batch(&texts).unwrap();
        for (i, t) in texts.iter().enumerate() {
            let single = embed(t).unwrap();
            let cos: f32 = batched[i].iter().zip(&single).map(|(a, b)| a * b).sum();
            assert!(
                cos > 0.99,
                "batched vs single cosine for {t:?} = {cos}, expected ~1.0"
            );
        }
    }

    #[test]
    #[ignore]
    fn split_to_token_cap_respects_cap() {
        // Build a string well over the cap, verify every piece is under.
        let line = "fn foo() { let x = 1; }\n".repeat(200);
        let pieces = split_to_token_cap(&line, 64).unwrap();
        for p in &pieces {
            assert!(count_tokens(p).unwrap() <= 64);
        }
    }
}
