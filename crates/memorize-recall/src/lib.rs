//! Recall pipeline: tokenize → expand synonyms → run BM25 + vector cosine →
//! fuse with RRF → diversify by session → hydrate observations.
//!
//! Defaults match the configuration agentmemory uses for its 95.2% R@5 on
//! LongMemEval-S, so production recall numbers transfer. The `RecallConfig`
//! struct exposes the knobs the eval harness varies for ablation; callers
//! who don't care use `RecallConfig::default()`.

pub mod expand;
pub mod rrf;
pub mod diversify;
pub mod pipeline;

pub use pipeline::{Recalled, recall, recall_with_config};

/// Which retrieval streams contribute to the fused result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Hybrid,
    Bm25Only,
    VectorOnly,
}

/// All recall knobs. Defaults reproduce agentmemory's published config.
#[derive(Debug, Clone, Copy)]
pub struct RecallConfig {
    pub mode: Mode,
    /// RRF constant. Agentmemory uses 60.
    pub rrf_k: f64,
    /// Per-stream top-K cutoff before fusion.
    pub per_stream_top_k: usize,
    /// Max results per session in the final ranking. `None` disables diversification.
    pub diversify_cap: Option<usize>,
    /// If false, the recall pipeline skips synonym expansion (caller is
    /// responsible for ensuring the synonyms table is empty if it wants a
    /// fully clean ablation).
    pub use_synonyms: bool,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            mode: Mode::Hybrid,
            rrf_k: 60.0,
            per_stream_top_k: 50,
            diversify_cap: Some(3),
            use_synonyms: true,
        }
    }
}
