//! Recall pipeline: tokenize → expand synonyms → run BM25 + vector cosine →
//! fuse with RRF (k=60) → diversify by session (max 3) → hydrate observations.
//!
//! Matches the same configuration agentmemory uses for its 95.2% R@5 on
//! LongMemEval-S, so eval numbers transfer.

pub mod expand;
pub mod rrf;
pub mod diversify;
pub mod pipeline;

pub use pipeline::{Recalled, recall};

/// RRF constant. Agentmemory uses 60; sticking with it makes our numbers
/// directly comparable.
pub const RRF_K: f64 = 60.0;

/// How many results we pull from each stream before fusion. Larger numbers
/// give RRF more material to find dual-stream hits but cost SQL time.
pub const PER_STREAM_TOP_K: usize = 50;

/// Session diversification cap. Matches agentmemory's hardcoded 3.
pub const MAX_PER_SESSION: usize = 3;
