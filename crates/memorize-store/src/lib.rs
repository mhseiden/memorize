//! DuckDB-backed storage for memorize. The only crate that depends on `duckdb`.
//!
//! Embeddings are stored as fixed-size `FLOAT[N]` arrays where `N` is the
//! Store's `embed_dim` (set at open time). Production opens with the default
//! 384 (MiniLM); the eval harness opens with whatever dim the chosen model
//! produces. We never read embeddings back into Rust — the only thing we ask
//! DuckDB to do with them is run `array_cosine_similarity` server-side and
//! return the scalar score, which sidesteps the awkward Arrow-backed
//! array-read path in `duckdb-rs`.

pub mod fts;
pub mod schema;
pub mod store;
pub mod synonyms_seed;

pub use store::{
    BM25Hit, CodeBM25Hit, CodeChunkRow, CodeVectorHit, FileMeta, Store, VectorHit,
};

/// Default embedding dimensionality (MiniLM-L6-v2). Stores opened via
/// `Store::open` / `Store::open_in_memory` use this. Use the `_with_dim`
/// variants to override.
pub const DEFAULT_EMBED_DIM: usize = 384;
