//! DuckDB-backed storage for memorize. The only crate that depends on `duckdb`.
//!
//! Embeddings are stored as fixed-size `FLOAT[EMBED_DIM]` arrays. We never read
//! them back into Rust — the only thing we ask DuckDB to do with them is run
//! `array_cosine_similarity` server-side and return the scalar score. That
//! sidesteps the awkward Arrow-backed array-read path in `duckdb-rs`.

pub mod schema;
pub mod store;
pub mod synonyms_seed;

pub use store::{Store, BM25Hit, VectorHit};

/// MiniLM-L6-v2 dimensionality. Hard-coded into the schema; if you ever swap
/// models you also need to migrate (or wipe) the `vec` table.
pub const EMBED_DIM: usize = 384;
