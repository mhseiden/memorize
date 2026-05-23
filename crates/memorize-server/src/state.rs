use anyhow::Result;
use memorize_core::{Dedup, PrivacyFilter};
use memorize_store::Store;
use std::path::PathBuf;
use std::sync::Arc;

/// Shared, thread-safe handle to all server-wide state. Cheap to clone.
#[derive(Clone)]
pub struct ServerState {
    pub store: Arc<Store>,
    pub dedup: Arc<Dedup>,
    pub privacy: Arc<PrivacyFilter>,
    pub token_budget: usize,
}

impl ServerState {
    pub fn new(db_path: PathBuf, token_budget: usize) -> Result<Self> {
        let store = Store::open(db_path)?;
        Ok(Self {
            store: Arc::new(store),
            dedup: Arc::new(Dedup::new()),
            privacy: Arc::new(PrivacyFilter::new()),
            token_budget,
        })
    }

    /// In-memory variant for tests.
    pub fn in_memory(token_budget: usize) -> Result<Self> {
        let store = Store::open_in_memory()?;
        Ok(Self {
            store: Arc::new(store),
            dedup: Arc::new(Dedup::new()),
            privacy: Arc::new(PrivacyFilter::new()),
            token_budget,
        })
    }
}
