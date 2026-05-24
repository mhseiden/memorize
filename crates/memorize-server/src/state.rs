use crate::config::UserConfig;
use crate::indexer_status::{IndexerSnapshot, IndexerStatus};
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
    /// Loaded once at startup; immutable until restart.
    pub config: Arc<UserConfig>,
    /// Live state of the indexer thread. Read by /status, written by the
    /// indexer.
    pub indexer_status: IndexerStatus,
}

impl ServerState {
    pub fn new(db_path: PathBuf, token_budget: usize) -> Result<Self> {
        let store = Store::open(db_path)?;
        // Hot vector recall lives in memory; ~5s startup, ~74MB resident,
        // ~3ms queries (vs ~80ms via DuckDB SQL int8 dot product).
        store.enable_vec_cache()?;
        let mut config = crate::config::load().unwrap_or_default();
        crate::config::apply_env_overrides(&mut config);
        let initial = IndexerSnapshot::initial(
            config.code_index.roots.clone(),
            config.code_index.enabled,
        );
        Ok(Self {
            store: Arc::new(store),
            dedup: Arc::new(Dedup::new()),
            privacy: Arc::new(PrivacyFilter::new()),
            token_budget,
            config: Arc::new(config),
            indexer_status: IndexerStatus::new(initial),
        })
    }

    /// In-memory variant for tests.
    pub fn in_memory(token_budget: usize) -> Result<Self> {
        let store = Store::open_in_memory()?;
        let config = UserConfig::default();
        let initial = IndexerSnapshot::initial(
            config.code_index.roots.clone(),
            config.code_index.enabled,
        );
        Ok(Self {
            store: Arc::new(store),
            dedup: Arc::new(Dedup::new()),
            privacy: Arc::new(PrivacyFilter::new()),
            token_budget,
            config: Arc::new(config),
            indexer_status: IndexerStatus::new(initial),
        })
    }
}
