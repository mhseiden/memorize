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

        // Validate the embed model matches whatever the vector tables were
        // built with. Cross-model vectors are silently incoherent — same
        // dim but different geometry, so cosine returns garbage rankings.
        // We refuse to start in that state. To intentionally switch models,
        // the operator runs `memorize reindex --confirm` which backs up
        // the DB, wipes the code index, updates the tag, and lets the
        // next startup cold-scan with the new model.
        let current = memorize_embed::model_tag();
        let stored = store.stored_model_tag()?;
        let vectors_present = store.count_code_chunks()? > 0;
        match stored.as_deref() {
            Some(s) if s == current => {}
            Some(s) if vectors_present => {
                anyhow::bail!(
                    "embed-model mismatch: vectors were built with '{s}' but binary is configured for '{current}'. \
                     Either revert the binary or run `memorize reindex --confirm` to wipe + rebuild."
                );
            }
            _ => {
                // Empty corpus or first run after the tag column landed —
                // stamp the current model and proceed.
                store.set_model_tag(&current)?;
            }
        }

        // Hot vector recall lives in memory; ~5s startup, ~74MB resident,
        // ~3ms queries (vs ~80ms via DuckDB SQL int8 dot product).
        store.enable_vec_cache()?;
        let store = Arc::new(store);
        // Background FTS rebuild loop. Polls every 5s; rebuilds take ~4s on
        // 192k chunks but happen on a cloned DuckDB connection so route
        // handlers don't block on them. DuckDB MVCC keeps the snapshots
        // coherent — search queries see either the pre- or post-rebuild
        // FTS state depending on when they snapshotted.
        Store::spawn_fts_worker(Arc::clone(&store), std::time::Duration::from_secs(5))?;
        let mut config = crate::config::load().unwrap_or_default();
        crate::config::apply_env_overrides(&mut config);
        let initial = IndexerSnapshot::initial(
            config.code_index.roots.clone(),
            config.code_index.enabled,
        );
        Ok(Self {
            store,
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
