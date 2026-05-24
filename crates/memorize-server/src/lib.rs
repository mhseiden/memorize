//! Loopback HTTP server. Exposes capture / recall / context / health for the
//! CLI and Claude Code hook stubs to talk to. Spawns a background code
//! indexer that watches configured repo roots.

pub mod code_indexer;
pub mod config;
pub mod indexer_log;
pub mod indexer_status;
pub mod routes;
pub mod state;

pub use config::{CodeIndexConfig, UserConfig};
pub use indexer_status::{IndexerPhase, IndexerSnapshot, IndexerStatus};
pub use routes::serve;
pub use state::ServerState;
