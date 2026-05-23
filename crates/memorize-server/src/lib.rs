//! Loopback HTTP server. Exposes capture / recall / context / health for the
//! CLI and Claude Code hook stubs to talk to.

pub mod routes;
pub mod state;

pub use routes::serve;
pub use state::ServerState;
