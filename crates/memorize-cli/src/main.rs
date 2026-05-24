mod bench;
mod capture;
mod client;
mod config;
mod install_hooks;
mod install_launchd;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::Config;

#[derive(Parser)]
#[command(name = "memorize", version, about = "Personal memory layer for Claude Code")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the HTTP server (foreground).
    Serve,
    /// Read a Claude Code hook event from stdin and route to the server.
    Capture {
        #[arg(long)]
        hook: String,
    },
    /// Search prior observations.
    Recall {
        query: String,
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
    },
    /// Save an arbitrary string as a `manual` observation.
    Remember { text: String },
    /// Manage synonyms (insert both directions).
    Syn {
        #[command(subcommand)]
        op: SynOp,
    },
    /// Install hook stubs into ~/.claude/hooks/ and patch ~/.claude/settings.json.
    InstallHooks {
        /// Dry-run: print what would change, don't write.
        #[arg(long)]
        dry_run: bool,
    },
    /// Install a launchd user-agent so `memorize serve` auto-starts at login
    /// and auto-restarts on crash. macOS only.
    InstallLaunchd {
        #[arg(long)]
        dry_run: bool,
    },
    /// Remove the launchd user-agent installed by `install-launchd`.
    UninstallLaunchd {
        #[arg(long)]
        dry_run: bool,
    },
    /// Print server liveness + corpus size.
    Status,
    /// Run the MCP stdio server (Claude Code spawns this).
    Mcp,
    /// Performance / memory profiling. Subcommands run isolated against an
    /// in-memory store — the live DB and daemon are not touched.
    Bench {
        #[command(subcommand)]
        op: BenchOp,
    },
}

#[derive(Subcommand)]
enum BenchOp {
    /// Walk a root, exercise the cold-scan pipeline, report per-phase timing.
    CodeIndex {
        /// Repo root to scan.
        #[arg(long)]
        root: std::path::PathBuf,
        /// Cap files processed (useful for quick sanity checks).
        #[arg(long)]
        limit: Option<usize>,
        /// Write the markdown report to this path.
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
    /// Tokenize every chunk produced by walking a root, report token-count
    /// distribution + overflow rate vs the model's max-seq cap (512 for MiniLM).
    /// Uses the embedder's own tokenizer.
    Tokens {
        /// Repo root to scan.
        #[arg(long)]
        root: std::path::PathBuf,
        /// Cap files processed (for quick checks).
        #[arg(long)]
        limit: Option<usize>,
        /// Token cap to overflow against. MiniLM = 512.
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        /// Write the markdown report to this path.
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
    /// Sweep embed batch sizes against synthetic chunks; isolate
    /// throughput-vs-batch-size from file I/O and tree-sitter.
    Embed {
        /// Comma-separated batch sizes to sweep.
        #[arg(long, default_value = "1,2,4,8,16,32,64,128,256")]
        sizes: String,
        /// Total chunks embedded at each batch size (rounded up to a multiple of the batch).
        #[arg(long, default_value_t = 512)]
        chunks: usize,
        /// Characters per synthetic chunk body (rough proxy for a real source-code chunk).
        #[arg(long, default_value_t = 800)]
        chunk_chars: usize,
        /// Write the markdown report to this path.
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
}

#[derive(Subcommand)]
enum SynOp {
    /// Add a bidirectional synonym pair.
    Add { term: String, expansion: String },
    /// Remove a pair (both directions) or all rows touching <term>.
    Remove { term: String, expansion: Option<String> },
    /// List all stored synonym rows.
    List,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load();
    match cli.cmd {
        Cmd::Serve => run_serve(&cfg),
        Cmd::Capture { hook } => capture::run(&cfg, &hook),
        Cmd::Recall { query, limit } => run_recall(&cfg, &query, limit),
        Cmd::Remember { text } => run_remember(&cfg, &text),
        Cmd::Status => run_status(&cfg),
        Cmd::Syn { op } => run_syn(&cfg, op),
        Cmd::InstallHooks { dry_run } => install_hooks::run(&cfg, dry_run),
        Cmd::InstallLaunchd { dry_run } => install_launchd::run(dry_run),
        Cmd::UninstallLaunchd { dry_run } => install_launchd::uninstall(dry_run),
        Cmd::Mcp => {
            let http = format!("http://127.0.0.1:{}", cfg.port);
            memorize_mcp::run_stdio(&http)
        }
        Cmd::Bench { op } => match op {
            BenchOp::CodeIndex { root, limit, out } => bench::code_index(bench::BenchOpts {
                root,
                limit,
                out,
                separate_embed_init: true,
            }),
            BenchOp::Tokens { root, limit, max_tokens, out } => {
                bench::tokens(bench::TokensOpts { root, limit, max_tokens, out })
            }
            BenchOp::Embed { sizes, chunks, chunk_chars, out } => {
                let parsed: Result<Vec<usize>, _> = sizes
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect();
                let parsed = parsed.map_err(|e| anyhow::anyhow!("invalid --sizes: {e}"))?;
                bench::embed_sweep(bench::EmbedSweepOpts {
                    batch_sizes: parsed,
                    chunks,
                    chunk_chars,
                    out,
                })
            }
        },
    }
}

fn run_serve(cfg: &Config) -> Result<()> {
    use anyhow::Context as _;
    let db = cfg.db_path();
    if let Some(parent) = db.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let state = memorize_server::ServerState::new(db, cfg.token_budget)
        .context("init server state")?;
    let bind = format!("127.0.0.1:{}", cfg.port);
    memorize_server::serve(state, &bind)
}

fn run_recall(cfg: &Config, query: &str, limit: usize) -> Result<()> {
    let body = serde_json::json!({ "query": query, "limit": limit });
    let resp = client::post_json(cfg, "/recall", &body.to_string())?;
    let parsed: serde_json::Value = serde_json::from_str(&resp)?;
    println!("{}", serde_json::to_string_pretty(&parsed)?);
    Ok(())
}

fn run_remember(cfg: &Config, text: &str) -> Result<()> {
    let session = format!("manual-{}", chrono::Utc::now().timestamp());
    let body = serde_json::json!({
        "session": session,
        "kind": "manual",
        "body": text,
    });
    let resp = client::post_json(cfg, "/capture", &body.to_string())?;
    println!("{resp}");
    Ok(())
}

fn run_syn(cfg: &Config, op: SynOp) -> Result<()> {
    match op {
        SynOp::Add { term, expansion } => {
            let body = serde_json::json!({
                "op": "add",
                "term": term,
                "expansion": expansion,
            });
            println!("{}", client::post_json(cfg, "/syn", &body.to_string())?);
        }
        SynOp::Remove { term, expansion } => {
            let body = serde_json::json!({
                "op": "remove",
                "term": term,
                "expansion": expansion,
            });
            println!("{}", client::post_json(cfg, "/syn", &body.to_string())?);
        }
        SynOp::List => {
            let raw = client::get(cfg, "/syn")?;
            let pairs: Vec<(String, String)> = serde_json::from_str(&raw)?;
            for (term, expansion) in &pairs {
                println!("{term:<24} → {expansion}");
            }
            println!("({} rows)", pairs.len());
        }
    }
    Ok(())
}

fn run_status(cfg: &Config) -> Result<()> {
    let alive = client::get(cfg, "/health").is_ok();
    let db = cfg.db_path();
    let db_size = std::fs::metadata(&db).map(|m| m.len()).unwrap_or(0);
    println!("server:       {}", if alive { "running" } else { "down" });
    println!("db_path:      {}", db.display());
    println!("db_size:      {} bytes", db_size);
    println!("port:         {}", cfg.port);
    println!("token_budget: {}", cfg.token_budget);
    Ok(())
}
