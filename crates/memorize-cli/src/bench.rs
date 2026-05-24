//! `memorize bench code-index` — exercise the cold-scan pipeline against an
//! arbitrary root and report per-phase timing.
//!
//! Runs entirely in-process against an in-memory Store; the user's real
//! `~/.memorize/db.duckdb` is never touched. Wrap with `/usr/bin/time -l`
//! for peak RSS, or `samply record` for a CPU flamegraph.

use anyhow::{Context, Result, bail};
use memorize_code::{CodeChunk, language_for_path};
use memorize_store::{CodeChunkRow, FileMeta, Store};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const TOP_SLOW_N: usize = 10;

#[derive(Debug, Default)]
struct PhaseTotals {
    walk_dir_ns: u128,
    stat_ns: u128,
    read_ns: u128,
    parse_ns: u128,
    embed_ns: u128,
    insert_ns: u128,
}

/// Per-call embed-batch instrumentation. One sample per `embed_batch` invocation.
#[derive(Debug, Default)]
struct EmbedBatchStats {
    samples: Vec<(usize, u128)>, // (batch_size, ns)
}

impl EmbedBatchStats {
    fn record(&mut self, batch_size: usize, ns: u128) {
        self.samples.push((batch_size, ns));
    }
    fn total_chunks(&self) -> usize {
        self.samples.iter().map(|(b, _)| *b).sum()
    }
    fn total_ns(&self) -> u128 {
        self.samples.iter().map(|(_, n)| *n).sum()
    }
}

impl PhaseTotals {
    fn sum_ns(&self) -> u128 {
        self.walk_dir_ns
            + self.stat_ns
            + self.read_ns
            + self.parse_ns
            + self.embed_ns
            + self.insert_ns
    }
}

#[derive(Debug)]
struct FileRecord {
    path: PathBuf,
    bytes: u64,
    chunks: usize,
    total_ms: u128,
}

#[derive(Debug, Default)]
struct ErrorRecord {
    phase: &'static str,
    path: PathBuf,
    message: String,
}

#[derive(Debug)]
pub struct BenchOpts {
    pub root: PathBuf,
    pub limit: Option<usize>,
    pub out: Option<PathBuf>,
    /// If true, embed-init time is reported separately from per-file time.
    /// Default true — almost always what you want.
    pub separate_embed_init: bool,
}

pub fn code_index(opts: BenchOpts) -> Result<()> {
    if !opts.root.exists() {
        bail!("root does not exist: {}", opts.root.display());
    }

    eprintln!("=== memorize bench: {} ===", opts.root.display());

    // Pre-warm the embedder so its init cost is separated from per-file time.
    let embed_init = Instant::now();
    if opts.separate_embed_init {
        // Tiny no-op embed to force model load.
        let _ = memorize_embed::embed("warm-up").context("warm up embedder")?;
    }
    let embed_init_elapsed = embed_init.elapsed();

    // Fresh in-memory store so we don't pollute the user's DB.
    let store = Store::open_in_memory().context("open in-memory store")?;

    let mut totals = PhaseTotals::default();
    let mut embed_stats = EmbedBatchStats::default();
    let mut files: Vec<FileRecord> = Vec::new();
    let mut errors: Vec<ErrorRecord> = Vec::new();
    let mut visited = 0u64;
    let skipped_excluded = 0u64;
    let mut skipped_too_big = 0u64;
    let mut skipped_wrong_lang = 0u64;
    let mut bytes_processed = 0u64;
    let mut chunks_emitted: u64 = 0;
    let max_file_bytes: u64 = 1_048_576;

    let started = Instant::now();
    let walk_started = Instant::now();
    let walker = ignore::WalkBuilder::new(&opts.root)
        .standard_filters(true)
        .follow_links(false)
        .build();

    let mut last_progress = Instant::now();
    let progress_interval = Duration::from_secs(5);

    for entry in walker {
        if let Some(limit) = opts.limit {
            if files.len() >= limit {
                break;
            }
        }
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                errors.push(ErrorRecord {
                    phase: "walk",
                    path: opts.root.clone(),
                    message: e.to_string(),
                });
                continue;
            }
        };
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.into_path();
        visited += 1;

        // Stat.
        let stat_t = Instant::now();
        let meta_fs = match fs::metadata(&path) {
            Ok(m) => m,
            Err(e) => {
                errors.push(ErrorRecord {
                    phase: "stat",
                    path: path.clone(),
                    message: e.to_string(),
                });
                continue;
            }
        };
        totals.stat_ns += stat_t.elapsed().as_nanos();

        // Filter: too big.
        if meta_fs.len() > max_file_bytes {
            skipped_too_big += 1;
            continue;
        }
        // Filter: language we don't index.
        let language = match language_for_path(&path) {
            Some(l) => l,
            None => {
                skipped_wrong_lang += 1;
                continue;
            }
        };

        let file_started = Instant::now();

        // Read.
        let read_t = Instant::now();
        let source = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                errors.push(ErrorRecord {
                    phase: "read",
                    path: path.clone(),
                    message: e.to_string(),
                });
                continue;
            }
        };
        totals.read_ns += read_t.elapsed().as_nanos();
        bytes_processed += meta_fs.len();

        // Parse + chunk.
        let parse_t = Instant::now();
        let chunks: Vec<CodeChunk> = match memorize_code::chunk_source(&source, language) {
            Ok(c) => c,
            Err(e) => {
                errors.push(ErrorRecord {
                    phase: "parse",
                    path: path.clone(),
                    message: e.to_string(),
                });
                continue;
            }
        };
        let chunks = match enforce_token_cap(chunks) {
            Ok(c) => c,
            Err(e) => {
                errors.push(ErrorRecord {
                    phase: "token_cap",
                    path: path.clone(),
                    message: e.to_string(),
                });
                continue;
            }
        };
        totals.parse_ns += parse_t.elapsed().as_nanos();
        if chunks.is_empty() {
            // Some files yield no chunkable nodes — count visit but don't bench downstream.
            continue;
        }

        // Embed (batched).
        let embed_t = Instant::now();
        let bodies: Vec<&str> = chunks.iter().map(|c| c.body.as_str()).collect();
        let embs = match memorize_embed::embed_batch(&bodies) {
            Ok(v) => v,
            Err(e) => {
                errors.push(ErrorRecord {
                    phase: "embed",
                    path: path.clone(),
                    message: e.to_string(),
                });
                continue;
            }
        };
        let embed_elapsed = embed_t.elapsed().as_nanos();
        totals.embed_ns += embed_elapsed;
        embed_stats.record(bodies.len(), embed_elapsed);

        // Insert.
        let insert_t = Instant::now();
        let rows: Vec<CodeChunkRow> = chunks
            .into_iter()
            .map(|c| CodeChunkRow {
                id: 0,
                path: path.to_string_lossy().into_owned(),
                language: c.language,
                line_start: c.line_start as i32,
                line_end: c.line_end as i32,
                kind: c.kind,
                qualified: c.qualified,
                body: c.body,
            })
            .collect();
        let meta = FileMeta {
            mtime_ns: meta_fs
                .modified()
                .ok()
                .and_then(|t| {
                    t.duration_since(std::time::UNIX_EPOCH)
                        .ok()
                        .map(|d| d.as_nanos() as i64)
                })
                .unwrap_or(0),
            size_bytes: meta_fs.len() as i64,
            git_rev: None,
        };
        if let Err(e) = store.upsert_code_file(
            &path.to_string_lossy(),
            &opts.root.to_string_lossy(),
            language,
            &meta,
            &rows,
            &embs,
        ) {
            errors.push(ErrorRecord {
                phase: "insert",
                path: path.clone(),
                message: e.to_string(),
            });
            continue;
        }
        totals.insert_ns += insert_t.elapsed().as_nanos();

        chunks_emitted += rows.len() as u64;
        files.push(FileRecord {
            path,
            bytes: meta_fs.len(),
            chunks: rows.len(),
            total_ms: file_started.elapsed().as_millis(),
        });

        // Progress every ~5s so long runs don't look frozen.
        if last_progress.elapsed() > progress_interval {
            let n = files.len();
            let elapsed_s = started.elapsed().as_secs_f64();
            let rate = n as f64 / elapsed_s;
            eprintln!(
                "  {n} files indexed ({:.1} files/s, {:.1} MB processed, elapsed {:.1}s)",
                rate,
                bytes_processed as f64 / 1_048_576.0,
                elapsed_s
            );
            last_progress = Instant::now();
        }
    }
    let _ = skipped_excluded; // `ignore` crate handles the exclude logic for us
    let total_elapsed = started.elapsed();
    // walk_dir is what's left after accounting for measured per-file phases —
    // i.e. time the indexer was waiting on the walker to yield + filtering
    // entries that never reach a phase.
    let other_phases_ns =
        totals.stat_ns + totals.read_ns + totals.parse_ns + totals.embed_ns + totals.insert_ns;
    totals.walk_dir_ns = walk_started
        .elapsed()
        .as_nanos()
        .saturating_sub(other_phases_ns);

    // Render the report. Goes to stderr + optional --out file.
    let report = render_report(
        &opts,
        embed_init_elapsed,
        total_elapsed,
        visited,
        skipped_too_big,
        skipped_wrong_lang,
        bytes_processed,
        chunks_emitted,
        &totals,
        &embed_stats,
        &files,
        &errors,
    );
    eprintln!("\n{report}");
    if let Some(out_path) = &opts.out {
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).ok();
        }
        let mut f = fs::File::create(out_path)
            .with_context(|| format!("create {}", out_path.display()))?;
        f.write_all(report.as_bytes())?;
        eprintln!("report written to {}", out_path.display());
    }
    Ok(())
}

/// Re-split each chunk so it fits the embedder's max-token window. Splits at
/// line boundaries when possible. Cheap when chunks already fit (one tokenize
/// per chunk via `count_tokens`).
fn enforce_token_cap(chunks: Vec<CodeChunk>) -> Result<Vec<CodeChunk>> {
    let cap = memorize_embed::max_seq_tokens();
    // Leave headroom for whatever special tokens the model prepends/appends
    // (CLS/SEP for BERT-family is 2). The buffer also covers slight drift
    // when joining lines with '\n' that the tokenizer might re-segment.
    let cap = cap.saturating_sub(8).max(1);

    let mut out: Vec<CodeChunk> = Vec::with_capacity(chunks.len());
    for c in chunks {
        let pieces = memorize_embed::split_to_token_cap(&c.body, cap)
            .with_context(|| format!("split chunk (kind={})", c.kind))?;
        if pieces.len() == 1 {
            out.push(c);
            continue;
        }
        let total = pieces.len();
        for (i, body) in pieces.into_iter().enumerate() {
            let qualified = if c.qualified.is_empty() {
                format!("part-{}/{total}", i + 1)
            } else {
                format!("{}#part-{}/{total}", c.qualified, i + 1)
            };
            out.push(CodeChunk {
                language: c.language.clone(),
                line_start: c.line_start,
                line_end: c.line_end,
                kind: c.kind.clone(),
                qualified,
                body,
            });
        }
    }
    Ok(out)
}

#[derive(Debug)]
pub struct TokensOpts {
    pub root: PathBuf,
    pub limit: Option<usize>,
    pub max_tokens: usize,
    pub out: Option<PathBuf>,
}

/// Walk a root, run the chunker, tokenize every chunk against the embedder's
/// own tokenizer, and report the token-count distribution + overflow rate.
/// This is what tells you whether `TARGET_CHARS` is set correctly for the
/// embedder you're using.
pub fn tokens(opts: TokensOpts) -> Result<()> {
    if !opts.root.exists() {
        bail!("root does not exist: {}", opts.root.display());
    }

    eprintln!("=== memorize bench tokens: {} ===", opts.root.display());
    eprintln!("model: {}", memorize_embed::model_tag());
    eprintln!("max-tokens cap: {}", opts.max_tokens);

    let max_file_bytes: u64 = 1_048_576;
    let walker = ignore::WalkBuilder::new(&opts.root)
        .standard_filters(true)
        .follow_links(false)
        .build();

    let mut token_counts: Vec<usize> = Vec::new();
    let mut overflow_files: Vec<(PathBuf, usize, usize)> = Vec::new(); // path, chunk_chars, tokens
    let mut overflow_chunks = 0u64;
    let mut files_processed = 0u64;
    let mut total_chunks = 0u64;
    let started = Instant::now();

    for entry in walker {
        if let Some(limit) = opts.limit {
            if files_processed as usize >= limit {
                break;
            }
        }
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.into_path();
        let meta_fs = match fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta_fs.len() > max_file_bytes {
            continue;
        }
        let language = match language_for_path(&path) {
            Some(l) => l,
            None => continue,
        };
        let source = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let chunks = match memorize_code::chunk_source(&source, language) {
            Ok(c) => c,
            Err(_) => continue,
        };
        // Apply the same token-cap re-split the indexer uses, so the bench
        // reflects the actual chunks that hit the embedder.
        let chunks = match enforce_token_cap(chunks) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if chunks.is_empty() {
            continue;
        }
        files_processed += 1;

        for c in &chunks {
            let n = match memorize_embed::count_tokens(&c.body) {
                Ok(n) => n,
                Err(_) => continue,
            };
            token_counts.push(n);
            total_chunks += 1;
            if n > opts.max_tokens {
                overflow_chunks += 1;
                overflow_files.push((path.clone(), c.body.len(), n));
            }
        }
    }

    if token_counts.is_empty() {
        bail!("no chunks produced — nothing to measure");
    }
    token_counts.sort_unstable();
    let elapsed = started.elapsed();

    let p = |q: f64| -> usize {
        let idx = ((token_counts.len() as f64 - 1.0) * q).round() as usize;
        token_counts[idx.min(token_counts.len() - 1)]
    };
    let total: usize = token_counts.iter().sum();
    let mean = total as f64 / token_counts.len() as f64;

    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "# memorize bench — chunk token distribution");
    let _ = writeln!(s);
    let _ = writeln!(s, "**root:** `{}`", opts.root.display());
    let _ = writeln!(s, "**model:** {}", memorize_embed::model_tag());
    let _ = writeln!(s, "**files processed:** {files_processed}");
    let _ = writeln!(s, "**chunks tokenized:** {total_chunks}");
    let _ = writeln!(s, "**elapsed:** {:.2}s", elapsed.as_secs_f64());
    let _ = writeln!(s);
    let _ = writeln!(s, "## Tokens per chunk");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "| mean | p50 | p75 | p90 | p95 | p99 | p99.9 | max |\n|---|---|---|---|---|---|---|---|\n| {:.0} | {} | {} | {} | {} | {} | {} | {} |",
        mean,
        p(0.50),
        p(0.75),
        p(0.90),
        p(0.95),
        p(0.99),
        p(0.999),
        token_counts.last().unwrap()
    );
    let _ = writeln!(s);

    // Overflow rate vs cap.
    let overflow_pct = (overflow_chunks as f64 / total_chunks as f64) * 100.0;
    let _ = writeln!(s, "## Overflow vs cap = {}", opts.max_tokens);
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "**chunks that would be truncated:** {overflow_chunks} / {total_chunks} ({overflow_pct:.2}%)"
    );
    let _ = writeln!(s);

    // Implied chars/token ratio at each percentile — useful for picking TARGET_CHARS.
    // We need char-counts in matching positions; cheapest path is a parallel sort by
    // index, but for a directional read we just report the mean ratio over all chunks.
    // Total chars / total tokens gives the population-mean — informative enough.
    let _ = writeln!(s, "## Calibration");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "Cap = {} tokens. To keep p99 under the cap with the current chunker, TARGET_CHARS should be ≈ p99_tokens × mean_chars_per_token of current corpus. Sample p99 = {} tokens.",
        opts.max_tokens,
        p(0.99)
    );
    let _ = writeln!(s);

    // Top overflow examples.
    overflow_files.sort_by(|a, b| b.2.cmp(&a.2));
    if !overflow_files.is_empty() {
        let _ = writeln!(s, "## Top overflow chunks");
        let _ = writeln!(s);
        let _ = writeln!(s, "| tokens | chunk_chars | path |");
        let _ = writeln!(s, "|---|---|---|");
        for (p, c, t) in overflow_files.iter().take(15) {
            let _ = writeln!(s, "| {t} | {c} | `{}` |", p.display());
        }
        let _ = writeln!(s);
    }

    eprintln!("\n{s}");
    if let Some(out) = &opts.out {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent).ok();
        }
        let mut f = fs::File::create(out)
            .with_context(|| format!("create {}", out.display()))?;
        f.write_all(s.as_bytes())?;
        eprintln!("report written to {}", out.display());
    }
    Ok(())
}

#[derive(Debug)]
pub struct EmbedSweepOpts {
    pub batch_sizes: Vec<usize>,
    pub chunks: usize,
    pub chunk_chars: usize,
    pub out: Option<PathBuf>,
}

/// Synthetic batch-size sweep. Builds a fixed pool of look-alike code chunks
/// and calls `embed_batch` at each requested batch size, reporting wall time
/// and chunks/s. Isolates the embedder from walker/parser/store.
pub fn embed_sweep(opts: EmbedSweepOpts) -> Result<()> {
    if opts.batch_sizes.is_empty() {
        bail!("--sizes must contain at least one batch size");
    }
    if opts.batch_sizes.iter().any(|n| *n == 0) {
        bail!("--sizes must not contain 0");
    }

    eprintln!(
        "=== memorize bench embed: {} sizes, {} chunks, {} chars/chunk ===",
        opts.batch_sizes.len(),
        opts.chunks,
        opts.chunk_chars,
    );

    // Warm the embedder so model-load time isn't billed to the first batch.
    let warm_t = Instant::now();
    let _ = memorize_embed::embed("warm-up").context("warm up embedder")?;
    eprintln!("warm-up: {:.2}s (model: {})", warm_t.elapsed().as_secs_f64(), memorize_embed::model_tag());

    // Build a deterministic pool of synthetic chunks. Each chunk is a unique
    // string so the tokenizer / embedder can't trivially cache results.
    let pool = build_synth_pool(opts.chunks, opts.chunk_chars);

    struct SweepRow {
        batch: usize,
        calls: usize,
        chunks: usize,
        wall_ms: u128,
        per_chunk_us: f64,
        chunks_per_sec: f64,
    }

    let mut rows: Vec<SweepRow> = Vec::new();
    for &batch in &opts.batch_sizes {
        // Round chunks up to the next multiple of batch so every call uses the
        // requested batch size exactly — keeps the throughput number clean.
        let calls = opts.chunks.div_ceil(batch);
        let total = calls * batch;
        let mut idx = 0;
        let t = Instant::now();
        for _ in 0..calls {
            let mut bodies: Vec<&str> = Vec::with_capacity(batch);
            for _ in 0..batch {
                bodies.push(&pool[idx % pool.len()]);
                idx += 1;
            }
            let _ = memorize_embed::embed_batch(&bodies).context("embed_batch")?;
        }
        let wall = t.elapsed();
        let wall_ms = wall.as_millis();
        let per_chunk_us = (wall.as_nanos() as f64 / total as f64) / 1_000.0;
        let chunks_per_sec = total as f64 / wall.as_secs_f64();
        eprintln!(
            "  batch={batch:<4} calls={calls:<5} chunks={total:<6} wall={wall_ms:>5}ms  per_chunk={per_chunk_us:>7.1}µs  {chunks_per_sec:>8.0} chunks/s"
        );
        rows.push(SweepRow {
            batch,
            calls,
            chunks: total,
            wall_ms,
            per_chunk_us,
            chunks_per_sec,
        });
    }

    // Render report.
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "# memorize bench — embed batch sweep");
    let _ = writeln!(s);
    let _ = writeln!(s, "**model:** {}", memorize_embed::model_tag());
    let _ = writeln!(s, "**chunk_chars:** {}", opts.chunk_chars);
    let _ = writeln!(s, "**target chunks per size:** {}", opts.chunks);
    let _ = writeln!(s);
    let _ = writeln!(s, "| batch | calls | chunks | wall ms | µs/chunk | chunks/s |");
    let _ = writeln!(s, "|---|---|---|---|---|---|");
    for r in &rows {
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {:.1} | {:.0} |",
            r.batch, r.calls, r.chunks, r.wall_ms, r.per_chunk_us, r.chunks_per_sec
        );
    }
    let _ = writeln!(s);

    // Speedup vs batch=1 (or whichever the smallest batch is).
    let baseline = rows.first().map(|r| r.per_chunk_us).unwrap_or(0.0);
    if baseline > 0.0 {
        let _ = writeln!(s, "**speedup vs batch={}:**", rows[0].batch);
        let _ = writeln!(s);
        let _ = writeln!(s, "| batch | speedup |");
        let _ = writeln!(s, "|---|---|");
        for r in &rows {
            let speedup = baseline / r.per_chunk_us;
            let _ = writeln!(s, "| {} | {:.2}× |", r.batch, speedup);
        }
        let _ = writeln!(s);
    }

    eprintln!("\n{s}");
    if let Some(out) = &opts.out {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent).ok();
        }
        let mut f = fs::File::create(out)
            .with_context(|| format!("create {}", out.display()))?;
        f.write_all(s.as_bytes())?;
        eprintln!("report written to {}", out.display());
    }
    Ok(())
}

/// Build a pool of synthetic, distinct chunks roughly the size of a real
/// source-code chunk. The bodies look enough like code that the tokenizer
/// produces a realistic token count distribution.
fn build_synth_pool(count: usize, chars: usize) -> Vec<String> {
    // Templates rotate so adjacent chunks aren't identical; the index suffix
    // forces every chunk to be unique even when the template repeats.
    const TEMPLATES: &[&str] = &[
        "fn process_event(event: &Event, state: &mut State) -> Result<Outcome> {\n    let key = event.key();\n    let entry = state.entries.entry(key).or_default();\n    entry.count += 1;\n    entry.last_seen = event.ts;\n    Ok(Outcome::Accepted)\n}\n",
        "class WorkbookController {\n    constructor(api, store) {\n        this.api = api;\n        this.store = store;\n        this.subscriptions = new Map();\n    }\n    async load(id) {\n        const wb = await this.api.fetchWorkbook(id);\n        this.store.upsert(wb);\n        return wb;\n    }\n}\n",
        "def hybrid_score(bm25_rank, vec_rank, k=60):\n    return 1.0 / (k + bm25_rank) + 1.0 / (k + vec_rank)\n\n\ndef diversify(results, cap_per_session=3):\n    seen = collections.Counter()\n    out = []\n    for r in results:\n        if seen[r.session] >= cap_per_session:\n            continue\n        seen[r.session] += 1\n        out.append(r)\n    return out\n",
    ];
    let mut pool = Vec::with_capacity(count);
    for i in 0..count {
        let tpl = TEMPLATES[i % TEMPLATES.len()];
        let mut body = String::with_capacity(chars + 32);
        body.push_str(&format!("// chunk #{i}\n"));
        while body.len() < chars {
            body.push_str(tpl);
        }
        body.truncate(chars);
        pool.push(body);
    }
    pool
}

fn render_report(
    opts: &BenchOpts,
    embed_init: Duration,
    total: Duration,
    visited: u64,
    skipped_too_big: u64,
    skipped_wrong_lang: u64,
    bytes: u64,
    chunks: u64,
    totals: &PhaseTotals,
    embed_stats: &EmbedBatchStats,
    files: &[FileRecord],
    errors: &[ErrorRecord],
) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "# memorize bench — code-index");
    let _ = writeln!(s);
    let _ = writeln!(s, "**root:** `{}`", opts.root.display());
    let _ = writeln!(s, "**files indexed:** {}", files.len());
    let _ = writeln!(s, "**files visited:** {visited}");
    let _ = writeln!(
        s,
        "**files skipped:** {} (too big), {} (wrong language)",
        skipped_too_big, skipped_wrong_lang
    );
    let _ = writeln!(s, "**bytes processed:** {} ({:.1} MB)", bytes, bytes as f64 / 1_048_576.0);
    let _ = writeln!(s, "**chunks emitted:** {chunks}");
    if files.is_empty() {
        let _ = writeln!(s, "**chunks/file mean:** —");
    } else {
        let _ = writeln!(
            s,
            "**chunks/file mean:** {:.1}",
            chunks as f64 / files.len() as f64
        );
    }
    let _ = writeln!(s, "**elapsed (per-file):** {:.2}s", total.as_secs_f64());
    let _ = writeln!(s, "**embed-model init:** {:.2}s (one-time, excluded from per-file)", embed_init.as_secs_f64());
    let _ = writeln!(s);

    // Phase share table.
    let _ = writeln!(s, "## Phase totals (wall ms across all files)");
    let _ = writeln!(s);
    let _ = writeln!(s, "| phase | ms | share |");
    let _ = writeln!(s, "|---|---|---|");
    let denom = totals.sum_ns().max(1) as f64;
    let row = |label: &str, ns: u128| {
        let ms = ns / 1_000_000;
        let share = (ns as f64 / denom) * 100.0;
        format!("| `{label}` | {ms} | {share:.1}% |")
    };
    let _ = writeln!(s, "{}", row("walk_dir", totals.walk_dir_ns));
    let _ = writeln!(s, "{}", row("stat", totals.stat_ns));
    let _ = writeln!(s, "{}", row("read", totals.read_ns));
    let _ = writeln!(s, "{}", row("tree_sitter", totals.parse_ns));
    let _ = writeln!(s, "{}", row("embed", totals.embed_ns));
    let _ = writeln!(s, "{}", row("insert", totals.insert_ns));
    let _ = writeln!(s);

    // Embed-batch utilization. A high call count with a small mean batch size
    // is the smoking gun for cross-file batching being worthwhile.
    if !embed_stats.samples.is_empty() {
        let calls = embed_stats.samples.len();
        let total_chunks = embed_stats.total_chunks();
        let total_ns = embed_stats.total_ns();
        let mean_batch = total_chunks as f64 / calls as f64;
        let ns_per_call = total_ns as f64 / calls as f64;
        let ns_per_chunk = total_ns as f64 / total_chunks.max(1) as f64;

        let mut sizes: Vec<usize> = embed_stats.samples.iter().map(|(b, _)| *b).collect();
        sizes.sort_unstable();
        let p = |q: f64| {
            let idx = ((sizes.len() as f64 - 1.0) * q).round() as usize;
            sizes[idx.min(sizes.len() - 1)]
        };

        let _ = writeln!(s, "## Embed batch utilization");
        let _ = writeln!(s);
        let _ = writeln!(s, "**embed_batch calls:** {calls}");
        let _ = writeln!(s, "**chunks embedded:** {total_chunks}");
        let _ = writeln!(s, "**mean batch size:** {mean_batch:.1}");
        let _ = writeln!(s, "**per-call:** {:.2} ms", ns_per_call / 1_000_000.0);
        let _ = writeln!(s, "**per-chunk:** {:.2} ms", ns_per_chunk / 1_000_000.0);
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "| batch p50 | p90 | p99 | max |\n|---|---|---|---|\n| {} | {} | {} | {} |",
            p(0.50),
            p(0.90),
            p(0.99),
            sizes.last().unwrap()
        );
        let _ = writeln!(s);

        // Histogram by power-of-two buckets, plus the share of *chunks* (not calls)
        // that landed in each bucket — what actually matters for batching upside.
        let buckets: &[(usize, usize, &str)] = &[
            (1, 1, "1"),
            (2, 4, "2-4"),
            (5, 8, "5-8"),
            (9, 16, "9-16"),
            (17, 32, "17-32"),
            (33, 64, "33-64"),
            (65, 128, "65-128"),
            (129, usize::MAX, "129+"),
        ];
        let _ = writeln!(s, "| batch | calls | call share | chunks | chunk share |");
        let _ = writeln!(s, "|---|---|---|---|---|");
        for (lo, hi, label) in buckets {
            let calls_in_bucket = embed_stats
                .samples
                .iter()
                .filter(|(b, _)| *b >= *lo && *b <= *hi)
                .count();
            let chunks_in_bucket: usize = embed_stats
                .samples
                .iter()
                .filter(|(b, _)| *b >= *lo && *b <= *hi)
                .map(|(b, _)| *b)
                .sum();
            let call_share = (calls_in_bucket as f64 / calls as f64) * 100.0;
            let chunk_share = (chunks_in_bucket as f64 / total_chunks.max(1) as f64) * 100.0;
            let _ = writeln!(
                s,
                "| {label} | {calls_in_bucket} | {call_share:.1}% | {chunks_in_bucket} | {chunk_share:.1}% |"
            );
        }
        let _ = writeln!(s);
    }

    // Per-file latency percentiles.
    if !files.is_empty() {
        let mut latencies: Vec<u128> = files.iter().map(|f| f.total_ms).collect();
        latencies.sort_unstable();
        let p = |q: f64| {
            let idx = ((latencies.len() as f64 - 1.0) * q).round() as usize;
            latencies[idx.min(latencies.len() - 1)]
        };
        let _ = writeln!(s, "## Per-file latency (ms, including all phases)");
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "| p50 | p90 | p95 | p99 | max |\n|---|---|---|---|---|\n| {} | {} | {} | {} | {} |",
            p(0.50), p(0.90), p(0.95), p(0.99), latencies.last().unwrap()
        );
        let _ = writeln!(s);

        // Top slow files.
        let mut by_slow: Vec<&FileRecord> = files.iter().collect();
        by_slow.sort_by(|a, b| b.total_ms.cmp(&a.total_ms));
        let _ = writeln!(s, "## Top {} slowest files", TOP_SLOW_N.min(by_slow.len()));
        let _ = writeln!(s);
        let _ = writeln!(s, "| ms | bytes | chunks | path |");
        let _ = writeln!(s, "|---|---|---|---|");
        for fr in by_slow.iter().take(TOP_SLOW_N) {
            let _ = writeln!(
                s,
                "| {} | {} | {} | `{}` |",
                fr.total_ms,
                fr.bytes,
                fr.chunks,
                fr.path.display()
            );
        }
        let _ = writeln!(s);

        // Chunks-per-file distribution.
        let mut chunks_per_file: Vec<usize> = files.iter().map(|f| f.chunks).collect();
        chunks_per_file.sort_unstable();
        let p = |q: f64| {
            let idx = ((chunks_per_file.len() as f64 - 1.0) * q).round() as usize;
            chunks_per_file[idx.min(chunks_per_file.len() - 1)]
        };
        let _ = writeln!(s, "## Chunks-per-file distribution");
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "| p50 | p90 | p99 | max |\n|---|---|---|---|\n| {} | {} | {} | {} |",
            p(0.50),
            p(0.90),
            p(0.99),
            chunks_per_file.last().unwrap()
        );
        let _ = writeln!(s);
    }

    // Language breakdown.
    let mut by_lang: HashMap<&'static str, (usize, u64)> = HashMap::new();
    for f in files {
        if let Some(lang) = language_for_path(&f.path) {
            let e = by_lang.entry(lang).or_default();
            e.0 += 1;
            e.1 += f.chunks as u64;
        }
    }
    if !by_lang.is_empty() {
        let mut langs: Vec<_> = by_lang.into_iter().collect();
        langs.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        let _ = writeln!(s, "## Indexed by language");
        let _ = writeln!(s);
        let _ = writeln!(s, "| language | files | chunks |");
        let _ = writeln!(s, "|---|---|---|");
        for (lang, (n, ch)) in langs {
            let _ = writeln!(s, "| `{lang}` | {n} | {ch} |");
        }
        let _ = writeln!(s);
    }

    if !errors.is_empty() {
        let _ = writeln!(s, "## Errors ({})", errors.len());
        let _ = writeln!(s);
        let take = errors.len().min(20);
        for e in errors.iter().take(take) {
            let _ = writeln!(s, "- `{}` {}: {}", e.phase, e.path.display(), e.message);
        }
        if errors.len() > take {
            let _ = writeln!(s, "- … and {} more", errors.len() - take);
        }
    }

    s
}
