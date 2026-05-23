# memorize — remaining phases

What's already shipped (Phases 1–3 + MCP):

- DuckDB-backed observation log with BM25 + MiniLM (384d) vector hybrid, RRF
  (k=60), session diversification (max 3/session), SQL-table-backed synonym
  expansion.
- 6 Claude Code hook stubs in `~/.claude/hooks/memorize-*.sh`,
  `~/.claude/settings.json` merged preserving prior entries.
- MCP stdio server at `memorize mcp` exposing `memory_recall` and
  `memory_save`, registered in `~/.claude.json`. Tool descriptions adopted
  verbatim from agentmemory.
- `~/Vibes/memorize/README.md` narrates the observation lifecycle.

This file covers what's left.

---

## Validation gate (before Phase 4)

Confirm the MCP tools self-trigger in real sessions. Open a fresh Claude
Code session in a repo with prior memorize captures and ask a question that
references prior work — "explain how the IR memo works", "what did we
decide about X", etc.

- If the model calls `memory_recall` on its own → MCP descriptions work as
  is, proceed to Phase 4.
- If it still defaults to Explore → the agentmemory-style descriptions
  aren't enough. Tune (more directive wording or keep the CLAUDE.md hint).
  Re-test before Phase 4.

This isn't a coding phase, just a checkpoint. Do not start Phase 4 until
the model proactively reaches for memory at least once per session on
prior-work prompts.

---

## Phase 4 — eval harness (LongMemEval parity)

Goal: reproducible quality measurement against the same dataset agentmemory
uses, with comparable outputs that let us tune knobs. Lives in
`crates/memorize-eval/`.

### Dataset

Download `longmemeval_s_cleaned.json` (~264MB, 500 questions, ~48 sessions
each) from `xiaowu0162/longmemeval-cleaned` on HuggingFace into
`crates/memorize-eval/data/`. Gate on file presence — skip if already
downloaded.

### Per-question harness

For each question:

1. Open a `:memory:` DuckDB via `memorize-store` (fresh ephemeral index).
2. Insert each of the question's ~48 session chunks as one obs row (raw
   text → `body`, embed → `emb`).
3. Run the question text through `memorize-recall::recall` exactly as
   production code does — no shortcuts.
4. Record: ranked list of obs ids, gold session match position, wall-clock
   latency.

### Metrics

Mirror agentmemory's `benchmark/lib/` calculations:

- `recall_any@K` for K ∈ {5, 10, 20} — does *any* gold session appear in top-K?
- NDCG@10
- MRR (first-relevant position)
- Precision@5
- All aggregated overall + by question category: `knowledge-update`,
  `multi-session`, `temporal-reasoning`, `single-session-user`,
  `single-session-assistant`, `single-session-preference`.

### Knobs (ablation surface)

| Flag | Default | Purpose |
|---|---|---|
| `--mode {bm25-only, vector-only, hybrid}` | `hybrid` | Isolate per-stream contribution |
| `--rrf-k <N>` | 60 | Vary RRF constant |
| `--top-k <N>` | 50 | Per-stream cutoff |
| `--diversify on|off` | `on` | Toggle session diversification |
| `--diversify-cap <N>` | 3 | Adjust per-session cap |
| `--synonyms on|off` | `on` | Measure synonym contribution |
| `--limit <N>` | 500 | Sample N questions for iteration speed |

### Outputs

Three artifacts per run, into `--out <dir>`:

- `report.md` — markdown table with overall + by-category rows. Configuration hash + git SHA in header so reports are attributable.
- `report.json` — per-question raw records (question_id, category, ranks of gold sessions, scores, latency). Slice externally without re-running.
- `summary.csv` — one line per run. Sticks into a tracking spreadsheet over time.

### Acceptance (calibration, not pass/fail)

| Mode | Target | Agentmemory's published |
|---|---|---|
| BM25-only | ≥ 85% R@5 | 86.2% |
| Hybrid | ≥ 94% R@5 | 95.2% |

If we land more than 2pp below either, something is wrong with tokenization
or synonyms — investigate before declaring Phase 4 complete.

### File layout

```
crates/memorize-eval/
├── Cargo.toml                  # depends on memorize-recall + memorize-store
├── data/.gitignore             # 264MB dataset not committed
└── src/
    ├── main.rs                 # produces `memorize-eval` binary
    ├── dataset.rs              # HuggingFace fetch + JSON parse
    ├── metrics.rs              # R@K, NDCG, MRR, Precision
    └── report.rs               # markdown/json/csv emitters
```

### Verification

`memorize-eval run --mode hybrid --limit 50 --out /tmp/r1/` produces
`report.md` with R@5 within 2pp of 95.2%. Re-run with `--mode bm25-only`
should drop to within 2pp of 86.2%. `--synonyms off` should produce a
measurable drop (1–4pp expected based on agentmemory's design).

---

## Phase 5 — optional LLM compression (opt-in)

Goal: enable raw-vs-summary A/B by populating compressed summaries for every
observation. Strictly additive to the working Phase-3 system. Default off.

### Schema migration

Idempotent on startup. Add columns and the second FTS index — none exist
pre-Phase-5:

```sql
ALTER TABLE obs ADD COLUMN IF NOT EXISTS compressed_at BIGINT;
ALTER TABLE obs ADD COLUMN IF NOT EXISTS summary VARCHAR;
ALTER TABLE obs ADD COLUMN IF NOT EXISTS title VARCHAR;
ALTER TABLE obs ADD COLUMN IF NOT EXISTS concepts VARCHAR[];
ALTER TABLE obs ADD COLUMN IF NOT EXISTS facts VARCHAR[];
ALTER TABLE vec ADD COLUMN IF NOT EXISTS emb_summary FLOAT[384];
PRAGMA create_fts_index('obs','id','summary', overwrite=1);
```

### LLM client (`crates/memorize-llm/`)

Single function: `compress_one(raw: &str) -> Result<Compressed>`. POST to
`https://api.anthropic.com/v1/messages` (configurable via
`ANTHROPIC_BASE_URL`) with:

- Headers: `x-api-key: $ANTHROPIC_API_KEY`, `anthropic-version: 2023-06-01`,
  `content-type: application/json`.
- Body:
  ```json
  {
    "model": "claude-haiku-4-5",
    "max_tokens": 1024,
    "system": [
      {"type": "text", "text": "<compression instructions>",
       "cache_control": {"type": "ephemeral"}}
    ],
    "tools": [{
      "name": "submit_compression",
      "description": "Submit the compressed observation",
      "input_schema": {
        "type": "object",
        "properties": {
          "title":    {"type": "string"},
          "summary":  {"type": "string"},
          "concepts": {"type": "array", "items": {"type": "string"}},
          "facts":    {"type": "array", "items": {"type": "string"}}
        },
        "required": ["title","summary","concepts","facts"]
      }
    }],
    "tool_choice": {"type": "tool", "name": "submit_compression"},
    "messages": [{"role": "user", "content": "<raw obs body>"}]
  }
  ```

Response parsing: find the `tool_use` block in `content[]`, deserialize
`input` into the `Compressed` struct. No fragile JSON-in-text parsing.

Prompt caching via `cache_control: ephemeral` on the system prompt — every
observation in a `memorize compress` batch hits the cache after the first
call, cutting cost ~10× on system-prompt tokens.

Plain `ureq` blocking. No streaming, no async. ~80 lines.

### CLI

```
memorize compress [--limit N]
```

- Checks `ANTHROPIC_API_KEY` is set; if not, prints `compression disabled`
  and exits 0.
- `SELECT id, body FROM obs WHERE compressed_at IS NULL ORDER BY ts ASC LIMIT N`
  (default N=100).
- For each row: call LLM, parse tool_use, UPDATE obs set
  summary/title/concepts/facts/compressed_at, embed summary into
  `vec.emb_summary`.
- Rebuild `summary` FTS index after batch.
- Progress: `compressed N/M (elapsed Xs)`.

Also exposed as `POST /compress?limit=N` for cron-style invocation.

### Recall mode switch

`/recall` gains optional `search_mode ∈ {"raw","summary"}` (default `"raw"`).

`summary` mode gated on 100% coverage:
`SELECT COUNT(*) FROM obs WHERE compressed_at IS NULL` must be 0. Otherwise
return HTTP 409 with the uncompressed count and instructions to run
`memorize compress`. When coverage is 100%, recall pipeline runs against
`summary`-column FTS + `emb_summary` vectors instead of `body` + `emb`.

### Status extension

`memorize status` shows compression coverage as `X/Y compressed`.

### Config additions

- `anthropic_api_key`
- `anthropic_model` (default `claude-haiku-4-5`)
- `anthropic_base_url` (optional, for local proxies)
- `compress_batch_size` (default 100)

### Crate layout

```
crates/memorize-llm/src/
├── lib.rs
├── client.rs          # Anthropic /v1/messages POST
├── compress.rs        # batch loop, schema validation, DB updates
└── prompt.rs          # compression system prompt
```

Plus extensions in:
- `crates/memorize-store/src/migrations.rs` — ALTER TABLE block
- `crates/memorize-cli/src/compress.rs` — CLI dispatch

### Verification

With `ANTHROPIC_API_KEY` set, `memorize compress --limit 10` populates ten
rows; `memorize status` shows `10/N compressed`. Once `N/N compressed`,
`recall --search-mode summary` returns results. Phase-4 eval harness re-runs
against summary indexing for a directly comparable R@5/R@10. Prompt-cache
hits visible in `usage.cache_read_input_tokens` on the API response.

---

## Phase 6 — deferred / revisit gates

Only revisit if dogfooding turns up the missing capability:

### JSONL replay

Seed memorize from existing `~/.claude/projects/*/conversations/*.jsonl`
history. One-time bulk import; useful when starting on a new machine or
recovering after a DB reset.

```
memorize replay [--from <dir>] [--limit N] [--dry-run]
```

Parses Claude Code's JSONL transcript format, materializes each turn as
synthetic obs rows (user_prompt, tool_use, etc.) into the existing tables.
Reuses the production privacy filter and dedup.

### HNSW vector index

Trigger: corpus passes ~50k observations and brute-force cosine starts
showing in recall latency (>100ms p95).

DuckDB VSS extension is the path of least resistance:

```sql
INSTALL vss; LOAD vss;
CREATE INDEX vec_hnsw ON vec USING HNSW (emb) WITH (metric = 'cosine');
```

Recall pipeline uses the index automatically once present; no API change.
Adds a per-startup `INSTALL vss; LOAD vss;` (downloads ~10MB on first run).

### Other items intentionally not planned

- LLM provider abstraction (we shipped Anthropic-only by design)
- Knowledge graph extraction (benchmark shows it regresses on internal data)
- 4-tier consolidation (brain-analogy theater; the table is the tier)
- Viewer UI (DuckDB CLI is sufficient for one user)
- Mesh sync / team memory (single user)
- Snapshots / audit / governance (DuckDB file is the artifact; `cp` is the backup story)
- Embedding providers beyond local MiniLM (the model is fine at this scale)
- iii runtime (replaced by tiny_http + std::sync::Mutex, intentionally)
