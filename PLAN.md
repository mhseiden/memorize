# memorize — session memory + code index

Two MCP tools, one process, one DuckDB.

- `session_recall(query, limit?)` — searches the user/assistant conversation
  corpus. Hybrid BM25 + per-chunk vector with RRF fusion. Returns prose
  observations (prompts, assistant messages, subagent results) plus compact
  references to files/lines touched during prior sessions.
- `code_recall(query, limit?, language?, path_prefix?)` — searches the
  local code index. AST-chunked via tree-sitter (cAST-style), hybrid BM25 +
  vector over function/class/block-level chunks, returns
  `{path, line_start, line_end, qualified, body}` per hit.

The agent decides which tool to reach for based on the query — we don't merge
them into one unified call. The cost asymmetry (session = cheap text, code
= structured local index that updates on file save) is something the agent
can reason about.

---

## What's already shipped

- Phase 1–3 of the original plan: hooks, MCP, hybrid retrieval (RRF k=60,
  session diversification, SQL-table synonyms), CPU-only embedding via
  `fastembed` + MiniLM-L6-v2 (384-d, 512-token context). On LongMemEval-S
  this scores R@5 = 0.964 with hybrid mode, beating agentmemory's published
  0.952.
- Embedding cache for eval iteration speed.
- CoreML investigation concluded: documented architectural mismatch for
  small encoders + dynamic batches; CPU is the right path. Wiring left
  behind `MEMORIZE_EMBED_COREML=1` env var for future tinkering on bigger
  models.

## What's changing

### 1. Session capture refactor (Phase A)

Today every `PostToolUse` hook stores a 4KB-truncated dump of tool I/O.
Most of that is garbage (file contents we already have on disk, bash
output the model could re-run). The corpus is noisier than it needs to be
and that drags down both BM25 signal-to-noise and the value of recall.

New shape:

| capture | stored as |
|---|---|
| `UserPromptSubmit` | full prose, `kind='user_prompt'` |
| `Stop` (extracted `last_assistant_message`) | full prose, `kind='assistant_message'` |
| `SubagentStop` | full prose, `kind='subagent_message'` |
| `PostToolUse` for `Read` | `Read(<path>:<lines>)` — path + line range only |
| `PostToolUse` for `Edit` | `Edit(<path>:<lines> → <intent>)` — path + range + first 80 chars from surrounding prompt as intent |
| `PostToolUse` for `Write` | `Write(<path>, <N> lines, created\|overwritten)` |
| `PostToolUse` for `Bash` | `Bash($ <cmd>, exit=<n>, <t>s)` — command + exit code + duration only |
| `PostToolUse` for `Grep`/`Glob` | skipped |
| `PostToolUse` for `WebFetch` | `WebFetch(<url>) → <status>` |
| `Task` (subagent spawn) | `Task(<type>, "<first 80 chars of prompt>")` |

**No tool output content is stored.** The session index becomes a record of
intent (prose) and pointers (refs). The agent dereferences refs via
`code_recall` or its own `Read` tool when it actually needs to see code.

Schema extension on `obs`:

```sql
ALTER TABLE obs ADD COLUMN IF NOT EXISTS ref_path VARCHAR;
ALTER TABLE obs ADD COLUMN IF NOT EXISTS ref_line_start INTEGER;
ALTER TABLE obs ADD COLUMN IF NOT EXISTS ref_line_end INTEGER;
```

Populated for tool-use obs, NULL for prose obs. Lets us filter recall by
file path cheaply (`WHERE ref_path LIKE 'crates/foo/%'`) without parsing body
strings.

### 2. Multi-chunk vectors for sessions (Phase A)

Today: one embedding per obs. fastembed truncates inputs over 512 tokens
silently — for long assistant messages and subagent outputs, the head is
embedded and the tail is lost.

New: char-windowed chunks (~1800 chars each, no overlap), one vector per
chunk, stored in a new `vec_chunks` table:

```sql
CREATE TABLE vec_chunks (
    obs_id    BIGINT NOT NULL,
    chunk_idx INTEGER NOT NULL,
    emb       FLOAT[384] NOT NULL,
    PRIMARY KEY (obs_id, chunk_idx)
);
```

Retrieval: cosine against every chunk, group by `obs_id`, take
`max(score)` per obs as the vector contribution. RRF fusion with BM25 is
unchanged.

BM25 indexing is unchanged — DuckDB FTS handles full body length without
truncation. We get full-body BM25 + chunked-vector recall.

### 3. Code index (Phase B)

New crate `memorize-code`. Tree-sitter via per-language Rust bindings.
cAST-style chunking: walk AST, emit chunks at function/class/method
boundaries. Oversized nodes recursively split; small siblings greedy-merge
up to ~1800 char budget.

Languages at v1: Rust, TypeScript/JavaScript, Python, Go, Bash. Adding a
language = one Cargo line + one parser registration.

Storage in the same DuckDB:

```sql
CREATE TABLE files (
    path       VARCHAR PRIMARY KEY,
    repo_root  VARCHAR NOT NULL,
    language   VARCHAR,
    mtime_ns   BIGINT NOT NULL,
    size_bytes BIGINT NOT NULL,
    git_rev    VARCHAR
);

CREATE TABLE code_chunks (
    id         BIGINT PRIMARY KEY,
    path       VARCHAR NOT NULL REFERENCES files(path),
    line_start INTEGER NOT NULL,
    line_end   INTEGER NOT NULL,
    kind       VARCHAR,
    qualified  VARCHAR,
    body       VARCHAR NOT NULL,
    body_hash  BLOB
);

CREATE TABLE vec_code (
    id  BIGINT PRIMARY KEY,
    emb FLOAT[384]
);

PRAGMA create_fts_index('code_chunks', 'id', 'body');
PRAGMA create_fts_index('code_chunks', 'id', 'qualified');
```

Watcher pipeline:

1. `notify` (Rust crate) watches configured roots (`~/Repos`, `~/Vibes` by
   default), with debouncing (200ms).
2. On modify/create: read file, detect language by extension, parse +
   chunk, hash each chunk's body, upsert any new or changed chunks, embed
   only the changed ones.
3. On delete/rename: remove all `code_chunks` for that path.
4. Cold-start: walk configured roots once on `memorize serve` startup,
   indexing anything not already present. Background thread; doesn't
   block the HTTP server.

Excludes (path-glob, fnmatch-style, configurable):
```
target, node_modules, .git, dist, build,
.env*, *.pem, *.key, secrets/, .aws/
```

Privacy: paths in the include set are indexed in full. Anything matching
an exclude pattern never enters `code_chunks`.

### 4. MCP surface (Phase C)

Drop `memory_recall` and `memory_save`. Replace with two distinct tools:

```text
session_recall(query, limit?)
  Search prior session memory (user prompts, assistant messages, subagent
  results, and compact tool references) for relevant context. Use when
  the user's question references prior work, decisions, or things "we"
  did — before exploring from scratch.

code_recall(query, limit?, language?, path_prefix?)
  Search the indexed local codebase for semantically relevant functions,
  classes, or code blocks. AST-chunked via tree-sitter, hybrid BM25 +
  vector. Returns {path, line_start, line_end, qualified, body}. Use when
  you need to find where something is defined or how a concept is
  implemented across the codebase.
```

The agent decides. Session recall is cheap (small corpus, often <10ms).
Code recall is local-fast (DuckDB FTS + brute-force cosine over a few
thousand chunks).

---

## What's NOT in this plan

- LLM-assisted compression (Phase 5 in the prior plan). Deferred until we
  see real signal-to-noise issues on the new corpus shape.
- Eval harness changes. The harness still runs against LongMemEval-S
  unchanged; it doesn't exercise the new tool-ref capture (the dataset has
  no tool calls) or the code index. Both new features will be measured by
  dogfooding, not by LongMemEval.
- LSIF/SCIP precise code intelligence. Out of scope.
- Cross-language symbol graph (Glean-style). Out of scope.
- "Intent extraction" via LLM for Edit references — today the intent
  string comes from surrounding prompt text; LLM is deferred.
- HNSW vector index. Brute-force cosine is fine until the corpora cross
  ~50k vectors total.

---

## Implementation phases

### Phase A: Session refactor (sequential)

1. **Capture refactor** (`memorize-cli/src/capture.rs`): rewrite per-tool
   handlers to emit compact ref-only bodies. Populate `ref_path` /
   `ref_line_start` / `ref_line_end` for tool obs.
2. **Multi-chunk vectors** (`memorize-store`, `memorize-recall`): drop
   `vec` schema, add `vec_chunks`. Update insert (chunk body, embed each)
   and search (group-by-obs max). Tests reflect new shape.
3. **MCP rename**: `memory_recall` → `session_recall`. Drop `memory_save`
   (kept implicitly by capturing user prompts). Update CLAUDE.md hint.

### Phase B: Code index (sequential)

1. **`memorize-code` crate**: tree-sitter dep + per-language registrations.
   `chunk_file(path) -> Vec<CodeChunk>` exported.
2. **Schema** (`memorize-store`): `files`, `code_chunks`, `vec_code`,
   indexes.
3. **Indexer + watcher**: `notify::Watcher` driven background thread.
   Cold-start scan. Incremental updates on file events.
4. **HTTP routes** (`memorize-server`): `POST /code/search`,
   `GET /code/at?path&line`.
5. **MCP tool** (`memorize-mcp`): `code_recall` registered alongside
   `session_recall`.

### Phase C: Build + install + dogfood

1. `cargo build --release`. Single binary at `~/.local/bin/memorize`.
2. `memorize install-hooks` refreshed: same 6 hook stubs (the capture-side
   refactor changes what they emit, not what fires).
3. Restart `memorize serve`. Cold-scan begins on the configured roots.
4. Smoke: trigger a few real Claude Code turns, confirm new obs shape;
   `code_recall "fastembed"` (or similar) returns hits from this very
   repo.

### What I'd track during dogfood

- Does the agent reach for `session_recall` and `code_recall` proactively?
- Is BM25 noticeably cleaner now that file dumps aren't in the corpus?
- Does `code_recall` actually surface useful chunks, or does the model
  prefer to `Grep` like it does today?

If `code_recall` isn't getting called, the next experiment is tuning the
tool description rather than the indexing — same posture as the agentmemory
descriptions we adopted earlier.
