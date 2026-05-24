# memorize bench — code-index

**root:** `/Users/max/Vibes/memorize`
**files indexed:** 37
**files visited:** 51
**files skipped:** 0 (too big), 14 (wrong language)
**bytes processed:** 246290 (0.2 MB)
**chunks emitted:** 225
**chunks/file mean:** 6.1
**elapsed (per-file):** 3.48s
**embed-model init:** 0.05s (one-time, excluded from per-file)

## Phase totals (wall ms across all files)

| phase | ms | share |
|---|---|---|
| `walk_dir` | 5 | 0.1% |
| `stat` | 0 | 0.0% |
| `read` | 4 | 0.1% |
| `tree_sitter` | 19 | 0.6% |
| `embed` | 3182 | 91.6% |
| `insert` | 263 | 7.6% |

## Per-file latency (ms, including all phases)

| p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|
| 69 | 172 | 282 | 510 | 510 |

## Top 10 slowest files

| ms | bytes | chunks | path |
|---|---|---|---|
| 510 | 31217 | 30 | `/Users/max/Vibes/memorize/crates/memorize-store/src/store.rs` |
| 291 | 18536 | 17 | `/Users/max/Vibes/memorize/crates/memorize-server/src/routes.rs` |
| 282 | 17882 | 18 | `/Users/max/Vibes/memorize/crates/memorize-mcp/src/lib.rs` |
| 255 | 15502 | 16 | `/Users/max/Vibes/memorize/crates/memorize-code/src/lib.rs` |
| 172 | 15255 | 11 | `/Users/max/Vibes/memorize/crates/memorize-server/src/code_indexer.rs` |
| 144 | 11474 | 9 | `/Users/max/Vibes/memorize/crates/memorize-cli/src/capture.rs` |
| 141 | 7578 | 8 | `/Users/max/Vibes/memorize/crates/memorize-server/src/config.rs` |
| 127 | 7691 | 8 | `/Users/max/Vibes/memorize/crates/memorize-eval/src/metrics.rs` |
| 119 | 6211 | 8 | `/Users/max/Vibes/memorize/crates/memorize-recall/src/pipeline.rs` |
| 111 | 6222 | 8 | `/Users/max/Vibes/memorize/crates/memorize-cli/src/main.rs` |

## Chunks-per-file distribution

| p50 | p90 | p99 | max |
|---|---|---|---|
| 5 | 11 | 30 | 30 |

## Indexed by language

| language | files | chunks |
|---|---|---|
| `rust` | 37 | 225 |

