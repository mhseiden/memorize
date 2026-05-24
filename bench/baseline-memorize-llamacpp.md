# memorize bench — code-index

**root:** `/Users/max/Vibes/memorize`
**files indexed:** 36
**files visited:** 65
**files skipped:** 0 (too big), 29 (wrong language)
**bytes processed:** 278800 (0.3 MB)
**chunks emitted:** 329
**chunks/file mean:** 9.1
**elapsed (per-file):** 1.60s
**embed-model init:** 0.46s (one-time, excluded from per-file)

## Phase totals (wall ms across all files)

| phase | ms | share |
|---|---|---|
| `walk_dir` | 6 | 0.4% |
| `stat` | 0 | 0.0% |
| `read` | 4 | 0.3% |
| `tree_sitter` | 164 | 10.3% |
| `embed` | 1127 | 70.4% |
| `insert` | 298 | 18.7% |

## Embed batch utilization

**embed_batch calls:** 36
**chunks embedded:** 329
**mean batch size:** 9.1
**per-call:** 31.32 ms
**per-chunk:** 3.43 ms

| batch p50 | p90 | p99 | max |
|---|---|---|---|
| 7 | 23 | 36 | 36 |

| batch | calls | call share | chunks | chunk share |
|---|---|---|---|---|
| 1 | 5 | 13.9% | 5 | 1.5% |
| 2-4 | 9 | 25.0% | 26 | 7.9% |
| 5-8 | 8 | 22.2% | 52 | 15.8% |
| 9-16 | 9 | 25.0% | 108 | 32.8% |
| 17-32 | 3 | 8.3% | 68 | 20.7% |
| 33-64 | 2 | 5.6% | 70 | 21.3% |
| 65-128 | 0 | 0.0% | 0 | 0.0% |
| 129+ | 0 | 0.0% | 0 | 0.0% |

## Per-file latency (ms, including all phases)

| p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|
| 31 | 90 | 140 | 288 | 288 |

## Top 10 slowest files

| ms | bytes | chunks | path |
|---|---|---|---|
| 288 | 34995 | 34 | `/Users/max/Vibes/memorize/crates/memorize-cli/src/bench.rs` |
| 174 | 31217 | 36 | `/Users/max/Vibes/memorize/crates/memorize-store/src/store.rs` |
| 140 | 18536 | 23 | `/Users/max/Vibes/memorize/crates/memorize-server/src/routes.rs` |
| 90 | 17882 | 24 | `/Users/max/Vibes/memorize/crates/memorize-mcp/src/lib.rs` |
| 84 | 19186 | 21 | `/Users/max/Vibes/memorize/crates/memorize-code/src/lib.rs` |
| 71 | 14272 | 16 | `/Users/max/Vibes/memorize/crates/memorize-embed/src/lib.rs` |
| 67 | 16467 | 16 | `/Users/max/Vibes/memorize/crates/memorize-server/src/code_indexer.rs` |
| 59 | 11230 | 13 | `/Users/max/Vibes/memorize/crates/memorize-cli/src/install_hooks.rs` |
| 51 | 11474 | 14 | `/Users/max/Vibes/memorize/crates/memorize-cli/src/capture.rs` |
| 51 | 9156 | 10 | `/Users/max/Vibes/memorize/crates/memorize-eval/src/harness.rs` |

## Chunks-per-file distribution

| p50 | p90 | p99 | max |
|---|---|---|---|
| 7 | 23 | 36 | 36 |

## Indexed by language

| language | files | chunks |
|---|---|---|
| `rust` | 36 | 329 |

