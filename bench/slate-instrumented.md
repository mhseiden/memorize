# memorize bench — code-index

**root:** `/Users/max/src/slate`
**files indexed:** 200
**files visited:** 303
**files skipped:** 3 (too big), 100 (wrong language)
**bytes processed:** 1724742 (1.6 MB)
**chunks emitted:** 1935
**chunks/file mean:** 9.7
**elapsed (per-file):** 27.25s
**embed-model init:** 0.05s (one-time, excluded from per-file)

## Phase totals (wall ms across all files)

| phase | ms | share |
|---|---|---|
| `walk_dir` | 17 | 0.1% |
| `stat` | 4 | 0.0% |
| `read` | 128 | 0.5% |
| `tree_sitter` | 116 | 0.4% |
| `embed` | 24353 | 89.4% |
| `insert` | 2631 | 9.7% |

## Embed batch utilization

**embed_batch calls:** 200
**chunks embedded:** 1935
**mean batch size:** 9.7
**per-call:** 121.77 ms
**per-chunk:** 12.59 ms

| batch p50 | p90 | p99 | max |
|---|---|---|---|
| 6 | 21 | 59 | 158 |

| batch | calls | call share | chunks | chunk share |
|---|---|---|---|---|
| 1 | 50 | 25.0% | 50 | 2.6% |
| 2-4 | 41 | 20.5% | 118 | 6.1% |
| 5-8 | 40 | 20.0% | 271 | 14.0% |
| 9-16 | 37 | 18.5% | 438 | 22.6% |
| 17-32 | 22 | 11.0% | 498 | 25.7% |
| 33-64 | 8 | 4.0% | 333 | 17.2% |
| 65-128 | 1 | 0.5% | 69 | 3.6% |
| 129+ | 1 | 0.5% | 158 | 8.2% |

## Per-file latency (ms, including all phases)

| p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|
| 74 | 291 | 456 | 940 | 2667 |

## Top 10 slowest files

| ms | bytes | chunks | path |
|---|---|---|---|
| 2667 | 82855 | 158 | `/Users/max/src/slate/packages/result/src/result.ts` |
| 1093 | 35275 | 69 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/__tests__/textMarkdown.test.ts` |
| 940 | 72354 | 59 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/__tests__/JoinSource.test.ts` |
| 720 | 39419 | 52 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/__tests__/errorPaths.test.ts` |
| 588 | 27705 | 37 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/shared/DataLabel.ts` |
| 554 | 40901 | 35 | `/Users/max/src/slate/tools/momo/src/pipeline/map.ts` |
| 541 | 36300 | 34 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/__tests__/Control.test.ts` |
| 509 | 23041 | 30 | `/Users/max/src/slate/tools/momo/src/pipeline/triageMerge.ts` |
| 487 | 25676 | 36 | `/Users/max/src/slate/tools/momo/src/collectors/collect.ts` |
| 482 | 39696 | 33 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/__tests__/Container.test.ts` |

## Chunks-per-file distribution

| p50 | p90 | p99 | max |
|---|---|---|---|
| 6 | 21 | 59 | 158 |

## Indexed by language

| language | files | chunks |
|---|---|---|
| `typescript` | 196 | 1918 |
| `javascript` | 2 | 4 |
| `bash` | 2 | 13 |

