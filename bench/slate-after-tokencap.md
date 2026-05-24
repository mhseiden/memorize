# memorize bench — code-index

**root:** `/Users/max/src/slate`
**files indexed:** 500
**files visited:** 674
**files skipped:** 4 (too big), 170 (wrong language)
**bytes processed:** 4818688 (4.6 MB)
**chunks emitted:** 6092
**chunks/file mean:** 12.2
**elapsed (per-file):** 82.35s
**embed-model init:** 0.06s (one-time, excluded from per-file)

## Phase totals (wall ms across all files)

| phase | ms | share |
|---|---|---|
| `walk_dir` | 33 | 0.0% |
| `stat` | 12 | 0.0% |
| `read` | 149 | 0.2% |
| `tree_sitter` | 1582 | 1.9% |
| `embed` | 71629 | 87.0% |
| `insert` | 8938 | 10.9% |

## Embed batch utilization

**embed_batch calls:** 500
**chunks embedded:** 6092
**mean batch size:** 12.2
**per-call:** 143.26 ms
**per-chunk:** 11.76 ms

| batch p50 | p90 | p99 | max |
|---|---|---|---|
| 6 | 24 | 84 | 705 |

| batch | calls | call share | chunks | chunk share |
|---|---|---|---|---|
| 1 | 100 | 20.0% | 100 | 1.6% |
| 2-4 | 109 | 21.8% | 309 | 5.1% |
| 5-8 | 98 | 19.6% | 631 | 10.4% |
| 9-16 | 92 | 18.4% | 1088 | 17.9% |
| 17-32 | 72 | 14.4% | 1565 | 25.7% |
| 33-64 | 19 | 3.8% | 830 | 13.6% |
| 65-128 | 8 | 1.6% | 704 | 11.6% |
| 129+ | 2 | 0.4% | 865 | 14.2% |

## Per-file latency (ms, including all phases)

| p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|
| 67 | 296 | 453 | 1249 | 13427 |

## Top 10 slowest files

| ms | bytes | chunks | path |
|---|---|---|---|
| 13427 | 676941 | 705 | `/Users/max/src/slate/packages/gql/src/mainChunkExtracted.ts` |
| 2498 | 82855 | 160 | `/Users/max/src/slate/packages/result/src/result.ts` |
| 2031 | 70282 | 126 | `/Users/max/src/slate/packages/slate/cypress/support/commands.ts` |
| 1548 | 67798 | 92 | `/Users/max/src/slate/packages/programmatic-api/src/v1/schema/Control.ts` |
| 1472 | 71083 | 95 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/Control.ts` |
| 1249 | 14849 | 78 | `/Users/max/src/slate/packages/utils/src/__tests__/hasher.test.ts` |
| 1143 | 35275 | 77 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/__tests__/textMarkdown.test.ts` |
| 1134 | 51125 | 84 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/__tests__/CondFormat.test.ts` |
| 957 | 72354 | 76 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/__tests__/JoinSource.test.ts` |
| 956 | 54300 | 60 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/preprocess.ts` |

## Chunks-per-file distribution

| p50 | p90 | p99 | max |
|---|---|---|---|
| 6 | 24 | 84 | 705 |

## Indexed by language

| language | files | chunks |
|---|---|---|
| `typescript` | 489 | 6055 |
| `javascript` | 9 | 24 |
| `bash` | 2 | 13 |

