# memorize bench — code-index

**root:** `/Users/max/src/slate`
**files indexed:** 500
**files visited:** 674
**files skipped:** 4 (too big), 170 (wrong language)
**bytes processed:** 4818688 (4.6 MB)
**chunks emitted:** 5290
**chunks/file mean:** 10.6
**elapsed (per-file):** 74.99s
**embed-model init:** 0.05s (one-time, excluded from per-file)

## Phase totals (wall ms across all files)

| phase | ms | share |
|---|---|---|
| `walk_dir` | 36 | 0.0% |
| `stat` | 16 | 0.0% |
| `read` | 302 | 0.4% |
| `tree_sitter` | 318 | 0.4% |
| `embed` | 66834 | 89.1% |
| `insert` | 7479 | 10.0% |

## Per-file latency (ms, including all phases)

| p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|
| 66 | 287 | 463 | 1206 | 8203 |

## Top 10 slowest files

| ms | bytes | chunks | path |
|---|---|---|---|
| 8203 | 676941 | 487 | `/Users/max/src/slate/packages/gql/src/mainChunkExtracted.ts` |
| 2652 | 82855 | 158 | `/Users/max/src/slate/packages/result/src/result.ts` |
| 1865 | 70282 | 120 | `/Users/max/src/slate/packages/slate/cypress/support/commands.ts` |
| 1437 | 71083 | 93 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/Control.ts` |
| 1410 | 67798 | 90 | `/Users/max/src/slate/packages/programmatic-api/src/v1/schema/Control.ts` |
| 1206 | 14849 | 77 | `/Users/max/src/slate/packages/utils/src/__tests__/hasher.test.ts` |
| 1104 | 51125 | 84 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/__tests__/CondFormat.test.ts` |
| 1077 | 35275 | 69 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/__tests__/textMarkdown.test.ts` |
| 946 | 72354 | 59 | `/Users/max/src/slate/packages/programmatic-api/src/v1/schema/__tests__/JoinSource.test.ts` |
| 938 | 72354 | 59 | `/Users/max/src/slate/packages/workbook-api/src/v1/schema/__tests__/JoinSource.test.ts` |

## Chunks-per-file distribution

| p50 | p90 | p99 | max |
|---|---|---|---|
| 5 | 20 | 84 | 487 |

## Indexed by language

| language | files | chunks |
|---|---|---|
| `typescript` | 489 | 5254 |
| `javascript` | 9 | 23 |
| `bash` | 2 | 13 |

