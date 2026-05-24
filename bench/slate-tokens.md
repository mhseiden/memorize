# memorize bench — chunk token distribution

**root:** `/Users/max/src/slate`
**tokenizer:** `/Users/max/.memorize/models/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/5f1b8cd78bc4fb444dd171e59b18f3a3af89a079/tokenizer.json`
**files processed:** 500
**chunks tokenized:** 5290
**elapsed:** 0.88s

## Tokens per chunk

| mean | p50 | p75 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|
| 251 | 175 | 309 | 475 | 571 | 790 | 2825 | 8336 |

## Overflow vs cap = 512

**chunks that would be truncated:** 437 / 5290 (8.26%)

## Calibration

Cap = 512 tokens. To keep p99 under the cap with the current chunker, TARGET_CHARS should be ≈ p99_tokens × mean_chars_per_token of current corpus. Sample p99 = 790 tokens.

## Top overflow chunks

| tokens | chunk_chars | path |
|---|---|---|
| 8336 | 24939 | `/Users/max/src/slate/oxlint.config.ts` |
| 6383 | 17814 | `/Users/max/src/slate/packages/slate/cypress/const.ts` |
| 6279 | 19438 | `/Users/max/src/slate/packages/gql/src/mainChunkExtracted.ts` |
| 3103 | 9428 | `/Users/max/src/slate/packages/gql/src/mainChunkExtracted.ts` |
| 3096 | 9330 | `/Users/max/src/slate/packages/slate/cypress/const.ts` |
| 2825 | 9000 | `/Users/max/src/slate/packages/gql/src/possibleTypes.ts` |
| 2392 | 12022 | `/Users/max/src/slate/packages/gql/src/mainChunkExtracted.ts` |
| 2269 | 5613 | `/Users/max/src/slate/packages/slate/cypress/live-check/workbook_row_level_security_spec.ts` |
| 2206 | 7736 | `/Users/max/src/slate/packages/slate/cypress/support/commands.ts` |
| 2205 | 5799 | `/Users/max/src/slate/packages/slate/cypress/const.ts` |
| 2129 | 6958 | `/Users/max/src/slate/packages/gql/src/mainChunkExtracted.ts` |
| 2092 | 10222 | `/Users/max/src/slate/packages/gql/src/mainChunkExtracted.ts` |
| 1937 | 5678 | `/Users/max/src/slate/packages/init/src/cloudHostProd.ts` |
| 1924 | 6503 | `/Users/max/src/slate/packages/gql/src/mainChunkExtracted.ts` |
| 1817 | 6077 | `/Users/max/src/slate/packages/tools/src/noRestrictedImports.ts` |

