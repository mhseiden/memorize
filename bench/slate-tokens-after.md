# memorize bench — chunk token distribution

**root:** `/Users/max/src/slate`
**tokenizer:** `/Users/max/.memorize/models/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/5f1b8cd78bc4fb444dd171e59b18f3a3af89a079/tokenizer.json`
**files processed:** 500
**chunks tokenized:** 5948
**elapsed:** 0.85s

## Tokens per chunk

| mean | p50 | p75 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|
| 223 | 167 | 283 | 426 | 476 | 532 | 665 | 822 |

## Overflow vs cap = 512

**chunks that would be truncated:** 107 / 5948 (1.80%)

## Calibration

Cap = 512 tokens. To keep p99 under the cap with the current chunker, TARGET_CHARS should be ≈ p99_tokens × mean_chars_per_token of current corpus. Sample p99 = 532 tokens.

## Top overflow chunks

| tokens | chunk_chars | path |
|---|---|---|
| 822 | 1440 | `/Users/max/src/slate/packages/utils/src/__tests__/round.test.ts` |
| 782 | 1411 | `/Users/max/src/slate/packages/utils/src/__tests__/round.test.ts` |
| 770 | 1423 | `/Users/max/src/slate/packages/utils/src/__tests__/round.test.ts` |
| 765 | 1454 | `/Users/max/src/slate/packages/utils/src/__tests__/round.test.ts` |
| 725 | 1496 | `/Users/max/src/slate/oxlint.config.ts` |
| 670 | 1474 | `/Users/max/src/slate/packages/slate/cypress/live-check/search_spec.ts` |
| 665 | 1493 | `/Users/max/src/slate/tools/momo/src/pipeline/rollup.ts` |
| 649 | 1471 | `/Users/max/src/slate/oxlint.config.ts` |
| 645 | 1442 | `/Users/max/src/slate/packages/utils/src/querycache/types.ts` |
| 632 | 1490 | `/Users/max/src/slate/packages/slate/cypress/const.ts` |
| 624 | 1482 | `/Users/max/src/slate/oxlint.config.ts` |
| 621 | 1483 | `/Users/max/src/slate/packages/slate/cypress/live-check/workbook_row_level_security_spec.ts` |
| 618 | 1499 | `/Users/max/src/slate/packages/slate/cypress/live-check/dynamic_connection_config_spec.ts` |
| 608 | 1466 | `/Users/max/src/slate/oxlint.config.ts` |
| 608 | 1441 | `/Users/max/src/slate/packages/slate/cypress/live-check/workbook_row_level_security_spec.ts` |

