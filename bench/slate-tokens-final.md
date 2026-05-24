# memorize bench — chunk token distribution

**root:** `/Users/max/src/slate`
**tokenizer:** `/Users/max/.memorize/models/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/5f1b8cd78bc4fb444dd171e59b18f3a3af89a079/tokenizer.json`
**files processed:** 500
**chunks tokenized:** 6092
**elapsed:** 2.08s

## Tokens per chunk

| mean | p50 | p75 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|
| 219 | 163 | 280 | 423 | 474 | 499 | 504 | 504 |

## Overflow vs cap = 512

**chunks that would be truncated:** 0 / 6092 (0.00%)

## Calibration

Cap = 512 tokens. To keep p99 under the cap with the current chunker, TARGET_CHARS should be ≈ p99_tokens × mean_chars_per_token of current corpus. Sample p99 = 499 tokens.

