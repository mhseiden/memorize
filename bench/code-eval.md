# memorize-eval — code recall A/B

Each query runs under several modes. The headline number is recall@5 — does any expected path appear in the top 5 hits. Modes that weren't enabled in this run are omitted.

## Overall

| mode | n | R@5 | R@10 | R@20 | MRR | mean latency (ms) |
|---|---|---|---|---|---|---|
| hybrid | 77 | 58.4% | 68.8% | 75.3% | 0.445 | 111.7 |
| bm25 | 77 | 58.4% | 63.6% | 72.7% | 0.430 | 31.0 |
| vector | 77 | 42.9% | 50.6% | 59.7% | 0.273 | 81.6 |
| int8-only | 77 | 41.6% | 49.4% | 58.4% | 0.311 | 7.1 |
| int8-hybrid | 77 | 57.1% | 68.8% | 75.3% | 0.445 | 34.4 |

## Hand-curated queries

| mode | n | R@5 | R@10 | R@20 | MRR | mean latency (ms) |
|---|---|---|---|---|---|---|
| hybrid | 25 | 76.0% | 88.0% | 92.0% | 0.526 | 111.4 |
| bm25 | 25 | 72.0% | 76.0% | 84.0% | 0.571 | 30.9 |
| vector | 25 | 68.0% | 76.0% | 76.0% | 0.383 | 81.5 |
| int8-only | 25 | 60.0% | 68.0% | 72.0% | 0.427 | 7.4 |
| int8-hybrid | 25 | 76.0% | 88.0% | 92.0% | 0.526 | 34.1 |

## Synthetic queries (qualified → path)

| mode | n | R@5 | R@10 | R@20 | MRR | mean latency (ms) |
|---|---|---|---|---|---|---|
| hybrid | 52 | 50.0% | 59.6% | 67.3% | 0.407 | 111.9 |
| bm25 | 52 | 51.9% | 57.7% | 67.3% | 0.362 | 31.1 |
| vector | 52 | 30.8% | 38.5% | 51.9% | 0.220 | 81.6 |
| int8-only | 52 | 32.7% | 40.4% | 51.9% | 0.256 | 6.9 |
| int8-hybrid | 52 | 48.1% | 59.6% | 67.3% | 0.407 | 34.5 |

