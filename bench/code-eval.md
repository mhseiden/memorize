# memorize-eval — code recall A/B

Each query runs under three modes (Hybrid / BM25-only / Vector-only). The headline number is recall@5 — does any expected path appear in the top 5 hits.

## Overall

| mode | n | R@5 | R@10 | R@20 | MRR | mean latency (ms) |
|---|---|---|---|---|---|---|
| hybrid | 47 | 68.1% | 76.6% | 80.9% | 0.513 | 113.2 |
| bm25 | 47 | 63.8% | 72.3% | 83.0% | 0.502 | 32.6 |
| vector | 47 | 53.2% | 66.0% | 66.0% | 0.337 | 81.1 |

## Hand-curated queries

| mode | n | R@5 | R@10 | R@20 | MRR | mean latency (ms) |
|---|---|---|---|---|---|---|
| hybrid | 25 | 76.0% | 80.0% | 88.0% | 0.520 | 112.2 |
| bm25 | 25 | 68.0% | 72.0% | 80.0% | 0.553 | 31.4 |
| vector | 25 | 68.0% | 76.0% | 76.0% | 0.383 | 80.9 |

## Synthetic queries (qualified → path)

| mode | n | R@5 | R@10 | R@20 | MRR | mean latency (ms) |
|---|---|---|---|---|---|---|
| hybrid | 22 | 59.1% | 72.7% | 72.7% | 0.505 | 114.4 |
| bm25 | 22 | 59.1% | 72.7% | 86.4% | 0.445 | 34.0 |
| vector | 22 | 36.4% | 54.5% | 54.5% | 0.284 | 81.2 |

