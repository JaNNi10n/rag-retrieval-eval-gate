| Arm | nDCG@10 | Recall@100 | MRR@10 | p (t-test vs BM25) | p (Fisher vs BM25) | p50 ms | p95 ms | Device |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.6762 | 0.9127 | 0.6413 | baseline | baseline | - | - | cpu |
| dense | 0.7127 | 0.9417 | 0.6822 | 0.031 * | 0.027 * | - | - | cuda |

Split: test (300 queries, 5183 docs). RRF k = None (tuned on train). `*` = p < 0.05.
