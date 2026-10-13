| Arm | nDCG@10 | Recall@100 | MRR@10 | p (t-test vs BM25) | p (Fisher vs BM25) | p50 ms | p95 ms | Device |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.6762 | 0.9127 | 0.6413 | baseline | baseline | 6.35 | 12.46 | cpu |
| dense | 0.7127 | 0.9417 | 0.6822 | 0.031 * | 0.031 * | 113.39 | 150.68 | cuda |
| hybrid | 0.7267 | 0.9650 | 0.6904 | 1.7e-06 * | <1e-04 * | 123.91 | 162.4 | cpu + cuda |
| rerank | 0.6868 | 0.9650 | 0.6575 | 0.49 | 0.5 | 732.46 | 866.03 | cpu + cuda |

Split: test (300 queries, 5183 docs). RRF k = 10 (tuned on train). `*` = p < 0.05.
