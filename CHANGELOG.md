# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-13

### Added
- `ragbench` package: SciFact loader (pinned HF revisions, local cache, size assertions), BM25 (bm25s, Lucene k1=0.9 b=0.4,
  English stopwords, Snowball stemming), dense retrieval (bge-small-en-v1.5, query instruction prefix, FAISS `IndexFlatIP`,
  fp16 on CUDA, on-disk `.faiss` + `.npy` cache), Reciprocal Rank Fusion with k tuned on the train qrels only,
  cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) of the hybrid top-100.
- Evaluation with ranx (nDCG@10, Recall@100, MRR@10), per-query scores in `runs/<arm>.json`, paired t-test and Fisher
  randomisation vs BM25 (and vs dense), per-query latency (10 warm-up queries, p50/p95), and an error dump of the 15
  hardest queries for the best arm, categorised by hand in `ERRORS.md`.
- CLI: `python -m ragbench run | gate | baseline`; the gate exits 1 when nDCG@10 drops more than `--tol` below the
  committed baseline.
- FastAPI service (`GET /search`, `GET /health`) with sync endpoints, input validation and a CPU Dockerfile (non-root).
- pytest suite (metrics, RRF, BM25, API, gate, pipeline on a fixture), GitHub Actions workflow (ruff, pytest with an
  80% coverage floor, BM25 + dense gate on CPU), and an HTTP latency benchmark script.
- Latency is timed with the arms interleaved per query (rotating order), in the pipeline and in the HTTP benchmark, so
  load from other processes on a shared GPU affects every arm equally.
- Ablations with saved logs: re-ranker in fp32, bge-small in fp32 (the CI precision), with and without the bge query
  prefix, and three deliberately broken `demo/*` branches run through the gate.

