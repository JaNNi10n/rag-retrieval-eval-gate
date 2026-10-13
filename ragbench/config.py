"""Central constants and paths. Everything tunable lives here or in env vars."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("RAGBENCH_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("RAGBENCH_DATA_DIR", ROOT / "data"))
INDEX_DIR = Path(os.environ.get("RAGBENCH_INDEX_DIR", ROOT / "indexes"))
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"

# Hugging Face sources, pinned to the exact revisions used for the reported numbers.
CORPUS_REPO = "BeIR/scifact"
CORPUS_REVISION = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
QRELS_REPO = "BeIR/scifact-qrels"
QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
RERANK_BATCH = 32
RERANK_MAX_LENGTH = 512

# Anserini BM25 defaults.
BM25_K1 = 0.9
BM25_B = 0.4

CANDIDATES = 100  # top-k pulled from each first-stage retriever and fed to RRF / reranker
RRF_K_GRID = (10, 30, 60, 100)
RRF_K_DEFAULT = 60

ARMS = ("bm25", "dense", "hybrid", "rerank")
METRICS = ("ndcg@10", "recall@100", "mrr@10")
SEED = 42

EXPECTED_DOCS = 5183
EXPECTED_TEST_QUERIES = 300
EXPECTED_TRAIN_QUERIES = 809


def device() -> str:
    """`RAGBENCH_DEVICE` overrides; otherwise CUDA when available."""
    forced = os.environ.get("RAGBENCH_DEVICE")
    if forced:
        return forced
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
