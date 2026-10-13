"""Single-query search over all arms. Used by the API and the latency benchmark."""

from __future__ import annotations

import json

from . import config
from .bm25 import BM25Index
from .dense import DenseIndex
from .hybrid import rrf
from .rerank import PairScorer, rerank

Hits = list[tuple[str, float]]


def frozen_rrf_k() -> int:
    """The k chosen on TRAIN qrels by `ragbench run` (results/rrf_tuning.json); 60 if never tuned."""
    path = config.RESULTS_DIR / "rrf_tuning.json"
    try:
        return int(json.loads(path.read_text())["best_k"])
    except (OSError, KeyError, ValueError):
        return config.RRF_K_DEFAULT


class Searcher:
    def __init__(
        self,
        docs: dict[str, str],
        titles: dict[str, str],
        bm25: BM25Index,
        dense: DenseIndex | None = None,
        scorer: PairScorer | None = None,
        rrf_k: int = config.RRF_K_DEFAULT,
        device: str = "cpu",
    ):
        self.docs, self.titles = docs, titles
        self.bm25, self.dense, self.scorer = bm25, dense, scorer
        self.rrf_k = rrf_k
        self.device = device

    @property
    def arms(self) -> list[str]:
        arms = ["bm25"]
        if self.dense is not None:
            arms += ["dense", "hybrid"]
            if self.scorer is not None:
                arms.append("rerank")
        return arms

    def search(self, query: str, arm: str, k: int = 10) -> Hits:
        if arm not in self.arms:
            raise ValueError(f"arm {arm!r} is not loaded (available: {self.arms})")
        if arm == "bm25":
            return self.bm25.search(query, k)
        if arm == "dense":
            return self.dense.search(query, k)
        fused = rrf([self.bm25.search(query, config.CANDIDATES), self.dense.search(query, config.CANDIDATES)], k=self.rrf_k)
        if arm == "hybrid":
            return fused[:k]
        return rerank(self.scorer, query, fused, self.docs)[:k]

    @classmethod
    def from_disk(cls, arms: list[str] | None = None, device: str | None = None) -> Searcher:
        """Load data, build BM25, load the cached dense index (encodes once if missing) and the reranker."""
        from .data import load_scifact
        from .dense import BgeEncoder
        from .rerank import CrossEncoderScorer

        arms = arms or list(config.ARMS)
        device = device or config.device()
        ds = load_scifact()
        bm25 = BM25Index(ds.doc_ids, ds.texts)
        dense = scorer = None
        if any(a in arms for a in ("dense", "hybrid", "rerank")):
            dense = DenseIndex.build_or_load(BgeEncoder(device=device), ds.doc_ids, ds.texts)
        if "rerank" in arms:
            scorer = CrossEncoderScorer(device=device)
        return cls(ds.docs, ds.titles, bm25, dense, scorer, rrf_k=frozen_rrf_k(), device=device)
