"""Cross-encoder re-ranking of the hybrid top-100 with ms-marco-MiniLM-L-6-v2."""

from __future__ import annotations

from typing import Protocol

from . import config

Hits = list[tuple[str, float]]


class PairScorer(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderScorer:
    def __init__(
        self,
        model_name: str = config.RERANK_MODEL,
        revision: str | None = config.RERANK_REVISION,
        device: str | None = None,
        batch_size: int = config.RERANK_BATCH,
        max_length: int = config.RERANK_MAX_LENGTH,
    ):
        from sentence_transformers import CrossEncoder

        self.device = device or config.device()
        self.model = CrossEncoder(model_name, revision=revision, device=self.device, max_length=max_length)
        if self.device.startswith("cuda"):
            self.model.model.half()
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)]


def rerank(scorer: PairScorer, query: str, candidates: Hits, docs: dict[str, str], top_n: int = config.CANDIDATES) -> Hits:
    """Re-score every candidate with the cross-encoder. Ties keep the first-stage order (stable sort)."""
    if not candidates:
        return []
    ids = [d for d, _ in candidates]
    scores = scorer.score(query, [docs[d] for d in ids])
    return sorted(zip(ids, scores), key=lambda x: -x[1])[:top_n]
