"""Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum over lists of 1 / (k + rank_d)."""

from __future__ import annotations

from collections.abc import Sequence

from . import config

Hits = list[tuple[str, float]]


def rrf(rankings: Sequence[Hits], k: int = config.RRF_K_DEFAULT, top_n: int = config.CANDIDATES) -> Hits:
    """Fuse ranked lists using ranks only (raw BM25 and cosine scores are on incomparable scales).

    - Each input is re-sorted by (score desc, doc_id asc), so tied input scores get a deterministic rank.
    - A doc missing from a list simply gets no contribution from that list.
    - Fused-score ties are broken by doc_id so the output is reproducible.
    """
    if k < 0:
        raise ValueError("RRF k must be non-negative")
    fused: dict[str, float] = {}
    for hits in rankings:
        for rank, (doc_id, _) in enumerate(sorted(hits, key=lambda h: (-h[1], h[0])), start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:top_n]


def fuse_runs(runs: Sequence[dict[str, Hits]], k: int, top_n: int = config.CANDIDATES) -> dict[str, Hits]:
    qids = sorted(set().union(*runs))
    return {q: rrf([r.get(q, []) for r in runs], k=k, top_n=top_n) for q in qids}


def tune_k(
    runs: Sequence[dict[str, Hits]],
    qrels: dict[str, dict[str, int]],
    grid: Sequence[int] = config.RRF_K_GRID,
    metric: str = "ndcg@10",
) -> tuple[int, dict[int, float]]:
    """Grid-search k on the given (train) qrels; returns (best_k, {k: score}). Ties keep the first k in grid."""
    from .evaluate import mean_scores

    scores = {k: mean_scores(fuse_runs(runs, k), qrels, [metric])[metric] for k in grid}
    best = max(grid, key=lambda k: scores[k])  # max() keeps the first maximum
    return best, scores
