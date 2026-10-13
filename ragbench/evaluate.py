"""Metrics (ranx), per-query scores and paired significance tests vs the BM25 baseline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from ranx import Qrels, Run, compare, evaluate

from . import config

Hits = list[tuple[str, float]]


def to_run(hits_by_qid: dict[str, Hits], name: str = "run") -> Run:
    """Rank-only metrics need only the order, so scores become strictly decreasing position values.

    This makes ranx evaluate exactly the order we produced (ranx's own tie-breaking never kicks in).
    Queries with no hits are omitted and scored 0 via make_comparable.
    """
    return Run({q: {d: float(len(h) - i) for i, (d, _) in enumerate(h)} for q, h in hits_by_qid.items() if h}, name=name)


def _qrels(qrels: dict[str, dict[str, int]]) -> Qrels:
    return Qrels({q: dict(docs) for q, docs in qrels.items()})


def _restrict(hits_by_qid: dict[str, Hits], qrels: dict[str, dict[str, int]]) -> dict[str, Hits]:
    return {q: hits_by_qid.get(q, []) for q in qrels}


def mean_scores(hits_by_qid: dict[str, Hits], qrels: dict[str, dict[str, int]], metrics: Sequence[str] = config.METRICS) -> dict[str, float]:
    res = evaluate(_qrels(qrels), to_run(_restrict(hits_by_qid, qrels)), list(metrics), make_comparable=True)
    return {metrics[0]: float(res)} if len(metrics) == 1 else {m: float(v) for m, v in res.items()}


def per_query_scores(hits_by_qid: dict[str, Hits], qrels: dict[str, dict[str, int]], metrics: Sequence[str] = config.METRICS) -> dict[str, dict[str, float]]:
    """{qid: {metric: score}} for every judged query (missing results score 0)."""
    run = to_run(_restrict(hits_by_qid, qrels))
    evaluate(_qrels(qrels), run, list(metrics), make_comparable=True)
    return {q: {m: float(run.scores[m][q]) for m in metrics} for q in qrels}


def significance(
    runs: dict[str, dict[str, Hits]],
    qrels: dict[str, dict[str, int]],
    baseline: str = "bm25",
    metric: str = "ndcg@10",
    n_permutations: int = 10_000,
) -> dict[str, dict[str, float]]:
    """Paired tests of every arm vs `baseline` on `metric`: Student's paired t-test and Fisher randomisation."""
    names = [baseline] + [a for a in runs if a != baseline]
    ranx_runs = [to_run(_restrict(runs[a], qrels), name=a) for a in names]
    out: dict[str, dict[str, float]] = {a: {} for a in names[1:]}
    for test in ("student", "fisher"):
        report = compare(
            _qrels(qrels),
            ranx_runs,
            [metric],
            stat_test=test,
            n_permutations=n_permutations,
            max_p=0.05,
            random_seed=config.SEED,
            make_comparable=True,
        ).to_dict()
        for a in names[1:]:
            out[a][f"p_{test}"] = float(report[baseline]["comparisons"][a][metric])
    return out


def save_run(path: Path, arm: str, split: str, per_query: dict[str, dict[str, float]], hits_by_qid: dict[str, Hits], keep: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arm": arm,
        "split": split,
        "per_query": per_query,
        f"top{keep}": {q: [[d, round(s, 6)] for d, s in hits_by_qid.get(q, [])[:keep]] for q in per_query},
    }
    path.write_text(json.dumps(payload, indent=1))
