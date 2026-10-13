"""Dump the hardest queries for the best arm so they can be categorised by hand (see ERRORS.md)."""

from __future__ import annotations

import json
from pathlib import Path

from .data import Dataset

Hits = list[tuple[str, float]]


def worst_queries(per_query: dict[str, dict[str, float]], hits_by_qid: dict[str, Hits], qrels: dict[str, dict[str, int]], n: int = 15) -> list[str]:
    """Lowest nDCG@10 first; ties (usually nDCG=0) broken by how deep the first gold doc sits (missing = worst)."""

    def first_gold_rank(q: str) -> int:
        for rank, (d, _) in enumerate(hits_by_qid.get(q, []), start=1):
            if qrels[q].get(d, 0) > 0:
                return rank
        return 10**6

    return sorted(per_query, key=lambda q: (per_query[q]["ndcg@10"], -first_gold_rank(q), q))[:n]


def dump(ds: Dataset, split: str, arm: str, per_query: dict[str, dict[str, float]], hits_by_qid: dict[str, Hits], out_dir: Path, n: int = 15) -> list[dict]:
    qrels = ds.qrels[split]
    rows = []
    for q in worst_queries(per_query, hits_by_qid, qrels, n):
        hits = hits_by_qid.get(q, [])
        gold_ranks = {d: r for r, (d, _) in enumerate(hits, start=1) if d in qrels[q]}
        rows.append(
            {
                "qid": q,
                "query": ds.queries[q],
                "ndcg@10": round(per_query[q]["ndcg@10"], 4),
                "gold": [{"doc_id": d, "title": ds.titles[d], "rank_in_top100": gold_ranks.get(d)} for d in qrels[q]],
                "top3": [{"doc_id": d, "title": ds.titles[d], "score": round(s, 4), "relevant": d in qrels[q]} for d, s in hits[:3]],
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "errors.json").write_text(json.dumps({"arm": arm, "split": split, "queries": rows}, indent=1))
    lines = [f"# Hardest {len(rows)} {split} queries for the `{arm}` arm (auto-generated)", ""]
    for i, r in enumerate(rows, 1):
        lines.append(f"## {i}. q{r['qid']} (nDCG@10 = {r['ndcg@10']})")
        lines.append(f"**Query:** {r['query']}")
        for g in r["gold"]:
            lines.append(f"- gold `{g['doc_id']}` (rank {g['rank_in_top100'] or '>100'}): {g['title']}")
        for j, t in enumerate(r["top3"], 1):
            lines.append(f"- top{j} `{t['doc_id']}`{' (relevant)' if t['relevant'] else ''}: {t['title']}")
        lines.append("")
    (out_dir / "errors_dump.md").write_text("\n".join(lines), encoding="utf-8")
    return rows
