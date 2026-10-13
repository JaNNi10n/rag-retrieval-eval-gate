"""End-to-end experiment: retrieve with every arm, evaluate, test significance, time, and write results/."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from . import config
from .bm25 import BM25Index
from .data import Dataset, load_scifact
from .dense import DenseIndex, Encoder
from .errors import dump
from .evaluate import mean_scores, per_query_scores, save_run, significance
from .hybrid import fuse_runs, tune_k
from .latency import measure_interleaved
from .rerank import PairScorer, rerank
from .search import Searcher

FISHER_PERMUTATIONS = 10_000
ARM_DEVICE = {"bm25": "cpu", "dense": "{dev}", "hybrid": "cpu + {dev}", "rerank": "cpu + {dev}"}


def environment(device: str) -> dict:
    from importlib.metadata import version

    import bm25s
    import faiss
    import numpy
    import ranx  # noqa: F401
    import sentence_transformers
    import torch

    env = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor(),
        "device": device,
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "faiss": faiss.__version__,
        "bm25s": bm25s.__version__,
        "ranx": version("ranx"),
        "numpy": numpy.__version__,
    }
    if device.startswith("cuda"):
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def fmt_p(p: float, n_permutations: int = FISHER_PERMUTATIONS) -> str:
    """Format a p-value; a randomisation p of 0 only means 'below 1 / n_permutations'."""
    text = f"<{1 / n_permutations:.0e}" if p == 0 else f"{p:.2g}"
    return text + (" *" if p < 0.05 else "")


def table_md(results: dict) -> str:
    rows = [
        "| Arm | nDCG@10 | Recall@100 | MRR@10 | p (t-test vs BM25) | p (Fisher vs BM25) | p50 ms | p95 ms | Device |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, r in results["arms"].items():
        p_t, p_f = ("baseline", "baseline") if arm == "bm25" else (fmt_p(r["p_student"]), fmt_p(r["p_fisher"]))
        lat = r.get("latency")
        rows.append(
            f"| {arm} | {r['ndcg@10']:.4f} | {r['recall@100']:.4f} | {r['mrr@10']:.4f} | {p_t} | {p_f} | "
            f"{lat['p50_ms'] if lat else '-'} | {lat['p95_ms'] if lat else '-'} | {r['device']} |"
        )
    rows.append("")
    rows.append(
        f"Split: {results['split']} ({results['n_queries']} queries, {results['n_docs']} docs). "
        f"RRF k = {results.get('rrf_k', '-')} (tuned on train). `*` = p < 0.05."
    )
    return "\n".join(rows) + "\n"


def run(
    arms: list[str],
    split: str = "test",
    out_dir: Path = config.RESULTS_DIR,
    runs_dir: Path = config.RUNS_DIR,
    ds: Dataset | None = None,
    encoder: Encoder | None = None,
    scorer: PairScorer | None = None,
    index_dir: Path | None = None,
    device: str | None = None,
    latency: bool = True,
    warmup: int = 10,
    n_errors: int = 15,
    log=print,
) -> dict:
    unknown = set(arms) - set(config.ARMS)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    out_dir, runs_dir = Path(out_dir), Path(runs_dir)
    device = device or ("cpu" if encoder is not None else config.device())
    timings: dict[str, float] = {}

    t = time.perf_counter()
    ds = ds or load_scifact()
    queries, qrels = ds.split_queries(split), ds.qrels[split]
    qids, qtexts = list(queries), list(queries.values())
    log(f"[data] {len(ds.doc_ids)} docs, {len(queries)} {split} queries ({time.perf_counter() - t:.1f}s)")

    t = time.perf_counter()
    bm25 = BM25Index(ds.doc_ids, ds.texts)
    timings["bm25_index_s"] = round(time.perf_counter() - t, 2)
    runs = {"bm25": dict(zip(qids, bm25.search_batch(qtexts)))}  # always computed: it is the significance baseline

    dense = None
    best_k, tuning = None, None
    if set(arms) & {"dense", "hybrid", "rerank"}:
        if encoder is None:
            from .dense import BgeEncoder

            encoder = BgeEncoder(device=device)
        t = time.perf_counter()
        dense = DenseIndex.build_or_load(encoder, ds.doc_ids, ds.texts, index_dir)
        timings["dense_index_build_or_load_s"] = round(time.perf_counter() - t, 2)
        runs["dense"] = dict(zip(qids, dense.search_batch(qtexts)))
        log(f"[dense] index ready in {timings['dense_index_build_or_load_s']}s")

    if set(arms) & {"hybrid", "rerank"}:
        train_q = ds.split_queries("train")
        train_runs = [dict(zip(train_q, r.search_batch(list(train_q.values())))) for r in (bm25, dense)]
        best_k, tuning = tune_k(train_runs, ds.qrels["train"])
        (out_dir).mkdir(parents=True, exist_ok=True)
        tuning_doc = {"tuned_on": "train", "n_queries": len(train_q), "metric": "ndcg@10", "grid": {str(k): round(v, 4) for k, v in tuning.items()}}
        (out_dir / "rrf_tuning.json").write_text(json.dumps({**tuning_doc, "best_k": best_k}, indent=1))
        log(f"[hybrid] RRF k tuned on train: {tuning} -> k={best_k}")
        runs["hybrid"] = fuse_runs([runs["bm25"], runs["dense"]], best_k)

    if "rerank" in arms:
        if scorer is None:
            from .rerank import CrossEncoderScorer

            scorer = CrossEncoderScorer(device=device)
        t = time.perf_counter()
        runs["rerank"] = {q: rerank(scorer, queries[q], runs["hybrid"][q], ds.docs) for q in qids}
        timings["rerank_all_queries_s"] = round(time.perf_counter() - t, 2)
        log(f"[rerank] {len(qids)} queries x {config.CANDIDATES} candidates in {timings['rerank_all_queries_s']}s")

    sig = significance({a: runs[a] for a in ["bm25", *[a for a in arms if a != "bm25"]]}, qrels, n_permutations=FISHER_PERMUTATIONS)
    vs_dense = {}
    if "dense" in runs and set(arms) & {"hybrid", "rerank"}:
        others = [a for a in arms if a in ("hybrid", "rerank")]
        vs_dense = significance({a: runs[a] for a in ["dense", *others]}, qrels, baseline="dense", n_permutations=FISHER_PERMUTATIONS)
    searcher = Searcher(ds.docs, ds.titles, bm25, dense, scorer, rrf_k=best_k or config.RRF_K_DEFAULT, device=device)

    results = {
        "dataset": "BEIR SciFact",
        "split": split,
        "n_queries": len(queries),
        "n_docs": len(ds.doc_ids),
        "rrf_k": best_k,
        "significance": {
            "baseline": "bm25",
            "metric": "ndcg@10",
            "tests": ["paired t-test", "Fisher randomisation"],
            "fisher_permutations": FISHER_PERMUTATIONS,
            "seed": config.SEED,
            "alpha": 0.05,
        },
        "environment": environment(device),
        "timings": timings,
        "arms": {},
    }
    lat = measure_interleaved({a: (lambda q, a=a: searcher.search(q, a, 10)) for a in arms}, qtexts, warmup=warmup) if latency else {}
    per_query_all = {}
    for arm in arms:
        pq = per_query_scores(runs[arm], qrels)
        per_query_all[arm] = pq
        save_run(runs_dir / f"{arm}.json", arm, split, pq, runs[arm])
        entry = {**mean_scores(runs[arm], qrels), "device": ARM_DEVICE[arm].format(dev=device)}
        entry.update(sig.get(arm, {}))
        if arm in vs_dense:
            entry["vs_dense"] = vs_dense[arm]
        if latency:
            entry["latency"] = lat[arm]
        results["arms"][arm] = entry
        lat_txt = f", p50={entry['latency']['p50_ms']}ms p95={entry['latency']['p95_ms']}ms" if latency else ""
        log(f"[{arm}] " + ", ".join(f"{m}={entry[m]:.4f}" for m in config.METRICS) + lat_txt)

    if latency and {"hybrid", "rerank"} <= set(arms):
        h, r = results["arms"]["hybrid"]["latency"], results["arms"]["rerank"]["latency"]
        results["rerank_latency_multiplier_vs_hybrid"] = {"p50": round(r["p50_ms"] / h["p50_ms"], 1), "p95": round(r["p95_ms"] / h["p95_ms"], 1)}

    best = max(arms, key=lambda a: results["arms"][a]["ndcg@10"])
    results["best_arm"] = best
    if n_errors:
        dump(ds, split, best, per_query_all[best], runs[best], out_dir, n=n_errors)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    (out_dir / "table.md").write_text(table_md(results), encoding="utf-8")
    return results
