"""Per-query wall-clock latency (batch size 1, as an online request would be)."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np


def summarise(times_ms: Sequence[float], warmup: int) -> dict[str, float]:
    arr = np.asarray(times_ms)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "n": len(arr),
        "warmup": warmup,
    }


def measure(fn: Callable[[str], object], queries: Sequence[str], warmup: int = 10) -> dict[str, float]:
    """Run `warmup` untimed queries, then time every query once. Returns p50/p95/mean in ms."""
    return measure_interleaved({"_": fn}, queries, warmup)["_"]


def measure_interleaved(fns: Mapping[str, Callable[[str], object]], queries: Sequence[str], warmup: int = 10) -> dict[str, dict[str, float]]:
    """Time several arms on the same queries, interleaved: for each query every arm runs once, in an order that rotates
    per query. On a shared machine, load that comes and goes then hits all arms in the same time window instead of
    whichever arm happened to be measured during a busy minute."""
    if not queries:
        raise ValueError("need at least one query")
    names = list(fns)
    for q in queries[:warmup]:
        for name in names:
            fns[name](q)
    times: dict[str, list[float]] = {name: [] for name in names}
    for i, q in enumerate(queries):
        for j in range(len(names)):
            name = names[(i + j) % len(names)]
            t0 = time.perf_counter()
            fns[name](q)
            times[name].append((time.perf_counter() - t0) * 1000)
    return {name: summarise(ts, min(warmup, len(queries))) for name, ts in times.items()}
