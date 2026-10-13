"""Per-query wall-clock latency (batch size 1, as an online request would be)."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np


def measure(fn: Callable[[str], object], queries: Sequence[str], warmup: int = 10) -> dict[str, float]:
    """Run `warmup` untimed queries, then time every query once. Returns p50/p95/mean in ms."""
    if not queries:
        raise ValueError("need at least one query")
    for q in queries[:warmup]:
        fn(q)
    times = []
    for q in queries:
        t0 = time.perf_counter()
        fn(q)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(times)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "n": len(arr),
        "warmup": min(warmup, len(queries)),
    }
