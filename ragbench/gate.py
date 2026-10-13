"""Regression gate: fail when an arm's nDCG@10 drops more than `tol` below the committed baseline."""

from __future__ import annotations

import json
from pathlib import Path

METRIC = "ndcg@10"


def make_baseline(results: dict, arms: list[str] | None = None) -> dict:
    arms = arms or list(results["arms"])
    return {"metric": METRIC, "split": results.get("split"), "arms": {a: round(results["arms"][a][METRIC], 4) for a in arms}}


def check(results: dict, baseline: dict, tol: float, arms: list[str] | None = None) -> tuple[bool, list[str]]:
    """Return (passed, report lines). Arms default to every arm in the baseline; a missing arm fails."""
    if tol < 0:
        raise ValueError("tol must be >= 0")
    metric = baseline.get("metric", METRIC)
    arms = arms or list(baseline["arms"])
    ok, lines = True, []
    for arm in arms:
        if arm not in baseline["arms"]:
            ok = False
            lines.append(f"FAIL {arm}: no baseline value")
            continue
        base = baseline["arms"][arm]
        if arm not in results.get("arms", {}):
            ok = False
            lines.append(f"FAIL {arm}: not present in results (baseline {base:.4f})")
            continue
        cur = results["arms"][arm][metric]
        delta = cur - base
        failed = delta < -tol - 1e-9
        ok &= not failed
        lines.append(f"{'FAIL' if failed else 'PASS'} {arm}: {metric} {cur:.4f} vs baseline {base:.4f} (delta {delta:+.4f}, tol {tol})")
    return ok, lines


def check_files(results_path: Path, baseline_path: Path, tol: float, arms: list[str] | None = None) -> tuple[bool, list[str]]:
    return check(json.loads(Path(results_path).read_text()), json.loads(Path(baseline_path).read_text()), tol, arms)
