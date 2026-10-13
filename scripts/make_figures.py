"""Render docs/results.png (quality + latency per arm) from results/results.json, and docs/gate_demo.png from the
saved gate logs. Every number drawn comes from files in results/.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, DOCS = ROOT / "results", ROOT / "docs"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = "#2a78d6"  # categorical slot 1; one series per panel, so no legend box


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def results_png() -> None:
    r = json.loads((RES / "results.json").read_text())
    arms = list(r["arms"])[::-1]
    ndcg = [r["arms"][a]["ndcg@10"] for a in arms]
    p50 = [r["arms"][a]["latency"]["p50_ms"] for a in arms]
    p95 = [r["arms"][a]["latency"]["p95_ms"] for a in arms]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    style(a1, "nDCG@10 on SciFact test (300 queries)")
    a1.barh(arms, ndcg, color=SERIES, height=0.5)
    a1.set_xlim(0, 0.8)  # bars start at zero so differences are not exaggerated
    for y, v in enumerate(ndcg):
        a1.text(v + 0.002, y, f"{v:.4f}", va="center", color=INK, fontsize=9)

    style(a2, "Per-query latency, ms (bar = p50, tick = p95, log scale)")
    a2.barh(arms, p50, color=SERIES, height=0.5)
    a2.scatter(p95, range(len(arms)), marker="|", s=300, color=INK, linewidths=2)
    a2.set_xscale("log")
    for y, (m, h) in enumerate(zip(p50, p95)):
        a2.text(h * 1.15, y, f"{m:g} / {h:g}", va="center", color=INK, fontsize=9)
    a2.set_xlim(1, max(p95) * 6)
    env = r["environment"]
    fig.text(0.01, 0.01, f"{env.get('gpu', env['device'])}, torch {env['torch']}; BM25 on CPU. Source: results/results.json", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(DOCS / "results.png", dpi=150, facecolor=SURFACE)


def gate_lines(path: Path) -> list[str]:
    """Branch header + the gate command and its output (the run output above it is in the same log file)."""
    lines = path.read_text().strip().splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith("$ python -m ragbench gate"))
    return [lines[0], *lines[i:]]


def gate_png() -> None:
    logs = [RES / "gate_main.log", RES / "gate_demo_truncate128" / "gate.log", RES / "gate_demo_drop_query_prefix" / "gate.log"]
    lines = []
    for path in logs:
        lines += [*gate_lines(path), ""]
    fig = plt.figure(figsize=(11, 0.24 * len(lines) + 0.3), facecolor="#1a1a19")
    for i, line in enumerate(lines):
        color = "#9fd8a8" if "PASS" in line else "#f2a3a3" if "FAIL" in line else "#f0efec" if line.startswith(("$", "branch")) else "#c3c2b7"
        fig.text(0.01, 1 - (i + 1) / (len(lines) + 1), line, family="monospace", fontsize=9, color=color, va="center")
    fig.savefig(DOCS / "gate_demo.png", dpi=150, facecolor="#1a1a19")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    results_png()
    gate_png()
    print("wrote docs/results.png and docs/gate_demo.png")
