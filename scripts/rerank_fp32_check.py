"""Ablation: does running the cross-encoder in fp16 explain why re-ranking does not beat hybrid on SciFact?

Re-runs the bm25 / hybrid / rerank arms on the test split with the cross-encoder kept in fp32 on the same GPU and
writes the metrics to results/rerank_fp32/ (no latency, no error dump).

    python scripts/rerank_fp32_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragbench.pipeline import run  # noqa: E402
from ragbench.rerank import CrossEncoderScorer  # noqa: E402

if __name__ == "__main__":
    scorer = CrossEncoderScorer()
    scorer.model.model.float()  # undo the fp16 cast that CrossEncoderScorer applies on CUDA
    out = ROOT / "results" / "rerank_fp32"
    res = run(["bm25", "hybrid", "rerank"], "test", out, out / "runs", scorer=scorer, latency=False, n_errors=0)
    print({a: round(r["ndcg@10"], 4) for a, r in res["arms"].items()})
