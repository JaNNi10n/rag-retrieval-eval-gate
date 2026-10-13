"""Precision check for the CI gate: run BM25 + dense with bge-small in fp32 (as on the CPU-only CI runner) and
compare against the committed fp16 baseline.

A full CPU encode of the corpus was too slow on this laptop while other jobs held the CPU, so the fp32 model is run
on the GPU instead: same weights and same fp32 arithmetic as the CPU path, only faster. Writes results/fp32_dense/.

    python scripts/fp32_dense_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragbench.dense import BgeEncoder  # noqa: E402
from ragbench.pipeline import run  # noqa: E402

if __name__ == "__main__":
    enc = BgeEncoder()
    enc.model.float()  # undo the fp16 cast applied on CUDA
    enc.fp16 = False
    enc.name = enc.name.replace("-fp16", "-fp32")  # separate on-disk cache from the fp16 index
    out = ROOT / "results" / "fp32_dense"
    res = run(["bm25", "dense"], "test", out, out / "runs", encoder=enc, device=enc.device, latency=False, n_errors=0)
    print({a: round(r["ndcg@10"], 4) for a, r in res["arms"].items()})
