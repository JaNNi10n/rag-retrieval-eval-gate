"""Where does single-query dense latency go? Times 100 SciFact test queries (after 10 warm-up) through
(a) the raw Hugging Face forward pass and (b) the full BgeEncoder.encode used by /search, on the current device.
If (a) is already slow, the cost is the GPU itself (e.g. shared with other processes), not ragbench code.

    python scripts/profile_dense_query.py > results/dense_query_profile.log
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragbench.data import load_scifact  # noqa: E402
from ragbench.dense import BgeEncoder  # noqa: E402


def timed(fn, queries, warmup=10):
    for q in queries[:warmup]:
        fn(q)
    xs = []
    for q in queries:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn(q)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        xs.append((time.perf_counter() - t) * 1000)
    return f"p50 {np.percentile(xs, 50):.2f} ms, p95 {np.percentile(xs, 95):.2f} ms (n={len(xs)})"


if __name__ == "__main__":
    queries = list(load_scifact().split_queries("test").values())[:100]
    enc = BgeEncoder()
    hf, tok = enc.model[0].auto_model, enc.model.tokenizer

    def forward(q):
        feats = tok([enc.query_prefix + q], return_tensors="pt").to(enc.device)
        with torch.inference_mode():
            return hf(**feats).last_hidden_state[:, 0]

    print(f"device: {enc.device} ({torch.cuda.get_device_name() if enc.device.startswith('cuda') else 'cpu'}), fp16={enc.fp16}")
    print("raw HF forward pass   :", timed(forward, queries))
    print("BgeEncoder.encode([q]):", timed(lambda q: enc.encode([q], is_query=True), queries))
    try:
        smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        print("other GPU compute processes at the time:\n" + smi.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
