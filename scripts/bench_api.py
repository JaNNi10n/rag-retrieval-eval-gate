"""End-to-end /search latency through a real uvicorn server (HTTP round-trip, not in-process).

Starts `uvicorn ragbench.api:app`, waits for /health, then for each arm sends 10 warm-up queries
followed by all 300 SciFact test queries one at a time (arms interleaved per query), and writes p50/p95 of both the client-side
round-trip and the server-reported `latency_ms` to results/api_latency.json.

    python scripts/bench_api.py --arms bm25,dense,hybrid --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragbench.data import load_scifact  # noqa: E402


def pct(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs)
    return {"p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2), "mean_ms": round(float(a.mean()), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="bm25,dense,hybrid,rerank")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "api_latency.json")
    args = ap.parse_args()
    arms = args.arms.split(",")

    ds = load_scifact()
    queries = list(ds.split_queries("test").values())
    base = f"http://127.0.0.1:{args.port}"
    env = {**os.environ, "RAGBENCH_ARMS": args.arms}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "ragbench.api:app", "--port", str(args.port), "--log-level", "warning"], cwd=ROOT, env=env)
    try:
        with httpx.Client(base_url=base, timeout=60) as client:
            t0 = time.time()
            while True:
                try:
                    health = client.get("/health").json()
                    break
                except httpx.TransportError:
                    if server.poll() is not None or time.time() - t0 > 600:
                        raise RuntimeError("server did not start") from None
                    time.sleep(1)
            out = {"endpoint": "GET /search", "k": args.k, "n_queries": len(queries), "warmup": args.warmup, "health": health, "arms": {}}
            for arm in arms:
                for q in queries[: args.warmup]:
                    client.get("/search", params={"q": q, "arm": arm, "k": args.k}).raise_for_status()
            rtt = {a: [] for a in arms}
            server_ms = {a: [] for a in arms}
            # Interleaved: each query hits every arm, in an order that rotates per query, so load from other
            # processes on a shared GPU affects all arms in the same time window.
            for i, q in enumerate(queries):
                for j in range(len(arms)):
                    arm = arms[(i + j) % len(arms)]
                    t = time.perf_counter()
                    r = client.get("/search", params={"q": q, "arm": arm, "k": args.k})
                    rtt[arm].append((time.perf_counter() - t) * 1000)
                    r.raise_for_status()
                    server_ms[arm].append(r.json()["latency_ms"])
            for arm in arms:
                out["arms"][arm] = {"round_trip": pct(rtt[arm]), "server_latency_ms": pct(server_ms[arm])}
                print(arm, out["arms"][arm], flush=True)
            sample = client.get("/search", params={"q": queries[0], "arm": "hybrid" if "hybrid" in arms else arms[0], "k": 3}).json()
            out["sample_response"] = sample
    finally:
        server.terminate()
        server.wait(timeout=30)
    args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
