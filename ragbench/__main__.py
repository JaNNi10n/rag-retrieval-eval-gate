"""CLI: `python -m ragbench run|gate|baseline`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config


def _arms(s: str) -> list[str]:
    arms = [a.strip() for a in s.split(",") if a.strip()]
    bad = [a for a in arms if a not in config.ARMS]
    if bad or not arms:
        raise argparse.ArgumentTypeError(f"arms must be a comma list of {config.ARMS}, got {s!r}")
    return arms


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ragbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="retrieve + evaluate + time the selected arms")
    r.add_argument("--arms", type=_arms, default=list(config.ARMS))
    r.add_argument("--split", choices=["test", "train"], default="test")
    r.add_argument("--out", type=Path, default=config.RESULTS_DIR, help="where results.json/table.md go")
    r.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    r.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    r.add_argument("--no-latency", action="store_true", help="skip the per-query latency benchmark")

    g = sub.add_parser("gate", help="fail (exit 1) if nDCG@10 regresses beyond --tol")
    g.add_argument("--baseline", type=Path, required=True)
    g.add_argument("--results", type=Path, default=config.RESULTS_DIR / "results.json")
    g.add_argument("--tol", type=float, default=0.02)
    g.add_argument("--arms", type=_arms, default=None, help="arms to gate (default: all arms in the baseline)")

    b = sub.add_parser("baseline", help="freeze nDCG@10 from a results.json into a baseline file")
    b.add_argument("--results", type=Path, default=config.RESULTS_DIR / "results.json")
    b.add_argument("--out", type=Path, default=config.RESULTS_DIR / "baseline.json")

    args = p.parse_args(argv)
    if args.cmd == "run":
        from .pipeline import run

        res = run(args.arms, args.split, args.out, args.runs_dir, device=args.device, latency=not args.no_latency)
        print((Path(args.out) / "table.md").read_text(encoding="utf-8"))
        print(f"best arm: {res['best_arm']}")
        return 0
    if args.cmd == "gate":
        from .gate import check_files

        ok, lines = check_files(args.results, args.baseline, args.tol, args.arms)
        print("\n".join(lines))
        print("GATE PASSED" if ok else "GATE FAILED")
        return 0 if ok else 1
    from .gate import make_baseline

    base = make_baseline(json.loads(args.results.read_text()))
    args.out.write_text(json.dumps(base, indent=1) + "\n")
    print(json.dumps(base, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
