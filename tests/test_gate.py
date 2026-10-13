import json

import pytest

from ragbench.__main__ import main
from ragbench.gate import check, make_baseline

BASE = {"metric": "ndcg@10", "arms": {"bm25": 0.68, "dense": 0.71}}


def results(bm25, dense):
    return {"arms": {"bm25": {"ndcg@10": bm25}, "dense": {"ndcg@10": dense}}}


def test_gate_fails_on_003_drop():
    ok, lines = check(results(0.68, 0.71 - 0.03), BASE, tol=0.02)
    assert not ok
    assert any(line.startswith("FAIL dense") for line in lines)
    assert any(line.startswith("PASS bm25") for line in lines)


def test_gate_passes_on_001_drop():
    ok, lines = check(results(0.68 - 0.01, 0.71 - 0.01), BASE, tol=0.02)
    assert ok and all(line.startswith("PASS") for line in lines)


def test_gate_missing_and_unknown_arms():
    ok, lines = check({"arms": {"bm25": {"ndcg@10": 0.68}}}, BASE, tol=0.02)
    assert not ok and "not present" in lines[-1]
    ok, _ = check(results(0.68, 0.71), BASE, tol=0.02, arms=["rerank"])
    assert not ok
    ok, _ = check({"arms": {"bm25": {"ndcg@10": 0.68}}}, BASE, tol=0.02, arms=["bm25"])
    assert ok
    with pytest.raises(ValueError):
        check(results(0.68, 0.71), BASE, tol=-1)


def test_gate_cli_exit_codes(tmp_path, capsys):
    (tmp_path / "base.json").write_text(json.dumps(BASE))
    (tmp_path / "bad.json").write_text(json.dumps(results(0.68, 0.68)))
    (tmp_path / "good.json").write_text(json.dumps(results(0.675, 0.70)))
    base = str(tmp_path / "base.json")
    assert main(["gate", "--baseline", base, "--results", str(tmp_path / "bad.json"), "--tol", "0.02"]) == 1
    assert "GATE FAILED" in capsys.readouterr().out
    assert main(["gate", "--baseline", base, "--results", str(tmp_path / "good.json"), "--tol", "0.02"]) == 0
    assert "GATE PASSED" in capsys.readouterr().out
    assert main(["baseline", "--results", str(tmp_path / "good.json"), "--out", str(tmp_path / "nb.json")]) == 0
    assert json.loads((tmp_path / "nb.json").read_text())["arms"] == {"bm25": 0.675, "dense": 0.7}
    assert make_baseline(results(0.5, 0.6), ["dense"])["arms"] == {"dense": 0.6}
    with pytest.raises(SystemExit):
        main(["run", "--arms", "bogus"])
