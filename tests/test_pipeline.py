import json

import numpy as np
import pytest

from ragbench.data import read_qrels
from ragbench.dense import DenseIndex, l2_normalise
from ragbench.latency import measure
from ragbench.pipeline import fmt_p, run
from ragbench.rerank import rerank

from .conftest import FakeEncoder, FakeScorer


def test_full_pipeline_on_fixture(toy_ds, tmp_path):
    res = run(
        ["bm25", "dense", "hybrid", "rerank"],
        "test",
        tmp_path / "res",
        tmp_path / "runs",
        ds=toy_ds,
        encoder=FakeEncoder(),
        scorer=FakeScorer(),
        index_dir=tmp_path / "idx",
        warmup=1,
        n_errors=2,
        log=lambda *_: None,
    )
    assert set(res["arms"]) == {"bm25", "dense", "hybrid", "rerank"}
    assert res["rrf_k"] in (10, 30, 60, 100)
    for arm, r in res["arms"].items():
        assert 0 <= r["ndcg@10"] <= 1 and r["latency"]["n"] == 3
        assert json.loads((tmp_path / "runs" / f"{arm}.json").read_text())["split"] == "test"
    assert "p_student" in res["arms"]["dense"] and "vs_dense" in res["arms"]["hybrid"]
    assert "rerank_latency_multiplier_vs_hybrid" in res
    assert (tmp_path / "res" / "table.md").read_text().startswith("| Arm")
    assert len(json.loads((tmp_path / "res" / "errors.json").read_text())["queries"]) == 2
    assert json.loads((tmp_path / "res" / "rrf_tuning.json").read_text())["tuned_on"] == "train"
    with pytest.raises(ValueError):
        run(["nope"], ds=toy_ds)


def test_pipeline_bm25_only_no_latency(toy_ds, tmp_path):
    res = run(["bm25"], "test", tmp_path, tmp_path / "runs", ds=toy_ds, latency=False, n_errors=0, log=lambda *_: None)
    assert "latency" not in res["arms"]["bm25"] and res["best_arm"] == "bm25"


def test_dense_cache_reused(toy_ds, tmp_path):
    a = DenseIndex.build_or_load(FakeEncoder(), toy_ds.doc_ids, toy_ds.texts, tmp_path)
    stem = DenseIndex.cache_stem(FakeEncoder(), toy_ds.doc_ids, tmp_path)
    assert (tmp_path / f"{stem.name}.faiss").exists() and (tmp_path / f"{stem.name}.npy").exists()
    # The model name's dots must survive in the file name (regression: with_suffix truncated "v1.5-fp16-<hash>").
    dotted = FakeEncoder()
    dotted.name = "fake-v1.5-fp16"
    DenseIndex.build_or_load(dotted, toy_ds.doc_ids, toy_ds.texts, tmp_path / "dotted")
    assert sorted(p.name.split("-")[1] for p in (tmp_path / "dotted").iterdir()) == ["v1.5"] * 3
    b = DenseIndex.build_or_load(FakeEncoder(), toy_ds.doc_ids, toy_ds.texts, tmp_path)
    assert np.allclose(a.embeddings, b.embeddings)
    assert b.search("statins cholesterol", k=2)[0][0] == "d5"
    assert b.search_batch([]) == []
    with pytest.raises(ValueError):
        DenseIndex(FakeEncoder(), ["a"], np.zeros((2, 4), dtype=np.float32))


def test_helpers(tmp_path):
    assert np.allclose(np.linalg.norm(l2_normalise(np.array([[3.0, 4.0]])), axis=1), 1.0)
    assert rerank(FakeScorer(), "q", [], {}) == []
    stats = measure(lambda q: None, ["a", "b", "c"], warmup=2)
    assert stats["n"] == 3 and stats["warmup"] == 2 and stats["p50_ms"] >= 0
    with pytest.raises(ValueError):
        measure(lambda q: None, [])
    assert fmt_p(0.0) == "<1e-04 *" and fmt_p(0.3) == "0.3"
    (tmp_path / "q.tsv").write_text("query-id\tcorpus-id\tscore\n1\t10\t1\n1\t11\t1\n2\t10\t1\n")
    assert read_qrels(tmp_path / "q.tsv") == {"1": {"10": 1, "11": 1}, "2": {"10": 1}}
