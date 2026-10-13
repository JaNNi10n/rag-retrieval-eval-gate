import pytest

from ragbench.hybrid import fuse_runs, rrf, tune_k


def test_rrf_expected_order():
    a = [("d1", 9.0), ("d2", 8.0), ("d3", 7.0)]
    b = [("d2", 0.9), ("d3", 0.8), ("d1", 0.7)]
    fused = rrf([a, b], k=60)
    # d2: 1/62 + 1/61, d1: 1/61 + 1/63, d3: 1/63 + 1/62
    assert [d for d, _ in fused] == ["d2", "d1", "d3"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_doc_missing_from_one_list():
    fused = dict(rrf([[("d1", 1.0), ("d2", 0.5)], [("d3", 1.0)]], k=10))
    assert fused["d1"] == pytest.approx(1 / 11)
    assert fused["d2"] == pytest.approx(1 / 12)
    assert fused["d3"] == pytest.approx(1 / 11)


def test_rrf_ties_are_deterministic():
    # Tied input scores are ranked by doc_id, so "a" is rank 1 in both lists.
    fused = rrf([[("b", 1.0), ("a", 1.0)], [("a", 1.0), ("b", 1.0)]], k=0)
    assert [d for d, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(2.0)
    # Tied fused scores are ordered by doc_id.
    sym = rrf([[("y", 1.0)], [("x", 1.0)]], k=60)
    assert [d for d, _ in sym] == ["x", "y"] and sym[0][1] == sym[1][1]


def test_rrf_top_n_and_validation():
    assert len(rrf([[(str(i), float(-i)) for i in range(50)]], k=60, top_n=10)) == 10
    assert rrf([], k=60) == []
    with pytest.raises(ValueError):
        rrf([[("a", 1.0)]], k=-1)


def test_fuse_runs_and_tune_k():
    runs = [{"q": [("x", 1.0), ("g", 0.5)]}, {"q": [("g", 1.0)], "q2": [("z", 1.0)]}]
    fused = fuse_runs(runs, 10)
    assert fused["q"][0][0] == "g" and fused["q2"] == [("z", pytest.approx(1 / 11))]
    best, scores = tune_k(runs, {"q": {"g": 1}}, grid=(10, 60))
    assert set(scores) == {10, 60} and best == 10  # equal scores -> first k in the grid
