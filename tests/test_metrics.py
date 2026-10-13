import math

import pytest

from ragbench.evaluate import mean_scores, per_query_scores, significance, to_run


def test_ndcg_matches_hand_computation():
    # One query, 3 docs, two relevant (binary). Ranking: a (rel), b (non-rel), c (rel).
    qrels = {"q": {"a": 1, "c": 1}}
    hits = {"q": [("a", 3.0), ("b", 2.0), ("c", 1.0)]}
    dcg = 1 / math.log2(2) + 1 / math.log2(4)  # gains at ranks 1 and 3
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # ideal: both relevant at ranks 1 and 2
    scores = mean_scores(hits, qrels)
    assert scores["ndcg@10"] == pytest.approx(dcg / idcg, abs=1e-9)
    assert scores["ndcg@10"] == pytest.approx(0.9197, abs=1e-4)
    assert scores["mrr@10"] == pytest.approx(1.0)
    assert scores["recall@100"] == pytest.approx(1.0)


def test_missing_query_scores_zero_and_list_order_wins_over_tied_scores():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    hits = {"q1": [("x", 5.0), ("a", 5.0)]}  # tied scores: our list order must be what gets evaluated
    pq = per_query_scores(hits, qrels)
    assert pq["q2"] == {"ndcg@10": 0.0, "recall@100": 0.0, "mrr@10": 0.0}
    assert pq["q1"]["mrr@10"] == pytest.approx(0.5)
    assert "q" not in to_run({"q": [], "r": [("a", 1.0)]}).keys()


def test_single_metric_mean():
    assert mean_scores({"q": [("a", 1.0)]}, {"q": {"a": 1}}, ["ndcg@10"]) == {"ndcg@10": 1.0}


def test_significance_returns_both_tests():
    qrels = {f"q{i}": {"good": 1} for i in range(20)}
    base = {q: [("bad", 2.0), ("good", 1.0)] for q in qrels}
    better = {q: [("good", 2.0)] for q in qrels}
    sig = significance({"bm25": base, "dense": better}, qrels, n_permutations=500)
    assert set(sig["dense"]) == {"p_student", "p_fisher"}
    assert sig["dense"]["p_fisher"] < 0.05
