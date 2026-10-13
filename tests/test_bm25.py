import pytest

from ragbench.bm25 import BM25Index


def test_exact_match_doc_ranks_first(toy_ds):
    idx = BM25Index(toy_ds.doc_ids, toy_ds.texts)
    for query, expected in [
        ("Tobacco smoking is the leading cause of lung cancer worldwide.", "d3"),
        ("Statins and cholesterol", "d5"),
        ("Vitamin D and bone density", "d2"),
    ]:
        assert idx.search(query, k=5)[0][0] == expected


def test_stemming_stopwords_and_edge_cases(toy_ds):
    idx = BM25Index(toy_ds.doc_ids, toy_ds.texts)
    assert idx.search("exercising", k=5)[0][0] == "d4"  # Snowball stems both to "exercis"
    assert idx.search("the and of", k=5) == []  # stopwords only -> no matches
    assert idx.search("zzzunknown", k=5) == []
    assert len(idx.search_batch(["aspirin", "statins"], k=100)) == 2  # k is clamped to corpus size
    assert idx.search_batch([]) == []


def test_length_mismatch():
    with pytest.raises(ValueError):
        BM25Index(["a"], ["x", "y"])
