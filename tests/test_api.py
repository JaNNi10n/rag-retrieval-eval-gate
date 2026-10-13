import pytest
from fastapi.testclient import TestClient

from ragbench.api import SearchResponse, create_app
from ragbench.bm25 import BM25Index
from ragbench.dense import DenseIndex
from ragbench.search import Searcher

from .conftest import FakeEncoder, FakeScorer


@pytest.fixture
def client(toy_ds, tmp_path):
    def factory():
        bm25 = BM25Index(toy_ds.doc_ids, toy_ds.texts)
        dense = DenseIndex.build_or_load(FakeEncoder(), toy_ds.doc_ids, toy_ds.texts, tmp_path)
        return Searcher(toy_ds.docs, toy_ds.titles, bm25, dense, FakeScorer(), rrf_k=10)

    with TestClient(create_app(factory)) as c:
        yield c


@pytest.mark.parametrize("arm", ["bm25", "dense", "hybrid", "rerank"])
def test_search_ok(client, arm):
    r = client.get("/search", params={"q": "aspirin heart attack", "arm": arm, "k": 3})
    assert r.status_code == 200
    body = SearchResponse.model_validate(r.json())
    assert body.arm == arm and 1 <= len(body.results) <= 3 and body.latency_ms >= 0
    assert [h.rank for h in body.results] == list(range(1, len(body.results) + 1))
    assert body.results[0].doc_id == "d1" and body.results[0].title == "Aspirin and heart attack"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["arms"] == ["bm25", "dense", "hybrid", "rerank"]
    assert r.json()["n_docs"] == 5 and r.json()["rrf_k"] == 10


@pytest.mark.parametrize(
    "params",
    [{"q": ""}, {"q": "   "}, {"q": "aspirin", "k": 0}, {"q": "aspirin", "k": 101}, {"q": "aspirin", "arm": "magic"}, {}],
)
def test_validation_422(client, params):
    assert client.get("/search", params=params).status_code == 422


def test_unloaded_arm_503(toy_ds):
    app = create_app(lambda: Searcher(toy_ds.docs, toy_ds.titles, BM25Index(toy_ds.doc_ids, toy_ds.texts)))
    with TestClient(app) as c:
        assert c.get("/search", params={"q": "aspirin", "arm": "dense"}).status_code == 503
        assert c.get("/search", params={"q": "aspirin", "arm": "bm25"}).status_code == 200


def test_searcher_rejects_unloaded_arm(toy_ds):
    s = Searcher(toy_ds.docs, toy_ds.titles, BM25Index(toy_ds.doc_ids, toy_ds.texts))
    with pytest.raises(ValueError):
        s.search("aspirin", "rerank")
