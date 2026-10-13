import hashlib
import re

import numpy as np
import pytest

from ragbench.data import Dataset

DOCS = {
    "d1": ("Aspirin and heart attack", "Daily aspirin reduces the risk of myocardial infarction in adults."),
    "d2": ("Vitamin D and bone density", "Vitamin D supplementation improves bone mineral density in elderly women."),
    "d3": ("Smoking and lung cancer", "Tobacco smoking is the leading cause of lung cancer worldwide."),
    "d4": ("Exercise and depression", "Regular aerobic exercise lowers symptoms of depression."),
    "d5": ("Statins and cholesterol", "Statins lower LDL cholesterol and cardiovascular events."),
}


class FakeEncoder:
    """Deterministic hashed bag-of-words encoder: lets the dense/hybrid code run without downloading a model."""

    name = "fake-hash-enc"
    dim = 64

    def encode(self, texts, is_query):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


class FakeScorer:
    """Word-overlap stand-in for the cross-encoder."""

    def score(self, query, texts):
        q = set(re.findall(r"\w+", query.lower()))
        return [float(len(q & set(re.findall(r"\w+", t.lower())))) for t in texts]


@pytest.fixture
def toy_ds():
    ids = list(DOCS)
    titles = {d: t for d, (t, _) in DOCS.items()}
    docs = {d: f"{t} {x}" for d, (t, x) in DOCS.items()}
    queries = {
        "q1": "does aspirin prevent heart attack",
        "q2": "vitamin D bone density",
        "q3": "smoking causes lung cancer",
        "q4": "exercise depression",
        "q5": "statins cholesterol",
        "q6": "aerobic exercise mood",
    }
    qrels = {
        "test": {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}},
        "train": {"q4": {"d4": 1}, "q5": {"d5": 1}, "q6": {"d4": 1}},
    }
    return Dataset(doc_ids=ids, titles=titles, docs=docs, queries=queries, qrels=qrels)
