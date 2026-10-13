"""Sparse BM25 retriever: lowercase + English stopwords + Snowball stemming, Lucene BM25 (k1=0.9, b=0.4)."""

from __future__ import annotations

import bm25s
import Stemmer

from . import config

Hits = list[tuple[str, float]]


class BM25Index:
    def __init__(self, doc_ids: list[str], texts: list[str], k1: float = config.BM25_K1, b: float = config.BM25_B):
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        self.doc_ids = list(doc_ids)
        self.stemmer = Stemmer.Stemmer("english")
        self.model = bm25s.BM25(k1=k1, b=b, method="lucene")
        self.model.index(self._tokenize(texts, return_ids=True), show_progress=False)

    def _tokenize(self, texts: list[str], return_ids: bool):
        # bm25s lowercases by default; "en" is its built-in English stopword list.
        return bm25s.tokenize(texts, lower=True, stopwords="en", stemmer=self.stemmer, return_ids=return_ids, show_progress=False)

    def search_batch(self, queries: list[str], k: int = config.CANDIDATES) -> list[Hits]:
        if not queries:
            return []
        k = min(k, len(self.doc_ids))
        idx, scores = self.model.retrieve(self._tokenize(queries, return_ids=False), k=k, show_progress=False, n_threads=1)
        # Docs with score 0 share no term with the query; they are padding, not matches.
        return [[(self.doc_ids[i], float(s)) for i, s in zip(row_idx, row_scores) if s > 0] for row_idx, row_scores in zip(idx, scores)]

    def search(self, query: str, k: int = config.CANDIDATES) -> Hits:
        return self.search_batch([query], k)[0]
