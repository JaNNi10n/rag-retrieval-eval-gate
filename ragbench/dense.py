"""Dense retriever: bge-small-en-v1.5 embeddings in an exact inner-product FAISS index, cached on disk."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from . import config

Hits = list[tuple[str, float]]


class Encoder(Protocol):
    name: str

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        """Return float32, L2-normalised embeddings of shape (len(texts), dim)."""


def l2_normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class BgeEncoder:
    """sentence-transformers wrapper. bge-v1.5 wants an instruction prefix on queries only."""

    def __init__(
        self,
        model_name: str = config.DENSE_MODEL,
        revision: str | None = config.DENSE_REVISION,
        device: str | None = None,
        query_prefix: str = config.BGE_QUERY_PREFIX,
        batch_size: int | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.device = device or config.device()
        self.model = SentenceTransformer(model_name, revision=revision, device=self.device)
        self.fp16 = self.device.startswith("cuda")
        if self.fp16:
            self.model.half()
        self.query_prefix = query_prefix
        # Smaller CPU batches keep peak RAM low (fp32 attention over 512-token docs is memory hungry).
        self.batch_size = batch_size or (64 if self.fp16 else 16)
        # Everything that changes document vectors goes into the name, which keys the on-disk index cache.
        self.name = f"{model_name.split('/')[-1]}-{'fp16' if self.fp16 else 'fp32'}-len{self.model.max_seq_length}"

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        if is_query:
            texts = [self.query_prefix + t for t in texts]
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )
        return l2_normalise(emb)  # re-normalise in fp32 after the fp16 forward pass


class DenseIndex:
    def __init__(self, encoder: Encoder, doc_ids: list[str], embeddings: np.ndarray):
        if embeddings.shape[0] != len(doc_ids):
            raise ValueError(f"{embeddings.shape[0]} embeddings for {len(doc_ids)} docs")
        self.encoder = encoder
        self.doc_ids = list(doc_ids)
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])  # exact search; cosine since vectors are unit-norm
        self.index.add(self.embeddings)

    @staticmethod
    def cache_stem(encoder: Encoder, doc_ids: list[str], texts: list[str], index_dir: Path) -> Path:
        # Hash ids AND texts: a change to what gets indexed (e.g. dropping titles) must not reuse stale embeddings.
        h = hashlib.sha1()
        for d, t in zip(doc_ids, texts):
            h.update(f"{d}\t{t}\n".encode())
        key = h.hexdigest()[:12]
        return Path(index_dir) / f"{encoder.name}-{key}"

    @classmethod
    def build_or_load(cls, encoder: Encoder, doc_ids: list[str], texts: list[str], index_dir: Path | None = None) -> DenseIndex:
        """Reuse <stem>.faiss/.npy when they exist for this exact model + corpus text, otherwise encode and save."""
        stem = cls.cache_stem(encoder, doc_ids, texts, Path(index_dir or config.INDEX_DIR))
        # Append (not with_suffix): the stem contains dots ("bge-small-en-v1.5-...") that with_suffix would eat.
        faiss_path, npy_path, ids_path = (stem.parent / f"{stem.name}{s}" for s in (".faiss", ".npy", ".ids.json"))
        if faiss_path.exists() and npy_path.exists() and ids_path.exists():
            if json.loads(ids_path.read_text()) == list(doc_ids):
                obj = cls.__new__(cls)
                obj.encoder, obj.doc_ids = encoder, list(doc_ids)
                obj.embeddings = np.load(npy_path)
                obj.index = faiss.read_index(str(faiss_path))
                if obj.index.ntotal == len(doc_ids) == obj.embeddings.shape[0]:
                    return obj
        obj = cls(encoder, doc_ids, encoder.encode(texts, is_query=False))
        stem.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(obj.index, str(faiss_path))
        np.save(npy_path, obj.embeddings)
        ids_path.write_text(json.dumps(list(doc_ids)))
        return obj

    def search_batch(self, queries: list[str], k: int = config.CANDIDATES) -> list[Hits]:
        if not queries:
            return []
        scores, idx = self.index.search(self.encoder.encode(queries, is_query=True), min(k, len(self.doc_ids)))
        return [[(self.doc_ids[i], float(s)) for i, s in zip(row_i, row_s) if i >= 0] for row_i, row_s in zip(idx, scores)]

    def search(self, query: str, k: int = config.CANDIDATES) -> Hits:
        return self.search_batch([query], k)[0]
