"""Load BEIR SciFact (corpus, queries, train/test qrels) from the Hugging Face Hub with a local cache."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Dataset:
    doc_ids: list[str]
    titles: dict[str, str]
    docs: dict[str, str]  # doc_id -> "title text" (what gets indexed)
    queries: dict[str, str]  # all queries (train + test)
    qrels: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)  # split -> qid -> {doc_id: rel}

    def split_queries(self, split: str) -> dict[str, str]:
        """Queries that have judgements in `split`, in qrels order."""
        return {qid: self.queries[qid] for qid in self.qrels[split]}

    @property
    def texts(self) -> list[str]:
        return [self.docs[d] for d in self.doc_ids]


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header: query-id corpus-id score
        for qid, doc_id, score in reader:
            qrels.setdefault(qid, {})[doc_id] = int(score)
    return qrels


def _download(repo: str, filename: str, revision: str, cache_dir: Path) -> Path:
    local_dir = cache_dir / repo.split("/")[-1]
    target = local_dir / filename
    if target.exists():
        return target
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision, local_dir=local_dir))


def load_scifact(cache_dir: Path | None = None, validate: bool = True) -> Dataset:
    import pandas as pd

    cache_dir = Path(cache_dir or config.DATA_DIR)
    corpus = pd.read_parquet(_download(config.CORPUS_REPO, "corpus/corpus-00000-of-00001.parquet", config.CORPUS_REVISION, cache_dir))
    queries = pd.read_parquet(_download(config.CORPUS_REPO, "queries/queries-00000-of-00001.parquet", config.CORPUS_REVISION, cache_dir))
    qrels = {split: read_qrels(_download(config.QRELS_REPO, f"{split}.tsv", config.QRELS_REVISION, cache_dir)) for split in ("train", "test")}

    doc_ids = [str(d) for d in corpus["_id"]]
    titles = dict(zip(doc_ids, (str(t) for t in corpus["title"])))
    docs = {d: f"{titles[d]} {text}".strip() for d, text in zip(doc_ids, corpus["text"])}
    ds = Dataset(doc_ids=doc_ids, titles=titles, docs=docs, queries=dict(zip(queries["_id"].astype(str), queries["text"])), qrels=qrels)

    if validate:
        assert len(ds.doc_ids) == config.EXPECTED_DOCS, f"expected {config.EXPECTED_DOCS} docs, got {len(ds.doc_ids)}"
        assert len(qrels["test"]) == config.EXPECTED_TEST_QUERIES, f"expected 300 test queries, got {len(qrels['test'])}"
        assert len(qrels["train"]) == config.EXPECTED_TRAIN_QUERIES, f"expected 809 train queries, got {len(qrels['train'])}"
        missing = [q for split in qrels.values() for q in split if q not in ds.queries]
        assert not missing, f"{len(missing)} qrels queries missing from queries file"
    return ds
