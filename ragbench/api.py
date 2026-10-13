"""FastAPI search service. Indexes load once at startup; endpoints are sync `def` so the CPU/GPU-bound
search runs in FastAPI's threadpool instead of blocking the event loop."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from .search import Searcher

Arm = Literal["bm25", "dense", "hybrid", "rerank"]


class Hit(BaseModel):
    doc_id: str
    title: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    arm: Arm
    k: int
    results: list[Hit]
    latency_ms: float


class Health(BaseModel):
    status: str
    arms: list[str]
    n_docs: int
    device: str
    rrf_k: int


def _default_factory() -> Searcher:
    arms = [a for a in os.environ.get("RAGBENCH_ARMS", "bm25,dense,hybrid,rerank").split(",") if a]
    return Searcher.from_disk(arms=arms)


def create_app(searcher_factory: Callable[[], Searcher] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.searcher = (searcher_factory or _default_factory)()
        yield

    app = FastAPI(title="ragbench search", version="1.0.0", lifespan=lifespan)

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        s: Searcher = request.app.state.searcher
        return Health(status="ok", arms=s.arms, n_docs=len(s.docs), device=s.device, rrf_k=s.rrf_k)

    @app.get("/search", response_model=SearchResponse)
    def search(
        request: Request,
        q: str = Query(..., min_length=1, max_length=1000, description="query text"),
        arm: Arm = "hybrid",
        k: int = Query(10, ge=1, le=100),
    ) -> SearchResponse:
        if not q.strip():
            raise HTTPException(status_code=422, detail="q must contain non-whitespace characters")
        s: Searcher = request.app.state.searcher
        if arm not in s.arms:
            raise HTTPException(status_code=503, detail=f"arm {arm!r} not loaded; available: {s.arms}")
        t0 = time.perf_counter()
        hits = s.search(q, arm, k)
        latency = (time.perf_counter() - t0) * 1000
        return SearchResponse(
            query=q,
            arm=arm,
            k=k,
            results=[Hit(doc_id=d, title=s.titles.get(d, ""), score=sc, rank=i) for i, (d, sc) in enumerate(hits, start=1)],
            latency_ms=round(latency, 3),
        )

    return app


app = create_app()
