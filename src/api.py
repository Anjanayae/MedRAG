"""
api.py — FastAPI backend.

  POST /query        retrieve (hybrid) -> rerank -> generate -> answer +
                      sources + confidence (optionally: citation
                      verification + composite confidence)
  GET  /documents     summary of what's currently indexed (source/topic
                      breakdown), read from data/processed/qa_pairs.jsonl
  POST /ingest        re-runs ingest -> chunk -> index for a strategy, in
                      the background, and hot-swaps the live retriever
                      when done — no server restart needed to pick up new
                      data. Long-running; poll GET /ingest/status.
  GET  /ingest/status current status of the last-triggered /ingest run
  GET  /health

Kept deliberately thin: this file wires modules together and handles the
HTTP layer; all the actual logic lives in hybrid_retriever.py,
generation.py, ingest.py, chunking.py, and index.py so it's testable
without spinning up a server.

Basic latency logging included now (retrieval_time_ms / generation_time_ms
on every response) rather than bolted on later — cheap to add while writing
the endpoint, and it's the first piece of the observability step.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from generation import generate_answer
from hybrid_retriever import HybridRetriever

_retriever: HybridRetriever | None = None
DEFAULT_STRATEGY = "sentence"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    # Loaded once at startup (embedding model + BM25 index are not cheap to
    # reload per-request) rather than per-request in the endpoint.
    print("Loading MedRAG retriever (embedding model, Chroma index, BM25 index)...")
    _retriever = HybridRetriever(strategy=DEFAULT_STRATEGY)
    # Force the embedding model AND the cross-encoder reranker to load now,
    # not on the first real user query. Both are lazy-loaded on first use
    # (see embed.py / reranker.py) — without this, the first person to hit
    # /query pays the full "load two ~90MB models + import torch" cost
    # inline with their request, which is what turned a normal sub-second
    # retrieval into an observed ~43-second one.
    print("Warming up reranker (loading cross-encoder model)...")
    _retriever.warmup()
    print("MedRAG ready.")
    yield
    _retriever = None


app = FastAPI(title="MedRAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True
    # None = use whichever LLM_PROVIDER is configured (.env), with that
    # provider's own default model. Set explicitly to override.
    model: str | None = None
    # Opt-in: costs one extra LLM-judge call per citation in the answer.
    # See generation.py's generate_answer() docstring.
    use_verification: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float
    refused: bool
    model: str
    retrieval_time_ms: float
    generation_time_ms: float
    # Populated only when use_verification=True was requested; None otherwise.
    citation_checks: list[dict] | None = None
    citation_coverage: float | None = None
    source_utilization: float | None = None
    composite_confidence: float | None = None


class IngestRequest(BaseModel):
    strategy: str = Field(default=DEFAULT_STRATEGY, pattern="^(fixed_size|sentence|semantic)$")
    reparse_raw: bool = Field(
        default=False,
        description="Re-run ingest.py's XML parse from data/raw/MedQuAD first "
                    "(slow, only needed if the raw source data changed). "
                    "False just re-chunks/re-indexes the existing qa_pairs.jsonl.",
    )


_ingest_status: dict = {"state": "idle", "strategy": None, "detail": None}


@app.get("/health")
def health():
    return {"status": "ok", "retriever_loaded": _retriever is not None}


@app.get("/documents")
def documents(source: str | None = Query(default=None, description="Filter to one source, e.g. 'NIDDK'")):
    """Summary of what's currently indexed — reads data/processed/qa_pairs.jsonl
    (the output of ingest.py), not the Chroma/BM25 indexes directly, since
    that file is the single source of truth both indexes are built from
    (see index.py / chunking.py docstrings)."""
    qa_path = DATA_DIR / "qa_pairs.jsonl"
    if not qa_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{qa_path} not found. Run ingest first (see POST /ingest).",
        )

    import json
    from collections import Counter

    doc_ids = set()
    focuses_by_source: dict[str, set] = {}
    total = 0
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if source and row["source"] != source:
                continue
            total += 1
            doc_ids.add(row["doc_id"])
            focuses_by_source.setdefault(row["source"], set()).add(row["focus"])

    return {
        "total_qa_pairs": total,
        "unique_documents": len(doc_ids),
        "sources": {
            src: {"unique_topics": len(focuses), "sample_topics": sorted(focuses)[:10]}
            for src, focuses in sorted(focuses_by_source.items())
        },
    }


def _run_ingest_pipeline(strategy: str, reparse_raw: bool) -> None:
    """Runs in the background (see POST /ingest). Reuses ingest.py's
    main(), chunking.py's build_chunks/save_chunks, and index.py's
    build_index as-is — this function only sequences them and hot-swaps
    the live retriever afterward, it doesn't reimplement any of them."""
    global _retriever, _ingest_status
    _ingest_status = {"state": "running", "strategy": strategy, "detail": "starting"}
    try:
        if reparse_raw:
            _ingest_status["detail"] = "parsing raw XML (ingest.py)"
            import ingest
            ingest.main()

        _ingest_status["detail"] = f"chunking (strategy={strategy})"
        import chunking
        if strategy == "semantic":
            from embed import embed_fn_factory
            chunks = chunking.build_chunks(strategy, embed_fn=embed_fn_factory())
        else:
            chunks = chunking.build_chunks(strategy)
        chunking.save_chunks(chunks, strategy)

        _ingest_status["detail"] = f"indexing (strategy={strategy})"
        import index as index_mod
        index_mod.build_index(strategy)

        _ingest_status["detail"] = "reloading live retriever"
        new_retriever = HybridRetriever(strategy=strategy)
        new_retriever.warmup()
        _retriever = new_retriever  # swap in only once the new one is ready

        _ingest_status = {
            "state": "done", "strategy": strategy,
            "detail": f"{len(chunks)} chunks indexed",
        }
    except Exception as e:
        _ingest_status = {"state": "error", "strategy": strategy, "detail": str(e)}


@app.post("/ingest")
def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    """Triggers a background re-ingest/re-chunk/re-index for `strategy`,
    then hot-swaps the live retriever — no server restart needed to pick
    up new/changed data. This is long-running (minutes, especially with
    reparse_raw=True on the full MedQuAD corpus); poll GET /ingest/status.
    Only one run at a time — returns 409 if one is already in progress."""
    if _ingest_status["state"] == "running":
        raise HTTPException(status_code=409, detail="An ingest run is already in progress.")
    background_tasks.add_task(_run_ingest_pipeline, request.strategy, request.reparse_raw)
    return {"status": "started", "strategy": request.strategy}


@app.get("/ingest/status")
def ingest_status():
    return _ingest_status


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    t0 = time.perf_counter()
    retrieved = _retriever.retrieve(
        request.question, top_k=request.top_k, use_reranker=request.use_reranker
    )
    retrieval_time_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    result = generate_answer(
        request.question, retrieved, model=request.model,
        verify_citations=request.use_verification,
    )
    generation_time_ms = (time.perf_counter() - t1) * 1000

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        confidence=result.confidence,
        refused=result.refused,
        model=result.model,
        retrieval_time_ms=round(retrieval_time_ms, 1),
        generation_time_ms=round(generation_time_ms, 1),
        citation_checks=result.citation_checks,
        citation_coverage=result.citation_coverage,
        source_utilization=result.source_utilization,
        composite_confidence=result.composite_confidence,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)