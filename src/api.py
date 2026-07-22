"""
api.py — FastAPI backend. Single /query endpoint: retrieve (hybrid) ->
rerank -> generate (Ollama) -> return answer + sources + confidence.

Kept deliberately thin: this file wires modules together and handles the
HTTP layer; all the actual logic lives in hybrid_retriever.py and
generation.py so it's testable without spinning up a server.

Basic latency logging included now (retrieval_time_ms / generation_time_ms
on every response) rather than bolted on later — cheap to add while writing
the endpoint, and it's the first piece of the observability step.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generation import generate_answer, DEFAULT_GROQ_MODEL
from hybrid_retriever import HybridRetriever

_retriever: HybridRetriever | None = None
DEFAULT_STRATEGY = "sentence"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    # Loaded once at startup (embedding model + BM25 index are not cheap to
    # reload per-request) rather than per-request in the endpoint.
    _retriever = HybridRetriever(strategy=DEFAULT_STRATEGY)
    yield
    _retriever = None


app = FastAPI(title="MedRAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True
    model: str = DEFAULT_GROQ_MODEL


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float
    refused: bool
    model: str
    retrieval_time_ms: float
    generation_time_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "retriever_loaded": _retriever is not None}


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
    result = generate_answer(request.question, retrieved, model=request.model)
    generation_time_ms = (time.perf_counter() - t1) * 1000

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        confidence=result.confidence,
        refused=result.refused,
        model=result.model,
        retrieval_time_ms=round(retrieval_time_ms, 1),
        generation_time_ms=round(generation_time_ms, 1),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)