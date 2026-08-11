"""
hybrid_retriever.py — Combines DenseRetriever + BM25Retriever via
Reciprocal Rank Fusion (RRF), then optionally reranks the fused candidates
with a cross-encoder.

Why RRF instead of a weighted score average: dense cosine similarity and
BM25 scores live on completely different, unbounded scales — averaging them
directly requires arbitrary normalization/weight tuning per dataset. RRF
sidesteps this entirely by only looking at *rank position* in each list:

    RRF_score(chunk) = sum over retrievers of  1 / (rrf_k + rank_in_that_list)

A chunk ranked #1 in both lists scores higher than one ranked #1 in only
one list — no tuning needed, and it's the standard approach used in
production hybrid search (Elasticsearch, Azure AI Search, Weaviate all
implement this).

Per-stage timing: set env var MEDRAG_DEBUG_TIMING=1 to print how long each
stage (query rewrite, dense retrieve, sparse retrieve, fusion, rerank)
takes. Added after a real slow-query report (~43s per query) — the
embedding model and cross-encoder reranker are lazy-loaded on first use
(see embed.py's get_model() / reranker.py's get_reranker()), so a query
landing on a cold process pays the full "load two ~90MB models from disk +
import torch" cost inline with that one request, which looks like a
retrieval bug but isn't. warmup() below exists to pay that cost once at
server startup instead.
"""

from __future__ import annotations

import os
import time

from retriever import DenseRetriever, RetrievedChunk
from sparse_retriever import BM25Retriever

_DEBUG_TIMING = os.environ.get("MEDRAG_DEBUG_TIMING", "0") == "1"


def _log_timing(label: str, elapsed_s: float) -> None:
    if _DEBUG_TIMING:
        print(f"[hybrid_retriever] {label}: {elapsed_s * 1000:.1f}ms")


def _log_timing_note(note: str) -> None:
    if _DEBUG_TIMING:
        print(f"[hybrid_retriever] {note}")


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], rrf_k: int = 60
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists of RetrievedChunk into one, ordered by RRF
    score. rrf_k=60 is the standard default from the original RRF paper —
    it dampens the impact of rank 1 vs rank 2 (without it, rank 1 would
    dominate almost entirely)."""
    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            fused_scores.setdefault(chunk.chunk_id, 0.0)
            fused_scores[chunk.chunk_id] += 1.0 / (rrf_k + rank)
            chunk_lookup[chunk.chunk_id] = chunk  # last write wins for text/metadata

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

    fused = []
    for cid in ranked_ids:
        chunk = chunk_lookup[cid]
        fused.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.chunk_text,
                score=fused_scores[cid],
                metadata=chunk.metadata,
            )
        )
    return fused


class HybridRetriever:
    def __init__(self, strategy: str = "sentence", model_name: str = "all-MiniLM-L6-v2"):
        self.dense = DenseRetriever(strategy=strategy, model_name=model_name)
        self.sparse = BM25Retriever(strategy=strategy)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k: int = 60,
        use_reranker: bool = True,
        use_query_rewrite: bool = True,
    ) -> list[RetrievedChunk]:
        """
        candidate_k: how many results each retriever contributes before fusion
                     (wider net than the final top_k so the reranker has
                     enough good candidates to choose from).
        top_k: final number of chunks returned after fusion (+ rerank).
        use_query_rewrite: normalize colloquial phrasing (e.g. "hairfall")
                     into clinical terms before retrieval — see
                     query_rewrite.py. Only affects what's searched with;
                     generation.py still receives the user's original
                     wording, so answers stay in the user's own terms.
        """
        search_query = query
        if use_query_rewrite:
            from query_rewrite import rewrite_query
            t_rw = time.perf_counter()
            search_query = rewrite_query(query)
            _log_timing("query rewrite", time.perf_counter() - t_rw)
            if search_query != query:
                _log_timing_note(f'rewrote "{query}" -> "{search_query}"')

        t0 = time.perf_counter()
        dense_results = self.dense.retrieve(search_query, top_k=candidate_k)
        _log_timing("dense retrieve", time.perf_counter() - t0)

        t1 = time.perf_counter()
        sparse_results = self.sparse.retrieve(search_query, top_k=candidate_k)
        _log_timing("sparse (BM25) retrieve", time.perf_counter() - t1)

        t2 = time.perf_counter()
        fused = reciprocal_rank_fusion([dense_results, sparse_results], rrf_k=rrf_k)
        _log_timing("RRF fusion", time.perf_counter() - t2)

        if not use_reranker:
            return fused[:top_k]

        from reranker import rerank
        t3 = time.perf_counter()
        result = rerank(search_query, fused, top_k=top_k)
        _log_timing("rerank", time.perf_counter() - t3)
        return result

    def warmup(self) -> None:
        """Forces the embedding model and cross-encoder reranker to load
        NOW, in this call, rather than lazily on the first real user query.
        Call this once at server startup (see api.py's lifespan) so the
        first actual /query request isn't the one paying for model loading
        — that's what turned a normal sub-second retrieval into an
        apparent 43-second one in practice. use_query_rewrite=False here:
        that step calls the Groq API, which has nothing to "warm up"
        locally and would just waste a call."""
        t0 = time.perf_counter()
        self.retrieve("warmup query to force model loading", top_k=1,
                       use_reranker=True, use_query_rewrite=False)
        _log_timing("full warmup (dense + sparse + rerank model load)", time.perf_counter() - t0)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--strategy", default="sentence")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--no_rerank", action="store_true")
    parser.add_argument("--no_rewrite", action="store_true")
    args = parser.parse_args()

    retriever = HybridRetriever(strategy=args.strategy)
    results = retriever.retrieve(
        args.query, top_k=args.top_k,
        use_reranker=not args.no_rerank,
        use_query_rewrite=not args.no_rewrite,
    )
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r.score:.3f} | {r.metadata['focus']} | source={r.metadata['source']}")
        print(f"    {r.chunk_text[:200]}...")