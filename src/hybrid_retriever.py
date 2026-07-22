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
"""

from __future__ import annotations

from retriever import DenseRetriever, RetrievedChunk
from sparse_retriever import BM25Retriever


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
    ) -> list[RetrievedChunk]:
        """
        candidate_k: how many results each retriever contributes before fusion
                     (wider net than the final top_k so the reranker has
                     enough good candidates to choose from).
        top_k: final number of chunks returned after fusion (+ rerank).
        """
        dense_results = self.dense.retrieve(query, top_k=candidate_k)
        sparse_results = self.sparse.retrieve(query, top_k=candidate_k)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], rrf_k=rrf_k)

        if not use_reranker:
            return fused[:top_k]

        from reranker import rerank
        # Rerank a modest slice of the fused list (not all of it — reranking
        # is the expensive step) then return the true top_k.
        # Rerank the WHOLE fused list, not a re-sliced portion of it. `fused`
        # is already bounded (at most 2*candidate_k unique chunks — dense's
        # contribution + sparse's, minus overlap), and reranking that many
        # is cheap. Re-slicing to candidate_k here was a real bug: a chunk
        # found only by dense (correct match, but no BM25 keyword overlap)
        # gets a modest RRF score and can get pushed below the candidate_k
        # cutoff by chunks appearing in BOTH lists at moderate individual
        # ranks — so the reranker never even saw the right answer. This is
        # what caused hybrid_recall_at_k (0.778) to come in worse than
        # dense_recall_at_k (0.972) in the first eval run.
        return rerank(query, fused, top_k=top_k)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--strategy", default="sentence")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--no_rerank", action="store_true")
    args = parser.parse_args()

    retriever = HybridRetriever(strategy=args.strategy)
    results = retriever.retrieve(args.query, top_k=args.top_k, use_reranker=not args.no_rerank)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r.score:.3f} | {r.metadata['focus']} | source={r.metadata['source']}")
        print(f"    {r.chunk_text[:200]}...")