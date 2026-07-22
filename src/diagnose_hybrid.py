"""
diagnose_hybrid.py — One-off diagnostic: traces a single query through every
stage (dense top-k, sparse top-k, RRF-fused rank, final reranked top-k) to
pinpoint EXACTLY where a gold chunk drops out, rather than guessing.

Usage:
    python src/diagnose_hybrid.py "What is the outlook for Primary Myelofibrosis ?" 0000287 --strategy sentence
"""

from __future__ import annotations

import argparse

from hybrid_retriever import reciprocal_rank_fusion
from reranker import rerank
from retriever import DenseRetriever
from sparse_retriever import BM25Retriever


def find_rank(chunks, gold_pair_uid: str) -> int | None:
    for i, c in enumerate(chunks, 1):
        if c.chunk_id.split("::")[0] == gold_pair_uid:
            return i
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("gold_pair_uid")
    parser.add_argument("--strategy", default="sentence")
    parser.add_argument("--candidate_k", type=int, default=20)
    args = parser.parse_args()

    dense = DenseRetriever(strategy=args.strategy)
    sparse = BM25Retriever(strategy=args.strategy)

    dense_results = dense.retrieve(args.query, top_k=args.candidate_k)
    sparse_results = sparse.retrieve(args.query, top_k=args.candidate_k)
    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    reranked = rerank(args.query, fused, top_k=5)

    print(f"Query: {args.query!r}")
    print(f"Gold pair_uid: {args.gold_pair_uid}\n")

    dense_rank = find_rank(dense_results, args.gold_pair_uid)
    sparse_rank = find_rank(sparse_results, args.gold_pair_uid)
    fused_rank = find_rank(fused, args.gold_pair_uid)
    reranked_rank = find_rank(reranked, args.gold_pair_uid)

    print(f"Dense top-{args.candidate_k}:    gold at rank {dense_rank}  (None = not in top-{args.candidate_k})")
    print(f"Sparse (BM25) top-{args.candidate_k}: gold at rank {sparse_rank}")
    print(f"Fused (RRF) list (len={len(fused)}): gold at rank {fused_rank}")
    print(f"Final reranked top-5:  gold at rank {reranked_rank}")

    print("\n--- Diagnosis ---")
    if fused_rank is None:
        print("Gold chunk isn't in dense OR sparse top-k at all — a genuine retrieval miss "
              "(neither retriever found it). Consider raising candidate_k, or this may be a "
              "chunking/embedding quality issue.")
    elif fused_rank <= args.candidate_k and reranked_rank is None:
        print("Gold chunk WAS available to the reranker (rank <= candidate_k in the fused list) "
              "but the reranker did NOT put it in the final top-5. This means the cross-encoder "
              "itself is scoring some other chunk as more relevant — a reranker quality issue, "
              "not a candidate-truncation issue.")
        print("\nWhat the reranker chose instead (top 5):")
        for i, c in enumerate(reranked, 1):
            print(f"  [{i}] score={c.score:.3f} focus={c.metadata.get('focus')} | {c.chunk_text[:100]}...")
    elif fused_rank > args.candidate_k:
        print(f"Gold chunk's RRF-fused rank ({fused_rank}) is beyond candidate_k ({args.candidate_k}) — "
              "if you're still seeing the OLD truncation bug's symptoms, this is the case it "
              "would affect. With the fix applied, it should still reach the reranker since we "
              "no longer slice `fused` before reranking.")
    elif reranked_rank is not None:
        print("Gold chunk made it through to the final top-5 — retrieval succeeded for this query.")


if __name__ == "__main__":
    main()