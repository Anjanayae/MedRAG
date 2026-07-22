"""
sparse_retriever.py — BM25 keyword-based retriever.

Why we need this alongside DenseRetriever: dense embeddings are great at
semantic similarity ("chest pain" ~ "cardiac discomfort") but weak at exact
term matching — drug names, acronyms (COPD, MI), dosages, rare disease names.
BM25 is the opposite: strong on exact/rare terms, blind to paraphrasing.
Combining both (hybrid_retriever.py) covers both failure modes.

Unlike the dense retriever, this needs no embedding model or network call —
it's pure term-frequency statistics, so we can build and test it fully in
any environment, no HF model hub required.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from retriever import RetrievedChunk

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple lowercase alphanumeric tokenizer. BM25 doesn't need anything
    fancier (no stemming) — medical terms are mostly literal, and stemming
    can actually hurt exact drug-name/acronym matching."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, strategy: str = "sentence"):
        self.strategy = strategy
        chunks_path = DATA_DIR / f"chunks_{strategy}.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"{chunks_path} not found — run `python src/chunking.py` first."
            )

        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]

        # Index on question + chunk_text (same fields dense embeds) so both
        # retrievers see comparable content.
        corpus = [
            tokenize(f"{c['question']} {c['chunk_text']}") for c in self.chunks
        ]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(tokenize(query))
        # argsort descending, take top_k
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            c = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    chunk_text=c["chunk_text"],
                    score=float(scores[idx]),
                    metadata={
                        "doc_id": c["doc_id"], "qid": c["qid"], "focus": c["focus"],
                        "qtype": c["qtype"], "source": c["source"], "url": c["url"],
                        "question": c["question"], "chunk_index": c["chunk_index"],
                    },
                )
            )
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--strategy", default="sentence")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    retriever = BM25Retriever(strategy=args.strategy)
    for i, r in enumerate(retriever.retrieve(args.query, top_k=args.top_k), 1):
        print(f"[{i}] score={r.score:.2f} | {r.metadata['focus']} | {r.chunk_text[:120]}...")