"""
retriever.py — Dense (embedding-based) retriever over a Chroma collection.

This is the "naive RAG" baseline retriever — pure vector similarity search.
Step 3 adds a BM25 sparse retriever alongside this one and fuses both into
a HybridRetriever, then reranks the fused results. Keeping DenseRetriever
as its own class (rather than folding hybrid logic in here from the start)
means the ablation in Step 3 can cleanly compare dense-only vs hybrid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from embed import embed_texts

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "chroma_db"


@dataclass
class RetrievedChunk:
    chunk_id: str
    chunk_text: str
    score: float  # similarity score, higher = more relevant
    metadata: dict

    def __repr__(self):
        preview = self.chunk_text[:80].replace("\n", " ")
        return f"RetrievedChunk(score={self.score:.3f}, text='{preview}...')"


class DenseRetriever:
    def __init__(self, strategy: str = "sentence", model_name: str = "all-MiniLM-L6-v2"):
        import chromadb

        self.strategy = strategy
        self.model_name = model_name
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection_name = f"medrag_{strategy}"
        try:
            self.collection = client.get_collection(collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Collection '{collection_name}' not found. "
                f"Run `python src/index.py --strategy {strategy}` first."
            ) from e

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = embed_texts([query], model_name=self.model_name)[0]

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        retrieved = []
        # Chroma returns cosine *distance* (0 = identical) since our vectors
        # are normalized; convert to a similarity score (1 = identical) so
        # it reads naturally and is comparable to the reranker's score later.
        for chunk_id, doc, meta, distance in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        ):
            similarity = 1 - distance
            retrieved.append(
                RetrievedChunk(chunk_id=chunk_id, chunk_text=doc, score=similarity, metadata=meta)
            )
        return retrieved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Question to test retrieval with")
    parser.add_argument("--strategy", default="sentence")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    retriever = DenseRetriever(strategy=args.strategy)
    results = retriever.retrieve(args.query, top_k=args.top_k)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r.score:.3f} | source={r.metadata['source']} | "
              f"focus={r.metadata['focus']}")
        print(f"    Q: {r.metadata['question']}")
        print(f"    {r.chunk_text[:200]}...")