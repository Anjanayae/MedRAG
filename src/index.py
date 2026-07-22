"""
index.py — Embed chunks and load them into a persistent Chroma collection.

Design notes:
  - One Chroma collection per chunking strategy (e.g. "medrag_sentence"),
    stored under data/processed/chroma_db/. This lets Step 3's ablation
    compare strategies by just pointing the retriever at a different
    collection name — no re-embedding needed mid-ablation.
  - We embed `chunk.embed_text()` (question + focus + chunk_text) rather
    than raw chunk_text — see chunking.py docstring for why.
  - Batched in groups of 256 to keep memory bounded on large chunk sets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embed import embed_texts  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"

BATCH_SIZE = 256


def load_chunks(strategy: str) -> list[dict]:
    path = DATA_DIR / f"chunks_{strategy}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python src/chunking.py` first."
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index(strategy: str, model_name: str = "all-MiniLM-L6-v2",
                 reset: bool = True) -> None:
    import chromadb

    chunks = load_chunks(strategy)
    print(f"Loaded {len(chunks)} chunks for strategy='{strategy}'")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = f"medrag_{strategy}"

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # collection didn't exist yet

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"strategy": strategy, "embedding_model": model_name},
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        # embed_text() = "Question: ...\nFocus: ...\n\n{chunk_text}" —
        # matches chunking.Chunk.embed_text(), reconstructed here since we
        # loaded raw dicts from JSONL rather than Chunk objects.
        texts_to_embed = [
            f"Question: {c['question']}\nFocus: {c['focus']}\n\n{c['chunk_text']}"
            for c in batch
        ]
        embeddings = embed_texts(texts_to_embed, model_name=model_name)

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings.tolist(),
            documents=[c["chunk_text"] for c in batch],
            metadatas=[
                {
                    "doc_id": c["doc_id"],
                    "qid": c["qid"],
                    "focus": c["focus"],
                    "qtype": c["qtype"],
                    "source": c["source"],
                    "url": c["url"],
                    "question": c["question"],
                    "chunk_index": c["chunk_index"],
                }
                for c in batch
            ],
        )
        print(f"  indexed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}", end="\r")

    print(f"\nDone. Collection '{collection_name}' has {collection.count()} vectors.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", default="sentence",
        choices=["fixed_size", "sentence", "semantic"],
        help="Which chunking strategy's output to index",
    )
    args = parser.parse_args()
    build_index(args.strategy)