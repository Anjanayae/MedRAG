"""
test_index_retriever.py — Integration test for the index -> retrieve
pipeline, using a fake deterministic embedder in place of the real
sentence-transformers model.

Why a fake embedder: this sandbox has no network access to the HuggingFace
model hub, so the real model can't be downloaded here. A hashed-bag-of-words
embedder is a reasonable stand-in for *testing pipeline plumbing* (does
indexing store the right vectors, does retrieval return the right chunk for
an obviously-matching query) — it is NOT a substitute for evaluating real
retrieval quality, which needs the actual model and is covered in Step 3's
ablation instead.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import index as index_mod  # noqa: E402
import retriever as retriever_mod  # noqa: E402

DIM = 64


def _hashed_bow_embed(texts: list[str]) -> np.ndarray:
    """Deterministic 'embedding': hash each word into a dimension, sum,
    L2-normalize. Similar word overlap -> similar vectors. Good enough to
    test plumbing, not a real semantic embedder."""
    vecs = np.zeros((len(texts), DIM), dtype="float32")
    for i, text in enumerate(texts):
        for word in text.lower().split():
            dim = hash(word) % DIM
            vecs[i, dim] += 1.0
        norm = np.linalg.norm(vecs[i])
        if norm > 0:
            vecs[i] /= norm
    return vecs


@pytest.fixture
def fake_index(tmp_path, monkeypatch):
    """Build a tiny index from 3 obviously-distinct fake chunks."""
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    chroma_dir = data_dir / "chroma_db"

    chunks = [
        {
            "chunk_id": "q1::sentence::0", "doc_id": "d1", "qid": "q1",
            "focus": "Diabetes", "qtype": "symptoms", "source": "NIDDK",
            "url": "http://example.com/diabetes",
            "question": "What are the symptoms of diabetes?",
            "chunk_text": "Common symptoms of diabetes include frequent urination and increased thirst.",
            "chunk_index": 0,
        },
        {
            "chunk_id": "q2::sentence::0", "doc_id": "d2", "qid": "q2",
            "focus": "Asthma", "qtype": "treatment", "source": "CDC",
            "url": "http://example.com/asthma",
            "question": "How is asthma treated?",
            "chunk_text": "Asthma is commonly treated with inhaled corticosteroids and bronchodilators.",
            "chunk_index": 0,
        },
        {
            "chunk_id": "q3::sentence::0", "doc_id": "d3", "qid": "q3",
            "focus": "Migraine", "qtype": "causes", "source": "NINDS",
            "url": "http://example.com/migraine",
            "question": "What causes migraines?",
            "chunk_text": "Migraines can be triggered by stress, certain foods, and hormonal changes.",
            "chunk_index": 0,
        },
    ]
    chunks_path = data_dir / "chunks_sentence.jsonl"
    with open(chunks_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    monkeypatch.setattr(index_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(index_mod, "CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(index_mod, "embed_texts", lambda texts, **kw: _hashed_bow_embed(texts))
    monkeypatch.setattr(retriever_mod, "CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(retriever_mod, "embed_texts", lambda texts, **kw: _hashed_bow_embed(texts))

    index_mod.build_index("sentence", reset=True)
    return chroma_dir


def test_index_builds_expected_count(fake_index):
    import chromadb
    client = chromadb.PersistentClient(path=str(fake_index))
    collection = client.get_collection("medrag_sentence")
    assert collection.count() == 3


def test_retriever_returns_most_relevant_chunk(fake_index, monkeypatch):
    retriever = retriever_mod.DenseRetriever(strategy="sentence")
    results = retriever.retrieve("What are diabetes symptoms like thirst?", top_k=1)
    assert len(results) == 1
    assert results[0].metadata["focus"] == "Diabetes"


def test_retriever_top_k_respected(fake_index):
    retriever = retriever_mod.DenseRetriever(strategy="sentence")
    results = retriever.retrieve("asthma treatment inhaler", top_k=2)
    assert len(results) == 2


def test_retriever_missing_collection_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever_mod, "CHROMA_DIR", tmp_path / "empty_db")
    with pytest.raises(RuntimeError, match="not found"):
        retriever_mod.DenseRetriever(strategy="nonexistent")