import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from fastapi.testclient import TestClient

import api as api_mod  # noqa: E402
from retriever import RetrievedChunk  # noqa: E402


class FakeRetriever:
    """Stands in for HybridRetriever so tests don't need Chroma/BM25/network."""
    def retrieve(self, query, top_k=5, use_reranker=True):
        return [
            RetrievedChunk(
                chunk_id="c1", chunk_text="Diabetes symptoms include thirst.",
                score=8.0, metadata={"focus": "Diabetes", "source": "NIDDK",
                                      "url": "u1", "question": "What are diabetes symptoms?"},
            )
        ]


@pytest.fixture
def client(monkeypatch):
    import generation
    monkeypatch.setattr(generation, "call_groq", lambda prompt, model: "Mocked grounded answer [1].")

    # Deliberately NOT using `with TestClient(...)` — that triggers the real
    # `lifespan` function, which builds an actual HybridRetriever (needs
    # Chroma + a downloaded embedding model, unavailable in this sandbox).
    # Instantiating without the context manager skips lifespan entirely, so
    # we set the module-level _retriever directly instead.
    monkeypatch.setattr(api_mod, "_retriever", FakeRetriever())
    c = TestClient(api_mod.app)
    yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_endpoint_happy_path(client):
    resp = client.post("/query", json={"question": "What are diabetes symptoms?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Mocked grounded answer [1]."
    assert body["refused"] is False
    assert len(body["sources"]) == 1
    assert body["sources"][0]["focus"] == "Diabetes"
    assert "retrieval_time_ms" in body
    assert "generation_time_ms" in body


def test_query_endpoint_rejects_short_question(client):
    resp = client.post("/query", json={"question": "hi"})
    assert resp.status_code == 422  # pydantic min_length=3 validation


def test_query_endpoint_top_k_bounds(client):
    resp = client.post("/query", json={"question": "valid question here", "top_k": 100})
    assert resp.status_code == 422  # le=20 validation