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


def test_query_endpoint_verification_off_by_default_leaves_new_fields_none(client):
    resp = client.post("/query", json={"question": "What are diabetes symptoms?"})
    body = resp.json()
    assert body["citation_checks"] is None
    assert body["composite_confidence"] is None


def test_query_endpoint_with_verification_populates_new_fields(client, monkeypatch):
    monkeypatch.setattr("citation_verify.call_llm", lambda prompt: '{"supported": true, "reason": "ok"}')
    resp = client.post("/query", json={
        "question": "What are diabetes symptoms?", "use_verification": True,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["composite_confidence"] is not None
    assert body["citation_coverage"] is not None


def test_documents_endpoint_404_when_no_data(client, tmp_path, monkeypatch):
    monkeypatch.setattr("api.DATA_DIR", tmp_path)
    resp = client.get("/documents")
    assert resp.status_code == 404


def test_documents_endpoint_summarizes_qa_pairs(client, tmp_path, monkeypatch):
    import json
    qa_path = tmp_path / "qa_pairs.jsonl"
    rows = [
        {"doc_id": "d1", "source": "NIDDK", "focus": "Diabetes"},
        {"doc_id": "d2", "source": "NIDDK", "focus": "Asthma"},
        {"doc_id": "d3", "source": "CDC", "focus": "Flu"},
    ]
    with open(qa_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    monkeypatch.setattr("api.DATA_DIR", tmp_path)

    resp = client.get("/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_qa_pairs"] == 3
    assert body["unique_documents"] == 3
    assert set(body["sources"].keys()) == {"NIDDK", "CDC"}
    assert body["sources"]["NIDDK"]["unique_topics"] == 2


def test_documents_endpoint_filters_by_source(client, tmp_path, monkeypatch):
    import json
    qa_path = tmp_path / "qa_pairs.jsonl"
    rows = [
        {"doc_id": "d1", "source": "NIDDK", "focus": "Diabetes"},
        {"doc_id": "d2", "source": "CDC", "focus": "Flu"},
    ]
    with open(qa_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    monkeypatch.setattr("api.DATA_DIR", tmp_path)

    resp = client.get("/documents", params={"source": "NIDDK"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_qa_pairs"] == 1
    assert set(body["sources"].keys()) == {"NIDDK"}


def test_ingest_status_endpoint_defaults_idle(client):
    resp = client.get("/ingest/status")
    assert resp.status_code == 200
    assert resp.json()["state"] == "idle"


def test_ingest_endpoint_rejects_concurrent_runs(client, monkeypatch):
    monkeypatch.setattr("api._ingest_status", {"state": "running", "strategy": "sentence", "detail": "x"})
    resp = client.post("/ingest", json={"strategy": "sentence"})
    assert resp.status_code == 409


def test_ingest_endpoint_rejects_unknown_strategy(client):
    resp = client.post("/ingest", json={"strategy": "not-a-real-strategy"})
    assert resp.status_code == 422