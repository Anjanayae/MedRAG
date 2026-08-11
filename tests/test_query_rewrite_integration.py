import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hybrid_retriever import HybridRetriever  # noqa: E402
import query_rewrite as query_rewrite_mod  # noqa: E402


class _FakeSubRetriever:
    def __init__(self):
        self.seen_queries = []

    def retrieve(self, query, top_k):
        self.seen_queries.append(query)
        return []


def test_retrieve_uses_rewritten_query_when_enabled(monkeypatch):
    monkeypatch.setattr(
        query_rewrite_mod, "rewrite_query",
        lambda q, model=None: "hair loss" if "hairfall" in q else q,
    )

    hr = HybridRetriever.__new__(HybridRetriever)
    hr.dense = _FakeSubRetriever()
    hr.sparse = _FakeSubRetriever()

    hr.retrieve("how does B12 play a role in hairfall", top_k=3, use_reranker=False,
                use_query_rewrite=True)

    assert hr.dense.seen_queries == ["hair loss"]
    assert hr.sparse.seen_queries == ["hair loss"]


def test_retrieve_skips_rewrite_when_disabled(monkeypatch):
    monkeypatch.setattr(
        query_rewrite_mod, "rewrite_query",
        lambda q, model=None: "SHOULD NOT BE CALLED",
    )

    hr = HybridRetriever.__new__(HybridRetriever)
    hr.dense = _FakeSubRetriever()
    hr.sparse = _FakeSubRetriever()

    hr.retrieve("original phrasing", top_k=3, use_reranker=False, use_query_rewrite=False)

    assert hr.dense.seen_queries == ["original phrasing"]
    assert hr.sparse.seen_queries == ["original phrasing"]