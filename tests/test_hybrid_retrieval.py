import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retriever import RetrievedChunk  # noqa: E402
import sparse_retriever as sparse_mod  # noqa: E402
from hybrid_retriever import reciprocal_rank_fusion  # noqa: E402
import reranker as reranker_mod  # noqa: E402


FAKE_CHUNKS = [
    {
        "chunk_id": "q1::sentence::0", "doc_id": "d1", "qid": "q1",
        "focus": "Diabetes", "qtype": "symptoms", "source": "NIDDK", "url": "u1",
        "question": "What are the symptoms of diabetes?",
        "chunk_text": "Common symptoms of diabetes include frequent urination and increased thirst.",
        "chunk_index": 0,
    },
    {
        "chunk_id": "q2::sentence::0", "doc_id": "d2", "qid": "q2",
        "focus": "Asthma", "qtype": "treatment", "source": "CDC", "url": "u2",
        "question": "How is asthma treated?",
        "chunk_text": "Asthma is commonly treated with inhaled corticosteroids and bronchodilators.",
        "chunk_index": 0,
    },
    {
        "chunk_id": "q3::sentence::0", "doc_id": "d3", "qid": "q3",
        "focus": "Migraine", "qtype": "causes", "source": "NINDS", "url": "u3",
        "question": "What causes migraines?",
        "chunk_text": "Migraines can be triggered by stress, certain foods, and hormonal changes.",
        "chunk_index": 0,
    },
]


@pytest.fixture
def bm25_retriever(tmp_path, monkeypatch):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    with open(data_dir / "chunks_sentence.jsonl", "w") as f:
        for c in FAKE_CHUNKS:
            f.write(json.dumps(c) + "\n")
    monkeypatch.setattr(sparse_mod, "DATA_DIR", data_dir)
    return sparse_mod.BM25Retriever(strategy="sentence")


def test_bm25_exact_term_match(bm25_retriever):
    # "urination" only appears in the diabetes chunk — BM25 should surface it top.
    results = bm25_retriever.retrieve("frequent urination symptom", top_k=1)
    assert results[0].metadata["focus"] == "Diabetes"


def test_bm25_returns_top_k(bm25_retriever):
    results = bm25_retriever.retrieve("treatment", top_k=2)
    assert len(results) == 2


def test_bm25_no_match_still_returns_results(bm25_retriever):
    # BM25 always returns *something* ranked (score may be 0), doesn't crash
    # on queries with no overlapping vocabulary.
    results = bm25_retriever.retrieve("completely unrelated gibberish xyz123", top_k=3)
    assert len(results) == 3


def test_rrf_prioritizes_chunk_ranked_high_in_both_lists():
    chunk_a = RetrievedChunk("a", "text a", 0.9, {})
    chunk_b = RetrievedChunk("b", "text b", 0.8, {})
    chunk_c = RetrievedChunk("c", "text c", 0.7, {})

    # 'a' ranked #1 in list1 but absent from list2.
    # 'b' ranked #2 in both lists.
    list1 = [chunk_a, chunk_b]
    list2 = [chunk_c, chunk_b]

    fused = reciprocal_rank_fusion([list1, list2])
    fused_ids = [c.chunk_id for c in fused]

    # 'b' appears in both lists (rank 2 + rank 2) vs 'a'/'c' each appearing
    # in only one list at rank 1 — b's combined RRF score should win.
    assert fused_ids[0] == "b"


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_dedupes_across_lists():
    chunk_a = RetrievedChunk("a", "text a", 0.9, {})
    fused = reciprocal_rank_fusion([[chunk_a], [chunk_a]])
    assert len(fused) == 1


def test_reranker_dedupes_candidates(monkeypatch):
    """Hybrid fusion can pass the same chunk_id in from both dense and
    sparse results (already deduped by RRF, but rerank() should be
    defensive) — verify rerank() doesn't score/return duplicates."""
    class FakeModel:
        def predict(self, pairs):
            return [1.0] * len(pairs)

    monkeypatch.setattr(reranker_mod, "get_reranker", lambda model_name=None: FakeModel())

    dup_chunk = RetrievedChunk("x", "some text", 0.5, {"focus": "Test"})
    candidates = [dup_chunk, dup_chunk]  # simulate accidental duplicate
    results = reranker_mod.rerank("query", candidates, top_k=5)
    assert len(results) == 1


def test_reranker_respects_top_k(monkeypatch):
    class FakeModel:
        def predict(self, pairs):
            # give increasing scores so order is deterministic
            return list(range(len(pairs)))

    monkeypatch.setattr(reranker_mod, "get_reranker", lambda model_name=None: FakeModel())

    candidates = [
        RetrievedChunk(f"id{i}", f"text {i}", 0.0, {}) for i in range(5)
    ]
    results = reranker_mod.rerank("query", candidates, top_k=2)
    assert len(results) == 2
    # highest fake score (last one, since predict returns increasing range) should be first
    assert results[0].chunk_id == "id4"


def test_reranker_includes_question_and_focus_context(monkeypatch):
    """Regression test for a real bug found via eval: rerank() used to feed
    the cross-encoder ONLY raw chunk_text, with no question/focus context —
    unlike embed.py, which deliberately includes both. Since MedQuAD splits
    each disease into several separate QA pairs by question type (symptoms/
    treatment/outlook/etc), a reranker judging on raw text alone can only
    tell "same disease" apart, not "same disease AND right question type" —
    so it kept preferring a wrong-facet chunk about the correct disease over
    the chunk that actually answered the asked question. This test verifies
    the metadata question/focus reach the cross-encoder input."""

    captured_pairs = []

    class FakeModel:
        def predict(self, pairs):
            captured_pairs.extend(pairs)
            return [0.0] * len(pairs)

    monkeypatch.setattr(reranker_mod, "get_reranker", lambda model_name=None: FakeModel())

    chunk = RetrievedChunk(
        chunk_id="c1",
        chunk_text="Primary myelofibrosis is a condition characterized by scar tissue.",
        score=0.0,
        metadata={"question": "What is the outlook for Primary Myelofibrosis ?", "focus": "Primary Myelofibrosis"},
    )
    reranker_mod.rerank("some query", [chunk], top_k=1)

    assert len(captured_pairs) == 1
    _, candidate_text = captured_pairs[0]
    assert "outlook" in candidate_text.lower(), (
        "reranker input must include the chunk's associated question "
        "(e.g. 'outlook') so it can distinguish question-type facets of "
        "the same disease, not just raw chunk_text"
    )
    assert "Primary Myelofibrosis" in candidate_text