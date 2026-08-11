import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import reranker as reranker_mod  # noqa: E402
from retriever import RetrievedChunk  # noqa: E402


class _FakeModel:
    def __init__(self, scores=None):
        self._scores = scores

    def predict(self, pairs):
        if self._scores is not None:
            return self._scores
        return [0.0] * len(pairs)


def test_rerank_dedupes_identical_chunk_text_across_different_pair_uids(monkeypatch):
    """Regression test for a real finding: MedQuAD itself contains
    duplicate content — different pair_uids (e.g. two adjacent source
    documents) sometimes have byte-identical answer text. This surfaced in
    a real query where 2 of 5 returned 'sources' for a metformin question
    turned out to be verbatim-identical Hypoglycemia chunks under two
    different pair_uids, wasting a citation slot. chunk_id-based dedup
    alone can't catch this since the ids genuinely differ — rerank() must
    also dedupe on chunk_text."""
    monkeypatch.setattr(reranker_mod, "get_reranker", lambda model_name=None: _FakeModel())

    duplicate_text = "Hypoglycemia occurs when blood glucose is low."
    c1 = RetrievedChunk("0012708::sentence::0", duplicate_text, 0.5,
                         {"question": "q1", "focus": "Hypoglycemia"})
    c2 = RetrievedChunk("0012709::sentence::0", duplicate_text, 0.5,
                         {"question": "q2", "focus": "Hypoglycemia"})
    c3 = RetrievedChunk("0099999::sentence::0", "Something totally different.", 0.5,
                         {"question": "q3", "focus": "Other"})

    results = reranker_mod.rerank("test query", [c1, c2, c3], top_k=5)

    assert len(results) == 2, (
        "expected chunk_id c2 to be dropped as a text-duplicate of c1, "
        "leaving only 2 unique chunks (c1 or c2, plus c3)"
    )
    result_texts = {r.chunk_text for r in results}
    assert duplicate_text in result_texts
    assert "Something totally different." in result_texts
    assert sum(1 for r in results if r.chunk_text == duplicate_text) == 1


def test_rerank_keeps_distinct_text_even_with_same_focus(monkeypatch):
    """Sanity check: text-dedup must not over-trigger — two genuinely
    different chunks about the same disease/focus should both survive."""
    monkeypatch.setattr(reranker_mod, "get_reranker", lambda model_name=None: _FakeModel())

    c1 = RetrievedChunk("id1", "Symptom text about hypoglycemia.", 0.5,
                         {"question": "q1", "focus": "Hypoglycemia"})
    c2 = RetrievedChunk("id2", "Treatment text about hypoglycemia.", 0.5,
                         {"question": "q2", "focus": "Hypoglycemia"})

    results = reranker_mod.rerank("test query", [c1, c2], top_k=5)
    assert len(results) == 2


def test_rerank_includes_question_and_focus_context(monkeypatch):
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

    class CapturingModel:
        def predict(self, pairs):
            captured_pairs.extend(pairs)
            return [0.0] * len(pairs)

    monkeypatch.setattr(reranker_mod, "get_reranker", lambda model_name=None: CapturingModel())

    chunk = RetrievedChunk(
        chunk_id="c1",
        chunk_text="Primary myelofibrosis is a condition characterized by scar tissue.",
        score=0.0,
        metadata={"question": "What is the outlook for Primary Myelofibrosis ?", "focus": "Primary Myelofibrosis"},
    )
    reranker_mod.rerank("some query", [chunk], top_k=1)

    assert len(captured_pairs) == 1
    _, candidate_text = captured_pairs[0]
    assert "outlook" in candidate_text.lower()
    assert "Primary Myelofibrosis" in candidate_text


def test_rerank_respects_top_k(monkeypatch):
    monkeypatch.setattr(reranker_mod, "get_reranker",
                         lambda model_name=None: _FakeModel(scores=list(range(5))))

    candidates = [RetrievedChunk(f"id{i}", f"text {i}", 0.0, {}) for i in range(5)]
    results = reranker_mod.rerank("query", candidates, top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "id4"