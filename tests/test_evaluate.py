import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evaluate as eval_mod  # noqa: E402
from retriever import RetrievedChunk  # noqa: E402


def make_chunk(pair_uid, score=5.0):
    return RetrievedChunk(
        chunk_id=f"{pair_uid}::sentence::0", chunk_text="text", score=score,
        metadata={"focus": "Test"},
    )


def test_extract_pair_uid():
    assert eval_mod.extract_pair_uid("0000123::sentence::2") == "0000123"


def test_recall_at_k_hit():
    chunks = [make_chunk("0000001"), make_chunk("0000002")]
    assert eval_mod.recall_at_k(chunks, "0000002") is True


def test_recall_at_k_miss():
    chunks = [make_chunk("0000001"), make_chunk("0000002")]
    assert eval_mod.recall_at_k(chunks, "0000999") is False


def test_evaluate_retrieval_aggregates_correctly(monkeypatch):
    """End-to-end test of the aggregation logic (recall@k across items,
    refusal-gate accuracy) using fully mocked retrievers — no Chroma/BM25/
    network needed, so this genuinely verifies the scoring math."""

    class FakeDense:
        def __init__(self, strategy): pass
        def retrieve(self, query, top_k=5):
            # Dense always finds the gold chunk for "findable" question,
            # misses for "hard" question — lets us verify recall != 1.0 or 0.0
            if "findable" in query:
                return [make_chunk("GOLD1")]
            return [make_chunk("WRONG")]

    class FakeHybrid:
        def __init__(self, strategy): pass
        def retrieve(self, query, top_k=5, use_reranker=True):
            if "findable" in query or "hard" in query:
                return [make_chunk("GOLD1", score=8.0)]  # hybrid finds both -> better recall
            # out-of-domain / borderline -> low score -> should trigger refusal
            return [make_chunk("IRRELEVANT", score=-10.0)]

    monkeypatch.setattr(eval_mod, "DenseRetriever", FakeDense)
    monkeypatch.setattr(eval_mod, "HybridRetriever", FakeHybrid)

    eval_items = [
        {"question": "findable question", "type": "real", "gold_pair_uid": "GOLD1", "expects_refusal": False},
        {"question": "hard question", "type": "real", "gold_pair_uid": "GOLD1", "expects_refusal": False},
        {"question": "nonsense question", "type": "out_of_domain", "expects_refusal": True},
    ]

    report = eval_mod.evaluate_retrieval(eval_items, strategy="sentence", top_k=5)

    assert report["n_gold_items"] == 2
    assert report["dense_recall_at_k"] == 0.5  # found 1/2 (only "findable")
    assert report["hybrid_recall_at_k"] == 1.0  # found 2/2 (both, since hybrid always returns GOLD1 for these)
    assert report["refusal_gate_accuracy"] == 1.0  # nonsense correctly refused
    assert report["false_refusal_rate_on_answerable"] == 0.0  # neither real item wrongly refused


def test_evaluate_retrieval_detects_false_refusal(monkeypatch):
    """If the confidence gate wrongly refuses an answerable question, the
    false-refusal metric should catch it — this is the metric most likely
    to matter once we tune RERANK_CONFIDENCE_THRESHOLD for real."""

    class FakeDense:
        def __init__(self, strategy): pass
        def retrieve(self, query, top_k=5):
            return [make_chunk("GOLD1")]

    class FakeHybrid:
        def __init__(self, strategy): pass
        def retrieve(self, query, top_k=5, use_reranker=True):
            # Even a real, answerable question gets a low score here —
            # simulates an overly strict threshold wrongly refusing it.
            return [make_chunk("GOLD1", score=-10.0)]

    monkeypatch.setattr(eval_mod, "DenseRetriever", FakeDense)
    monkeypatch.setattr(eval_mod, "HybridRetriever", FakeHybrid)

    eval_items = [
        {"question": "a real answerable question", "type": "real", "gold_pair_uid": "GOLD1", "expects_refusal": False},
    ]
    report = eval_mod.evaluate_retrieval(eval_items, strategy="sentence", top_k=5)
    assert report["false_refusal_rate_on_answerable"] == 1.0

def test_judge_calibration_check_passes_when_judge_scores_low(monkeypatch):
    """The calibration check should PASS when the judge correctly scores a
    deliberately fabricated/ungrounded answer as low-groundedness."""

    class FakeHybrid:
        def __init__(self, strategy): pass
        def retrieve(self, query, top_k=5):
            return [make_chunk("GOLD1")]

    monkeypatch.setattr(eval_mod, "HybridRetriever", FakeHybrid)
    monkeypatch.setattr(eval_mod, "call_groq", lambda prompt: '{"groundedness": 1, "relevance": 2}')

    eval_items = [{"question": "real q", "type": "real", "gold_pair_uid": "GOLD1", "expects_refusal": False}]
    result = eval_mod.judge_calibration_check(eval_items, strategy="sentence")

    assert result["judge_correctly_flagged_as_ungrounded"] is True
    assert result["judge_scores"]["groundedness"] == 1


def test_judge_calibration_check_fails_when_judge_too_lenient(monkeypatch):
    """If the judge scores an obviously fabricated answer as high-
    groundedness, the calibration check should FAIL — this is the signal
    that the earlier 'perfect scores' on real answers might not mean much,
    since the judge isn't discriminating between good and bad answers."""

    class FakeHybrid:
        def __init__(self, strategy): pass
        def retrieve(self, query, top_k=5):
            return [make_chunk("GOLD1")]

    monkeypatch.setattr(eval_mod, "HybridRetriever", FakeHybrid)
    # Simulates a too-lenient judge giving a fabricated answer a perfect score
    monkeypatch.setattr(eval_mod, "call_groq", lambda prompt: '{"groundedness": 5, "relevance": 5}')

    eval_items = [{"question": "real q", "type": "real", "gold_pair_uid": "GOLD1", "expects_refusal": False}]
    result = eval_mod.judge_calibration_check(eval_items, strategy="sentence")

    assert result["judge_correctly_flagged_as_ungrounded"] is False