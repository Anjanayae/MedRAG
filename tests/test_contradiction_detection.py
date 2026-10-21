import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import contradiction_detection as cd  # noqa: E402


class FakeChunk:
    def __init__(self, text):
        self.chunk_text = text


def test_find_contradictions_returns_empty_for_fewer_than_two_chunks():
    assert cd.find_contradictions([]) == []
    assert cd.find_contradictions([FakeChunk("only one")]) == []


def test_find_contradictions_returns_empty_when_none_conflict(monkeypatch):
    monkeypatch.setattr(cd, "call_llm", lambda prompt: '[{"index": 0, "conflicting": false, "explanation": ""}]')
    chunks = [FakeChunk("Take 500mg daily."), FakeChunk("Take 500mg daily, same as source A.")]
    flags = cd.find_contradictions(chunks)
    assert flags == []


def test_find_contradictions_flags_a_detected_conflict(monkeypatch):
    monkeypatch.setattr(
        cd, "call_llm",
        lambda prompt: '[{"index": 0, "conflicting": true, "explanation": "different dosage guidance"}]',
    )
    chunks = [FakeChunk("Take 500mg daily."), FakeChunk("Take 1000mg daily.")]
    flags = cd.find_contradictions(chunks)
    assert len(flags) == 1
    assert flags[0].source_index_a == 1
    assert flags[0].source_index_b == 2
    assert flags[0].explanation == "different dosage guidance"


def test_find_contradictions_uses_exactly_one_llm_call_for_multiple_pairs(monkeypatch):
    calls = []

    def fake_call_llm(prompt):
        calls.append(prompt)
        return ('[{"index": 0, "conflicting": true, "explanation": "a"}, '
                '{"index": 1, "conflicting": false, "explanation": ""}, '
                '{"index": 2, "conflicting": true, "explanation": "c"}]')

    monkeypatch.setattr(cd, "call_llm", fake_call_llm)
    chunks = [FakeChunk("A"), FakeChunk("B"), FakeChunk("C")]  # 3 chunks -> C(3,2)=3 pairs
    flags = cd.find_contradictions(chunks)
    assert len(calls) == 1
    assert len(flags) == 2


def test_find_contradictions_respects_max_chunks_bound(monkeypatch):
    calls = []

    def fake_call_llm(prompt):
        calls.append(prompt)
        return "[]"

    monkeypatch.setattr(cd, "call_llm", fake_call_llm)
    chunks = [FakeChunk(f"chunk {i}") for i in range(10)]
    cd.find_contradictions(chunks, max_chunks=3)
    assert '"index": 3' not in calls[0]


def test_find_contradictions_fails_safe_on_unparseable_response(monkeypatch):
    monkeypatch.setattr(cd, "call_llm", lambda prompt: "not valid json at all")
    chunks = [FakeChunk("A"), FakeChunk("B")]
    flags = cd.find_contradictions(chunks)
    assert flags == []


def test_find_contradictions_ignores_malformed_verdict_entries(monkeypatch):
    monkeypatch.setattr(
        cd, "call_llm",
        lambda prompt: '[{"index": 0, "conflicting": true, "explanation": "ok"}, {"conflicting": true}]',
    )
    chunks = [FakeChunk("A"), FakeChunk("B")]
    flags = cd.find_contradictions(chunks)
    assert len(flags) == 1


def test_find_contradictions_ignores_out_of_range_index(monkeypatch):
    monkeypatch.setattr(
        cd, "call_llm",
        lambda prompt: '[{"index": 99, "conflicting": true, "explanation": "bogus index"}]',
    )
    chunks = [FakeChunk("A"), FakeChunk("B")]
    flags = cd.find_contradictions(chunks)
    assert flags == []


def test_summarize_contradictions_none_when_no_flags():
    assert cd.summarize_contradictions([]) is None


def test_summarize_contradictions_includes_source_indices_and_explanation():
    flags = [cd.ContradictionFlag(source_index_a=1, source_index_b=3, explanation="conflicting dosage advice")]
    summary = cd.summarize_contradictions(flags)
    assert "[1]" in summary and "[3]" in summary
    assert "conflicting dosage advice" in summary
    assert "acknowledge" in summary.lower()


def test_summarize_contradictions_handles_multiple_flags():
    flags = [
        cd.ContradictionFlag(source_index_a=1, source_index_b=2, explanation="dosage"),
        cd.ContradictionFlag(source_index_a=2, source_index_b=3, explanation="cause"),
    ]
    summary = cd.summarize_contradictions(flags)
    assert "dosage" in summary and "cause" in summary