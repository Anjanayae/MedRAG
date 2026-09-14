import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import citation_verify as cv  # noqa: E402


def test_extract_cited_sentences_single_marker():
    answer = "Diabetes causes increased thirst [1]. It also causes fatigue [2]."
    result = cv.extract_cited_sentences(answer)
    assert result[0] == ("Diabetes causes increased thirst [1].", [1])
    assert result[1] == ("It also causes fatigue [2].", [2])


def test_extract_cited_sentences_multiple_markers_one_sentence():
    answer = "This is supported by two sources [1][2]."
    result = cv.extract_cited_sentences(answer)
    assert result[0][1] == [1, 2]


def test_extract_cited_sentences_no_marker():
    answer = "This sentence cites nothing."
    result = cv.extract_cited_sentences(answer)
    assert result[0][1] == []


def test_verify_citation_parses_supported_true(monkeypatch):
    monkeypatch.setattr(cv, "call_llm", lambda prompt: '{"supported": true, "reason": "ok"}')
    supported, raw = cv.verify_citation("claim", "source text")
    assert supported is True


def test_verify_citation_parses_supported_false(monkeypatch):
    monkeypatch.setattr(cv, "call_llm", lambda prompt: '{"supported": false, "reason": "no"}')
    supported, raw = cv.verify_citation("claim", "source text")
    assert supported is False


def test_verify_citation_handles_unparseable_response(monkeypatch):
    monkeypatch.setattr(cv, "call_llm", lambda prompt: "not json")
    supported, raw = cv.verify_citation("claim", "source text")
    assert supported is None
    assert raw == "not json"


def test_verify_answer_citations_flags_missing_index_without_llm_call(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("should not call the judge for a citation index that doesn't exist")
    monkeypatch.setattr(cv, "call_llm", boom)

    answer = "This claims something from a source that isn't there [9]."
    checks = cv.verify_answer_citations(answer, chunk_texts={1: "Some real source text."})
    assert len(checks) == 1
    assert checks[0].supported is False
    assert checks[0].marker == 9


def test_verify_answer_citations_calls_judge_for_valid_index(monkeypatch):
    monkeypatch.setattr(cv, "call_llm", lambda prompt: '{"supported": true, "reason": "ok"}')
    answer = "Diabetes causes thirst [1]."
    checks = cv.verify_answer_citations(answer, chunk_texts={1: "Diabetes is associated with excessive thirst."})
    assert len(checks) == 1
    assert checks[0].supported is True
    assert checks[0].source_index == 1


def test_citation_coverage_empty_checks_returns_full_credit():
    assert cv.citation_coverage([]) == 1.0


def test_citation_coverage_mixed_results():
    checks = [
        cv.CitationCheck(marker=1, claim_text="a", source_index=1, supported=True),
        cv.CitationCheck(marker=2, claim_text="b", source_index=2, supported=False),
    ]
    assert cv.citation_coverage(checks) == 0.5


def test_citation_coverage_all_unparseable_gives_half_credit():
    checks = [
        cv.CitationCheck(marker=1, claim_text="a", source_index=1, supported=None),
    ]
    assert cv.citation_coverage(checks) == 0.5


def test_uncited_claim_fraction_all_cited():
    answer = "Diabetes causes thirst [1]. It also causes fatigue [2]."
    assert cv.uncited_claim_fraction(answer) == 0.0


def test_uncited_claim_fraction_none_cited():
    answer = "Diabetes causes increased thirst. It also causes fatigue."
    assert cv.uncited_claim_fraction(answer) == 1.0


def test_uncited_claim_fraction_ignores_trivial_sentences():
    # Short filler sentences (<=3 words) shouldn't count as "substantive
    # claims" that need a citation.
    answer = "Sure. Diabetes causes increased thirst [1]."
    assert cv.uncited_claim_fraction(answer) == 0.0