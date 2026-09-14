import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import generation as gen  # noqa: E402
from retriever import RetrievedChunk  # noqa: E402


def make_chunk(score, focus="Diabetes", text="Sample chunk text."):
    return RetrievedChunk(
        chunk_id="c1", chunk_text=text, score=score,
        metadata={"focus": focus, "source": "NIDDK", "url": "u1", "question": "q?"},
    )


def test_sigmoid_bounds():
    assert 0 < gen.sigmoid(-100) < 0.01
    assert 0.99 < gen.sigmoid(100) <= 1.0
    assert abs(gen.sigmoid(0) - 0.5) < 1e-9


def test_no_chunks_refuses_without_calling_groq(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("call_groq should not be called when there are no chunks")
    monkeypatch.setattr(gen, "call_groq", boom)

    result = gen.generate_answer("What is diabetes?", [])
    assert result.refused is True
    assert result.confidence == 0.0
    assert result.answer == gen.REFUSAL_MESSAGE


def test_low_confidence_refuses_without_calling_groq(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("call_groq should not be called below confidence threshold")
    monkeypatch.setattr(gen, "call_groq", boom)

    # A very negative rerank score -> sigmoid near 0 -> below threshold
    low_conf_chunk = make_chunk(score=-10.0)
    result = gen.generate_answer("obscure query", [low_conf_chunk])
    assert result.refused is True
    assert result.confidence < gen.RERANK_CONFIDENCE_THRESHOLD


def test_high_confidence_calls_groq_and_returns_answer(monkeypatch):
    monkeypatch.setattr(gen, "call_groq", lambda prompt, model: "Mocked answer citing [1].")

    high_conf_chunk = make_chunk(score=10.0)
    result = gen.generate_answer("What is diabetes?", [high_conf_chunk])
    assert result.refused is False
    assert result.answer == "Mocked answer citing [1]."
    assert result.confidence > gen.RERANK_CONFIDENCE_THRESHOLD


def test_build_sources_includes_expected_fields():
    chunks = [make_chunk(score=5.0, focus="Asthma")]
    sources = gen.build_sources(chunks)
    assert sources[0]["focus"] == "Asthma"
    assert sources[0]["index"] == 1
    assert "score" in sources[0]


def test_build_context_block_numbers_sources():
    chunks = [make_chunk(score=1.0, text="First chunk"), make_chunk(score=2.0, text="Second chunk")]
    block = gen.build_context_block(chunks)
    assert "[1]" in block and "[2]" in block
    assert "First chunk" in block and "Second chunk" in block


# --- provider dispatch (call_llm) ---

def test_call_llm_defaults_to_groq_and_respects_monkeypatch(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(gen, "call_groq", lambda prompt, model: f"groq:{model}")
    assert gen.call_llm("hi") == f"groq:{gen.DEFAULT_GROQ_MODEL}"


def test_call_llm_routes_to_openai_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(gen, "call_openai", lambda prompt, model: f"openai:{model}")
    assert gen.call_llm("hi") == f"openai:{gen.DEFAULT_OPENAI_MODEL}"


def test_call_llm_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    try:
        gen.call_llm("hi")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_call_llm_explicit_model_overrides_provider_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(gen, "call_groq", lambda prompt, model: f"groq:{model}")
    assert gen.call_llm("hi", model="custom-model") == "groq:custom-model"


# --- citation verification / composite confidence integration ---

def test_generate_answer_without_verify_citations_leaves_new_fields_none(monkeypatch):
    monkeypatch.setattr(gen, "call_groq", lambda prompt, model: "Answer citing [1].")
    result = gen.generate_answer("What is diabetes?", [make_chunk(score=10.0)])
    assert result.citation_checks is None
    assert result.composite_confidence is None


def test_generate_answer_with_verify_citations_populates_composite_confidence(monkeypatch):
    monkeypatch.setattr(gen, "call_groq", lambda prompt, model: "Diabetes causes thirst [1].")
    monkeypatch.setattr("citation_verify.call_llm", lambda prompt: '{"supported": true, "reason": "ok"}')

    chunk = make_chunk(score=10.0, text="Diabetes is associated with excessive thirst.")
    result = gen.generate_answer("What is diabetes?", [chunk], verify_citations=True)

    assert result.citation_coverage == 1.0
    assert result.source_utilization == 1.0
    assert result.composite_confidence is not None
    assert 0.0 <= result.composite_confidence <= 1.0


def test_refused_answers_never_run_citation_verification(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("verification should not run on a refused answer")
    monkeypatch.setattr("citation_verify.verify_answer_citations", boom)

    low_conf_chunk = make_chunk(score=-10.0)
    result = gen.generate_answer("obscure query", [low_conf_chunk], verify_citations=True)
    assert result.refused is True
    assert result.citation_checks is None


def test_compute_composite_confidence_weights_sum_matches_inputs():
    score = gen.compute_composite_confidence(
        retrieval_confidence=1.0, citation_coverage_score=1.0, source_utilization_score=1.0
    )
    assert abs(score - 1.0) < 1e-9
    score_zero = gen.compute_composite_confidence(0.0, 0.0, 0.0)
    assert score_zero == 0.0


def test_compute_source_utilization_counts_unique_cited_sources():
    checks = [
        gen_citation_check(marker=1, source_index=1),
        gen_citation_check(marker=1, source_index=1),  # duplicate marker, same source
    ]
    util = gen.compute_source_utilization({1: "a", 2: "b"}, checks)
    assert util == 0.5  # only source 1 of 2 was ever cited


def gen_citation_check(marker, source_index):
    from citation_verify import CitationCheck
    return CitationCheck(marker=marker, claim_text="x", source_index=source_index, supported=True)