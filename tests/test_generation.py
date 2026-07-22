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