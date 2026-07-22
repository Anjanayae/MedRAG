import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chunking import split_sentences, fixed_size_chunks, sentence_chunks  # noqa: E402


def test_split_sentences_basic():
    text = "Hypopharyngeal cancer is rare. It affects the throat. Treatment varies."
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "Hypopharyngeal cancer is rare."


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_fixed_size_chunks_respects_size():
    text = "word " * 1000  # 5000 chars
    chunks = fixed_size_chunks(text, chunk_size=800, overlap=100)
    assert all(len(c) <= 800 for c in chunks)
    assert len(chunks) > 1


def test_fixed_size_chunks_short_text_single_chunk():
    text = "Short answer."
    chunks = fixed_size_chunks(text, chunk_size=800)
    assert chunks == [text]


def test_sentence_chunks_never_exceeds_budget():
    # A pathological "table-like" answer with no sentence punctuation —
    # this is the real edge case found in MedQuAD (symptom tables).
    text = ("Symptom Frequency " * 200).strip()
    chunks = sentence_chunks(text, max_tokens=50)
    max_chars = 50 * 4
    assert all(len(c) <= max_chars for c in chunks), (
        "sentence_chunks must fall back to hard-splitting oversized "
        "'sentences' (e.g. tabular content with no terminal punctuation)"
    )


def test_sentence_chunks_no_text_loss():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = sentence_chunks(text, max_tokens=100)
    rejoined = " ".join(chunks)
    for word in ["First", "Second", "Third"]:
        assert word in rejoined
