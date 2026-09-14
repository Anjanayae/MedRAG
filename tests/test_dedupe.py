import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from chunking import Chunk  # noqa: E402
from dedupe import remove_near_duplicates  # noqa: E402


def make_chunk(chunk_id, focus, text, strategy="sentence"):
    return Chunk(
        chunk_id=chunk_id, doc_id="d1", qid="q1", focus=focus, qtype="info",
        source="NIDDK", url="http://x", question="q?", chunk_text=text,
        chunk_index=0, strategy=strategy,
    )


def fake_embed_fn(vectors_by_text: dict[str, list[float]]):
    """Returns an embed_fn (list[str] -> ndarray) driven by a lookup table,
    so tests control exact similarity without a real model."""
    def _fn(texts):
        return np.array([vectors_by_text[t] for t in texts], dtype="float32")
    return _fn


def test_no_duplicates_within_bucket_keeps_all():
    chunks = [
        make_chunk("c1", "Diabetes", "Diabetes causes thirst."),
        make_chunk("c2", "Diabetes", "Diabetes causes fatigue."),
    ]
    embed_fn = fake_embed_fn({
        "Diabetes causes thirst.": [1.0, 0.0],
        "Diabetes causes fatigue.": [0.0, 1.0],  # orthogonal -> similarity 0
    })
    deduped, report = remove_near_duplicates(chunks, embed_fn, similarity_threshold=0.95)
    assert len(deduped) == 2
    assert report.n_removed == 0


def test_near_identical_chunks_in_same_bucket_are_deduped():
    chunks = [
        make_chunk("c1", "Diabetes", "Diabetes causes thirst."),
        make_chunk("c2", "Diabetes", "Diabetes causes thirst and fatigue."),  # near-dup of c1
    ]
    embed_fn = fake_embed_fn({
        "Diabetes causes thirst.": [1.0, 0.0],
        "Diabetes causes thirst and fatigue.": [0.999, 0.001],  # cosine ~1.0 with c1
    })
    deduped, report = remove_near_duplicates(chunks, embed_fn, similarity_threshold=0.95)
    assert len(deduped) == 1
    assert deduped[0].chunk_id == "c1"  # first occurrence wins
    assert report.n_removed == 1


def test_different_focus_never_compared_even_if_similar_text(monkeypatch):
    # Two DIFFERENT focuses with byte-identical text — should NOT be
    # deduped against each other, because near-dups are only ever
    # meaningful within the same topic (see dedupe.py docstring).
    def boom(texts):
        raise AssertionError("embed_fn should be called once per bucket, not across buckets")

    chunks = [
        make_chunk("c1", "Diabetes", "Some info."),
        make_chunk("c2", "Asthma", "Some info."),
    ]
    # Buckets of size 1 never call embed_fn at all (see remove_near_duplicates
    # short-circuit), so this should just pass both through untouched.
    deduped, report = remove_near_duplicates(chunks, boom, similarity_threshold=0.95)
    assert len(deduped) == 2
    assert report.n_removed == 0


def test_three_way_duplicate_cluster_keeps_only_first():
    chunks = [
        make_chunk("c1", "Diabetes", "text a"),
        make_chunk("c2", "Diabetes", "text b"),
        make_chunk("c3", "Diabetes", "text c"),
    ]
    embed_fn = fake_embed_fn({
        "text a": [1.0, 0.0],
        "text b": [1.0, 0.0],  # identical to a
        "text c": [1.0, 0.0],  # identical to a and b
    })
    deduped, report = remove_near_duplicates(chunks, embed_fn, similarity_threshold=0.95)
    assert len(deduped) == 1
    assert deduped[0].chunk_id == "c1"
    assert report.n_removed == 2


def test_report_counts_are_consistent():
    chunks = [
        make_chunk("c1", "Diabetes", "text a"),
        make_chunk("c2", "Diabetes", "text a duplicate"),
        make_chunk("c3", "Asthma", "unrelated"),
    ]
    embed_fn = fake_embed_fn({
        "text a": [1.0, 0.0],
        "text a duplicate": [1.0, 0.0],
        "unrelated": [0.0, 1.0],
    })
    deduped, report = remove_near_duplicates(chunks, embed_fn, similarity_threshold=0.95)
    assert report.n_before == 3
    assert report.n_after == len(deduped) == 2
    assert report.n_removed == 1
    assert report.n_buckets_with_dupes == 1