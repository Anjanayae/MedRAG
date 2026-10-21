import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import eval_holdout as eh  # noqa: E402
from retriever import RetrievedChunk  # noqa: E402


def make_chunk(pair_uid, strategy="sentence"):
    return RetrievedChunk(chunk_id=f"{pair_uid}::{strategy}::0", chunk_text="text", score=1.0, metadata={})


def make_rows():
    """Synthetic corpus: SourceA has 10 docs x 2 QA pairs each, SourceB has
    6 docs x 1 QA pair each -- enough documents per source for a stratified
    split with a small heldout_fraction to have >=1 heldout doc per source."""
    rows = []
    pair_id = 0
    for d in range(10):
        for q in range(2):
            rows.append({
                "pair_uid": f"{pair_id:07d}", "doc_id": f"A_doc{d}", "source": "SourceA",
                "focus": f"FocusA{d}", "qtype": "info", "question": f"Q A{d}-{q}",
                "answer": " ".join(["word"] * 50),
            })
            pair_id += 1
    for d in range(6):
        rows.append({
            "pair_uid": f"{pair_id:07d}", "doc_id": f"B_doc{d}", "source": "SourceB",
            "focus": f"FocusB{d}", "qtype": "info", "question": f"Q B{d}",
            "answer": " ".join(["word"] * 50),
        })
        pair_id += 1
    return rows


# --- split_doc_ids ---

def test_split_is_a_full_partition_with_no_overlap():
    rows = make_rows()
    dev, heldout = eh.split_doc_ids(rows=rows, heldout_fraction=0.2, seed=42)
    all_doc_ids = {r["doc_id"] for r in rows}
    assert dev | heldout == all_doc_ids
    assert dev & heldout == set()


def test_split_is_stratified_across_sources():
    rows = make_rows()
    _, heldout = eh.split_doc_ids(rows=rows, heldout_fraction=0.2, seed=42)
    heldout_sources = {r["source"] for r in rows if r["doc_id"] in heldout}
    assert heldout_sources == {"SourceA", "SourceB"}  # both sources represented, not just the bigger one


def test_split_is_deterministic_for_same_seed():
    rows = make_rows()
    dev1, heldout1 = eh.split_doc_ids(rows=rows, heldout_fraction=0.2, seed=42)
    dev2, heldout2 = eh.split_doc_ids(rows=rows, heldout_fraction=0.2, seed=42)
    assert dev1 == dev2
    assert heldout1 == heldout2


def test_split_differs_for_different_seed():
    rows = make_rows()
    _, heldout_a = eh.split_doc_ids(rows=rows, heldout_fraction=0.2, seed=42)
    _, heldout_b = eh.split_doc_ids(rows=rows, heldout_fraction=0.2, seed=999)
    assert heldout_a != heldout_b


def test_no_qa_pair_doc_id_appears_on_both_sides():
    rows = make_rows()
    dev, heldout = eh.split_doc_ids(rows=rows, heldout_fraction=0.2, seed=42)
    for r in rows:
        assert not (r["doc_id"] in dev and r["doc_id"] in heldout)


def test_heldout_fraction_roughly_respected_per_source():
    rows = make_rows()
    dev, heldout = eh.split_doc_ids(rows=rows, heldout_fraction=0.2, seed=42)
    a_heldout = {d for d in heldout if d.startswith("A_doc")}
    b_heldout = {d for d in heldout if d.startswith("B_doc")}
    assert 1 <= len(a_heldout) <= 3   # ~20% of 10 docs
    assert 1 <= len(b_heldout) <= 2   # ~20% of 6 docs, floor of >=1 per source


# --- build_holdout_items ---

def test_holdout_items_only_come_from_heldout_doc_ids():
    rows = make_rows()
    dev, heldout = eh.split_doc_ids(rows=rows, heldout_fraction=0.3, seed=42)
    items = eh.build_holdout_items(heldout, rows=rows, n_per_source=2, seed=42)
    assert len(items) > 0
    for item in items:
        assert item["gold_doc_id"] in heldout
        assert item["gold_doc_id"] not in dev


def test_holdout_items_have_expected_schema():
    rows = make_rows()
    _, heldout = eh.split_doc_ids(rows=rows, heldout_fraction=0.3, seed=42)
    items = eh.build_holdout_items(heldout, rows=rows, n_per_source=2, seed=42)
    required = {"question", "type", "expects_refusal", "gold_pair_uid", "gold_doc_id", "gold_focus", "gold_source"}
    for item in items:
        assert required.issubset(item.keys())
        assert item["type"] == "real"
        assert item["expects_refusal"] is False


def test_holdout_items_respects_n_per_source_cap():
    rows = make_rows()
    _, heldout = eh.split_doc_ids(rows=rows, heldout_fraction=0.5, seed=42)  # bigger heldout pool
    items = eh.build_holdout_items(heldout, rows=rows, n_per_source=2, seed=42)
    by_source = {}
    for item in items:
        by_source.setdefault(item["gold_source"], 0)
        by_source[item["gold_source"]] += 1
    for count in by_source.values():
        assert count <= 2


def test_holdout_items_deterministic_for_same_seed():
    rows = make_rows()
    _, heldout = eh.split_doc_ids(rows=rows, heldout_fraction=0.3, seed=42)
    items1 = eh.build_holdout_items(heldout, rows=rows, n_per_source=2, seed=7)
    items2 = eh.build_holdout_items(heldout, rows=rows, n_per_source=2, seed=7)
    assert [i["gold_pair_uid"] for i in items1] == [i["gold_pair_uid"] for i in items2]


# --- reciprocal_rank ---

def test_reciprocal_rank_first_position():
    chunks = [make_chunk("0000001"), make_chunk("0000002")]
    assert eh.reciprocal_rank(chunks, "0000001") == 1.0


def test_reciprocal_rank_second_position():
    chunks = [make_chunk("0000002"), make_chunk("0000001")]
    assert eh.reciprocal_rank(chunks, "0000001") == 0.5


def test_reciprocal_rank_not_found_returns_zero():
    chunks = [make_chunk("0000002")]
    assert eh.reciprocal_rank(chunks, "0000001") == 0.0


def test_reciprocal_rank_empty_list_returns_zero():
    assert eh.reciprocal_rank([], "0000001") == 0.0

    