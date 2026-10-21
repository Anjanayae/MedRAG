import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import golden_eval as g  # noqa: E402
from retriever import RetrievedChunk  # noqa: E402


def make_chunk(pair_uid, score=5.0, text="chunk text"):
    return RetrievedChunk(chunk_id=f"{pair_uid}::sentence::0", chunk_text=text, score=score, metadata={})


def test_golden_set_has_expected_size_range():
    assert 30 <= len(g.GOLDEN_SET) <= 40


def test_golden_set_covers_all_six_categories():
    categories = {c["category"] for c in g.GOLDEN_SET}
    assert categories == {
        "answerable", "unanswerable", "ambiguous",
        "difficult_retrieval", "multi_source", "adversarial_edge",
    }


def test_golden_set_ids_are_unique():
    ids = [c["id"] for c in g.GOLDEN_SET]
    assert len(ids) == len(set(ids))


def test_golden_set_every_case_has_required_fields():
    required = {"id", "category", "must_abstain", "question", "key_facts", "relevant_sources", "note"}
    for case in g.GOLDEN_SET:
        assert required.issubset(case.keys()), f"{case.get('id')} missing fields"


def test_answerable_cases_do_not_require_abstention():
    for case in g.GOLDEN_SET:
        if case["category"] == "answerable":
            assert case["must_abstain"] is False


def test_unanswerable_and_adversarial_cases_require_abstention():
    for case in g.GOLDEN_SET:
        if case["category"] in ("unanswerable", "adversarial_edge"):
            assert case["must_abstain"] is True


def test_multi_source_cases_have_at_least_two_relevant_sources():
    for case in g.GOLDEN_SET:
        if case["category"] == "multi_source":
            assert len(case["relevant_sources"]) >= 2, case["id"]


def test_pair_uid_hit_true_when_any_source_matches():
    chunks = [make_chunk("0000123"), make_chunk("0000999")]
    assert g._pair_uid_hit(chunks, ["0000123"]) is True


def test_pair_uid_hit_true_with_multi_source_matching_second():
    chunks = [make_chunk("0000999")]
    assert g._pair_uid_hit(chunks, ["0000123", "0000999"]) is True


def test_pair_uid_hit_false_when_no_source_matches():
    chunks = [make_chunk("0000999")]
    assert g._pair_uid_hit(chunks, ["0000123"]) is False


def test_pair_uid_hit_none_when_relevant_sources_empty():
    chunks = [make_chunk("0000999")]
    assert g._pair_uid_hit(chunks, []) is None


def test_curated_fact_recall_full_match():
    facts = ["pinworm", "children", "hygiene"]
    answer = "Pinworm infection is common in children and spreads without good hygiene."
    assert g.curated_fact_recall(answer, facts) == 1.0


def test_curated_fact_recall_partial_match():
    facts = ["pinworm", "unicorn"]
    answer = "Pinworm infection is common in children."
    assert g.curated_fact_recall(answer, facts) == 0.5


def test_curated_fact_recall_none_when_no_facts():
    assert g.curated_fact_recall("some answer", []) is None


class FakeHybridRetriever:
    def __init__(self, strategy):
        self.strategy = strategy

    def retrieve(self, query, top_k=5):
        if "pinworm" in query.lower():
            return [make_chunk("0016318")]
        return [make_chunk("0000000")]


def test_run_golden_eval_without_generation_runs_all_cases(monkeypatch):
    monkeypatch.setattr(g, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(g, "check_confidence", lambda chunks: (0.9, False))

    report = g.run_golden_eval(strategy="sentence", with_generation=False)
    assert report["n_cases"] == len(g.GOLDEN_SET)
    assert "generation_sample" not in report
    assert len(report["per_item"]) == len(g.GOLDEN_SET)


def test_run_golden_eval_by_category_reports_all_categories(monkeypatch):
    monkeypatch.setattr(g, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(g, "check_confidence", lambda chunks: (0.9, False))

    report = g.run_golden_eval(strategy="sentence")
    assert set(report["by_category"].keys()) == {
        "answerable", "unanswerable", "ambiguous",
        "difficult_retrieval", "multi_source", "adversarial_edge",
    }


def test_run_golden_eval_abstain_accuracy_reflects_refusal_gate(monkeypatch):
    monkeypatch.setattr(g, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(g, "check_confidence", lambda chunks: (0.0, True))

    report = g.run_golden_eval(strategy="sentence")
    answerable_stats = report["by_category"]["answerable"]
    unanswerable_stats = report["by_category"]["unanswerable"]
    assert answerable_stats["abstain_accuracy"] == 0.0
    assert unanswerable_stats["abstain_accuracy"] == 1.0


def test_run_golden_eval_with_generation_calls_sample_helper(monkeypatch):
    monkeypatch.setattr(g, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(g, "check_confidence", lambda chunks: (0.9, False))
    monkeypatch.setattr(g, "_run_generation_sample", lambda hybrid, top_k, n: [{"id": "g_ans_01", "fake": True}])

    report = g.run_golden_eval(strategy="sentence", with_generation=True, generation_sample_size=3)
    assert report["generation_sample"] == [{"id": "g_ans_01", "fake": True}]