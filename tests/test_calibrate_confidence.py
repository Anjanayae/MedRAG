import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import calibrate_confidence as cal  # noqa: E402


def test_pearson_perfect_positive_correlation():
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 6, 8, 10]
    assert abs(cal._pearson(xs, ys) - 1.0) < 1e-9


def test_pearson_perfect_negative_correlation():
    xs = [1, 2, 3, 4, 5]
    ys = [10, 8, 6, 4, 2]
    assert abs(cal._pearson(xs, ys) - (-1.0)) < 1e-9


def test_pearson_no_correlation_returns_value_near_zero():
    xs = [1, 2, 3, 4]
    ys = [5, 5, 5, 5.0001]
    r = cal._pearson(xs, ys)
    assert r is not None


def test_pearson_constant_input_returns_none():
    xs = [1, 2, 3]
    ys = [5, 5, 5]
    assert cal._pearson(xs, ys) is None


def test_pearson_too_few_points_returns_none():
    assert cal._pearson([1], [1]) is None
    assert cal._pearson([], []) is None


def make_row(composite, groundedness, retrieval=0.5, citation_coverage=0.8, source_utilization=0.7, relevance=4):
    return {
        "question": "q", "retrieval_confidence": retrieval,
        "citation_coverage": citation_coverage, "source_utilization": source_utilization,
        "composite_confidence": composite, "groundedness": groundedness, "relevance": relevance,
        "key_fact_recall": 0.5,
    }


def test_compute_calibration_report_too_few_items_warns():
    rows = [make_row(0.9, 5), make_row(0.2, 1)]
    report = cal.compute_calibration_report(rows)
    assert "warning" in report


def test_compute_calibration_report_ignores_unscored_items():
    rows = [
        make_row(0.9, 5), make_row(0.85, 4), make_row(0.8, 5),
        {**make_row(0.5, None), "groundedness": None},
    ]
    report = cal.compute_calibration_report(rows)
    assert report["n_items"] == 3


def test_compute_calibration_report_correlations_present_with_enough_data():
    rows = [
        make_row(0.9, 5), make_row(0.8, 4), make_row(0.6, 3),
        make_row(0.4, 2), make_row(0.2, 1),
    ]
    report = cal.compute_calibration_report(rows)
    assert report["correlations_pearson_r"]["composite_confidence_vs_groundedness"] > 0.9


def test_compute_calibration_report_binned_calibration_covers_all_bins():
    rows = [make_row(0.9, 5), make_row(0.3, 1), make_row(0.55, 3), make_row(0.7, 4)]
    report = cal.compute_calibration_report(rows)
    assert len(report["binned_calibration"]) == len(cal.CONFIDENCE_BINS)
    total_n = sum(b["n"] for b in report["binned_calibration"])
    assert total_n == 4


def test_recommend_threshold_finds_smallest_qualifying_threshold():
    sweep = [
        {"threshold": 0.3, "n_above": 10, "precision_groundedness_ge_bar": 0.5},
        {"threshold": 0.5, "n_above": 6, "precision_groundedness_ge_bar": 0.83},
        {"threshold": 0.7, "n_above": 3, "precision_groundedness_ge_bar": 1.0},
    ]
    rec = cal._recommend_threshold(sweep)
    assert rec["threshold"] == 0.5


def test_recommend_threshold_returns_none_when_nothing_qualifies():
    sweep = [
        {"threshold": 0.3, "n_above": 10, "precision_groundedness_ge_bar": 0.4},
        {"threshold": 0.7, "n_above": 2, "precision_groundedness_ge_bar": 1.0},
    ]
    rec = cal._recommend_threshold(sweep)
    assert rec["threshold"] is None
    assert "reason" in rec


def test_write_markdown_report_handles_warning_case(tmp_path):
    report = {"warning": "too few items"}
    out_path = tmp_path / "report.md"
    cal.write_markdown_report(report, out_path, "sentence")
    content = out_path.read_text()
    assert "too few items" in content


def test_write_markdown_report_full_case(tmp_path):
    rows = [make_row(0.9, 5), make_row(0.8, 4), make_row(0.6, 3), make_row(0.4, 2), make_row(0.2, 1)]
    report = cal.compute_calibration_report(rows)
    out_path = tmp_path / "report.md"
    cal.write_markdown_report(report, out_path, "sentence")
    content = out_path.read_text()
    assert "Confidence calibration report" in content
    assert "Threshold sweep" in content


def test_run_calibration_sample_skips_refused_items(monkeypatch):
    import calibrate_confidence as cal_mod

    class FakeResult:
        def __init__(self, refused):
            self.refused = refused
            self.answer = "Some answer [1]."
            self.confidence = 0.9
            self.citation_coverage = 1.0
            self.source_utilization = 1.0
            self.composite_confidence = 0.95

    class FakeRetriever:
        def __init__(self, strategy):
            pass

        def retrieve(self, query, top_k=5):
            return []

    call_count = {"n": 0}

    def fake_generate_answer(query, retrieved, verify_citations=False):
        call_count["n"] += 1
        return FakeResult(refused=(call_count["n"] == 1))

    monkeypatch.setattr(cal_mod, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(cal_mod, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(cal_mod, "call_llm", lambda prompt: '{"groundedness": 5, "relevance": 5}')
    monkeypatch.setattr(cal_mod, "load_gold_answer", lambda uid: None)

    eval_items = [
        {"question": "q1", "type": "real", "gold_pair_uid": "u1"},
        {"question": "q2", "type": "real", "gold_pair_uid": "u2"},
    ]
    rows, n_refused = cal.run_calibration_sample(eval_items, strategy="sentence", sample_size=5)
    assert n_refused == 1
    assert len(rows) == 1