import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run_ablations as ra  # noqa: E402
from retriever import RetrievedChunk  # noqa: E402


def make_chunk(pair_uid):
    return RetrievedChunk(chunk_id=f"{pair_uid}::sentence::0", chunk_text="text", score=1.0, metadata={})


def test_build_gold_cases_merges_both_sources(monkeypatch):
    monkeypatch.setattr(ra, "build_eval_set", lambda: [
        {"type": "real", "question": "q1", "gold_pair_uid": "1111"},
        {"type": "out_of_domain", "question": "q2", "expects_refusal": True},
    ])
    monkeypatch.setattr(ra, "GOLDEN_SET", [
        {"category": "answerable", "question": "q3", "relevant_sources": ["2222"]},
        {"category": "unanswerable", "question": "q4", "relevant_sources": []},
        {"category": "multi_source", "question": "q5", "relevant_sources": ["3333", "4444"]},
    ])
    cases = ra.build_gold_cases()
    questions = {c["question"]: c["gold_pair_uids"] for c in cases}
    assert questions["q1"] == {"1111"}
    assert questions["q3"] == {"2222"}
    assert questions["q5"] == {"3333", "4444"}
    assert "q2" not in questions
    assert "q4" not in questions


def test_recall_hit_true_on_any_match():
    chunks = [make_chunk("1111"), make_chunk("9999")]
    assert ra.recall_hit(chunks, {"1111", "2222"}) is True


def test_recall_hit_false_when_no_match():
    chunks = [make_chunk("9999")]
    assert ra.recall_hit(chunks, {"1111"}) is False


class Recorder:
    def __init__(self, tag):
        self.tag = tag
        self.calls = []

    def retrieve(self, query, top_k=5, **kwargs):
        self.calls.append({"query": query, "top_k": top_k, **kwargs})
        return [f"{self.tag}-result"]


def test_retrieve_with_config_dense_only_calls_dense_retriever():
    dense, sparse, hybrid = Recorder("dense"), Recorder("sparse"), Recorder("hybrid")
    result = ra._retrieve_with_config(hybrid, dense, sparse, "dense_only", "q", 5)
    assert result == ["dense-result"]
    assert len(dense.calls) == 1
    assert len(sparse.calls) == 0 and len(hybrid.calls) == 0


def test_retrieve_with_config_bm25_only_calls_sparse_retriever():
    dense, sparse, hybrid = Recorder("dense"), Recorder("sparse"), Recorder("hybrid")
    result = ra._retrieve_with_config(hybrid, dense, sparse, "bm25_only", "q", 5)
    assert result == ["sparse-result"]


def test_retrieve_with_config_hybrid_rrf_disables_rerank_and_rewrite():
    dense, sparse, hybrid = Recorder("dense"), Recorder("sparse"), Recorder("hybrid")
    ra._retrieve_with_config(hybrid, dense, sparse, "hybrid_rrf", "q", 5)
    call = hybrid.calls[0]
    assert call["use_reranker"] is False
    assert call["use_query_rewrite"] is False


def test_retrieve_with_config_full_pipeline_enables_both():
    dense, sparse, hybrid = Recorder("dense"), Recorder("sparse"), Recorder("hybrid")
    ra._retrieve_with_config(hybrid, dense, sparse, "full_pipeline", "q", 5)
    call = hybrid.calls[0]
    assert call["use_reranker"] is True
    assert call["use_query_rewrite"] is True


def test_retrieve_with_config_hybrid_rrf_rewrite_only_enables_rewrite():
    dense, sparse, hybrid = Recorder("dense"), Recorder("sparse"), Recorder("hybrid")
    ra._retrieve_with_config(hybrid, dense, sparse, "hybrid_rrf_rewrite", "q", 5)
    call = hybrid.calls[0]
    assert call["use_reranker"] is False
    assert call["use_query_rewrite"] is True


def test_retrieve_with_config_hybrid_rrf_rerank_only_enables_rerank():
    dense, sparse, hybrid = Recorder("dense"), Recorder("sparse"), Recorder("hybrid")
    ra._retrieve_with_config(hybrid, dense, sparse, "hybrid_rrf_rerank", "q", 5)
    call = hybrid.calls[0]
    assert call["use_reranker"] is True
    assert call["use_query_rewrite"] is False


def test_retrieve_with_config_unknown_raises():
    dense, sparse, hybrid = Recorder("dense"), Recorder("sparse"), Recorder("hybrid")
    try:
        ra._retrieve_with_config(hybrid, dense, sparse, "not-a-config", "q", 5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_all_retrieval_configs_are_dispatchable():
    dense, sparse, hybrid = Recorder("dense"), Recorder("sparse"), Recorder("hybrid")
    for config in ra.RETRIEVAL_CONFIGS:
        result = ra._retrieve_with_config(hybrid, dense, sparse, config, "q", 5)
        assert result


class FakeGoodRetriever:
    def __init__(self, strategy):
        pass

    def retrieve(self, query, top_k=5, **kwargs):
        return [make_chunk("1111")]


def test_run_retrieval_ablation_skips_missing_index(monkeypatch):
    def boom(strategy):
        raise RuntimeError("no such collection")
    monkeypatch.setattr(ra, "DenseRetriever", boom)
    monkeypatch.setattr(ra, "BM25Retriever", FakeGoodRetriever)
    monkeypatch.setattr(ra, "HybridRetriever", FakeGoodRetriever)

    result = ra.run_retrieval_ablation("sentence", [{"question": "q", "gold_pair_uids": {"1111"}}])
    assert result is None


def test_run_retrieval_ablation_computes_recall_and_latency(monkeypatch):
    monkeypatch.setattr(ra, "DenseRetriever", FakeGoodRetriever)
    monkeypatch.setattr(ra, "BM25Retriever", FakeGoodRetriever)
    monkeypatch.setattr(ra, "HybridRetriever", FakeGoodRetriever)

    gold_cases = [{"question": "q1", "gold_pair_uids": {"1111"}}, {"question": "q2", "gold_pair_uids": {"1111"}}]
    result = ra.run_retrieval_ablation("sentence", gold_cases, top_k=5)
    assert set(result.keys()) == set(ra.RETRIEVAL_CONFIGS)
    for config, stats in result.items():
        assert stats["recall_at_k"] == 1.0
        assert stats["n_cases"] == 2
        assert stats["avg_latency_ms"] >= 0


def test_write_markdown_report_renders_retrieval_and_generation(tmp_path):
    report = {
        "n_gold_cases": 10, "top_k": 5,
        "by_strategy": {
            "sentence": {
                "retrieval": {"dense_only": {"recall_at_k": 0.8, "avg_latency_ms": 12.3}},
                "generation": {"full_pipeline": {"avg_groundedness": 4.5, "avg_relevance": 4.2,
                                                   "avg_citation_coverage": 0.9, "n_refused": 1}},
            }
        },
    }
    out_path = tmp_path / "report.md"
    ra.write_markdown_report(report, out_path)
    content = out_path.read_text()
    assert "dense_only" in content
    assert "full_pipeline" in content
    assert "ablation report" in content.lower()