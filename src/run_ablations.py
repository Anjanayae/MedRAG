"""
run_ablations.py — Retrieval/chunking ablation study.

THIS SCRIPT IS FOR YOU TO RUN AGAINST YOUR OWN LIVE INDEX. It doesn't
simulate, estimate, or fabricate any numbers — every result comes from
actually calling DenseRetriever / BM25Retriever / HybridRetriever /
generate_answer against whatever's really indexed. If a strategy's index
isn't built yet, that strategy is skipped with a warning, not a crash.

NOTE (added after eval_holdout.py existed): the gold cases here come from
eval_dataset.py's real items + golden_eval.py's cases — the SAME corpus
your questions' answers are indexed in (see eval_holdout.py's docstring on
why that's a real leakage concern). If you want a leakage-clean version of
this comparison, swap build_gold_cases() below to pull from
eval_holdout.build_holdout_items() + point the retrievers at
eval_holdout.HOLDOUT_CHROMA_DIR instead — not done here to keep this
recreation faithful to what was actually built and tested before.

No new retrieval code: every configuration is just a different way of
calling DenseRetriever / BM25Retriever / HybridRetriever with their
existing toggles (use_reranker, use_query_rewrite).

Retrieval configurations tested:
  dense_only, bm25_only, hybrid_rrf, hybrid_rrf_rewrite, hybrid_rrf_rerank, full_pipeline

Usage:
    python src/run_ablations.py --strategies sentence fixed_size semantic
    python src/run_ablations.py --strategies sentence --with-generation --generation-sample-size 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval_dataset import build_eval_set
from evaluate import JUDGE_PROMPT_TEMPLATE, extract_pair_uid
from generation import build_context_block, call_llm, generate_answer
from golden_eval import GOLDEN_SET
from hybrid_retriever import HybridRetriever
from retriever import DenseRetriever
from sparse_retriever import BM25Retriever

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"

RETRIEVAL_CONFIGS = [
    "dense_only", "bm25_only", "hybrid_rrf",
    "hybrid_rrf_rewrite", "hybrid_rrf_rerank", "full_pipeline",
]
GENERATION_ABLATION_CONFIGS = ["full_pipeline", "hybrid_rrf", "dense_only"]


def build_gold_cases() -> list[dict]:
    cases = []
    for item in build_eval_set():
        if item["type"] == "real" and item.get("gold_pair_uid"):
            cases.append({"question": item["question"], "gold_pair_uids": {item["gold_pair_uid"]}})
    for case in GOLDEN_SET:
        if case["category"] in ("answerable", "difficult_retrieval", "multi_source") and case["relevant_sources"]:
            cases.append({"question": case["question"], "gold_pair_uids": set(case["relevant_sources"])})
    return cases


def recall_hit(chunks, gold_pair_uids: set[str]) -> bool:
    retrieved = {extract_pair_uid(c.chunk_id) for c in chunks}
    return bool(retrieved & gold_pair_uids)


def _retrieve_with_config(hybrid: HybridRetriever, dense: DenseRetriever, sparse: BM25Retriever,
                           config: str, query: str, top_k: int):
    if config == "dense_only":
        return dense.retrieve(query, top_k=top_k)
    if config == "bm25_only":
        return sparse.retrieve(query, top_k=top_k)
    if config == "hybrid_rrf":
        return hybrid.retrieve(query, top_k=top_k, use_reranker=False, use_query_rewrite=False)
    if config == "hybrid_rrf_rewrite":
        return hybrid.retrieve(query, top_k=top_k, use_reranker=False, use_query_rewrite=True)
    if config == "hybrid_rrf_rerank":
        return hybrid.retrieve(query, top_k=top_k, use_reranker=True, use_query_rewrite=False)
    if config == "full_pipeline":
        return hybrid.retrieve(query, top_k=top_k, use_reranker=True, use_query_rewrite=True)
    raise ValueError(f"unknown config: {config}")


def run_retrieval_ablation(strategy: str, gold_cases: list[dict], top_k: int = 5) -> dict | None:
    try:
        dense = DenseRetriever(strategy=strategy)
        sparse = BM25Retriever(strategy=strategy)
        hybrid = HybridRetriever(strategy=strategy)
    except Exception as e:
        print(f"[skip] strategy={strategy}: {e}  (index not built? see README 'Ingest data')")
        return None

    results = {}
    for config in RETRIEVAL_CONFIGS:
        hits = 0
        latencies = []
        for case in gold_cases:
            t0 = time.perf_counter()
            chunks = _retrieve_with_config(hybrid, dense, sparse, config, case["question"], top_k)
            latencies.append((time.perf_counter() - t0) * 1000)
            hits += recall_hit(chunks, case["gold_pair_uids"])
        results[config] = {
            "recall_at_k": round(hits / len(gold_cases), 3),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "n_cases": len(gold_cases),
        }
    return results


def run_generation_ablation(strategy: str, gold_cases: list[dict], sample_size: int, top_k: int = 5) -> dict:
    dense = DenseRetriever(strategy=strategy)
    sparse = BM25Retriever(strategy=strategy)
    hybrid = HybridRetriever(strategy=strategy)
    sample = gold_cases[:sample_size]

    results = {}
    for config in GENERATION_ABLATION_CONFIGS:
        rows = []
        for case in sample:
            chunks = _retrieve_with_config(hybrid, dense, sparse, config, case["question"], top_k)
            result = generate_answer(case["question"], chunks, verify_citations=True)
            if result.refused:
                rows.append({"refused": True, "groundedness": None, "relevance": None, "citation_coverage": None})
                continue

            context_block = build_context_block(chunks)
            judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=case["question"], context=context_block, answer=result.answer
            )
            judge_response = call_llm(judge_prompt)
            try:
                scores = json.loads(judge_response)
                groundedness, relevance = scores.get("groundedness"), scores.get("relevance")
            except json.JSONDecodeError:
                groundedness = relevance = None

            rows.append({
                "refused": False, "groundedness": groundedness, "relevance": relevance,
                "citation_coverage": result.citation_coverage,
            })

        valid = [r for r in rows if not r["refused"] and r["groundedness"] is not None]
        cov_valid = [r["citation_coverage"] for r in valid if r["citation_coverage"] is not None]
        results[config] = {
            "n_sample": len(sample),
            "n_refused": sum(1 for r in rows if r["refused"]),
            "avg_groundedness": round(sum(r["groundedness"] for r in valid) / len(valid), 2) if valid else None,
            "avg_relevance": round(sum(r["relevance"] for r in valid) / len(valid), 2) if valid else None,
            "avg_citation_coverage": round(sum(cov_valid) / len(cov_valid), 3) if cov_valid else None,
        }
    return results


def write_markdown_report(report: dict, out_path: Path) -> None:
    lines = ["# Retrieval / chunking ablation report", "",
             f"n_gold_cases={report['n_gold_cases']}, top_k={report['top_k']}", ""]
    for strategy, data in report["by_strategy"].items():
        lines.append(f"## strategy = {strategy}")
        lines.append("")
        lines.append("| Config | Recall@k | Avg latency (ms) |")
        lines.append("|---|---|---|")
        for config, stats in data["retrieval"].items():
            lines.append(f"| {config} | {stats['recall_at_k']} | {stats['avg_latency_ms']} |")
        lines.append("")
        if "generation" in data:
            lines.append("| Config | Avg groundedness | Avg relevance | Avg citation coverage | N refused |")
            lines.append("|---|---|---|---|---|")
            for config, stats in data["generation"].items():
                lines.append(
                    f"| {config} | {stats['avg_groundedness']} | {stats['avg_relevance']} | "
                    f"{stats['avg_citation_coverage']} | {stats['n_refused']} |"
                )
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval/chunking ablation study — run against your real, already-built index."
    )
    parser.add_argument("--strategies", nargs="+", default=["sentence", "fixed_size", "semantic"],
                         choices=["fixed_size", "sentence", "semantic"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--generation-sample-size", type=int, default=8)
    args = parser.parse_args()

    gold_cases = build_gold_cases()
    print(f"{len(gold_cases)} gold-labeled retrieval cases "
          f"(eval_dataset.py real items + golden_eval.py answerable/difficult_retrieval/multi_source)\n")

    report = {"top_k": args.top_k, "n_gold_cases": len(gold_cases), "by_strategy": {}}

    for strategy in args.strategies:
        print(f"=== strategy={strategy} ===")
        retrieval_results = run_retrieval_ablation(strategy, gold_cases, top_k=args.top_k)
        if retrieval_results is None:
            continue
        for config, stats in retrieval_results.items():
            print(f"  {config:20s} recall@{args.top_k}={stats['recall_at_k']:.3f}  "
                  f"avg_latency={stats['avg_latency_ms']:.1f}ms")

        strategy_report = {"retrieval": retrieval_results}

        if args.with_generation:
            n = min(args.generation_sample_size, len(gold_cases))
            print(f"  Running generation ablation on {n} cases x "
                  f"{len(GENERATION_ABLATION_CONFIGS)} configs — real API cost...")
            gen_results = run_generation_ablation(strategy, gold_cases, args.generation_sample_size, top_k=args.top_k)
            for config, stats in gen_results.items():
                print(f"    {config:20s} avg_groundedness={stats['avg_groundedness']} "
                      f"avg_citation_coverage={stats['avg_citation_coverage']} n_refused={stats['n_refused']}")
            strategy_report["generation"] = gen_results

        report["by_strategy"][strategy] = strategy_report
        print()

    if not report["by_strategy"]:
        print("No strategy produced results — build at least one index first.")
        return

    out_path = EVAL_DIR / "ablation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Full report -> {out_path}")

    md_path = EVAL_DIR / "ablation_report.md"
    write_markdown_report(report, md_path)
    print(f"Markdown report -> {md_path}")


if __name__ == "__main__":
    main()