"""
calibrate_confidence.py — Empirical calibration experiment for the
composite confidence score in generation.py.

Question this answers: does composite_confidence (0.5*retrieval +
0.3*citation_coverage + 0.2*source_utilization) actually predict answer
correctness better than retrieval_confidence alone (the existing
pre-generation refusal-gate signal)? That combination of weights was
never validated against real outcomes when it was added — this script
produces that evidence.

This is READ-ONLY measurement. It does not change generation.py's
COMPOSITE_WEIGHTS, RERANK_CONFIDENCE_THRESHOLD, or the refusal gate.
Whether to act on the recommendation this script prints is a follow-up
decision, not something automated here.

Reuses, rather than reimplements:
  - eval_dataset.build_eval_set()               real/answerable items
  - hybrid_retriever.HybridRetriever             same retrieval as evaluate.py
  - generation.generate_answer(verify_citations=True)
        already computes retrieval confidence, citation_coverage,
        source_utilization, and composite_confidence in one call
  - generation.build_context_block               exact same context the
        answer was generated from, so the judge sees what the model saw
  - evaluate.JUDGE_PROMPT_TEMPLATE + call_llm     groundedness/relevance
        judge score, used here as the "actual correctness" signal
  - evaluate.key_fact_recall / load_gold_answer   second, free correctness
        proxy against the original MedQuAD answer

Usage:
    python src/calibrate_confidence.py --strategy sentence --sample-size 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_dataset import build_eval_set
from evaluate import JUDGE_PROMPT_TEMPLATE, key_fact_recall, load_gold_answer
from generation import build_context_block, call_llm, generate_answer
from hybrid_retriever import HybridRetriever

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"

CONFIDENCE_BINS = [(0.0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)]
THRESHOLD_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GOOD_GROUNDEDNESS_BAR = 4


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Plain-Python Pearson correlation — no numpy/scipy dependency needed
    for a sample this small. Returns None if undefined (fewer than 2
    points, or one variable is constant across the whole sample)."""
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x**0.5 * var_y**0.5)


def run_calibration_sample(eval_items: list[dict], strategy: str, sample_size: int = 20) -> tuple[list[dict], int]:
    """Runs up to sample_size real/answerable items through full generation
    WITH citation verification on, and grades each with the existing
    groundedness/relevance judge + key-fact recall.

    Returns (rows, n_refused). Refused items are counted but not scored
    here — their confidence-vs-outcome tradeoff is a different question
    (already covered by evaluate.py's refusal-gate-accuracy metric)."""
    hybrid = HybridRetriever(strategy=strategy)
    real_items = [i for i in eval_items if i["type"] == "real"][:sample_size]

    rows = []
    n_refused = 0
    for item in real_items:
        query = item["question"]
        retrieved = hybrid.retrieve(query, top_k=5)
        result = generate_answer(query, retrieved, verify_citations=True)

        if result.refused:
            n_refused += 1
            continue

        context_block = build_context_block(retrieved)
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(question=query, context=context_block, answer=result.answer)
        judge_response = call_llm(judge_prompt)
        try:
            scores = json.loads(judge_response)
            groundedness, relevance = scores.get("groundedness"), scores.get("relevance")
        except json.JSONDecodeError:
            groundedness = relevance = None

        gold_answer = load_gold_answer(item.get("gold_pair_uid", ""))
        kf_recall = key_fact_recall(result.answer, gold_answer) if gold_answer else None

        rows.append({
            "question": query,
            "retrieval_confidence": result.confidence,
            "citation_coverage": result.citation_coverage,
            "source_utilization": result.source_utilization,
            "composite_confidence": result.composite_confidence,
            "groundedness": groundedness,
            "relevance": relevance,
            "key_fact_recall": kf_recall,
        })

    return rows, n_refused


def _recommend_threshold(sweep: list[dict]) -> dict:
    """Picks the smallest threshold that reaches 80% precision (fraction of
    above-threshold answers judged groundedness >= GOOD_GROUNDEDNESS_BAR)
    while keeping at least 3 items above it. Reports "insufficient data"
    rather than guessing if nothing clears that bar."""
    candidates = [
        s for s in sweep
        if s["n_above"] >= 3 and s["precision_groundedness_ge_bar"] is not None
        and s["precision_groundedness_ge_bar"] >= 0.8
    ]
    if not candidates:
        return {
            "threshold": None,
            "reason": (
                "No threshold in the sweep reached 80% precision with >=3 items "
                "above it. Either the sample is too small, or composite_confidence "
                "isn't yet a reliable abstention signal at the weights currently "
                "used in generation.py.COMPOSITE_WEIGHTS."
            ),
        }
    best = min(candidates, key=lambda s: s["threshold"])
    return {"threshold": best["threshold"], "precision": best["precision_groundedness_ge_bar"], "n_above": best["n_above"]}


def compute_calibration_report(rows: list[dict]) -> dict:
    resolved = [r for r in rows if r["groundedness"] is not None]
    if len(resolved) < 3:
        return {
            "n_items": len(resolved),
            "warning": f"Only {len(resolved)} items with a valid judge score — too few for "
                       f"meaningful correlation. Re-run with a larger --sample-size.",
        }

    composite = [r["composite_confidence"] for r in resolved]
    retrieval = [r["retrieval_confidence"] for r in resolved]
    citation_cov = [r["citation_coverage"] for r in resolved]
    source_util = [r["source_utilization"] for r in resolved]
    groundedness = [r["groundedness"] for r in resolved]
    relevance = [r["relevance"] for r in resolved]

    correlations = {
        "composite_confidence_vs_groundedness": _pearson(composite, groundedness),
        "retrieval_confidence_vs_groundedness": _pearson(retrieval, groundedness),
        "citation_coverage_vs_groundedness": _pearson(citation_cov, groundedness),
        "source_utilization_vs_groundedness": _pearson(source_util, groundedness),
        "composite_confidence_vs_relevance": _pearson(composite, relevance),
    }

    bin_table = []
    for lo, hi in CONFIDENCE_BINS:
        bucket = [r for r in resolved if lo <= r["composite_confidence"] < hi]
        avg_ground = round(sum(r["groundedness"] for r in bucket) / len(bucket), 2) if bucket else None
        bin_table.append({"range": f"[{lo:.2f}, {hi:.2f})", "n": len(bucket), "avg_groundedness": avg_ground})

    threshold_sweep = []
    for t in THRESHOLD_SWEEP:
        above = [r for r in resolved if r["composite_confidence"] >= t]
        precision = (
            sum(1 for r in above if r["groundedness"] >= GOOD_GROUNDEDNESS_BAR) / len(above)
            if above else None
        )
        threshold_sweep.append({
            "threshold": t, "n_above": len(above),
            "precision_groundedness_ge_bar": round(precision, 3) if precision is not None else None,
        })

    return {
        "n_items": len(resolved),
        "correlations_pearson_r": {k: (round(v, 3) if v is not None else None) for k, v in correlations.items()},
        "binned_calibration": bin_table,
        "threshold_sweep": threshold_sweep,
        "recommended_threshold": _recommend_threshold(threshold_sweep),
        "caveat": (
            f"n={len(resolved)} answered items — a small-sample eval. Treat correlation "
            f"coefficients as directional signal, not statistically robust estimates. "
            f"Re-run with a larger --sample-size for a firmer number before acting on it."
        ),
    }


def maybe_plot(rows: list[dict], out_path: Path) -> Path | None:
    """Best-effort PNG scatter plot. Returns None without erroring if
    matplotlib isn't installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    resolved = [r for r in rows if r["groundedness"] is not None]
    if not resolved:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([r["composite_confidence"] for r in resolved], [r["groundedness"] for r in resolved],
               label="composite_confidence", alpha=0.75)
    ax.scatter([r["retrieval_confidence"] for r in resolved], [r["groundedness"] for r in resolved],
               label="retrieval_confidence (existing gate signal)", alpha=0.75, marker="x")
    ax.set_xlabel("confidence score")
    ax.set_ylabel("judge groundedness (1-5)")
    ax.set_title("Confidence vs. actual groundedness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def write_markdown_report(report: dict, out_path: Path, strategy: str) -> None:
    lines = [f"# Confidence calibration report — strategy={strategy}", ""]
    if "warning" in report:
        lines.append(f"**{report['warning']}**")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append(f"n = {report['n_items']} answered items (refused items excluded — see `evaluate.py`).")
    lines.append("")
    lines.append("## Correlation with judge-scored groundedness (Pearson r)")
    lines.append("")
    lines.append("| Signal | r |")
    lines.append("|---|---|")
    for k, v in report["correlations_pearson_r"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Binned calibration (composite_confidence bucket -> avg groundedness)")
    lines.append("")
    lines.append("| Confidence range | n | Avg groundedness |")
    lines.append("|---|---|---|")
    for b in report["binned_calibration"]:
        lines.append(f"| {b['range']} | {b['n']} | {b['avg_groundedness']} |")
    lines.append("")
    lines.append(f"## Threshold sweep (precision = fraction of items above threshold with groundedness >= {GOOD_GROUNDEDNESS_BAR})")
    lines.append("")
    lines.append("| Threshold | n above | Precision |")
    lines.append("|---|---|---|")
    for s in report["threshold_sweep"]:
        lines.append(f"| {s['threshold']} | {s['n_above']} | {s['precision_groundedness_ge_bar']} |")
    lines.append("")
    rec = report["recommended_threshold"]
    if rec["threshold"] is not None:
        lines.append(f"**Recommended threshold: {rec['threshold']}** "
                      f"(precision {rec['precision']} over {rec['n_above']} items above it)")
    else:
        lines.append(f"**No threshold recommended.** {rec['reason']}")
    lines.append("")
    lines.append(f"> {report['caveat']}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Empirically calibrate composite_confidence against actual judge-scored correctness."
    )
    parser.add_argument("--strategy", default="sentence", choices=["fixed_size", "sentence", "semantic"])
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()

    eval_items = build_eval_set()
    print(f"Running calibration on up to {args.sample_size} real items (strategy={args.strategy})...")
    print("Calls the LLM for generation + citation verification + judging — real API cost.\n")

    rows, n_refused = run_calibration_sample(eval_items, strategy=args.strategy, sample_size=args.sample_size)
    print(f"{len(rows)} answered items scored, {n_refused} refused "
          f"(excluded here — see evaluate.py's refusal-gate-accuracy for that).")

    report = compute_calibration_report(rows)
    report["n_refused_excluded"] = n_refused
    report["raw_rows"] = rows

    print(json.dumps({k: v for k, v in report.items() if k != "raw_rows"}, indent=2))

    out_json = EVAL_DIR / f"calibration_report_{args.strategy}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report -> {out_json}")

    md_path = EVAL_DIR / f"calibration_report_{args.strategy}.md"
    write_markdown_report(report, md_path, args.strategy)
    print(f"Markdown report -> {md_path}")

    plot_path = EVAL_DIR / f"calibration_plot_{args.strategy}.png"
    saved = maybe_plot(rows, plot_path)
    print(f"Calibration plot -> {saved}" if saved else
          "matplotlib not installed — skipped plot (the binned_calibration table has the same information).")


if __name__ == "__main__":
    main()