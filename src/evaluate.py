"""
evaluate.py — Runs the eval set (src/eval_dataset.py) against the retrieval
pipeline and reports:

  1. Retrieval recall@k (for 'real' + 'paraphrase' items with a gold chunk):
     did any of the top-k retrieved chunks come from the same source QA pair
     (matched via pair_uid, extracted from chunk_id) as the question?
     Computed for BOTH dense-only and hybrid+reranked retrieval, so we get
     an immediate ablation number for free.

  2. Refusal-gate accuracy (for 'out_of_domain' + 'borderline' items, no
     Groq calls needed): does the confidence check correctly refuse?
     Also reports on 'real'/'paraphrase' items: did we wrongly refuse a
     question we should have answered? (false-refusal rate)

  3. (optional, needs GROQ_API_KEY) LLM-judge groundedness/relevance score
     on a small subset — real generation quality signal, at real API cost,
     so kept to a handful of items rather than the whole eval set.

Usage:
    python src/evaluate.py --strategy sentence
    python src/evaluate.py --strategy sentence --skip-llm-judge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation import check_confidence, call_groq, build_context_block, SYSTEM_PROMPT
from hybrid_retriever import HybridRetriever
from retriever import DenseRetriever

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"

LLM_JUDGE_SAMPLE_SIZE = 8  # keep small — this is the part that costs real API calls

JUDGE_PROMPT_TEMPLATE = """You will be shown a QUESTION, the CONTEXT used to answer it, and an ANSWER.

Rate two things, each on a 1-5 scale:
- groundedness: is the ANSWER fully supported by the CONTEXT (5), or does it add unsupported claims (1)?
- relevance: does the ANSWER directly address the QUESTION (5), or is it off-topic/incomplete (1)?

Respond with ONLY a JSON object, no other text: {{"groundedness": <1-5>, "relevance": <1-5>}}

QUESTION: {question}

CONTEXT: {context}

ANSWER: {answer}
"""


def extract_pair_uid(chunk_id: str) -> str:
    return chunk_id.split("::")[0]


def recall_at_k(retrieved_chunks, gold_pair_uid: str) -> bool:
    return any(extract_pair_uid(c.chunk_id) == gold_pair_uid for c in retrieved_chunks)


def evaluate_retrieval(eval_items: list[dict], strategy: str, top_k: int = 5) -> dict:
    dense = DenseRetriever(strategy=strategy)
    hybrid = HybridRetriever(strategy=strategy)

    gold_items = [i for i in eval_items if i["type"] in ("real", "paraphrase")]
    refusal_items = [i for i in eval_items if i["type"] in ("out_of_domain", "borderline")]

    dense_hits, hybrid_hits = 0, 0
    per_item_results = []

    for item in gold_items:
        query = item["question"]
        gold = item.get("gold_pair_uid")

        dense_results = dense.retrieve(query, top_k=top_k)
        hybrid_results = hybrid.retrieve(query, top_k=top_k)

        if gold:  # 'paraphrase' items may or may not carry a gold id
            dense_hit = recall_at_k(dense_results, gold)
            hybrid_hit = recall_at_k(hybrid_results, gold)
            dense_hits += dense_hit
            hybrid_hits += hybrid_hit
        else:
            dense_hit = hybrid_hit = None

        conf, refused = check_confidence(hybrid_results)
        per_item_results.append(
            {
                "question": query, "type": item["type"], "gold_pair_uid": gold,
                "dense_hit": dense_hit, "hybrid_hit": hybrid_hit,
                "confidence": round(conf, 3), "refused": refused,
                "wrongly_refused": refused,  # these SHOULD have been answered
            }
        )

    n_gold_with_id = sum(1 for i in gold_items if i.get("gold_pair_uid"))
    false_refusal_count = sum(1 for r in per_item_results if r["wrongly_refused"])

    refusal_correct = 0
    for item in refusal_items:
        hybrid_results = hybrid.retrieve(item["question"], top_k=top_k)
        conf, refused = check_confidence(hybrid_results)
        correct = refused == item["expects_refusal"]
        refusal_correct += correct
        per_item_results.append(
            {
                "question": item["question"], "type": item["type"],
                "confidence": round(conf, 3), "refused": refused,
                "expected_refusal": item["expects_refusal"], "correct": correct,
            }
        )

    return {
        "n_gold_items": n_gold_with_id,
        "dense_recall_at_k": round(dense_hits / n_gold_with_id, 3) if n_gold_with_id else None,
        "hybrid_recall_at_k": round(hybrid_hits / n_gold_with_id, 3) if n_gold_with_id else None,
        "false_refusal_rate_on_answerable": round(false_refusal_count / len(gold_items), 3),
        "n_refusal_test_items": len(refusal_items),
        "refusal_gate_accuracy": round(refusal_correct / len(refusal_items), 3) if refusal_items else None,
        "per_item": per_item_results,
    }


def llm_judge_sample(eval_items: list[dict], strategy: str, sample_size: int = LLM_JUDGE_SAMPLE_SIZE) -> list[dict]:
    """Runs a handful of real, answerable items through full generation and
    an LLM-judge grading pass. Costs real Groq API calls — kept small
    on purpose. Skips items the confidence gate would refuse."""
    hybrid = HybridRetriever(strategy=strategy)
    real_items = [i for i in eval_items if i["type"] == "real"][:sample_size]

    results = []
    for item in real_items:
        query = item["question"]
        retrieved = hybrid.retrieve(query, top_k=5)
        conf, refused = check_confidence(retrieved)
        if refused:
            continue  # skip — nothing to judge, would just fail the recall check above

        context_block = build_context_block(retrieved)
        prompt = f"Sources:\n{context_block}\n\nQuestion: {query}"
        answer = call_groq(prompt)

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(question=query, context=context_block, answer=answer)
        judge_response = call_groq(judge_prompt)
        try:
            scores = json.loads(judge_response)
        except json.JSONDecodeError:
            scores = {"groundedness": None, "relevance": None, "raw_judge_response": judge_response}

        results.append({"question": query, "answer": answer, "confidence": round(conf, 3), **scores})

    return results

def judge_calibration_check(eval_items: list[dict], strategy: str) -> dict:
    """Sanity-checks the LLM-judge itself, not just the system under test.
    A judge that scores every real answer 5/5 might mean the system is
    genuinely excellent, or might mean the judge is too lenient to be
    trusted. We test this by feeding it ONE deliberately fabricated,
    ungrounded answer (a real question, real context, but a made-up claim
    that contradicts/isn't in the context) and confirming groundedness
    scores low. If the judge still gives this a 5, the judge itself is
    unreliable and the earlier perfect scores shouldn't be trusted at
    face value."""
    hybrid = HybridRetriever(strategy=strategy)
    real_items = [i for i in eval_items if i["type"] == "real"]
    item = real_items[0]

    retrieved = hybrid.retrieve(item["question"], top_k=5)
    context_block = build_context_block(retrieved)

    fabricated_answer = (
        "Based on the sources, this condition is caused entirely by exposure "
        "to microwave radiation from household appliances, and can be cured "
        "within 24 hours by drinking exactly 3 liters of pineapple juice. "
        "No medical consultation is necessary."
    )

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=item["question"], context=context_block, answer=fabricated_answer
    )
    judge_response = call_groq(judge_prompt)
    try:
        scores = json.loads(judge_response)
    except json.JSONDecodeError:
        scores = {"groundedness": None, "relevance": None, "raw_judge_response": judge_response}

    passed = scores.get("groundedness") is not None and scores["groundedness"] <= 2
    return {
        "question": item["question"],
        "fabricated_answer": fabricated_answer,
        "judge_scores": scores,
        "judge_correctly_flagged_as_ungrounded": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="sentence")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--skip-llm-judge", action="store_true",
                         help="skip the LLM-judge pass (no Groq API calls, no key needed)")
    args = parser.parse_args()

    eval_path = EVAL_DIR / "eval_set.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"{eval_path} not found — run `python src/eval_dataset.py` first.")
    with open(eval_path, encoding="utf-8") as f:
        eval_items = json.load(f)

    print(f"Loaded {len(eval_items)} eval items. Running retrieval + refusal-gate eval "
          f"(strategy='{args.strategy}', no API calls needed for this part)...\n")
    retrieval_report = evaluate_retrieval(eval_items, strategy=args.strategy, top_k=args.top_k)

    print(f"Retrieval recall@{args.top_k}:")
    print(f"  Dense-only:   {retrieval_report['dense_recall_at_k']}")
    print(f"  Hybrid+rerank: {retrieval_report['hybrid_recall_at_k']}")
    print(f"\nFalse-refusal rate on answerable questions: "
          f"{retrieval_report['false_refusal_rate_on_answerable']} "
          f"(lower is better — this is questions we WRONGLY refused)")
    print(f"Refusal-gate accuracy (out-of-domain/borderline, should refuse): "
          f"{retrieval_report['refusal_gate_accuracy']}")

    report = {"retrieval": retrieval_report}

    if not args.skip_llm_judge:
        print(f"\nRunning LLM-judge pass on up to {LLM_JUDGE_SAMPLE_SIZE} real items "
              f"(this DOES call the Groq API)...")
        try:
            judge_results = llm_judge_sample(eval_items, strategy=args.strategy)
            report["llm_judge"] = judge_results
            valid = [r for r in judge_results if r.get("groundedness") is not None]
            if valid:
                avg_ground = sum(r["groundedness"] for r in valid) / len(valid)
                avg_rel = sum(r["relevance"] for r in valid) / len(valid)
                print(f"Avg groundedness: {avg_ground:.2f}/5 | Avg relevance: {avg_rel:.2f}/5 "
                      f"(n={len(valid)})")

            print("\nRunning judge calibration check (1 fabricated/ungrounded answer, "
                  "should score LOW if the judge is discriminating properly)...")
            calibration = judge_calibration_check(eval_items, strategy=args.strategy)
            report["judge_calibration_check"] = calibration
            status = "PASSED" if calibration["judge_correctly_flagged_as_ungrounded"] else "FAILED"
            print(f"Judge calibration check: {status} "
                  f"(scored fabricated answer's groundedness as "
                  f"{calibration['judge_scores'].get('groundedness')}/5 — should be <= 2)")
            if not calibration["judge_correctly_flagged_as_ungrounded"]:
                print("WARNING: the judge did not catch an obviously fabricated answer. "
                      "Treat the earlier groundedness/relevance averages with caution — "
                      "the judge itself may be too lenient to trust at face value.")
        except Exception as e:
            print(f"Skipped LLM-judge pass: {e}")

    out_path = EVAL_DIR / "eval_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved -> {out_path}")


if __name__ == "__main__":
    main()