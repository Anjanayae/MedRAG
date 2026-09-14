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

  4. Key-fact recall (free, no extra API calls): for 'real' items, does the
     generated answer actually mention the key terms present in the
     original MedQuAD answer it's graded against? Groundedness (#3) checks
     the answer doesn't say anything UNSUPPORTED; this checks the answer
     didn't OMIT the substance of the source either — a different failure
     mode a groundedness-only score can miss (a technically-grounded but
     thin/incomplete answer still scores well on groundedness).

  5. (optional, --verify-citations, extra API calls) Citation accuracy: for
     the same LLM-judge sample, runs each inline [N] citation in the
     generated answer through citation_verify.py's independent judge and
     reports the fraction that are actually supported by their cited
     source — distinct from #3's groundedness, which grades the answer as
     a whole rather than citation-by-citation.

Usage:
    python src/evaluate.py --strategy sentence
    python src/evaluate.py --strategy sentence --skip-llm-judge
    python src/evaluate.py --strategy sentence --verify-citations
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from generation import check_confidence, call_llm, build_context_block, build_chunk_text_map, SYSTEM_PROMPT
from hybrid_retriever import HybridRetriever
from retriever import DenseRetriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"

LLM_JUDGE_SAMPLE_SIZE = 8  # keep small — this is the part that costs real API calls

# Common words excluded from key-fact term extraction — not exhaustive,
# just enough to stop trivial words from diluting the signal. Medical
# terms, numbers, and drug/disease names (the things that actually matter
# for "did the answer cover the substance of the source") are rarely in
# this list.
_STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "your",
    "have", "has", "can", "may", "will", "may", "not", "you", "may",
    "these", "their", "about", "some", "other", "than", "into", "more",
    "which", "such", "also", "been", "when", "what", "each", "were",
}
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


def extract_key_terms(text: str, max_terms: int = 12) -> list[str]:
    """Cheap, no-model keyword extraction: pulls out numbers and words
    5+ characters that aren't in the stopword list, dedups (case-
    insensitive) preserving first-seen order, caps at max_terms. This is
    deliberately simple — the goal is a rough "did the answer touch on
    the source's substance" signal, not precision NLP."""
    seen = set()
    terms = []
    for word in _WORD_RE.findall(text):
        key = word.lower()
        if key in _STOPWORDS or (not word.isdigit() and len(word) < 5):
            continue
        if key in seen:
            continue
        seen.add(key)
        terms.append(word)
        if len(terms) >= max_terms:
            break
    return terms


def key_fact_recall(generated_answer: str, gold_answer: str) -> float | None:
    """Fraction of gold_answer's key terms that appear (case-insensitive
    substring match) somewhere in generated_answer. Returns None if the
    gold answer had no extractable key terms (nothing to check).

    Known limitation: this is plain substring matching, not stemming/
    lemmatization — "causes" in the gold answer won't match "cause" or
    "caused" in the generated one. That undercounts recall on legitimate
    paraphrases. Treat this as a rough, free (no API call) signal to
    complement the LLM-judge groundedness/relevance score, not a precise
    metric on its own."""
    key_terms = extract_key_terms(gold_answer)
    if not key_terms:
        return None
    lowered = generated_answer.lower()
    hits = sum(1 for term in key_terms if term.lower() in lowered)
    return round(hits / len(key_terms), 3)


_qa_answer_lookup: dict[str, str] | None = None


def load_gold_answer(pair_uid: str) -> str | None:
    """Looks up the ORIGINAL MedQuAD answer text for a gold_pair_uid, for
    key_fact_recall() to compare against. Cached at module level since
    eval runs look this up once per item, not once per line of the file."""
    global _qa_answer_lookup
    if _qa_answer_lookup is None:
        qa_path = DATA_DIR / "qa_pairs.jsonl"
        _qa_answer_lookup = {}
        if qa_path.exists():
            with open(qa_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    _qa_answer_lookup[row["pair_uid"]] = row["answer"]
    return _qa_answer_lookup.get(pair_uid)

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


def llm_judge_sample(
    eval_items: list[dict],
    strategy: str,
    sample_size: int = LLM_JUDGE_SAMPLE_SIZE,
    verify_citations: bool = False,
) -> list[dict]:
    """Runs a handful of real, answerable items through full generation and
    an LLM-judge grading pass. Costs real Groq API calls — kept small
    on purpose. Skips items the confidence gate would refuse.

    key_fact_recall is always computed (free — string matching against the
    item's own gold MedQuAD answer, no extra API call). verify_citations
    adds one extra judge call per citation the answer contains — opt-in via
    --verify-citations since it can meaningfully add to API cost on top of
    the groundedness/relevance judge call already made per item."""
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
        answer = call_llm(prompt)

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(question=query, context=context_block, answer=answer)
        judge_response = call_llm(judge_prompt)
        try:
            scores = json.loads(judge_response)
        except json.JSONDecodeError:
            scores = {"groundedness": None, "relevance": None, "raw_judge_response": judge_response}

        gold_answer = load_gold_answer(item.get("gold_pair_uid", ""))
        kf_recall = key_fact_recall(answer, gold_answer) if gold_answer else None

        entry = {
            "question": query, "answer": answer, "confidence": round(conf, 3),
            "key_fact_recall": kf_recall, **scores,
        }

        if verify_citations:
            from citation_verify import verify_answer_citations, citation_coverage
            chunk_texts = build_chunk_text_map(retrieved)
            checks = verify_answer_citations(answer, chunk_texts)
            entry["citation_coverage"] = round(citation_coverage(checks), 3)
            entry["n_citations_checked"] = len(checks)

        results.append(entry)

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
    judge_response = call_llm(judge_prompt)
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
    parser.add_argument("--verify-citations", action="store_true",
                         help="also run independent citation verification on the LLM-judge "
                              "sample (extra API calls, one per citation in each answer)")
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
            judge_results = llm_judge_sample(
                eval_items, strategy=args.strategy, verify_citations=args.verify_citations
            )
            report["llm_judge"] = judge_results
            valid = [r for r in judge_results if r.get("groundedness") is not None]
            if valid:
                avg_ground = sum(r["groundedness"] for r in valid) / len(valid)
                avg_rel = sum(r["relevance"] for r in valid) / len(valid)
                print(f"Avg groundedness: {avg_ground:.2f}/5 | Avg relevance: {avg_rel:.2f}/5 "
                      f"(n={len(valid)})")

            kf_valid = [r["key_fact_recall"] for r in judge_results if r.get("key_fact_recall") is not None]
            if kf_valid:
                print(f"Avg key-fact recall: {sum(kf_valid)/len(kf_valid):.2f} "
                      f"(n={len(kf_valid)}) — fraction of the gold answer's key terms "
                      f"the generated answer actually mentioned")

            if args.verify_citations:
                cc_valid = [r["citation_coverage"] for r in judge_results if "citation_coverage" in r]
                if cc_valid:
                    print(f"Avg citation accuracy: {sum(cc_valid)/len(cc_valid):.2f} "
                          f"(n={len(cc_valid)}) — fraction of inline citations an independent "
                          f"judge confirmed are actually supported by their cited source")

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