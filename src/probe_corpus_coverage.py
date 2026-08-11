"""
probe_corpus_coverage.py — One-off diagnostic: runs a diverse batch of
questions through the retrieval pipeline (rewrite -> hybrid retrieve ->
rerank -> confidence check) to assess where MedQuAD's coverage is strong
vs weak, WITHOUT calling the generation LLM (cheap/fast — just retrieval +
the confidence gate).

Motivation: after finding that "how does B12 play a role in hairfall" gets
correctly refused because MedQuAD has no chunk connecting the two, the
open question is whether that's a one-off gap or a systematic pattern
(e.g. MedQuAD is strong on single-disease factual QA, but weak on
drug-specific info since the MPlusDrugs subset was excluded for copyright,
and weak on cross-concept/relationship questions since each chunk answers
one narrow question about one topic). This script tests a spread of
question types in one batch run (loading models ONCE, not once per
question) so the pattern is visible at a glance instead of guessing from
one query at a time.

Usage:
    python src/probe_corpus_coverage.py --strategy sentence
"""

from __future__ import annotations

import argparse

from generation import check_confidence
from hybrid_retriever import HybridRetriever

# (question, category) — category labels the TYPE of question, to spot
# whether refusals cluster by type rather than being random noise.
PROBE_QUESTIONS = [
    ("What are the symptoms of diabetes?", "single-disease factual (baseline — should work)"),
    ("What causes high blood pressure?", "single-disease factual (baseline — should work)"),
    ("How does vitamin B12 deficiency affect hair loss?", "cross-concept association"),
    ("Can vitamin D deficiency cause depression?", "cross-concept association"),
    ("What medication is commonly used to treat high blood pressure?", "drug-specific info"),
    ("What are alternatives to metformin for treating diabetes?", "drug alternatives"),
    ("Is it safe to take ibuprofen if I have high blood pressure?", "drug interaction"),
    ("What is the difference between type 1 and type 2 diabetes?", "comparison across two topics"),
    ("What causes hair loss in women?", "common patient question, plain language"),
    ("What are the side effects of chemotherapy?", "treatment side effects (general, not drug-name-specific)"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="sentence")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading retriever (strategy='{args.strategy}')... this loads the embedding "
          f"model + reranker ONCE for the whole batch.\n")
    retriever = HybridRetriever(strategy=args.strategy)
    retriever.warmup()  # pay model-load cost once, up front, not on question 1

    print(f"{'='*100}\nRunning {len(PROBE_QUESTIONS)} probe questions...\n{'='*100}\n")

    results = []
    for question, category in PROBE_QUESTIONS:
        retrieved = retriever.retrieve(question, top_k=args.top_k)
        confidence, refused = check_confidence(retrieved)
        top = retrieved[0] if retrieved else None

        status = "REFUSED" if refused else "ANSWERED"
        print(f"[{status}] ({category})")
        print(f"  Q: {question}")
        print(f"  confidence: {confidence:.3f}")
        if top:
            print(f"  top match: focus={top.metadata.get('focus')!r} | "
                  f"source={top.metadata.get('source')} | score={top.score:.3f}")
            print(f"  preview: {top.chunk_text[:150]}...")
        print()

        results.append({"question": question, "category": category,
                         "confidence": round(confidence, 3), "refused": refused})

    refused_count = sum(1 for r in results if r["refused"])
    print(f"{'='*100}")
    print(f"Summary: {refused_count}/{len(results)} refused")
    print("\nBy category:")
    for r in results:
        mark = "REFUSED" if r["refused"] else "ok"
        print(f"  [{mark:8s}] {r['category']:55s} conf={r['confidence']}")


if __name__ == "__main__":
    main()