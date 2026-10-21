"""
eval_holdout.py — Held-out retrieval evaluation benchmark (Recall@k, MRR).

THE LEAKAGE THIS FIXES (see eval_dataset.py's own docstring, which already
flags part of this as a known limitation):
  1. eval_dataset.py's "real" items are questions sampled directly from the
     SAME qa_pairs.jsonl that chunking.py/index.py chunk and index into the
     PRODUCTION chroma_db + bm25 (100% of documents — there has never been
     any dev/held-out separation).
  2. Sharper problem: chunking.py's Chunk.embed_text() embeds
     "Question: {question}\\nFocus: {focus}\\n\\n{chunk_text}" — the
     LITERAL, VERBATIM question text is embedded as part of the document.
     eval_dataset.py's "real" items then query with that exact same
     verbatim string. Retrieving with a query that's a substring of what's
     embedded in the target document is close to string-matching, not a
     real test of semantic retrieval. (Not fixed here — see module-level
     "residual risk" note at the bottom; fixing it well needs either
     hand-paraphrased held-out questions or a change to what gets embedded,
     both bigger than this change's scope.)

THE FIX — a document-level held-out split:
  - qa_pairs.jsonl's doc_ids (source documents) are partitioned into `dev`
    (majority) and `heldout` (minority), STRATIFIED BY SOURCE, deterministic
    seed. Every QA pair sharing a doc_id goes entirely to one side — a
    document's chunks can never end up split across dev and heldout.
  - Held-out eval QUESTIONS are sampled ONLY from `heldout` doc_ids.
  - The benchmark builds its OWN dedicated Chroma collection at
    data/eval/chroma_db_holdout/ (separate from and never touching the
    production data/processed/chroma_db/, chunks_<strategy>.jsonl, or bm25
    — those stay exactly as they are, still serving the live app).

Why the benchmark's corpus still includes ALL documents (dev + heldout
merged), not just the heldout ones: recall@k/MRR need the gold chunk to be
findable among a realistic pool of distractors, not a handful of
hand-picked candidates — this is standard IR-benchmark practice (BEIR,
MS MARCO, Natural Questions all search a full corpus that DOES contain the
gold documents; what's "held out" in those benchmarks, and here, is which
documents/questions were available when the SYSTEM was being tuned, not
whether the corpus contains the answer). Concretely: this held-out
partition and its exact chunks are excluded from the PRODUCTION index used
for the app / any prior chunking-strategy or threshold decisions — they
only exist inside this dedicated, separate benchmark index, built for
measurement. Going forward, `dev_doc_ids` are what any future retrieval/
chunking tuning should use; `heldout_doc_ids` are reserved for reporting
the "real" benchmark number.

Reuses, not reimplements: chunking.py's build_chunks() (same chunking
functions/logic), embed.py's embed_fn_factory() (same embedding model),
and DenseRetriever/BM25Retriever/HybridRetriever (same RRF/rerank/query
rewrite code) — only the *corpus* they point at is swapped for a separate
one, via the small optional constructor params added to those three
classes (chroma_dir on DenseRetriever/HybridRetriever, chunks on
BM25Retriever/HybridRetriever — both default to production behavior when
omitted).

Usage:
    python src/eval_holdout.py --strategy sentence --top-k 5
    python src/eval_holdout.py --strategy sentence --rebuild   # force rebuild of the benchmark index
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
HOLDOUT_CHROMA_DIR = EVAL_DIR / "chroma_db_holdout"
HOLDOUT_CHUNKS_CACHE = EVAL_DIR / "holdout_chunks_cache"  # one file per strategy, see below

HELDOUT_FRACTION = 0.15  # ~15% of documents per source reserved as held-out
N_PER_SOURCE = 4
SEED = 42


def load_qa_pairs() -> list[dict]:
    qa_path = DATA_DIR / "qa_pairs.jsonl"
    with open(qa_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_doc_ids(
    rows: list[dict] | None = None,
    heldout_fraction: float = HELDOUT_FRACTION,
    seed: int = SEED,
) -> tuple[set[str], set[str]]:
    """Partitions doc_ids into (dev_doc_ids, heldout_doc_ids), stratified by
    source, deterministic. Document-level: every QA pair sharing a doc_id
    goes entirely to one side. rows=None loads qa_pairs.jsonl directly;
    pass explicit rows to test this without touching disk."""
    if rows is None:
        rows = load_qa_pairs()

    doc_id_to_source: dict[str, str] = {}
    for row in rows:
        doc_id_to_source.setdefault(row["doc_id"], row["source"])

    by_source: dict[str, list[str]] = {}
    for doc_id, source in doc_id_to_source.items():
        by_source.setdefault(source, []).append(doc_id)

    rng = random.Random(seed)
    dev_doc_ids: set[str] = set()
    heldout_doc_ids: set[str] = set()
    for source, doc_ids in sorted(by_source.items()):
        doc_ids = sorted(doc_ids)  # deterministic order before shuffling
        rng.shuffle(doc_ids)
        n_heldout = max(1, round(len(doc_ids) * heldout_fraction))
        heldout_doc_ids.update(doc_ids[:n_heldout])
        dev_doc_ids.update(doc_ids[n_heldout:])
    return dev_doc_ids, heldout_doc_ids


def build_holdout_items(
    heldout_doc_ids: set[str],
    rows: list[dict] | None = None,
    n_per_source: int = N_PER_SOURCE,
    seed: int = SEED,
) -> list[dict]:
    """Samples eval QUESTIONS only from heldout_doc_ids. Same dict shape as
    eval_dataset.py's "real" items (type/gold_pair_uid/gold_focus/
    gold_source) plus an added gold_doc_id, so evaluate.py-style consumers
    keep working with minimal/no changes if pointed at this instead."""
    if rows is None:
        rows = load_qa_pairs()

    heldout_rows = [r for r in rows if r["doc_id"] in heldout_doc_ids]
    by_source: dict[str, list[dict]] = {}
    for row in heldout_rows:
        by_source.setdefault(row["source"], []).append(row)

    rng = random.Random(seed)
    sample = []
    for source, source_rows in sorted(by_source.items()):
        candidates = [r for r in source_rows if len(r["answer"].split()) >= 40]
        if not candidates:
            candidates = source_rows
        picked = rng.sample(candidates, min(n_per_source, len(candidates)))
        for row in picked:
            sample.append({
                "question": row["question"], "type": "real", "expects_refusal": False,
                "gold_pair_uid": row["pair_uid"], "gold_doc_id": row["doc_id"],
                "gold_focus": row["focus"], "gold_source": row["source"],
            })
    return sample


def reciprocal_rank(retrieved_chunks, gold_pair_uid: str) -> float:
    """1/rank of the first chunk matching gold_pair_uid (1-indexed), 0.0 if
    not found in the retrieved list at all. Reuses evaluate.extract_pair_uid
    rather than re-parsing chunk_id here."""
    from evaluate import extract_pair_uid
    for rank, c in enumerate(retrieved_chunks, start=1):
        if extract_pair_uid(c.chunk_id) == gold_pair_uid:
            return 1.0 / rank
    return 0.0


def build_holdout_corpus_chunks(strategy: str) -> list[dict]:
    """Builds the FULL corpus (dev + heldout doc_ids merged — every
    document) as plain chunk dicts, via chunking.py's EXISTING
    build_chunks() — reused unmodified, not reimplemented. This becomes the
    benchmark's own chunk set; it never touches
    data/processed/chunks_<strategy>.jsonl."""
    import chunking

    embed_fn = None
    if strategy == "semantic":
        from embed import embed_fn_factory
        embed_fn = embed_fn_factory()

    chunks = chunking.build_chunks(strategy, embed_fn=embed_fn)
    return [asdict(c) for c in chunks]


def build_or_load_holdout_index(strategy: str, rebuild: bool = False) -> list[dict]:
    """Builds (or reuses, if already cached) the benchmark's dedicated
    Chroma collection + chunk cache, entirely separate from production.
    Returns the chunk dicts (also needed for BM25Retriever(chunks=...))."""
    import chromadb
    from embed import embed_texts

    HOLDOUT_CHUNKS_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = HOLDOUT_CHUNKS_CACHE / f"chunks_{strategy}.jsonl"

    client = chromadb.PersistentClient(path=str(HOLDOUT_CHROMA_DIR))
    # Same collection name DenseRetriever always looks up ("medrag_{strategy}")
    # -- isolation from production comes from HOLDOUT_CHROMA_DIR being a
    # separate directory, not from a different collection name.
    collection_name = f"medrag_{strategy}"

    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        if cache_path.exists():
            cache_path.unlink()

    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        try:
            client.get_collection(collection_name)
            return chunks  # already built and cached — nothing to do
        except Exception:
            pass  # cache file exists but collection doesn't (e.g. deleted) — rebuild the index below
    else:
        chunks = build_holdout_corpus_chunks(strategy)
        with open(cache_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Building held-out benchmark index ({len(chunks)} chunks, strategy={strategy}) "
          f"at {HOLDOUT_CHROMA_DIR} — this is separate from the production index.")
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"strategy": strategy, "benchmark": "holdout"}
    )
    batch_size = 256
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts_to_embed = [f"Question: {c['question']}\nFocus: {c['focus']}\n\n{c['chunk_text']}" for c in batch]
        embeddings = embed_texts(texts_to_embed)
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings.tolist(),
            documents=[c["chunk_text"] for c in batch],
            metadatas=[
                {"doc_id": c["doc_id"], "qid": c["qid"], "focus": c["focus"], "qtype": c["qtype"],
                 "source": c["source"], "url": c["url"], "question": c["question"],
                 "chunk_index": c["chunk_index"]}
                for c in batch
            ],
        )
        print(f"  indexed {min(i + batch_size, len(chunks))}/{len(chunks)}", end="\r")
    print(f"\nDone. Held-out benchmark collection has {collection.count()} vectors.")
    return chunks


def run_holdout_evaluation(strategy: str, top_k: int = 5, rebuild: bool = False) -> dict:
    from evaluate import recall_at_k
    from hybrid_retriever import HybridRetriever
    from retriever import DenseRetriever

    dev_doc_ids, heldout_doc_ids = split_doc_ids()
    heldout_items = build_holdout_items(heldout_doc_ids)

    chunks = build_or_load_holdout_index(strategy, rebuild=rebuild)

    dense = DenseRetriever(strategy=strategy, chroma_dir=HOLDOUT_CHROMA_DIR)
    hybrid = HybridRetriever(strategy=strategy, chroma_dir=HOLDOUT_CHROMA_DIR, chunks=chunks)

    per_item = []
    dense_hits = hybrid_hits = 0
    mrr_dense_sum = mrr_hybrid_sum = 0.0

    for item in heldout_items:
        query = item["question"]
        gold = item["gold_pair_uid"]

        dense_results = dense.retrieve(query, top_k=top_k)
        hybrid_results = hybrid.retrieve(query, top_k=top_k)

        d_hit = recall_at_k(dense_results, gold)
        h_hit = recall_at_k(hybrid_results, gold)
        dense_hits += d_hit
        hybrid_hits += h_hit

        d_mrr = reciprocal_rank(dense_results, gold)
        h_mrr = reciprocal_rank(hybrid_results, gold)
        mrr_dense_sum += d_mrr
        mrr_hybrid_sum += h_mrr

        per_item.append({
            "question": query, "gold_pair_uid": gold, "gold_doc_id": item["gold_doc_id"],
            "gold_source": item["gold_source"],
            "dense_hit": d_hit, "hybrid_hit": h_hit,
            "dense_reciprocal_rank": round(d_mrr, 3), "hybrid_reciprocal_rank": round(h_mrr, 3),
        })

    n = len(heldout_items)
    return {
        "strategy": strategy, "top_k": top_k,
        "n_dev_doc_ids": len(dev_doc_ids), "n_heldout_doc_ids": len(heldout_doc_ids),
        "n_heldout_items": n,
        "dense_recall_at_k": round(dense_hits / n, 3) if n else None,
        "hybrid_recall_at_k": round(hybrid_hits / n, 3) if n else None,
        "dense_mrr": round(mrr_dense_sum / n, 3) if n else None,
        "hybrid_mrr": round(mrr_hybrid_sum / n, 3) if n else None,
        "per_item": per_item,
    }


def main():
    parser = argparse.ArgumentParser(description="Held-out retrieval evaluation benchmark (Recall@k, MRR).")
    parser.add_argument("--strategy", default="sentence", choices=["fixed_size", "sentence", "semantic"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true",
                         help="force rebuild of the held-out benchmark index (otherwise reused if cached)")
    args = parser.parse_args()

    report = run_holdout_evaluation(strategy=args.strategy, top_k=args.top_k, rebuild=args.rebuild)

    print(f"\n=== Held-out benchmark — strategy={args.strategy}, top_k={args.top_k} ===")
    print(f"dev doc_ids: {report['n_dev_doc_ids']} | heldout doc_ids: {report['n_heldout_doc_ids']} "
          f"| heldout eval items: {report['n_heldout_items']}")
    print(f"Dense  — Recall@{args.top_k}: {report['dense_recall_at_k']}  MRR: {report['dense_mrr']}")
    print(f"Hybrid — Recall@{args.top_k}: {report['hybrid_recall_at_k']}  MRR: {report['hybrid_mrr']}")

    out_path = EVAL_DIR / f"holdout_eval_report_{args.strategy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report -> {out_path}")


if __name__ == "__main__":
    main()

# --- Residual leakage risk, not fixed here (see module docstring) ---
# chunking.py's Chunk.embed_text() embeds the literal question text
# ("Question: {question}\n...") into what gets indexed. Because the
# held-out items above still query with that SAME verbatim text, a chunk
# whose embed_text() contains an identical question string will score
# artificially high regardless of whether the underlying retrieval/
# embedding approach actually generalizes. Document-level held-out
# splitting (this file) does NOT fix this specific mechanism — only
# rephrasing held-out queries away from verbatim source text, or changing
# what's embedded, would. Flagged here rather than solved, since a good
# fix needs either hand-paraphrased held-out questions or a change to
# chunking.py's embed_text(), both bigger than this change's scope.