"""
dedupe.py — Near-duplicate chunk removal at INGEST time (before indexing),
extending the exact-duplicate collapse reranker.py already does at QUERY
time.

reranker.py already dedups on chunk_id AND chunk_text at query time (see
its docstring — a real eval case found two byte-identical Hypoglycemia
chunks under different pair_uids, wasting a citation slot meant for a 5th,
more useful source). This module closes the same gap earlier and more
broadly: NEAR-duplicate (not just byte-identical) chunks never get
embedded/indexed at all, instead of being indexed and relying on every
single future query's reranker pass to filter them back out.

Why this needs re-indexing: it changes the actual chunk set written to
chunks_<strategy>.jsonl. index.py must be rerun for any strategy you dedup
— see README "Dedup and re-indexing".

Scalability note: comparing every chunk against every other chunk is
O(n^2), which doesn't scale to the tens of thousands of chunks a full
MedQuAD ingest produces. But the near-duplicates that actually occur in
this corpus are same-TOPIC duplicates (reranker.py's documented case: two
different pair_uids, same disease, adjacent source documents) — a diabetes
chunk and an asthma chunk are never going to be near-duplicates of each
other. So we bucket by `focus` (already a Chunk field) first and only
compare within each bucket, turning O(n^2) over the whole corpus into
O(k^2) per bucket, where k is however many chunks share one focus
(typically a handful to a few dozen, not thousands).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from chunking import Chunk


def _cosine_sim(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


@dataclass
class DedupeReport:
    strategy: str
    n_before: int
    n_after: int
    n_removed: int
    n_buckets_with_dupes: int


def remove_near_duplicates(
    chunks: list[Chunk],
    embed_fn,
    similarity_threshold: float = 0.95,
) -> tuple[list[Chunk], DedupeReport]:
    """embed_fn: same injected-function pattern chunking.semantic_chunks()
    already uses (list[str] -> embeddings) — keeps this module free of a
    hard ML dependency and unit-testable with a fake embedder, same as the
    rest of chunking.py.

    Within each focus-bucket, chunks are compared in their original order;
    a chunk is dropped if its similarity to any chunk ALREADY KEPT in that
    bucket is >= similarity_threshold (greedy, first-occurrence wins —
    keep the first copy, drop later near-identical ones)."""
    buckets: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        buckets[c.focus].append(c)

    kept: list[Chunk] = []
    n_removed = 0
    n_buckets_with_dupes = 0

    for focus, bucket in buckets.items():
        if len(bucket) == 1:
            kept.extend(bucket)
            continue

        texts = [c.chunk_text for c in bucket]
        embeddings = embed_fn(texts)

        kept_indices: list[int] = []
        bucket_removed = 0
        for i, c in enumerate(bucket):
            is_dup = any(
                _cosine_sim(embeddings[i], embeddings[j]) >= similarity_threshold
                for j in kept_indices
            )
            if is_dup:
                n_removed += 1
                bucket_removed += 1
            else:
                kept_indices.append(i)
                kept.append(c)

        if bucket_removed:
            n_buckets_with_dupes += 1

    report = DedupeReport(
        strategy=chunks[0].strategy if chunks else "unknown",
        n_before=len(chunks), n_after=len(kept), n_removed=n_removed,
        n_buckets_with_dupes=n_buckets_with_dupes,
    )
    return kept, report


if __name__ == "__main__":
    import argparse

    from chunking import DATA_DIR, save_chunks
    import json
    from dataclasses import asdict

    parser = argparse.ArgumentParser(
        description="Remove near-duplicate chunks from an already-built "
                     "chunks_<strategy>.jsonl file, in place. Run "
                     "`python src/index.py --strategy <s>` afterward to "
                     "re-index the deduped chunk set."
    )
    parser.add_argument("--strategy", default="sentence",
                         choices=["fixed_size", "sentence", "semantic"])
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    from embed import embed_fn_factory

    chunks_path = DATA_DIR / f"chunks_{args.strategy}.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path} not found — run `python src/chunking.py` first."
        )
    with open(chunks_path, encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]

    deduped, report = remove_near_duplicates(
        chunks, embed_fn=embed_fn_factory(), similarity_threshold=args.threshold
    )
    save_chunks(deduped, args.strategy)

    print(f"[{args.strategy}] {report.n_before} -> {report.n_after} chunks "
          f"({report.n_removed} near-duplicates removed, threshold={args.threshold}, "
          f"across {report.n_buckets_with_dupes} focus-buckets with at least one dupe)")
    print(f"chunks_{args.strategy}.jsonl updated in place. "
          f"Run `python src/index.py --strategy {args.strategy}` to re-index.")