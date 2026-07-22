"""
reranker.py — Cross-encoder reranker.

Why rerank at all: bi-encoders (what DenseRetriever uses) embed the query
and each chunk *independently* then compare vectors — fast (good for
searching thousands of chunks) but less accurate, because the model never
sees the query and chunk together. A cross-encoder feeds (query, chunk)
pairs jointly through the model and outputs a relevance score directly —
much more accurate, but too slow to run over the whole corpus. So the
pattern is: cheap retrieval (dense + BM25) narrows thousands of chunks down
to ~20-40 candidates, then the expensive-but-accurate reranker picks the
true best k from those.
"""

from __future__ import annotations

from retriever import RetrievedChunk

_RERANKER_CACHE: dict[str, object] = {}

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker(model_name: str = DEFAULT_MODEL):
    if model_name not in _RERANKER_CACHE:
        from sentence_transformers import CrossEncoder
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int = 5,
           model_name: str = DEFAULT_MODEL) -> list[RetrievedChunk]:
    """Re-score `candidates` against `query` with a cross-encoder, return
    the top_k re-sorted by the new score. Deduplicates by chunk_id first
    (hybrid fusion can otherwise pass the same chunk to the reranker twice).

    IMPORTANT: we feed the cross-encoder the chunk's *question + focus*
    context alongside its raw text, not chunk_text alone. Found via a real
    eval failure: MedQuAD splits each disease into several separate QA pairs
    by question type (symptoms / treatment / outlook / causes, etc). Feeding
    only raw chunk_text meant the reranker could only judge topical overlap
    ("myelofibrosis" vs "myelofibrosis"), so it consistently preferred a
    "what is it" or "treatment" chunk about the right disease over the chunk
    that actually answered the specific question asked (e.g. "outlook").
    embed.py already avoids this (it embeds "Question: ...\\nFocus: ...\\n\\n
    {chunk_text}") — the reranker just hadn't been given the same context.
    Passing the same enriched text here lets the cross-encoder weigh
    question-type match, not just disease-name overlap.
    """
    if not candidates:
        return []

    seen = set()
    deduped = []
    for c in candidates:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            deduped.append(c)

    model = get_reranker(model_name)
    pairs = [
        [query, f"Question: {c.metadata.get('question', '')}\n"
                f"Focus: {c.metadata.get('focus', '')}\n\n{c.chunk_text}"]
        for c in deduped
    ]
    scores = model.predict(pairs)

    for c, score in zip(deduped, scores):
        c.score = float(score)

    deduped.sort(key=lambda c: c.score, reverse=True)
    return deduped[:top_k]