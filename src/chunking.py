"""
chunking.py — Three swappable chunking strategies over MedQuAD QA pairs.

Design decision: we chunk the *answer* text (since that's what can run to
4000+ words), but every chunk keeps its parent question/focus/source as
metadata. This matters for retrieval quality — a chunk that's just a
paragraph like "Radiation therapy uses two types..." is meaningless without
knowing which disease/question it belongs to, so we prepend light context
when embedding (see chunk.embed_text) while keeping the raw chunk_text clean
for display/citation.

Strategies implemented:
  1. fixed_size   — naive baseline: split by character count with overlap.
                    Fast, no NLP, but can cut mid-sentence. This is the
                    strategy most tutorial RAG projects stop at.
  2. sentence      — split into sentences, then greedily pack sentences into
                    chunks up to a token budget. Respects sentence boundaries.
  3. semantic      — embed each sentence, walk through the answer, and start
                    a new chunk when cosine similarity between consecutive
                    sentences drops below a threshold (i.e. topic shift).
                    Needs an embedding model, so it's the most expensive but
                    should produce the most coherent chunks.

We keep all three so Day 3's ablation can measure retrieval quality
(recall@k) across strategies instead of just asserting one is better.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Rough chars-per-token estimate for English (avoids pulling in a tokenizer
# just for chunk-sizing decisions; good enough for chunking purposes).
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    qid: str
    focus: str
    qtype: str
    source: str
    url: str
    question: str
    chunk_text: str
    chunk_index: int
    strategy: str

    def embed_text(self) -> str:
        """Text actually sent to the embedding model — includes light
        context so the chunk is self-contained for semantic search."""
        return f"Question: {self.question}\nFocus: {self.focus}\n\n{self.chunk_text}"


_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!\b[A-Z])(?<=[.!?])\s+(?=[A-Z0-9])"
)


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (regex-based, no NLTK download needed —
    keeps the pipeline runnable offline/without extra data downloads).
    Not perfect on abbreviations (e.g. 'Dr.', 'e.g.') but good enough for
    medical answer text, which is mostly plain declarative sentences."""
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def fixed_size_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split by raw character count with overlap. chunk_size/overlap in chars.
    ~800 chars ≈ 200 tokens, a common baseline chunk size."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def sentence_chunks(text: str, max_tokens: int = 200) -> list[str]:
    """Greedily pack sentences into chunks up to a token budget, never
    splitting a sentence mid-way."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    max_chars = max_tokens * CHARS_PER_TOKEN
    chunks, current = [], []
    current_len = 0
    for sent in sentences:
        # Safety net: MedQuAD has tabular/list-style answers (e.g. symptom
        # tables with no terminal punctuation) that the regex splitter can't
        # break up, producing one giant "sentence". Hard-split those with
        # the fixed-size chunker rather than emitting an oversized chunk.
        if len(sent) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(fixed_size_chunks(sent, chunk_size=max_chars, overlap=0))
            continue
        sent_len = len(sent)
        if current and current_len + sent_len > max_chars:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def semantic_chunks(text: str, embed_fn, similarity_threshold: float = 0.55,
                     max_tokens: int = 250) -> list[str]:
    """Split on topic shifts: embed each sentence, compare cosine similarity
    of consecutive sentences, start a new chunk when similarity drops below
    `similarity_threshold` OR the chunk hits max_tokens (safety cap so a
    single chunk can't grow unbounded on a very self-similar answer).

    `embed_fn` is injected (rather than importing sentence-transformers here)
    so this module has no hard ML dependency and can be unit-tested with a
    fake embedder.
    """
    import numpy as np

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences

    embeddings = embed_fn(sentences)  # shape (n_sentences, dim)
    max_chars = max_tokens * CHARS_PER_TOKEN

    chunks, current = [], [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        sent_len = len(sentences[i])
        if sim < similarity_threshold or current_len + sent_len > max_chars:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentences[i])
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def _cosine_sim(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def build_chunks(strategy: str, embed_fn=None) -> list[Chunk]:
    """Read qa_pairs.jsonl and produce Chunk records for the given strategy."""
    qa_path = DATA_DIR / "qa_pairs.jsonl"
    chunks: list[Chunk] = []

    with open(qa_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    for row in rows:
        if strategy == "fixed_size":
            pieces = fixed_size_chunks(row["answer"])
        elif strategy == "sentence":
            pieces = sentence_chunks(row["answer"])
        elif strategy == "semantic":
            if embed_fn is None:
                raise ValueError("semantic strategy requires embed_fn")
            pieces = semantic_chunks(row["answer"], embed_fn)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{row['pair_uid']}::{strategy}::{i}",
                    doc_id=row["doc_id"],
                    qid=row["qid"],
                    focus=row["focus"],
                    qtype=row["qtype"],
                    source=row["source"],
                    url=row["url"],
                    question=row["question"],
                    chunk_text=piece,
                    chunk_index=i,
                    strategy=strategy,
                )
            )
    return chunks


def save_chunks(chunks: list[Chunk], strategy: str) -> Path:
    out_path = DATA_DIR / f"chunks_{strategy}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return out_path


if __name__ == "__main__":
    for strategy in ["fixed_size", "sentence", "semantic"]:
        if strategy == "semantic":
            from embed import embed_fn_factory

            chunks = build_chunks(strategy, embed_fn=embed_fn_factory())
        else:
            chunks = build_chunks(strategy)
        path = save_chunks(chunks, strategy)
        lengths = [len(c.chunk_text.split()) for c in chunks]
        print(f"[{strategy}] {len(chunks)} chunks | "
              f"avg words/chunk: {sum(lengths)/len(lengths):.1f} | "
              f"max: {max(lengths)} -> saved to {path}")
