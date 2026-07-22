"""
embed.py — Thin wrapper around sentence-transformers so the rest of the
pipeline (chunking, indexing, retrieval) depends on a simple function
signature (`list[str] -> np.ndarray`) instead of the library directly.

Why wrap it:
  - chunking.py's semantic_chunks() takes an injected embed_fn so it has no
    hard ML dependency and is unit-testable with a fake embedder.
  - If we swap embedding models later (ablation: MiniLM vs mpnet, e.g.),
    only this file changes.

Model choice: 'all-MiniLM-L6-v2' — 384-dim, ~80MB, fast on CPU. Good default
for a portfolio project since it doesn't need a GPU. Noted as an ablation
candidate against a larger model (e.g. 'multi-qa-mpnet-base-dot-v1') in
Step 3.
"""

from __future__ import annotations

import numpy as np

_MODEL_CACHE: dict[str, object] = {}

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def get_model(model_name: str = DEFAULT_MODEL):
    """Lazily load and cache the sentence-transformers model (loading it is
    slow — ~1-2s — so we don't want to reinstantiate it per call)."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL,
                 batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Embed a list of texts -> (n, dim) float32 array, L2-normalized so
    cosine similarity == dot product (matters for Chroma's default metric
    and for our own _cosine_sim in chunking.py)."""
    if not texts:
        return np.empty((0, 384), dtype="float32")
    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def embed_fn_factory(model_name: str = DEFAULT_MODEL):
    """Returns a callable matching the `embed_fn` signature chunking.py
    expects: list[str] -> np.ndarray. Used to plug the real model into
    semantic_chunks()."""
    def _fn(texts: list[str]) -> np.ndarray:
        return embed_texts(texts, model_name=model_name)
    return _fn