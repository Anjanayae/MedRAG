"""
query_rewrite.py — Optional query normalization step: rewrites informal or
colloquial user phrasing into terms more likely to match MedQuAD's clinical
vocabulary, BEFORE retrieval.

Why this exists (found via a real failure, not speculation): dense
embeddings and BM25 both struggle when a user's wording doesn't overlap
with how the corpus describes the same concept. E.g. "hairfall" (informal,
one word) vs the corpus's "hair loss" — BM25 finds zero token overlap, and
the embedding model may not map an informal, rarer term as tightly to "hair
loss" as a more literal synonym would. A real query, "how does B12 play a
role in hairfall", got refused (confidence 0.02) even though B12/hair-loss
is a legitimate, well-documented relationship — while "vitamin B12" alone
retrieved fine. Rewriting the query to canonical phrasing BEFORE retrieval
fixes this at the source, instead of loosening the confidence threshold to
force an answer through what would still be weak retrieval.

Uses a small, fast Groq model (not the main generation model) since this
adds a network round-trip to every query and should stay cheap/quick.
Fails open: any error here returns the original query unchanged rather
than blocking retrieval.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

REWRITE_MODEL = "llama-3.1-8b-instant"

REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's medical question using standard medical terms, "
    "preserving the original meaning and intent exactly. Expand colloquial "
    "or informal words into their common medical name (e.g. 'hairfall' -> "
    "'hair loss', 'sugar' -> 'blood glucose'). Use plain terms a patient "
    "education website (like NIH or CDC) would use — NOT obscure clinical "
    "or academic jargon (e.g. prefer 'hair loss' over 'alopecia' or "
    "'pathogenesis of hair loss'). Keep it as a single question, same "
    "length or shorter. Output ONLY the rewritten question, nothing else "
    "— no preamble, no quotes, no explanation."
)


def rewrite_query(query: str, model: str = REWRITE_MODEL) -> str:
    """Returns a rewritten query, or the original query unchanged if the
    rewrite call fails for any reason (network error, empty response,
    missing API key, etc.) — this step must never block retrieval."""
    try:
        from generation import get_groq_client
        client = get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else query
    except Exception:
        return query