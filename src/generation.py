"""
generation.py — Builds the grounded prompt, decides whether to answer at
all (confidence-based refusal), and calls the Groq API.

Confidence-based refusal: rather than always generating an answer and
hoping the LLM says "I don't know" when it should, we check the top
reranked chunk's relevance score *before* calling the LLM. If it's below
threshold, we skip generation entirely and return a canned refusal —
cheaper, faster, and more reliable than trusting the LLM to self-police,
which is especially important in a medical domain where a confident-sounding
wrong answer is worse than no answer.

The threshold below (RERANK_CONFIDENCE_THRESHOLD) is a starting guess, not a
tuned value — the eval-harness step is what actually calibrates it against
real pass/fail cases instead of vibes.

Groq requires GROQ_API_KEY to be set as an environment variable (put it in
a .env file at the project root — see .env.example — loaded via
python-dotenv so it's never hardcoded or committed).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from retriever import RetrievedChunk

load_dotenv()  # reads .env into os.environ if present

# llama-3.3-70b-versatile: good quality/speed tradeoff on Groq for this use
# case. llama-3.1-8b-instant is a faster/cheaper alternative worth an
# ablation entry if you want to compare quality vs latency later.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Cross-encoder scores are raw logits, not probabilities — squash with a
# sigmoid to get an interpretable 0-1 "confidence" for the threshold check.
RERANK_CONFIDENCE_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are a medical information assistant. Answer the user's question using ONLY the numbered sources below.

Rules:
- Cite sources inline like [1], [2] for every claim you make.
- If the sources don't contain enough information to answer, say "I don't have enough information in my knowledge base to answer that confidently" — do not guess or use outside knowledge.
- Keep the answer concise and clearly structured.
- This is general medical information, not a diagnosis or a substitute for professional medical advice — do not present it as either."""

REFUSAL_MESSAGE = (
    "I don't have confident enough information in my knowledge base to answer "
    "that question. Please consult a healthcare professional, or try rephrasing "
    "your question with more specific medical terms."
)


@dataclass
class GenerationResult:
    answer: str
    sources: list[dict]
    confidence: float
    refused: bool
    model: str | None = None
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] (Topic: {c.metadata.get('focus', 'unknown')}) {c.chunk_text}")
    return "\n\n".join(lines)


def build_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "index": i,
            "focus": c.metadata.get("focus"),
            "source": c.metadata.get("source"),
            "url": c.metadata.get("url"),
            "question": c.metadata.get("question"),
            "score": round(c.score, 4),
        }
        for i, c in enumerate(chunks, 1)
    ]


_groq_client = None


def get_groq_client():
    """Lazily construct the Groq client (fails loudly and clearly if
    GROQ_API_KEY isn't set, rather than a confusing error deep in the SDK)."""
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Create a .env file at the project "
                "root with GROQ_API_KEY=your_key_here (see .env.example), or "
                "export it in your shell."
            )
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def call_groq(prompt: str, model: str = DEFAULT_GROQ_MODEL) -> str:
    """Isolated in its own function so it's the one thing we mock in tests
    (no real API key/network call needed for unit tests)."""
    client = get_groq_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,  # low temperature — favor grounded/consistent over creative
    )
    return response.choices[0].message.content


def check_confidence(
    retrieved_chunks: list[RetrievedChunk],
    confidence_threshold: float = RERANK_CONFIDENCE_THRESHOLD,
) -> tuple[float, bool]:
    """Compute confidence and the refusal decision WITHOUT calling the LLM.
    Split out from generate_answer() so the eval harness can test retrieval
    + refusal-gate behavior cheaply (no API calls) across the whole eval
    set, saving real LLM calls for the smaller quality-scoring pass."""
    if not retrieved_chunks:
        return 0.0, True
    confidence = sigmoid(retrieved_chunks[0].score)
    return confidence, confidence < confidence_threshold


def generate_answer(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    model: str = DEFAULT_GROQ_MODEL,
    confidence_threshold: float = RERANK_CONFIDENCE_THRESHOLD,
) -> GenerationResult:
    confidence, refused = check_confidence(retrieved_chunks, confidence_threshold)

    if refused:
        return GenerationResult(
            answer=REFUSAL_MESSAGE,
            sources=build_sources(retrieved_chunks),
            confidence=confidence,
            refused=True,
            model=model,
        )

    context_block = build_context_block(retrieved_chunks)
    prompt = f"Sources:\n{context_block}\n\nQuestion: {query}"
    answer_text = call_groq(prompt, model=model)

    return GenerationResult(
        answer=answer_text,
        sources=build_sources(retrieved_chunks),
        confidence=confidence,
        refused=False,
        model=model,
    )