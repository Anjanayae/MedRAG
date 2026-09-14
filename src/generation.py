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

Provider swap: set LLM_PROVIDER=openai|anthropic|deepseek (default: groq) in
.env to use a different LLM for generation, with the matching *_API_KEY.
call_groq/call_openai/call_anthropic/call_deepseek stay as separate,
independently-testable functions (each mockable on its own, same pattern as
the original call_groq) — call_llm() just picks which one to invoke based on
LLM_PROVIDER. Nothing about the retrieval/reranking pipeline changes; this
only affects which API generate_answer() calls.
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

# Provider swap (LLM_PROVIDER env var, default "groq"): each provider gets
# its own default model so switching providers doesn't require also passing
# a model string by hand. These are reasonable current picks, not tuned.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

DEFAULT_MODEL_BY_PROVIDER = {
    "groq": DEFAULT_GROQ_MODEL,
    "openai": DEFAULT_OPENAI_MODEL,
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
    "deepseek": DEFAULT_DEEPSEEK_MODEL,
}

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
    confidence: float  # retrieval-only signal, unchanged: sigmoid(top rerank score).
    #   Kept as-is (name and meaning) so existing callers (api.py's
    #   QueryResponse, app.py's UI) don't break. Everything below is new
    #   and additive.
    refused: bool
    model: str | None = None
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    # --- citation verification (populated only when verify_citations=True
    # is passed to generate_answer(); None otherwise, so callers not using
    # this feature see no behavior/shape change) ---
    citation_checks: list[dict] | None = None
    citation_coverage: float | None = None
    source_utilization: float | None = None
    composite_confidence: float | None = None


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


_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file, or "
                "export it in your shell."
            )
        from openai import OpenAI
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def call_openai(prompt: str, model: str = DEFAULT_OPENAI_MODEL) -> str:
    """Same shape as call_groq: isolated so it's the one thing mocked in
    tests, no real API key/network call needed."""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


_anthropic_client = None


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file, or "
                "export it in your shell."
            )
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def call_anthropic(prompt: str, model: str = DEFAULT_ANTHROPIC_MODEL) -> str:
    client = get_anthropic_client()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.content[0].text


_deepseek_client = None


def get_deepseek_client():
    """DeepSeek's API is OpenAI-compatible, so we reuse the openai SDK
    pointed at DeepSeek's base_url rather than pulling in another SDK."""
    global _deepseek_client
    if _deepseek_client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Add it to your .env file, or "
                "export it in your shell."
            )
        from openai import OpenAI
        _deepseek_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return _deepseek_client


def call_deepseek(prompt: str, model: str = DEFAULT_DEEPSEEK_MODEL) -> str:
    client = get_deepseek_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def call_llm(prompt: str, model: str | None = None, provider: str | None = None) -> str:
    """Provider dispatcher. provider defaults to the LLM_PROVIDER env var
    (default "groq" — unchanged behavior for anyone not using this feature).
    model defaults to that provider's own default model if not given, so
    switching providers via env var alone (no code/call-site change) works
    without accidentally sending a Groq model name to another provider's API.

    Deliberately calls call_groq/call_openai/call_anthropic/call_deepseek by
    name (not via a dict built at import time) so monkeypatching any of
    those functions directly — e.g. in tests — is still respected here."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "groq")).lower()
    if provider not in DEFAULT_MODEL_BY_PROVIDER:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Expected one of: "
            f"{sorted(DEFAULT_MODEL_BY_PROVIDER)}"
        )
    resolved_model = model or DEFAULT_MODEL_BY_PROVIDER[provider]

    if provider == "groq":
        return call_groq(prompt, model=resolved_model)
    elif provider == "openai":
        return call_openai(prompt, model=resolved_model)
    elif provider == "anthropic":
        return call_anthropic(prompt, model=resolved_model)
    elif provider == "deepseek":
        return call_deepseek(prompt, model=resolved_model)


def build_chunk_text_map(chunks: list[RetrievedChunk]) -> dict[int, str]:
    """1-based index -> chunk_text, matching the numbering build_sources()
    and build_context_block() already use. citation_verify.py needs plain
    (index -> text) strings, not RetrievedChunk objects, so this is the one
    place that bridges retriever internals to that module."""
    return {i: c.chunk_text for i, c in enumerate(chunks, 1)}


# Composite confidence weights. retrieval carries the most weight (it's
# what the existing pre-generation refusal gate already relies on and has
# been the sole signal so far); citation_coverage and source_utilization
# are the two new signals that catch failures retrieval confidence alone
# can't see — e.g. retrieval was fine but the model cited the wrong chunk,
# or ignored most of what was retrieved. These weights are a starting
# point, not tuned — same "eval harness calibrates it, not vibes" spirit
# as RERANK_CONFIDENCE_THRESHOLD above.
COMPOSITE_WEIGHTS = {"retrieval": 0.5, "citation_coverage": 0.3, "source_utilization": 0.2}


def compute_composite_confidence(
    retrieval_confidence: float,
    citation_coverage_score: float,
    source_utilization_score: float,
) -> float:
    """Combines three signals the pre-generation refusal gate can't see
    together, since it only has retrieval_confidence available before the
    LLM has even run:
      - retrieval_confidence: same sigmoid(top rerank score) as `confidence`
      - citation_coverage_score: fraction of the answer's citations that an
        independent judge confirmed are actually supported by their source
      - source_utilization_score: fraction of retrieved sources the answer
        actually cited at least once (low = model ignored most of the
        evidence it was given)
    This is a POST-generation score — it doesn't replace or feed back into
    the existing pre-generation refusal gate (check_confidence), which
    stays exactly as it was so the "skip the LLM call entirely on weak
    retrieval" cost-saving behavior is unaffected."""
    w = COMPOSITE_WEIGHTS
    return (
        w["retrieval"] * retrieval_confidence
        + w["citation_coverage"] * citation_coverage_score
        + w["source_utilization"] * source_utilization_score
    )


def compute_source_utilization(chunk_texts: dict[int, str], checks: list) -> float:
    """Fraction of provided sources (1..len(chunk_texts)) that were cited
    at least once anywhere in the answer (i.e. appear as a marker in
    `checks`, regardless of whether that citation was judged supported —
    this measures USE of evidence, not accuracy of it; citation_coverage
    already measures accuracy)."""
    if not chunk_texts:
        return 0.0
    cited_indices = {c.source_index for c in checks}
    return len(cited_indices & set(chunk_texts)) / len(chunk_texts)


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
    model: str | None = None,
    confidence_threshold: float = RERANK_CONFIDENCE_THRESHOLD,
    verify_citations: bool = False,
) -> GenerationResult:
    """model=None (default) means "use whatever LLM_PROVIDER is configured,
    with that provider's own default model" — see call_llm(). Pass an
    explicit model string to override. Provider itself is picked up from
    the LLM_PROVIDER env var (default "groq", i.e. unchanged behavior for
    anyone not using the multi-provider feature).

    verify_citations=False by default: citation verification costs one
    extra LLM-judge call per citation in the answer, so it's opt-in rather
    than run on every request automatically. Set True (or pass
    use_verification=True through api.py) to get citation_checks,
    citation_coverage, source_utilization, and composite_confidence
    populated on the result. The existing pre-generation refusal gate
    (check_confidence, right below) is completely unaffected either way —
    verification only ever runs AFTER an answer is already generated."""
    confidence, refused = check_confidence(retrieved_chunks, confidence_threshold)
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    resolved_model = model or DEFAULT_MODEL_BY_PROVIDER.get(provider, DEFAULT_GROQ_MODEL)

    if refused:
        return GenerationResult(
            answer=REFUSAL_MESSAGE,
            sources=build_sources(retrieved_chunks),
            confidence=confidence,
            refused=True,
            model=resolved_model,
        )

    context_block = build_context_block(retrieved_chunks)
    prompt = f"Sources:\n{context_block}\n\nQuestion: {query}"
    answer_text = call_llm(prompt, model=model, provider=provider)

    result = GenerationResult(
        answer=answer_text,
        sources=build_sources(retrieved_chunks),
        confidence=confidence,
        refused=False,
        model=resolved_model,
    )

    if verify_citations:
        from citation_verify import (
            verify_answer_citations, citation_coverage as compute_citation_coverage,
        )
        chunk_texts = build_chunk_text_map(retrieved_chunks)
        checks = verify_answer_citations(answer_text, chunk_texts)
        cov = compute_citation_coverage(checks)
        util = compute_source_utilization(chunk_texts, checks)

        result.citation_checks = [
            {
                "marker": c.marker, "claim_text": c.claim_text,
                "source_index": c.source_index, "supported": c.supported,
            }
            for c in checks
        ]
        result.citation_coverage = round(cov, 3)
        result.source_utilization = round(util, 3)
        result.composite_confidence = round(
            compute_composite_confidence(confidence, cov, util), 3
        )

    return result