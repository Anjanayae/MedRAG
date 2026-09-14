"""
citation_verify.py — Independent verification that each inline citation
([1], [2], ...) in a generated answer is actually supported by the chunk
it points to.

Why this exists: generation.py's SYSTEM_PROMPT *instructs* the model to
cite every claim accurately, but nothing today checks that it did. A model
can put [1] next to a claim [1] doesn't actually support, cite the wrong
source, or make a claim with no citation at all, and nothing in the
pipeline catches it — this matters more here than in a generic RAG demo
because a confidently-cited wrong claim in a medical answer is exactly the
failure mode generation.py's own docstring calls out as worst-case.

This is a SECOND, independent LLM-judge pass, separate from the model call
that produced the answer — same "don't trust the model to self-police"
principle generation.py already applies to the retrieval confidence gate,
just applied to citations instead of retrieval strength.

Mirrors the existing judge pattern in evaluate.py (JUDGE_PROMPT_TEMPLATE +
call_llm) rather than inventing a new calling convention.

Consumers:
  - generation.py: computes the citation_coverage component of the
    composite confidence score.
  - evaluate.py: citation-accuracy eval metric.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from generation import call_llm

_CITATION_RE = re.compile(r"\[(\d+)\]")

# Same lightweight, no-NLTK sentence splitter approach as chunking.py's
# split_sentences — good enough for citation-checking purposes, no reason
# to pull in a heavier splitter just for this.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class CitationCheck:
    marker: int  # which [N] this citation is
    claim_text: str  # the sentence the marker appeared in
    source_index: int  # 1-based index into the sources list (== marker)
    supported: bool | None  # None only if the judge's response wasn't parseable JSON
    raw_judge_response: str | None = None


def extract_cited_sentences(answer: str) -> list[tuple[str, list[int]]]:
    """Split the answer into sentences; for each, extract which citation
    marker numbers appear in it. Sentences with no marker are still
    returned (empty marker list) so uncited_claim_fraction() can use the
    same split without re-parsing the answer a second time."""
    sentences = _SENT_SPLIT_RE.split(answer.strip())
    return [(s.strip(), [int(m) for m in _CITATION_RE.findall(s)]) for s in sentences if s.strip()]


CITATION_JUDGE_PROMPT = """You will be shown a CLAIM (one sentence from a generated answer) and the SOURCE TEXT it was cited against.

Does the SOURCE TEXT actually support the CLAIM? A source supports a claim if a careful reader would agree the source text contains the information stated in the claim (paraphrasing is fine; contradiction, or the source simply not mentioning it, is NOT support).

Respond with ONLY a JSON object, no other text: {{"supported": true or false, "reason": "<one short sentence>"}}

CLAIM: {claim}

SOURCE TEXT: {source_text}
"""


def verify_citation(claim: str, source_text: str) -> tuple[bool | None, str | None]:
    """Runs ONE claim/source pair through the judge. Isolated in its own
    function so it's the one thing mocked in tests — same pattern as
    generation.call_groq / evaluate.call_llm."""
    prompt = CITATION_JUDGE_PROMPT.format(claim=claim, source_text=source_text)
    raw = call_llm(prompt)
    try:
        parsed = json.loads(raw)
        return bool(parsed.get("supported")), raw
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None, raw


def verify_answer_citations(
    answer: str,
    chunk_texts: dict[int, str],
) -> list[CitationCheck]:
    """
    answer: generated answer text with [1]/[2]-style inline citations.
    chunk_texts: maps 1-based source index -> the chunk text that index
        points to (same numbering as generation.build_sources() /
        build_context_block()). Passed in rather than re-derived from
        RetrievedChunk here, so this module has zero dependency on
        retriever internals — it only needs plain strings.

    Only sentences that carry a citation marker are checked here; a
    sentence with no marker isn't claiming to be grounded in a numbered
    source at all — see uncited_claim_fraction() for that separate case.
    A sentence citing two sources (e.g. "...[1][2].") is checked against
    each source it cites, once per source.
    """
    checks: list[CitationCheck] = []
    for sentence, markers in extract_cited_sentences(answer):
        for marker in markers:
            source_text = chunk_texts.get(marker)
            if source_text is None:
                # Model cited a source number that isn't in our sources
                # list at all (hallucinated citation index) — automatic
                # fail, no need to spend a judge call confirming it.
                checks.append(CitationCheck(
                    marker=marker, claim_text=sentence, source_index=marker,
                    supported=False, raw_judge_response="cited index not in sources list",
                ))
                continue
            supported, raw = verify_citation(sentence, source_text)
            checks.append(CitationCheck(
                marker=marker, claim_text=sentence, source_index=marker,
                supported=supported, raw_judge_response=raw,
            ))
    return checks


def citation_coverage(checks: list[CitationCheck]) -> float:
    """Fraction of citation checks that came back supported=True.
    Returns 1.0 if there were no citations to check (nothing to penalize
    here — "no citations present at all" is a distinct failure mode,
    caught by uncited_claim_fraction() instead, not this function)."""
    if not checks:
        return 1.0
    resolved = [c for c in checks if c.supported is not None]
    if not resolved:
        return 0.5  # every judge call failed to parse — genuinely unknown, not full credit
    return sum(1 for c in resolved if c.supported) / len(resolved)


def uncited_claim_fraction(answer: str) -> float:
    """Fraction of substantive sentences (>3 words) carrying NO citation
    marker at all. High fraction = model asserting things without
    pointing to any source — different from citation_coverage(), which
    only measures accuracy of citations that ARE present."""
    substantive = [(s, m) for s, m in extract_cited_sentences(answer) if len(s.split()) > 3]
    if not substantive:
        return 0.0
    uncited = sum(1 for _, markers in substantive if not markers)
    return uncited / len(substantive)