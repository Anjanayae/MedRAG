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

Batched, not one-call-per-citation: verify_answer_citations() collects
every (claim, cited source text) pair in the answer and sends them to the
judge in a SINGLE call (a JSON array in, a JSON array of verdicts out),
rather than one judge call per citation. An answer with 5 citations costs
1 judge call, not 5 — same verification depth, far less latency/API cost.
verify_citation() (single-pair) is kept as its own function for direct use
and for the mock-a-single-thing-in-tests pattern the rest of this codebase
uses; verify_answer_citations() just doesn't loop over it anymore.

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

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class CitationCheck:
    marker: int
    claim_text: str
    source_index: int
    supported: bool | None
    raw_judge_response: str | None = None


def extract_cited_sentences(answer: str) -> list[tuple[str, list[int]]]:
    sentences = _SENT_SPLIT_RE.split(answer.strip())
    return [(s.strip(), [int(m) for m in _CITATION_RE.findall(s)]) for s in sentences if s.strip()]


CITATION_JUDGE_PROMPT = """You will be shown a CLAIM (one sentence from a generated answer) and the SOURCE TEXT it was cited against.

Does the SOURCE TEXT actually support the CLAIM? A source supports a claim if a careful reader would agree the source text contains the information stated in the claim (paraphrasing is fine; contradiction, or the source simply not mentioning it, is NOT support).

Respond with ONLY a JSON object, no other text: {{"supported": true or false, "reason": "<one short sentence>"}}

CLAIM: {claim}

SOURCE TEXT: {source_text}
"""


def verify_citation(claim: str, source_text: str) -> tuple[bool | None, str | None]:
    """Runs ONE claim/source pair through the judge. Kept for direct use
    and the mock-a-single-thing-in-tests pattern; not used in a loop by
    verify_answer_citations() anymore (see _run_batched_judge)."""
    prompt = CITATION_JUDGE_PROMPT.format(claim=claim, source_text=source_text)
    raw = call_llm(prompt)
    try:
        parsed = json.loads(raw)
        return bool(parsed.get("supported")), raw
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None, raw


CITATION_JUDGE_BATCH_PROMPT = """You will be shown a JSON array of claim/source pairs, each with an "index". For EACH pair, decide whether the SOURCE TEXT actually supports the CLAIM. A source supports a claim if a careful reader would agree the source text contains the information stated in the claim (paraphrasing is fine; contradiction, or the source simply not mentioning it, is NOT support).

Respond with ONLY a JSON array, no other text, one object per pair, same "index" values: [{{"index": <int>, "supported": true or false, "reason": "<one short sentence>"}}, ...]

PAIRS:
{pairs_json}
"""


def _run_batched_judge(pairs: list[tuple[str, str]]) -> dict[int, tuple[bool | None, str | None]]:
    """pairs: list of (claim, source_text). Returns {pair_index: (supported, reason)}.
    Exactly ONE call_llm() call total, regardless of how many pairs."""
    if not pairs:
        return {}
    payload = [{"index": i, "claim": claim, "source_text": text} for i, (claim, text) in enumerate(pairs)]
    prompt = CITATION_JUDGE_BATCH_PROMPT.format(pairs_json=json.dumps(payload))
    raw = call_llm(prompt)
    try:
        verdicts = json.loads(raw)
        result: dict[int, tuple[bool | None, str | None]] = {}
        for v in verdicts:
            idx = int(v["index"])
            result[idx] = (bool(v.get("supported")), v.get("reason"))
        return result
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        return {i: (None, raw) for i in range(len(pairs))}


def verify_answer_citations(
    answer: str,
    chunk_texts: dict[int, str],
) -> list[CitationCheck]:
    """Only sentences carrying a citation marker are checked. A marker
    with no matching source is resolved locally (no LLM call needed); all
    valid pairs go through ONE batched judge call (_run_batched_judge)."""
    entries: list[tuple[str, int, str | None]] = []
    for sentence, markers in extract_cited_sentences(answer):
        for marker in markers:
            entries.append((sentence, marker, chunk_texts.get(marker)))

    pairs = [(sentence, source_text) for sentence, _marker, source_text in entries if source_text is not None]
    verdicts = _run_batched_judge(pairs)

    checks: list[CitationCheck] = []
    pair_cursor = 0
    for sentence, marker, source_text in entries:
        if source_text is None:
            checks.append(CitationCheck(
                marker=marker, claim_text=sentence, source_index=marker,
                supported=False, raw_judge_response="cited index not in sources list",
            ))
            continue
        supported, reason = verdicts.get(pair_cursor, (None, None))
        checks.append(CitationCheck(
            marker=marker, claim_text=sentence, source_index=marker,
            supported=supported, raw_judge_response=reason,
        ))
        pair_cursor += 1
    return checks


def citation_coverage(checks: list[CitationCheck]) -> float:
    if not checks:
        return 1.0
    resolved = [c for c in checks if c.supported is not None]
    if not resolved:
        return 0.5
    return sum(1 for c in resolved if c.supported) / len(resolved)


def uncited_claim_fraction(answer: str) -> float:
    substantive = [(s, m) for s, m in extract_cited_sentences(answer) if len(s.split()) > 3]
    if not substantive:
        return 0.0
    uncited = sum(1 for _, markers in substantive if not markers)
    return uncited / len(substantive)