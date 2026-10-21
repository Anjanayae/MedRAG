"""
contradiction_detection.py — Checks whether the highly-relevant retrieved
sources for a query contain MATERIALLY conflicting information, before
generation runs.

Why this matters here specifically: hybrid retrieval can legitimately
return two chunks that both look relevant to the query but disagree —
different documents on the same topic, differing dosage guidance, or one
source updated more recently than another. Without a check like this, the
generator just silently picks whichever source it happens to lean on and
the conflict is invisible to the person asking. In a medical context that's
a real, not cosmetic, gap.

Design mirrors citation_verify.py's batched judge pattern (one LLM call
covering all pairs, not one call per pair) — same "an answer with N
retrieved chunks shouldn't cost N^2 judge calls" principle, and reuses
generation.call_llm rather than inventing a new LLM-calling convention.

ADDITIVE ONLY: this module doesn't touch retrieval or generation itself —
generation.py calls find_contradictions() as an opt-in pre-generation step
(check_contradictions=True) and, if conflicts are found, adds a short note
to the PROMPT (not the system prompt, so this doesn't change behavior for
any request that doesn't opt in) instructing the model to acknowledge the
conflict explicitly instead of silently picking one source's version. If
this module fails (parse error, no chunks, etc.) it fails safe — returns no
detected contradictions rather than blocking generation, since a missed
contradiction check should degrade to "no info available", not break the
answer pipeline entirely.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

from generation import call_llm


@dataclass
class ContradictionFlag:
    source_index_a: int  # 1-based, matching generation.build_sources() numbering
    source_index_b: int
    explanation: str


CONTRADICTION_JUDGE_PROMPT = """You will be shown a JSON array of SOURCE PAIRS, each retrieved for the same medical question, with an "index". For EACH pair, decide whether the two sources contain MATERIALLY CONFLICTING information — e.g. different dosage recommendations, contradictory causes, opposing treatment advice — as opposed to merely different emphasis, different level of detail, or complementary (non-contradictory) information.

Respond with ONLY a JSON array, no other text, one object per pair, same "index" values: [{{"index": <int>, "conflicting": true or false, "explanation": "<one short sentence, empty string if not conflicting>"}}, ...]

PAIRS:
{pairs_json}
"""


def find_contradictions(chunks: list, max_chunks: int = 5) -> list[ContradictionFlag]:
    """Compares every pair among the first max_chunks chunks (these are
    assumed to already be the highly-relevant, reranked top-k the caller
    is about to generate from — not the full candidate pool) in a SINGLE
    batched LLM call. max_chunks=5 keeps the number of pairs bounded
    (C(5,2)=10) even if a caller passes a larger chunk list.

    Returns only the pairs judged conflicting (empty list if none, or if
    fewer than 2 chunks, or if the judge call fails to parse — fails safe,
    doesn't raise)."""
    considered = chunks[:max_chunks]
    if len(considered) < 2:
        return []

    pairs = list(itertools.combinations(range(len(considered)), 2))
    payload = [
        {"index": i, "source_a": considered[a].chunk_text, "source_b": considered[b].chunk_text}
        for i, (a, b) in enumerate(pairs)
    ]
    prompt = CONTRADICTION_JUDGE_PROMPT.format(pairs_json=json.dumps(payload))

    try:
        raw = call_llm(prompt)
        verdicts = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ConnectionError, RuntimeError):
        # Fail safe: a broken/unparseable judge call means "couldn't check",
        # not "definitely no conflict" -- but blocking generation on a
        # judge hiccup would be worse than just proceeding without the
        # (unavailable) contradiction info this call. See module docstring.
        return []

    flags = []
    for v in verdicts:
        try:
            idx = int(v["index"])
            conflicting = bool(v.get("conflicting"))
        except (KeyError, TypeError, ValueError):
            continue
        if not conflicting or not (0 <= idx < len(pairs)):
            continue
        a, b = pairs[idx]
        flags.append(ContradictionFlag(
            source_index_a=a + 1, source_index_b=b + 1,
            explanation=v.get("explanation") or "",
        ))
    return flags


def summarize_contradictions(flags: list[ContradictionFlag]) -> str | None:
    """Human-readable note for the generation prompt (see generation.py's
    check_contradictions integration) — instructs the model to acknowledge
    the conflict explicitly rather than silently choosing one source.
    Returns None if there's nothing to summarize (nothing appended to the
    prompt in that case)."""
    if not flags:
        return None
    lines = [f"- Sources [{f.source_index_a}] and [{f.source_index_b}]: {f.explanation}" for f in flags]
    return (
        "\n\nNOTE — the following retrieved sources may contain conflicting information:\n"
        + "\n".join(lines)
        + "\nIf your answer relies on information from these sources, explicitly acknowledge "
        "the conflict (e.g. 'sources disagree on X') rather than silently picking one source's version."
    )