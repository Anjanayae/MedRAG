"""
ingest.py — Parse the raw MedQuAD XML files into a single clean, structured
dataset (JSONL + CSV) ready for chunking/embedding.

Why this exists as its own module (not inline in a notebook):
  - It's the reproducible, testable entry point of the pipeline.
  - Someone reviewing the repo should be able to run `python src/ingest.py`
    and regenerate data/processed/qa_pairs.jsonl from scratch.

Notes on the data:
  - MedQuAD ships 12 subsets. 3 of them (A.D.A.M., MedlinePlus Drugs,
    MedlinePlus Herbs & Supplements) have their <Answer> tags stripped for
    copyright reasons — we exclude those folders entirely rather than
    silently keeping empty-answer rows.
  - Each XML "Document" can contain multiple QAPairs, and Documents are
    nested under source-specific subfolders.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from lxml import etree
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "MedQuAD"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# These subsets have no answers (copyright restriction from MedlinePlus) — skip.
EXCLUDED_SUBSETS = {
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
}


@dataclass
class QAPair:
    pair_uid: str  # globally unique across the whole corpus — see note below
    doc_id: str
    qid: str
    focus: str
    qtype: str
    question: str
    answer: str
    source: str
    url: str
    semantic_group: str | None = None


def clean_text(text: str | None) -> str:
    """Collapse whitespace/newlines that XML pretty-printing introduces."""
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_file(path: Path) -> list[QAPair]:
    pairs: list[QAPair] = []
    try:
        tree = etree.parse(str(path))
    except etree.XMLSyntaxError:
        return pairs

    root = tree.getroot()
    doc_id = root.get("id", path.stem)
    source = root.get("source", "unknown")
    url = root.get("url", "")

    focus_el = root.find("Focus")
    focus = clean_text(focus_el.text) if focus_el is not None else ""

    sem_group_el = root.find(".//SemanticGroup")
    semantic_group = clean_text(sem_group_el.text) if sem_group_el is not None else None

    for qa_pair_el in root.findall(".//QAPair"):
        q_el = qa_pair_el.find("Question")
        a_el = qa_pair_el.find("Answer")
        if q_el is None or a_el is None:
            continue
        question = clean_text(q_el.text)
        answer = clean_text(a_el.text)
        if not question or not answer:
            continue  # copyright-stripped or malformed entries
        pairs.append(
            QAPair(
                pair_uid="",  # placeholder — assigned a real global unique id in main()
                doc_id=doc_id,
                qid=q_el.get("qid", f"{doc_id}-{len(pairs)}"),
                focus=focus,
                qtype=q_el.get("qtype", "unknown"),
                question=question,
                answer=answer,
                source=source,
                url=url,
                semantic_group=semantic_group,
            )
        )
    return pairs


def list_xml_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Deterministically ordered list of XML files to parse. Order here
    directly determines `pair_uid` assignment in main(), so it MUST be
    stable across runs/machines/filesystems — hence sorting both the
    subset directories and the files within each (rglob's own order is
    filesystem-dependent, not guaranteed alphabetical)."""
    return [
        p
        for subset_dir in sorted(raw_dir.iterdir())
        if subset_dir.is_dir() and subset_dir.name not in EXCLUDED_SUBSETS
        for p in sorted(subset_dir.rglob("*.xml"))
    ]


def main():
    xml_files = list_xml_files()
    print(f"Found {len(xml_files)} XML files across {12 - len(EXCLUDED_SUBSETS)} subsets "
          f"(excluded {len(EXCLUDED_SUBSETS)} copyright-restricted subsets)")

    all_pairs: list[QAPair] = []
    for f in tqdm(xml_files, desc="Parsing XML"):
        all_pairs.extend(parse_file(f))

    print(f"Parsed {len(all_pairs)} clean QA pairs")

    # MedQuAD's own `qid`/`doc_id` values are only unique *within* the source
    # file they came from — several sources (CancerGov notably) reuse the
    # same numbering scheme across genuinely different disease topics. ~41%
    # of pairs collide on qid if you rely on it directly (confirmed by a
    # DuplicateIDError from Chroma downstream). We assign our own
    # sequential, guaranteed-unique id here so nothing downstream (chunk_id,
    # vector store ids) has to worry about it. Original doc_id/qid are kept
    # as metadata for reference/display only.
    for i, pair in enumerate(all_pairs):
        pair.pair_uid = f"{i:07d}"

    jsonl_path = OUT_DIR / "qa_pairs.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")

    import pandas as pd
    df = pd.DataFrame([asdict(p) for p in all_pairs])
    df.to_csv(OUT_DIR / "qa_pairs.csv", index=False)

    # Quick stats — useful both for sanity-checking and for the README later.
    print("\n--- Dataset stats ---")
    print(f"Total QA pairs: {len(df)}")
    print(f"Unique documents (diseases/drugs/topics): {df['doc_id'].nunique()}")
    print(f"Unique sources: {df['source'].nunique()} -> {sorted(df['source'].unique())}")
    print(f"Top 10 question types:\n{df['qtype'].value_counts().head(10)}")
    print(f"Answer length (words) — mean: {df['answer'].str.split().str.len().mean():.1f}, "
          f"median: {df['answer'].str.split().str.len().median():.0f}, "
          f"max: {df['answer'].str.split().str.len().max()}")
    dup_qid_count = df['qid'].duplicated(keep=False).sum()
    print(f"Note: {dup_qid_count} pairs ({100*dup_qid_count/len(df):.1f}%) share a duplicated "
          f"'qid' across different topics — MedQuAD's own ids are only locally unique. "
          f"We assign our own 'pair_uid' (sequential, globally unique) instead of relying on it.")

    print(f"\nSaved -> {jsonl_path}")
    print(f"Saved -> {OUT_DIR / 'qa_pairs.csv'}")


if __name__ == "__main__":
    main()
