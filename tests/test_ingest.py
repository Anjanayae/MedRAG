import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ingest  # noqa: E402


def test_list_xml_files_is_sorted_and_deterministic(tmp_path):
    """Regression test for a real reproducibility bug: pair_uid is assigned
    by enumeration order over the XML files, but `rglob()`'s own traversal
    order is filesystem-dependent (not guaranteed alphabetical) — so without
    explicit sorting, the SAME disease's QA pair could get a DIFFERENT
    pair_uid on a different machine or even a different run on the same
    machine. Caught this when a Windows-generated eval set's gold_pair_uid
    didn't match the same id in a separately-regenerated copy of the data."""
    subset_b = tmp_path / "2_SubsetB_QA"
    subset_a = tmp_path / "1_SubsetA_QA"
    subset_b.mkdir()
    subset_a.mkdir()

    (subset_b / "zzz.xml").write_text("<QAPairs/>")
    (subset_b / "aaa.xml").write_text("<QAPairs/>")
    (subset_a / "ccc.xml").write_text("<QAPairs/>")
    (subset_a / "bbb.xml").write_text("<QAPairs/>")

    result = ingest.list_xml_files(raw_dir=tmp_path)
    result_names = [p.name for p in result]

    assert result_names == ["bbb.xml", "ccc.xml", "aaa.xml", "zzz.xml"], (
        "expected files sorted by subset dir name, then by filename within "
        "each subset — got a different order, which would make pair_uid "
        "non-reproducible across runs"
    )

    result2 = ingest.list_xml_files(raw_dir=tmp_path)
    assert [p.name for p in result2] == result_names


def test_list_xml_files_excludes_copyright_restricted_subsets(tmp_path):
    restricted = tmp_path / "10_MPlus_ADAM_QA"
    allowed = tmp_path / "1_CancerGov_QA"
    restricted.mkdir()
    allowed.mkdir()
    (restricted / "doc.xml").write_text("<QAPairs/>")
    (allowed / "doc.xml").write_text("<QAPairs/>")

    result = ingest.list_xml_files(raw_dir=tmp_path)
    assert len(result) == 1
    assert result[0].parent.name == "1_CancerGov_QA"