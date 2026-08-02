# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from export_glm_hierarchy_adjudication import adjudication_package, low_confidence_indices  # noqa: E402
from glm_hierarchical_labels import HierarchicalAssignment  # noqa: E402
from glm_semantic_labels import SampleDocument  # noqa: E402


def assignment(sample_index: int, confidence: float) -> HierarchicalAssignment:
    return HierarchicalAssignment(
        sample_index=sample_index,
        primary_parent_id="SCIENCE",
        secondary_parent_ids=[],
        primary_leaf_id="BIOLOGY",
        secondary_leaf_ids=[],
        form_id="ARTICLE",
        confidence=confidence,
        rationale="Test",
    )


def test_low_confidence_indices_select_exact_bottom_fraction() -> None:
    assignments = [assignment(index, confidence) for index, confidence in enumerate([0.9, 0.1, 0.8, 0.2, 0.7])]

    selected = low_confidence_indices(assignments, 0.4)

    assert selected == [1, 3]


def test_low_confidence_indices_round_up_small_tail() -> None:
    assignments = [assignment(index, 0.5) for index in range(21)]

    selected = low_confidence_indices(assignments, 0.05)

    assert len(selected) == 2
    assert set(selected).issubset(range(21))


def test_low_confidence_indices_reject_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="tail fraction"):
        low_confidence_indices([assignment(0, 0.5)], 0)


def test_adjudication_package_hides_sources_and_aligns_rows() -> None:
    documents = [
        SampleDocument(index, f"hash-{index}", f"source-{index}", "standard", index, f"text-{index}")
        for index in range(4)
    ]
    assignments = [assignment(index, confidence) for index, confidence in enumerate([0.9, 0.1, 0.8, 0.2])]
    taxonomy = {
        "parents": [{"bucket_id": "SCIENCE"}],
        "leaves": [{"bucket_id": "BIOLOGY", "parent_id": "SCIENCE"}],
        "precedence_rules": [],
    }

    package = adjudication_package(documents, assignments, taxonomy, 0.5)

    assert package["samples"] == {"adjudication": [1, 3]}
    assert [row["sample_index"] for row in package["documents"]] == [1, 3]
    assert [row["text"] for row in package["documents"]] == ["text-1", "text-3"]
    assert package["source_metadata_in_package"] is False
    assert all("source" not in row for row in package["documents"])
