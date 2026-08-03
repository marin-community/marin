# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from glm_hierarchical_labels import HierarchicalAssignment  # noqa: E402
from train_semantic_projection_prefix import validated_assignment_prefix  # noqa: E402


def assignment(index: int) -> HierarchicalAssignment:
    return HierarchicalAssignment(index, "P", [], "L", [], "GENERAL_PROSE", 0.9, "valid")


def test_assignment_prefix_selects_exact_rows_when_later_checkpoints_exist() -> None:
    rows = [assignment(index) for index in range(7)]

    result = validated_assignment_prefix(rows, expected_documents=5)

    assert [row.sample_index for row in result] == list(range(5))


def test_assignment_prefix_rejects_a_missing_row() -> None:
    rows = [assignment(0), assignment(2)]

    with pytest.raises(ValueError, match="not complete"):
        validated_assignment_prefix(rows, expected_documents=3)
