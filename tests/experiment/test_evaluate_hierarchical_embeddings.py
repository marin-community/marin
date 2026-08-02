# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from evaluate_hierarchical_embeddings import label_levels  # noqa: E402
from glm_hierarchical_labels import HierarchicalAssignment  # noqa: E402


def test_label_levels_keep_domains_and_forms_separate() -> None:
    assignments = [
        HierarchicalAssignment(0, "SCIENCE", ["CODE"], "BIOLOGY", ["PYTHON"], "RESEARCH", 0.9, "a"),
        HierarchicalAssignment(1, "CODE", [], "PYTHON", [], "CODE", 0.8, "b"),
    ]

    levels = label_levels(assignments)

    assert levels["parent"][0].tolist() == ["SCIENCE", "CODE"]
    assert levels["parent"][1] == [frozenset({"SCIENCE", "CODE"}), frozenset({"CODE"})]
    assert levels["leaf"][0].tolist() == ["BIOLOGY", "PYTHON"]
    assert levels["leaf"][1] == [frozenset({"BIOLOGY", "PYTHON"}), frozenset({"PYTHON"})]
    assert levels["form"][0].tolist() == ["RESEARCH", "CODE"]
    assert levels["form"][1] == [frozenset({"RESEARCH"}), frozenset({"CODE"})]
