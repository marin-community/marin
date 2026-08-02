# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from evaluate_hierarchical_embeddings import group_f1_gates, label_levels  # noqa: E402
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


def test_group_f1_gates_compare_each_large_group_with_its_best_teacher() -> None:
    def metrics(a: float, b: float):
        return {
            "cross_group_nearest_primary_per_label": {
                "A": {"support": 40, "f1": a},
                "B": {"support": 20, "f1": b},
            }
        }

    model_metrics = {
        "fast_arctic_3m": metrics(0.77, 0.0),
        "arctic_medium": metrics(0.80, 1.0),
        "qwen3_embedding_0.6b": metrics(0.79, 1.0),
        "lfm2.5_embedding_350m": metrics(0.78, 1.0),
    }

    gates = group_f1_gates(model_metrics)

    assert set(gates) == {"A"}
    assert gates["A"]["best_teacher"] == "arctic_medium"
    assert gates["A"]["delta"] == pytest.approx(-0.03)
    assert gates["A"]["passed"]
