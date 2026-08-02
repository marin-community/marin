# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from evaluate_hierarchical_embeddings import (  # noqa: E402
    adjudicated_assignments,
    group_f1_gates,
    label_levels,
    neighborhood_review_indices,
    strongest_reference_model,
)
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


def test_adjudicated_assignments_replace_only_reviewed_rows() -> None:
    assignments = [
        HierarchicalAssignment(0, "SCIENCE", [], "BIOLOGY", [], "GENERAL_PROSE", 0.4, "GLM"),
        HierarchicalAssignment(1, "ARTS", [], "FICTION", [], "NARRATIVE", 0.9, "GLM"),
    ]
    taxonomy = {
        "parents": [{"bucket_id": "SCIENCE"}, {"bucket_id": "ARTS"}],
        "leaves": [
            {"bucket_id": "BIOLOGY", "parent_id": "SCIENCE"},
            {"bucket_id": "FICTION", "parent_id": "ARTS"},
        ],
    }
    review = {
        "adjudication": {"documents": 1},
        "claude_assignments": [
            {
                "sample_index": 0,
                "primary_parent_id": "ARTS",
                "secondary_parent_ids": [],
                "primary_leaf_id": "FICTION",
                "secondary_leaf_ids": [],
                "form_id": "NARRATIVE",
                "confidence": 0.8,
                "rationale": "Claude",
            }
        ],
    }

    result = adjudicated_assignments(assignments, taxonomy, review)

    assert result[0].primary_parent_id == "ARTS"
    assert result[0].rationale == "Claude"
    assert result[1] == assignments[1]


def test_adjudicated_assignments_reject_unknown_sample() -> None:
    assignments = [HierarchicalAssignment(0, "SCIENCE", [], "BIOLOGY", [], "GENERAL_PROSE", 0.4, "GLM")]
    taxonomy = {
        "parents": [{"bucket_id": "SCIENCE"}],
        "leaves": [{"bucket_id": "BIOLOGY", "parent_id": "SCIENCE"}],
    }
    review = {
        "adjudication": {"documents": 1},
        "claude_assignments": [
            {
                "sample_index": 2,
                "primary_parent_id": "SCIENCE",
                "secondary_parent_ids": [],
                "primary_leaf_id": "BIOLOGY",
                "secondary_leaf_ids": [],
                "form_id": "GENERAL_PROSE",
                "confidence": 0.8,
                "rationale": "Claude",
            }
        ],
    }

    with pytest.raises(ValueError, match="unknown sample"):
        adjudicated_assignments(assignments, taxonomy, review)


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

    model_metrics["fast_arctic_10m"] = model_metrics.pop("fast_arctic_3m")
    assert group_f1_gates(model_metrics, "fast_arctic_10m") == gates


def test_strongest_reference_model_uses_all_levels_and_fixed_metrics() -> None:
    metric_names = (
        "cross_group_neighbor_any_label_fraction",
        "cross_group_neighbor_label_jaccard",
        "cross_group_nearest_primary_macro_f1",
        "cluster_nmi",
    )
    levels = {}
    for level in ("parent", "leaf", "form"):
        levels[level] = {"models": {}}
        for model, value in (
            ("arctic_medium", 0.7),
            ("qwen3_embedding_0.6b", 0.8),
            ("lfm2.5_embedding_350m", 0.6),
        ):
            levels[level]["models"][model] = {metric: value for metric in metric_names}

    model, scores = strongest_reference_model(levels)

    assert model == "qwen3_embedding_0.6b"
    assert scores[model] == pytest.approx(0.8)


def test_neighborhood_review_indices_include_code_then_stable_population() -> None:
    assignments = [
        HierarchicalAssignment(index, "P", [], "L", [], "CODE" if index < 3 else "GENERAL_PROSE", 0.9, "")
        for index in range(20)
    ]

    selected = neighborhood_review_indices(assignments, 10)

    assert len(selected) == 10
    assert len(set(selected)) == 10
    assert {0, 1, 2}.issubset(selected)
