# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from semantic_embedding_metrics import (  # noqa: E402
    cosine_order_fidelity,
    semantic_metrics,
    stored_vector_rows,
    student_gates,
)


def test_stored_vector_rows_allow_duplicate_hashes() -> None:
    assert stored_vector_rows(
        raw_hashes=["duplicate", "duplicate", "other"],
        eval_ranks=[3, 8, 12],
        requested_rows=[(8, "duplicate"), (12, "other")],
    ) == [1, 2]


def test_semantic_metrics_find_coherent_neighbors_and_clusters() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [-1.0, 0.0],
            [-0.9, 0.1],
            [-0.8, 0.2],
        ]
    )
    primary = np.asarray(["A", "A", "A", "B", "B", "B"])
    label_sets = [frozenset((label,)) for label in primary]

    metrics, neighbors = semantic_metrics(vectors, primary, label_sets, neighbor_count=1, cluster_count=2, seed=42)

    assert metrics["neighbor_any_label_fraction"] == 1.0
    assert metrics["nearest_primary_macro_f1"] == 1.0
    assert metrics["cluster_nmi"] == 1.0
    assert metrics["cluster_purity"] == 1.0
    assert np.all(primary[neighbors[:, 0]] == primary)


def test_cosine_order_fidelity_is_one_for_rotated_vectors() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])

    assert cosine_order_fidelity(vectors @ rotation, vectors) == 1.0


def test_student_gates_compare_semantics_health_and_speed() -> None:
    teacher = {
        "finite_fraction": 1.0,
        "unique_fraction_4dp": 1.0,
        "effective_rank": 40.0,
        "neighbor_any_label_fraction": 0.8,
        "neighbor_label_jaccard": 0.6,
        "nearest_primary_macro_f1": 0.5,
        "cluster_nmi": 0.4,
    }
    student = {
        "finite_fraction": 1.0,
        "unique_fraction_4dp": 0.999,
        "effective_rank": 20.0,
        "neighbor_any_label_fraction": 0.78,
        "neighbor_label_jaccard": 0.58,
        "nearest_primary_macro_f1": 0.48,
        "cluster_nmi": 0.38,
    }

    assert all(student_gates(student, teacher, speed_ratio=0.8).values())
