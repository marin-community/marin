# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from semantic_embedding_metrics import (  # noqa: E402
    cosine_order_fidelity,
    fixed_bucket_metrics,
    nearest_neighbors_outside_groups,
    normalize_embeddings,
    sampled_pairs,
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

    groups = np.asarray(["one", "one", "two", "three", "three", "four"])
    metrics, neighbors = semantic_metrics(
        vectors,
        primary,
        label_sets,
        neighbor_count=1,
        cluster_count=2,
        seed=42,
        exclusion_groups=groups,
    )

    assert metrics["neighbor_any_label_fraction"] == 1.0
    assert metrics["nearest_primary_macro_f1"] == 1.0
    assert metrics["cluster_nmi"] == 1.0
    assert metrics["cluster_purity"] == 1.0
    assert metrics["cross_group_neighbor_any_label_fraction"] == 1.0
    assert metrics["neighbor_same_group_fraction"] == 2 / 3
    assert metrics["nearest_primary_per_label"] == {
        "A": {"support": 3, "f1": 1.0},
        "B": {"support": 3, "f1": 1.0},
    }
    assert np.all(primary[neighbors[:, 0]] == primary)

    cross_group_neighbors = nearest_neighbors_outside_groups(normalize_embeddings(vectors), groups, 1)
    cached_metrics, cached_neighbors = semantic_metrics(
        vectors,
        primary,
        label_sets,
        neighbor_count=1,
        cluster_count=2,
        seed=42,
        exclusion_groups=groups,
        precomputed_neighbors=neighbors,
        precomputed_cross_group_neighbors=cross_group_neighbors,
    )
    assert cached_metrics == metrics
    np.testing.assert_array_equal(cached_neighbors, neighbors)


def test_fixed_bucket_metrics_use_one_clustering_for_all_semantic_levels() -> None:
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
    labels = np.asarray(["A", "A", "A", "B", "B", "B"])

    metrics = fixed_bucket_metrics(
        vectors,
        {"parent": labels, "form": labels},
        cluster_count=2,
        seed=42,
    )

    assert metrics["cluster_count"] == 2
    assert metrics["cluster_sizes_descending"] == [3, 3]
    assert metrics["effective_cluster_count"] == 2.0
    assert metrics["levels"] == {
        "parent": {"cluster_nmi": 1.0, "cluster_purity": 1.0},
        "form": {"cluster_nmi": 1.0, "cluster_purity": 1.0},
    }


def test_cosine_order_fidelity_is_one_for_rotated_vectors() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])

    assert cosine_order_fidelity(vectors @ rotation, vectors) == 1.0


def test_sampled_pairs_are_stable_unique_and_bounded() -> None:
    left, right = sampled_pairs(row_count=1_000, maximum=2_000, seed=42)
    repeated_left, repeated_right = sampled_pairs(row_count=1_000, maximum=2_000, seed=42)

    assert len(left) == 2_000
    assert np.all(left < right)
    assert len(set(zip(left, right, strict=True))) == 2_000
    np.testing.assert_array_equal(left, repeated_left)
    np.testing.assert_array_equal(right, repeated_right)


def test_student_gates_compare_semantics_health_and_speed() -> None:
    teacher = {
        "finite_fraction": 1.0,
        "unique_fraction_4dp": 1.0,
        "effective_rank": 40.0,
        "effective_rank_fraction": 0.4,
        "neighbor_any_label_fraction": 0.8,
        "neighbor_label_jaccard": 0.6,
        "nearest_primary_macro_f1": 0.5,
        "cluster_nmi": 0.4,
        "cross_group_neighbor_any_label_fraction": 0.8,
        "cross_group_neighbor_label_jaccard": 0.6,
        "cross_group_nearest_primary_macro_f1": 0.5,
    }
    student = {
        "finite_fraction": 1.0,
        "unique_fraction_4dp": 0.999,
        "effective_rank": 20.0,
        "effective_rank_fraction": 0.2,
        "neighbor_any_label_fraction": 0.78,
        "neighbor_label_jaccard": 0.58,
        "nearest_primary_macro_f1": 0.48,
        "cluster_nmi": 0.38,
        "cross_group_neighbor_any_label_fraction": 0.78,
        "cross_group_neighbor_label_jaccard": 0.58,
        "cross_group_nearest_primary_macro_f1": 0.48,
    }

    assert all(student_gates(student, teacher, speed_ratio=0.8).values())
