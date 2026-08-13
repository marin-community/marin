# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.datakit.cluster.domain.v1.coarsen import CoarseningConfig, coarsen_centroids
from experiments.datakit.cluster.domain.v1.sample import proportional_quotas
from experiments.datakit.cluster.domain.v1.train import stratified_indices, training_quotas


def _separated_centroids() -> tuple[np.ndarray, np.ndarray]:
    angles = np.array([-0.03, 0.03, 1.52, 1.57, 1.62, 3.11, 3.17, 4.65, 4.71, 4.77])
    centroids = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    weights = np.array([8, 17, 7, 8, 10, 11, 14, 7, 8, 10], dtype=np.float64)
    return centroids, weights


def _config() -> CoarseningConfig:
    return CoarseningConfig(
        clusters=4,
        minimum_fraction=0.2,
        seeds=(40, 41, 42),
        split_restarts=10,
        split_iterations=20,
    )


def _weighted_cosine_loss(centroids: np.ndarray, weights: np.ndarray, assignments: np.ndarray) -> float:
    normalized = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    loss = weights.sum()
    for cluster in range(int(assignments.max()) + 1):
        selected = assignments == cluster
        loss -= np.linalg.norm((normalized[selected] * weights[selected, None]).sum(axis=0))
    return float(loss)


def test_proportional_quotas_give_small_sources_one_document_and_hit_target():
    counts = {"tiny": 3, "small": 1_000, "medium": 1_011, "large": 10_000}

    quotas = proportional_quotas(counts, 5_000, small_source_max_rows=1_000, small_source_quota=1)

    assert quotas == {"tiny": 1, "small": 1, "medium": 459, "large": 4_539}


def test_training_quotas_preserve_every_source_and_hit_target():
    counts = {"one": 1, "tiny": 3, "medium": 100, "large": 900}

    quotas = training_quotas(counts, 100)

    assert quotas == {"one": 1, "tiny": 1, "medium": 11, "large": 87}


def test_stratified_indices_realize_source_quotas_deterministically():
    source_codes = np.repeat(np.arange(4), [1, 3, 100, 900])
    source_names = ("one", "tiny", "medium", "large")

    selected, quotas = stratified_indices(source_codes, source_names, 100, seed=42)
    repeated, _ = stratified_indices(source_codes, source_names, 100, seed=42)

    assert np.bincount(source_codes[selected], minlength=4).tolist() == [1, 1, 11, 87]
    assert quotas == {"one": 1, "tiny": 1, "medium": 11, "large": 87}
    assert np.array_equal(selected, repeated)


def test_coarsen_centroids_recovers_separated_groups_and_preserves_weight_floor():
    centroids, weights = _separated_centroids()
    expected_groups = ((0, 1), (2, 3, 4), (5, 6), (7, 8, 9))

    result = coarsen_centroids(centroids, weights, _config())

    cluster_weights = np.bincount(result.fine_to_coarse, weights=weights, minlength=4)
    assert np.all(cluster_weights >= 20)
    assert all(len({result.fine_to_coarse[index] for index in group}) == 1 for group in expected_groups)
    assert len({result.fine_to_coarse[group[0]] for group in expected_groups}) == 4
    assert result.final.loss <= result.initial.loss
    assert result.selected_seed == min(result.divisive_runs, key=lambda run: run.stats.loss).seed


def test_coarsen_centroids_stops_at_legal_single_move_optimum():
    rng = np.random.default_rng(7)
    centroids = rng.normal(size=(30, 5)).astype(np.float32)
    weights = rng.integers(1, 10, size=len(centroids)).astype(np.float64)
    config = CoarseningConfig(4, 0.1, (3,), 4, 8)

    result = coarsen_centroids(centroids, weights, config)

    assert result.moves > 0
    assert result.final.loss < result.initial.loss
    minimum_weight = weights.sum() * config.minimum_fraction
    final_loss = _weighted_cosine_loss(centroids, weights, result.fine_to_coarse)
    cluster_weights = np.bincount(result.fine_to_coarse, weights=weights, minlength=config.clusters)
    for point, source in enumerate(result.fine_to_coarse):
        if cluster_weights[source] - weights[point] < minimum_weight:
            continue
        for target in range(config.clusters):
            if target == source:
                continue
            candidate = result.fine_to_coarse.copy()
            candidate[point] = target
            assert _weighted_cosine_loss(centroids, weights, candidate) >= final_loss - 1e-10


def test_coarsen_centroids_rejects_impossible_common_minimum():
    with pytest.raises(ValueError, match="infeasible"):
        coarsen_centroids(
            np.eye(3, dtype=np.float32),
            np.ones(3),
            CoarseningConfig(3, 0.4, (42,), 1, 1),
        )
