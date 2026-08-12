# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coarsen weighted domain centroids into broad groups."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoarseningConfig:
    clusters: int
    minimum_fraction: float
    seeds: tuple[int, ...]
    split_restarts: int
    split_iterations: int


@dataclass(frozen=True)
class ClusteringStats:
    loss: float
    weighted_mean_cosine_distance: float
    minimum_weight_fraction: float
    maximum_weight_fraction: float
    weight_fractions: np.ndarray
    fine_centroid_counts: np.ndarray


@dataclass(frozen=True)
class SeedResult:
    seed: int
    stats: ClusteringStats


@dataclass(frozen=True)
class CoarseningResult:
    fine_to_coarse: np.ndarray
    centers: np.ndarray
    selected_seed: int
    divisive_runs: tuple[SeedResult, ...]
    initial: ClusteringStats
    final: ClusteringStats
    moves: int
    sweeps: int


@dataclass(frozen=True)
class _Clustering:
    assignments: np.ndarray
    centers: np.ndarray
    loss: float


@dataclass(frozen=True)
class _Cluster:
    indices: np.ndarray
    center: np.ndarray
    loss: float
    weight: float


def _normalized_centroids(centroids: np.ndarray, weights: np.ndarray, config: CoarseningConfig) -> np.ndarray:
    vectors = np.asarray(centroids, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(weights):
        raise ValueError("centroids must be two-dimensional and aligned with weights")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("centroids must be finite")
    if config.clusters < 2 or len(vectors) < config.clusters or not 0 < config.minimum_fraction <= 1 / config.clusters:
        raise ValueError("cluster count and minimum fraction are infeasible")
    if not config.seeds or config.split_restarts <= 0 or config.split_iterations <= 0:
        raise ValueError("seeds, split restarts, and split iterations must be positive")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("centroids must have nonzero norm")
    return np.asarray(vectors / norms, dtype=np.float32)


def _validated_weights(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be a one-dimensional array of positive finite values")
    return values


def _cluster(indices: np.ndarray, vectors: np.ndarray, weights: np.ndarray) -> _Cluster:
    weighted_sum = (vectors[indices] * weights[indices, None]).sum(axis=0)
    norm = np.linalg.norm(weighted_sum)
    if norm == 0:
        raise ValueError("cluster has a zero-norm weighted centroid")
    center = weighted_sum / norm
    weight = float(weights[indices].sum())
    return _Cluster(indices, center, weight - float(norm), weight)


def _repair_split(
    labels: np.ndarray,
    similarities: np.ndarray,
    weights: np.ndarray,
    minimum_weight: float,
) -> np.ndarray | None:
    repaired = labels.copy()
    child_weights = np.bincount(repaired, weights=weights, minlength=2)
    for recipient in np.argsort(child_weights):
        if child_weights[recipient] >= minimum_weight:
            continue
        donor = 1 - int(recipient)
        donor_indices = np.flatnonzero(repaired == donor)
        move_costs = similarities[donor_indices, donor] - similarities[donor_indices, recipient]
        for index in donor_indices[np.argsort(move_costs)]:
            weight = weights[index]
            if child_weights[donor] - weight < minimum_weight:
                continue
            repaired[index] = recipient
            child_weights[donor] -= weight
            child_weights[recipient] += weight
            if child_weights[recipient] >= minimum_weight:
                break
        if child_weights[recipient] < minimum_weight:
            return None
    return repaired


def _split_once(
    parent: _Cluster,
    vectors: np.ndarray,
    weights: np.ndarray,
    minimum_weight: float,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[_Cluster, _Cluster] | None:
    indices = parent.indices
    local_vectors = vectors[indices]
    local_weights = weights[indices]
    first = int(rng.choice(len(indices), p=local_weights / local_weights.sum()))
    first_distances = np.maximum(1.0 - local_vectors @ local_vectors[first], 0.0)
    second_weights = local_weights * first_distances
    if second_weights.sum() == 0:
        return None
    second = int(rng.choice(len(indices), p=second_weights / second_weights.sum()))
    centers = local_vectors[[first, second]].copy()
    labels = np.full(len(indices), -1, dtype=np.int8)

    for _ in range(iterations):
        similarities = local_vectors @ centers.T
        proposed = np.argmax(similarities, axis=1).astype(np.int8)
        proposed = _repair_split(proposed, similarities, local_weights, minimum_weight)
        if proposed is None:
            return None
        if np.array_equal(labels, proposed):
            break
        labels = proposed
        for child in range(2):
            selected = labels == child
            weighted_sum = (local_vectors[selected] * local_weights[selected, None]).sum(axis=0)
            norm = np.linalg.norm(weighted_sum)
            if norm == 0:
                return None
            centers[child] = weighted_sum / norm

    if np.any(labels < 0):
        return None
    return (
        _cluster(indices[labels == 0], vectors, weights),
        _cluster(indices[labels == 1], vectors, weights),
    )


def _best_split(
    parent: _Cluster,
    vectors: np.ndarray,
    weights: np.ndarray,
    minimum_weight: float,
    config: CoarseningConfig,
    rng: np.random.Generator,
) -> tuple[_Cluster, _Cluster] | None:
    best = None
    best_loss = np.inf
    for _ in range(config.split_restarts):
        candidate = _split_once(parent, vectors, weights, minimum_weight, config.split_iterations, rng)
        if candidate is None:
            continue
        loss = candidate[0].loss + candidate[1].loss
        if loss < best_loss:
            best = candidate
            best_loss = loss
    return best


def _divisive_clustering(
    vectors: np.ndarray,
    weights: np.ndarray,
    config: CoarseningConfig,
    seed: int,
) -> _Clustering:
    minimum_weight = float(weights.sum() * config.minimum_fraction)
    rng = np.random.default_rng(seed)
    partition = [_cluster(np.arange(len(vectors)), vectors, weights)]
    while len(partition) < config.clusters:
        candidates = sorted(
            (cluster for cluster in partition if cluster.weight >= 2 * minimum_weight),
            key=lambda cluster: cluster.loss,
            reverse=True,
        )
        split = None
        parent = None
        for candidate in candidates:
            split = _best_split(candidate, vectors, weights, minimum_weight, config, rng)
            if split is not None:
                parent = candidate
                break
        if split is None or parent is None:
            raise ValueError(f"could not produce {config.clusters} clusters while preserving the minimum fraction")
        parent_index = next(index for index, cluster in enumerate(partition) if cluster is parent)
        partition[parent_index : parent_index + 1] = split

    partition.sort(key=lambda cluster: int(cluster.indices.min()))
    assignments = np.empty(len(vectors), dtype=np.int32)
    for label, cluster in enumerate(partition):
        assignments[cluster.indices] = label
    return _Clustering(
        assignments=assignments,
        centers=np.stack([cluster.center for cluster in partition]),
        loss=sum(cluster.loss for cluster in partition),
    )


def _hill_climb(
    vectors: np.ndarray,
    weights: np.ndarray,
    initial: _Clustering,
    minimum_fraction: float,
) -> tuple[_Clustering, int, int]:
    current = initial.assignments.copy()
    clusters = len(initial.centers)
    minimum_weight = float(weights.sum() * minimum_fraction)
    cluster_weights = np.bincount(current, weights=weights, minlength=clusters)
    sums = np.zeros((clusters, vectors.shape[1]), dtype=np.float64)
    np.add.at(sums, current, vectors * weights[:, None])
    norms = np.linalg.norm(sums, axis=1)
    tolerance = np.finfo(np.float64).eps * weights.sum()
    moves = 0
    sweeps = 0

    while True:
        sweep_moves = 0
        for point, vector in enumerate(vectors):
            source = int(current[point])
            weight = weights[point]
            if cluster_weights[source] - weight < minimum_weight:
                continue
            similarities = sums @ vector
            removal_norm = math.sqrt(max(norms[source] ** 2 + weight**2 - 2 * weight * similarities[source], 0))
            if removal_norm == 0:
                continue
            addition_norms = np.sqrt(np.maximum(norms**2 + weight**2 + 2 * weight * similarities, 0))
            gains = removal_norm + addition_norms - norms[source] - norms
            gains[source] = -np.inf
            target = int(np.argmax(gains))
            if gains[target] <= tolerance:
                continue

            weighted_vector = weight * vector
            sums[source] -= weighted_vector
            sums[target] += weighted_vector
            cluster_weights[source] -= weight
            cluster_weights[target] += weight
            norms[source] = removal_norm
            norms[target] = addition_norms[target]
            current[point] = target
            moves += 1
            sweep_moves += 1
        sweeps += 1
        if sweep_moves == 0:
            break

    if moves == 0:
        return initial, moves, sweeps
    return _Clustering(current, sums / norms[:, None], float(weights.sum() - norms.sum())), moves, sweeps


def _stats(clustering: _Clustering, weights: np.ndarray) -> ClusteringStats:
    cluster_count = len(clustering.centers)
    cluster_weights = np.bincount(clustering.assignments, weights=weights, minlength=cluster_count)
    fractions = cluster_weights / weights.sum()
    return ClusteringStats(
        loss=clustering.loss,
        weighted_mean_cosine_distance=clustering.loss / weights.sum(),
        minimum_weight_fraction=float(fractions.min()),
        maximum_weight_fraction=float(fractions.max()),
        weight_fractions=fractions,
        fine_centroid_counts=np.bincount(clustering.assignments, minlength=cluster_count),
    )


def coarsen_centroids(
    centroids: np.ndarray,
    weights: np.ndarray,
    config: CoarseningConfig,
) -> CoarseningResult:
    """Run multi-seed divisive clustering followed by exact single-centroid hill climbing."""
    validated_weights = _validated_weights(weights)
    vectors = _normalized_centroids(centroids, validated_weights, config)
    divisive_runs = []
    selected = None
    selected_seed = None
    for seed in config.seeds:
        clustering = _divisive_clustering(vectors, validated_weights, config, seed)
        divisive_runs.append(SeedResult(seed, _stats(clustering, validated_weights)))
        if selected is None or clustering.loss < selected.loss:
            selected = clustering
            selected_seed = seed
    if selected is None or selected_seed is None:
        raise RuntimeError("no divisive clustering run completed")

    refined, moves, sweeps = _hill_climb(vectors, validated_weights, selected, config.minimum_fraction)
    return CoarseningResult(
        fine_to_coarse=refined.assignments,
        centers=refined.centers,
        selected_seed=selected_seed,
        divisive_runs=tuple(divisive_runs),
        initial=_stats(selected, validated_weights),
        final=_stats(refined, validated_weights),
        moves=moves,
        sweeps=sweeps,
    )
