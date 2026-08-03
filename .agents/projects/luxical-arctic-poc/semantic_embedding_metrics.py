# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure source-free semantic coherence and embedding health."""

import math
from collections import Counter

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score


def stored_vector_rows(
    raw_hashes: list[str],
    eval_ranks: list[int],
    requested_rows: list[tuple[int, str]],
) -> list[int]:
    """Return stored row indices aligned by evaluation rank and checked by hash."""
    if len(raw_hashes) != len(eval_ranks):
        raise ValueError("Stored hash and evaluation-rank counts differ")
    row_by_rank = {rank: index for index, rank in enumerate(eval_ranks)}
    if len(row_by_rank) != len(eval_ranks):
        raise ValueError("A stored embedding table has duplicate evaluation ranks")
    selected = []
    for eval_rank, expected_hash in requested_rows:
        try:
            row = row_by_rank[eval_rank]
        except KeyError as error:
            raise ValueError(f"Stored embeddings do not contain evaluation rank {eval_rank}") from error
        if raw_hashes[row] != expected_hash:
            raise ValueError(f"Stored hash differs at evaluation rank {eval_rank}")
        selected.append(row)
    return selected


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    """Return finite row-normalized embeddings."""
    if vectors.ndim != 2 or len(vectors) < 2:
        raise ValueError(f"Expected an embedding matrix with at least two rows, got {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Embeddings contain a zero vector")
    return vectors / norms


def effective_rank(vectors: np.ndarray) -> float:
    """Return the entropy-based effective rank of centered embeddings."""
    singular_values = np.linalg.svd(vectors - vectors.mean(axis=0, keepdims=True), compute_uv=False)
    eigenvalues = np.square(singular_values)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 0] / total
    return float(math.exp(float(-np.sum(probabilities * np.log(probabilities)))))


def nearest_neighbors(vectors: np.ndarray, count: int) -> np.ndarray:
    """Return ordered cosine-neighbor indices without each query row."""
    count = min(count, len(vectors) - 1)
    if count < 1:
        raise ValueError("The neighbor count must be positive")
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -np.inf)
    candidates = np.argpartition(similarities, -count, axis=1)[:, -count:]
    candidate_scores = np.take_along_axis(similarities, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def nearest_neighbors_outside_groups(vectors: np.ndarray, groups: np.ndarray, count: int) -> np.ndarray:
    """Return cosine-neighbor indices after excluding each query group."""
    if len(groups) != len(vectors):
        raise ValueError("Embedding and group counts differ")
    count = min(count, len(vectors) - max(Counter(groups).values()))
    if count < 1:
        raise ValueError("No cross-group neighbor is available")
    similarities = vectors @ vectors.T
    similarities[groups[:, None] == groups[None, :]] = -np.inf
    candidates = np.argpartition(similarities, -count, axis=1)[:, -count:]
    candidate_scores = np.take_along_axis(similarities, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def cluster_purity(primary_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Return the fraction assigned to the largest primary label in each cluster."""
    correct = 0
    for cluster in np.unique(cluster_labels):
        counts = Counter(primary_labels[cluster_labels == cluster])
        correct += max(counts.values())
    return correct / len(primary_labels)


def fixed_bucket_metrics(
    vectors: np.ndarray,
    primary_labels_by_level: dict[str, np.ndarray],
    cluster_count: int,
    seed: int,
) -> dict[str, object]:
    """Return semantic quality for one fixed production-style clustering."""
    normalized = normalize_embeddings(vectors)
    if cluster_count < 2 or cluster_count > len(normalized):
        raise ValueError("The fixed bucket count must be between two and the document count")
    for labels in primary_labels_by_level.values():
        if len(labels) != len(normalized):
            raise ValueError("Embedding and fixed-bucket label counts differ")
    clustering = KMeans(n_clusters=cluster_count, n_init=10, random_state=seed).fit_predict(normalized)
    cluster_counts = np.bincount(clustering, minlength=cluster_count)
    cluster_fractions = cluster_counts / len(clustering)
    levels = {
        level: {
            "cluster_nmi": float(normalized_mutual_info_score(labels, clustering)),
            "cluster_purity": float(cluster_purity(labels, clustering)),
        }
        for level, labels in primary_labels_by_level.items()
    }
    return {
        "cluster_count": cluster_count,
        "largest_cluster_fraction": float(cluster_fractions.max()),
        "effective_cluster_count": float(
            math.exp(-sum(fraction * math.log(fraction) for fraction in cluster_fractions if fraction > 0))
        ),
        "cluster_sizes_descending": sorted((int(value) for value in cluster_counts), reverse=True),
        "levels": levels,
    }


def label_neighborhood_metrics(
    neighbors: np.ndarray,
    primary_labels: np.ndarray,
    label_sets: list[frozenset[str]],
) -> dict[str, object]:
    """Return semantic label agreement for one neighbor matrix."""
    any_matches = []
    jaccards = []
    primary_matches = []
    for index, row in enumerate(neighbors):
        query_labels = label_sets[index]
        for neighbor in row:
            neighbor_labels = label_sets[int(neighbor)]
            intersection = query_labels & neighbor_labels
            any_matches.append(bool(intersection))
            jaccards.append(len(intersection) / len(query_labels | neighbor_labels))
        primary_matches.append(primary_labels[index] == primary_labels[row[0]])
    return {
        "neighbor_any_label_fraction": float(np.mean(any_matches)),
        "neighbor_label_jaccard": float(np.mean(jaccards)),
        "nearest_primary_accuracy": float(np.mean(primary_matches)),
        "nearest_primary_macro_f1": float(
            f1_score(primary_labels, primary_labels[neighbors[:, 0]], average="macro", zero_division=0)
        ),
        "nearest_primary_per_label": per_label_f1(primary_labels, primary_labels[neighbors[:, 0]]),
    }


def per_label_f1(expected: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float | int]]:
    """Return support and one-vs-rest F1 for each primary label."""
    labels = np.unique(expected)
    scores = f1_score(expected, predicted, labels=labels, average=None, zero_division=0)
    return {
        str(label): {"support": int(np.sum(expected == label)), "f1": float(score)}
        for label, score in zip(labels, scores, strict=True)
    }


def sampled_pairs(row_count: int, maximum: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return all pairs or a stable uniform sample of unordered row pairs."""
    pair_count = row_count * (row_count - 1) // 2
    if maximum is None or pair_count <= maximum:
        return np.triu_indices(row_count, k=1)
    if maximum < 1:
        raise ValueError("The maximum pair count must be positive")
    random = np.random.default_rng(seed)
    pairs = np.empty((0, 2), dtype=np.int64)
    while len(pairs) < maximum:
        remaining = maximum - len(pairs)
        left = random.integers(0, row_count, size=remaining * 2)
        right = random.integers(0, row_count, size=remaining * 2)
        unequal = left != right
        batch = np.column_stack((np.minimum(left[unequal], right[unequal]), np.maximum(left[unequal], right[unequal])))
        pairs = np.unique(np.concatenate((pairs, batch)), axis=0)
    if len(pairs) > maximum:
        pairs = pairs[random.choice(len(pairs), size=maximum, replace=False)]
    return pairs[:, 0], pairs[:, 1]


def semantic_metrics(
    vectors: np.ndarray,
    primary_labels: np.ndarray,
    label_sets: list[frozenset[str]],
    neighbor_count: int,
    cluster_count: int,
    seed: int,
    exclusion_groups: np.ndarray | None = None,
    maximum_pair_count: int | None = None,
    precomputed_neighbors: np.ndarray | None = None,
    precomputed_cross_group_neighbors: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    """Return semantic-coherence metrics and nearest-neighbor indices."""
    normalized = normalize_embeddings(vectors)
    if len(primary_labels) != len(normalized) or len(label_sets) != len(normalized):
        raise ValueError("Embedding and label counts differ")
    neighbors = precomputed_neighbors
    if neighbors is None:
        neighbors = nearest_neighbors(normalized, neighbor_count)
    if len(neighbors) != len(normalized):
        raise ValueError("The precomputed neighbor count differs from the embeddings")
    neighborhood = label_neighborhood_metrics(neighbors, primary_labels, label_sets)

    clustering = KMeans(n_clusters=cluster_count, n_init=10, random_state=seed).fit_predict(normalized)
    cluster_counts = np.bincount(clustering, minlength=cluster_count)
    cluster_fractions = cluster_counts / len(clustering)
    pair_left, pair_right = sampled_pairs(len(normalized), maximum_pair_count, seed)
    pair_cosines = np.sum(normalized[pair_left] * normalized[pair_right], axis=1)
    rounded_unique = np.unique(np.round(normalized, decimals=4), axis=0).shape[0] / len(normalized)
    rank = effective_rank(normalized)
    metrics: dict[str, object] = {
        "documents": len(normalized),
        "dimension": normalized.shape[1],
        "finite_fraction": float(np.isfinite(normalized).all(axis=1).mean()),
        "unique_fraction_4dp": float(rounded_unique),
        "effective_rank": rank,
        "effective_rank_fraction": rank / min(len(normalized) - 1, normalized.shape[1]),
        "total_variance": float(np.var(normalized, axis=0).sum()),
        "pair_cosine_mean": float(pair_cosines.mean()),
        "pair_cosine_standard_deviation": float(pair_cosines.std()),
        "pair_count": len(pair_left),
        "neighbor_count": neighbors.shape[1],
        **neighborhood,
        "cluster_count": cluster_count,
        "cluster_nmi": float(normalized_mutual_info_score(primary_labels, clustering)),
        "cluster_purity": float(cluster_purity(primary_labels, clustering)),
        "largest_cluster_fraction": float(cluster_fractions.max()),
        "effective_cluster_count": float(
            math.exp(-sum(fraction * math.log(fraction) for fraction in cluster_fractions if fraction > 0))
        ),
        "cluster_sizes_descending": sorted((int(value) for value in cluster_counts), reverse=True),
    }
    if exclusion_groups is not None:
        cross_group_neighbors = precomputed_cross_group_neighbors
        if cross_group_neighbors is None:
            cross_group_neighbors = nearest_neighbors_outside_groups(normalized, exclusion_groups, neighbor_count)
        if len(cross_group_neighbors) != len(normalized):
            raise ValueError("The precomputed cross-group neighbor count differs from the embeddings")
        cross_group = label_neighborhood_metrics(cross_group_neighbors, primary_labels, label_sets)
        metrics["neighbor_same_group_fraction"] = float(
            np.mean(exclusion_groups[neighbors] == exclusion_groups[:, None])
        )
        metrics.update({f"cross_group_{name}": value for name, value in cross_group.items()})
    return metrics, neighbors


def cosine_order_fidelity(
    vectors: np.ndarray,
    reference: np.ndarray,
    maximum_pair_count: int | None = None,
    seed: int = 0,
) -> float:
    """Return Spearman fidelity for all pairwise cosine values."""
    normalized = normalize_embeddings(vectors)
    normalized_reference = normalize_embeddings(reference)
    if len(normalized) != len(normalized_reference):
        raise ValueError("Embedding and reference counts differ")
    left, right = sampled_pairs(len(normalized), maximum_pair_count, seed)
    values = np.sum(normalized[left] * normalized[right], axis=1)
    reference_values = np.sum(normalized_reference[left] * normalized_reference[right], axis=1)
    correlation = spearmanr(values, reference_values).statistic
    if not np.isfinite(correlation):
        raise ValueError("Pairwise cosine fidelity is not finite")
    return float(correlation)


def student_gates(student: dict[str, object], teacher: dict[str, object], speed_ratio: float) -> dict[str, bool]:
    """Compare one fast student with a selected semantic teacher."""
    numeric_student = {name: float(value) for name, value in student.items() if isinstance(value, int | float)}
    numeric_teacher = {name: float(value) for name, value in teacher.items() if isinstance(value, int | float)}
    return {
        "finite": numeric_student["finite_fraction"] == 1.0,
        "unique": numeric_student["unique_fraction_4dp"] >= 0.99,
        "effective_rank": numeric_student["effective_rank_fraction"] >= 0.5 * numeric_teacher["effective_rank_fraction"],
        "cross_group_neighbor_any_label": (
            numeric_student["cross_group_neighbor_any_label_fraction"]
            >= numeric_teacher["cross_group_neighbor_any_label_fraction"] - 0.02
        ),
        "cross_group_neighbor_label_jaccard": (
            numeric_student["cross_group_neighbor_label_jaccard"]
            >= numeric_teacher["cross_group_neighbor_label_jaccard"] - 0.02
        ),
        "cross_group_nearest_primary_macro_f1": (
            numeric_student["cross_group_nearest_primary_macro_f1"]
            >= numeric_teacher["cross_group_nearest_primary_macro_f1"] - 0.02
        ),
        "cluster_nmi": numeric_student["cluster_nmi"] >= numeric_teacher["cluster_nmi"] - 0.02,
        "cpu_speed": speed_ratio >= 0.8,
    }
