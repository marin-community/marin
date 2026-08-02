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


def cluster_purity(primary_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Return the fraction assigned to the largest primary label in each cluster."""
    correct = 0
    for cluster in np.unique(cluster_labels):
        counts = Counter(primary_labels[cluster_labels == cluster])
        correct += max(counts.values())
    return correct / len(primary_labels)


def semantic_metrics(
    vectors: np.ndarray,
    primary_labels: np.ndarray,
    label_sets: list[frozenset[str]],
    neighbor_count: int,
    cluster_count: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    """Return semantic-coherence metrics and nearest-neighbor indices."""
    normalized = normalize_embeddings(vectors)
    if len(primary_labels) != len(normalized) or len(label_sets) != len(normalized):
        raise ValueError("Embedding and label counts differ")
    neighbors = nearest_neighbors(normalized, neighbor_count)
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

    clustering = KMeans(n_clusters=cluster_count, n_init=10, random_state=seed).fit_predict(normalized)
    cluster_counts = np.bincount(clustering, minlength=cluster_count)
    cluster_fractions = cluster_counts / len(clustering)
    pair_left, pair_right = np.triu_indices(len(normalized), k=1)
    pair_cosines = np.sum(normalized[pair_left] * normalized[pair_right], axis=1)
    rounded_unique = np.unique(np.round(normalized, decimals=4), axis=0).shape[0] / len(normalized)
    metrics: dict[str, object] = {
        "documents": len(normalized),
        "dimension": normalized.shape[1],
        "finite_fraction": float(np.isfinite(normalized).all(axis=1).mean()),
        "unique_fraction_4dp": float(rounded_unique),
        "effective_rank": effective_rank(normalized),
        "total_variance": float(np.var(normalized, axis=0).sum()),
        "pair_cosine_mean": float(pair_cosines.mean()),
        "pair_cosine_standard_deviation": float(pair_cosines.std()),
        "neighbor_count": neighbors.shape[1],
        "neighbor_any_label_fraction": float(np.mean(any_matches)),
        "neighbor_label_jaccard": float(np.mean(jaccards)),
        "nearest_primary_accuracy": float(np.mean(primary_matches)),
        "nearest_primary_macro_f1": float(
            f1_score(primary_labels, primary_labels[neighbors[:, 0]], average="macro", zero_division=0)
        ),
        "cluster_count": cluster_count,
        "cluster_nmi": float(normalized_mutual_info_score(primary_labels, clustering)),
        "cluster_purity": float(cluster_purity(primary_labels, clustering)),
        "largest_cluster_fraction": float(cluster_fractions.max()),
        "effective_cluster_count": float(
            math.exp(-sum(fraction * math.log(fraction) for fraction in cluster_fractions if fraction > 0))
        ),
        "cluster_sizes_descending": sorted((int(value) for value in cluster_counts), reverse=True),
    }
    return metrics, neighbors


def cosine_order_fidelity(vectors: np.ndarray, reference: np.ndarray) -> float:
    """Return Spearman fidelity for all pairwise cosine values."""
    normalized = normalize_embeddings(vectors)
    normalized_reference = normalize_embeddings(reference)
    if len(normalized) != len(normalized_reference):
        raise ValueError("Embedding and reference counts differ")
    left, right = np.triu_indices(len(normalized), k=1)
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
        "effective_rank": numeric_student["effective_rank"] >= 0.5 * numeric_teacher["effective_rank"],
        "neighbor_any_label": (
            numeric_student["neighbor_any_label_fraction"] >= numeric_teacher["neighbor_any_label_fraction"] - 0.02
        ),
        "neighbor_label_jaccard": (
            numeric_student["neighbor_label_jaccard"] >= numeric_teacher["neighbor_label_jaccard"] - 0.02
        ),
        "nearest_primary_macro_f1": (
            numeric_student["nearest_primary_macro_f1"] >= numeric_teacher["nearest_primary_macro_f1"] - 0.02
        ),
        "cluster_nmi": numeric_student["cluster_nmi"] >= numeric_teacher["cluster_nmi"] - 0.02,
        "cpu_speed": speed_ratio >= 0.8,
    }
