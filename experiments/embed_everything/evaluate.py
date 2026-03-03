# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation of embedding quality for quality classification and topic clustering.

Quality probe: ridge regression on embeddings → ordinal quality labels.
Topic clustering: K-Means on embeddings, evaluated against ground-truth topic labels.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


@dataclass
class QualityProbeResult:
    """Artifact returned by evaluate_quality_probe."""

    spearman_rho: float
    kendall_tau: float
    n_train: int
    n_test: int


@dataclass
class TopicClusterResult:
    """Artifact returned by evaluate_topic_clusters."""

    ari: float
    nmi: float
    n_clusters: int
    n_docs: int


def evaluate_quality_probe(
    output_path: str,
    embeddings_path: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> QualityProbeResult:
    """Train a linear probe on embeddings to predict ordinal quality labels.

    Loads embeddings from a .npz file, splits into train/test, fits a ridge regression,
    and reports Spearman rank correlation and Kendall's Tau on the test set.

    Args:
        output_path: Directory to write results JSON.
        embeddings_path: Path to .npz file with "embeddings" and "labels" arrays.
        test_size: Fraction of data for test split.
        seed: Random seed for train/test split.

    Returns:
        QualityProbeResult with correlation metrics.
    """
    os.makedirs(output_path, exist_ok=True)

    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]

    # Convert string labels to ordinal values
    le = LabelEncoder()
    y = le.fit_transform(labels).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=test_size, random_state=seed, stratify=labels
    )

    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rho, _ = spearmanr(y_test, preds)
    tau, _ = kendalltau(y_test, preds)

    result = QualityProbeResult(
        spearman_rho=float(rho),
        kendall_tau=float(tau),
        n_train=len(X_train),
        n_test=len(X_test),
    )

    results_file = os.path.join(output_path, "results.json")
    with open(results_file, "w") as f:
        json.dump(asdict(result), f, indent=2)

    logger.info(f"Quality probe: Spearman={rho:.3f}, Kendall={tau:.3f}")
    return result


def evaluate_topic_clusters(
    output_path: str,
    embeddings_path: str,
    n_clusters: int,
    seed: int = 42,
) -> TopicClusterResult:
    """Run K-Means on embeddings and evaluate cluster quality against ground-truth labels.

    Args:
        output_path: Directory to write results JSON.
        embeddings_path: Path to .npz file with "embeddings" and "labels" arrays.
        n_clusters: Number of clusters for K-Means.
        seed: Random seed for K-Means initialization.

    Returns:
        TopicClusterResult with ARI and NMI metrics.
    """
    os.makedirs(output_path, exist_ok=True)

    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    predicted = km.fit_predict(embeddings)

    ari = adjusted_rand_score(labels, predicted)
    nmi = normalized_mutual_info_score(labels, predicted)

    result = TopicClusterResult(
        ari=float(ari),
        nmi=float(nmi),
        n_clusters=n_clusters,
        n_docs=len(labels),
    )

    results_file = os.path.join(output_path, "results.json")
    with open(results_file, "w") as f:
        json.dump(asdict(result), f, indent=2)

    logger.info(f"Topic clustering: ARI={ari:.3f}, NMI={nmi:.3f}")
    return result
