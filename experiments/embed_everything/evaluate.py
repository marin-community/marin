# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate embedding quality for downstream tasks.

Quality classification: linear probe on embeddings → Spearman/Kendall correlation
with oracle quality scores.

Topic clustering: K-Means on embeddings → ARI/NMI against oracle topic labels.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass

import numpy as np
from iris.marin_fs import open_url

from marin.execution import THIS_OUTPUT_PATH

logger = logging.getLogger(__name__)


def _load_npz(path: str) -> dict[str, np.ndarray]:
    """Load a .npz file from local or GCS path."""
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(path))
    with open_url(path, "rb") as src:
        content = src.read()
    with open(local_path, "wb") as dst:
        dst.write(content)
    data = dict(np.load(local_path, allow_pickle=True))
    os.remove(local_path)
    return data


def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return list of dicts."""
    docs = []
    with open_url(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def _write_json(data: dict, path: str) -> None:
    """Write a dict as JSON to a file."""
    with open_url(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# Ordinal mapping for Nemotron quality buckets
QUALITY_ORDINAL = {
    "low": 0,
    "medium_low": 1,
    "medium": 2,
    "medium_high": 3,
    "high": 4,
}


@dataclass(frozen=True)
class EvalQualityConfig:
    """Config for quality probe evaluation."""

    embeddings_path: str
    oracle_path: str
    output_path: str = THIS_OUTPUT_PATH


def evaluate_quality_probe(config: EvalQualityConfig) -> None:
    """Train a linear probe on embeddings and evaluate against oracle quality scores.

    Reports:
    - Spearman rank correlation between predicted and oracle scores
    - Kendall's Tau between predicted and oracle scores
    - Ridge regression R^2
    - Per-bucket accuracy
    """
    from scipy import stats
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    # Load embeddings
    emb_file = os.path.join(config.embeddings_path, "quality_embeddings.npz")
    emb_data = _load_npz(emb_file)
    embeddings = emb_data["embeddings"]
    doc_ids = emb_data["doc_ids"].tolist()
    splits = emb_data["splits"].tolist()

    # Load oracle labels
    oracle_file = os.path.join(config.oracle_path, "quality_labeled.jsonl")
    oracle_docs = _read_jsonl(oracle_file)
    oracle_by_id = {d["doc_id"]: d for d in oracle_docs}

    # Align embeddings with oracle scores
    train_emb, train_scores = [], []
    test_emb, test_scores = [], []
    train_buckets, test_buckets = [], []

    for i, doc_id in enumerate(doc_ids):
        oracle = oracle_by_id.get(doc_id)
        if oracle is None or oracle.get("oracle_quality_score", -1) == -1:
            continue

        score = oracle["oracle_quality_score"]
        bucket = oracle.get("quality_bucket", "unknown")

        if splits[i] == "train":
            train_emb.append(embeddings[i])
            train_scores.append(score)
            train_buckets.append(bucket)
        else:
            test_emb.append(embeddings[i])
            test_scores.append(score)
            test_buckets.append(bucket)

    if not train_emb or not test_emb:
        logger.error("Insufficient data: %d train, %d test", len(train_emb), len(test_emb))
        return

    X_train = np.array(train_emb)
    y_train = np.array(train_scores, dtype=float)
    X_test = np.array(test_emb)
    y_test = np.array(test_scores, dtype=float)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train ridge regression
    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    spearman_r, spearman_p = stats.spearmanr(y_test, y_pred)
    kendall_tau, kendall_p = stats.kendalltau(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Also compute correlation using bucket ordinals as a sanity check
    test_ordinals = np.array([QUALITY_ORDINAL.get(b, 2) for b in test_buckets], dtype=float)
    bucket_spearman, _ = stats.spearmanr(test_ordinals, y_pred)

    results = {
        "task": "quality_probe",
        "n_train": len(train_emb),
        "n_test": len(test_emb),
        "ridge_alpha": float(model.alpha_),
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_tau),
        "kendall_p": float(kendall_p),
        "bucket_ordinal_spearman": float(bucket_spearman),
        "r2": float(r2),
        "mse": float(mse),
        "embedding_dim": int(X_train.shape[1]),
    }

    logger.info("Quality probe results: Spearman=%.4f, Kendall=%.4f, R2=%.4f", spearman_r, kendall_tau, r2)

    os.makedirs(config.output_path, exist_ok=True)
    _write_json(results, os.path.join(config.output_path, "quality_results.json"))


@dataclass(frozen=True)
class EvalTopicConfig:
    """Config for topic clustering evaluation."""

    embeddings_path: str
    oracle_path: str
    output_path: str = THIS_OUTPUT_PATH
    n_clusters: int = 15


def evaluate_topic_clusters(config: EvalTopicConfig) -> None:
    """Run K-Means on embeddings and evaluate against oracle topic labels.

    Reports:
    - Adjusted Rand Index (ARI)
    - Normalized Mutual Information (NMI)
    - Homogeneity, completeness, V-measure
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
        v_measure_score,
    )
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Load embeddings
    emb_file = os.path.join(config.embeddings_path, "topic_embeddings.npz")
    emb_data = _load_npz(emb_file)
    embeddings = emb_data["embeddings"]
    doc_ids = emb_data["doc_ids"].tolist()

    # Load oracle labels
    oracle_file = os.path.join(config.oracle_path, "topic_labeled.jsonl")
    oracle_docs = _read_jsonl(oracle_file)
    oracle_by_id = {d["doc_id"]: d for d in oracle_docs}

    # Align and filter
    valid_emb = []
    valid_labels = []
    for i, doc_id in enumerate(doc_ids):
        oracle = oracle_by_id.get(doc_id)
        if oracle is None or oracle.get("oracle_topic") == "labeling_failed":
            continue
        valid_emb.append(embeddings[i])
        valid_labels.append(oracle["oracle_topic"])

    if len(valid_emb) < config.n_clusters:
        logger.error("Too few valid documents (%d) for %d clusters", len(valid_emb), config.n_clusters)
        return

    X = np.array(valid_emb)
    true_labels = valid_labels

    # Encode labels
    le = LabelEncoder()
    y_true = le.fit_transform(true_labels)
    n_true_clusters = len(le.classes_)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means with k = number of oracle topics
    kmeans = KMeans(n_clusters=config.n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_scaled)

    # Metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    homogeneity = homogeneity_score(y_true, y_pred)
    completeness = completeness_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)

    results = {
        "task": "topic_clustering",
        "n_documents": len(valid_emb),
        "n_true_clusters": int(n_true_clusters),
        "n_kmeans_clusters": config.n_clusters,
        "true_label_distribution": {
            label: int(count) for label, count in zip(*np.unique(true_labels, return_counts=True), strict=True)
        },
        "ari": float(ari),
        "nmi": float(nmi),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "embedding_dim": int(X.shape[1]),
    }

    logger.info("Topic clustering results: ARI=%.4f, NMI=%.4f, V-measure=%.4f", ari, nmi, v_measure)

    os.makedirs(config.output_path, exist_ok=True)
    _write_json(results, os.path.join(config.output_path, "topic_results.json"))
