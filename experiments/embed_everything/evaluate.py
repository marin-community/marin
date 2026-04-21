# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate embedding quality for downstream tasks.

Quality classification: linear probe on embeddings -> Spearman/Kendall correlation
with oracle quality scores.

Topic clustering: K-Means on embeddings -> ARI/NMI against oracle topic labels.
"""

import json
import logging
import os
import tempfile

import numpy as np
from rigging.filesystem import open_url
from zephyr.readers import load_parquet

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


def _load_quality_data(
    embeddings_path: str,
    oracle_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    """Load and align quality embeddings with oracle scores, returning standardized train/test splits."""
    from sklearn.preprocessing import StandardScaler

    emb_file = os.path.join(embeddings_path, "quality_embeddings.npz")
    emb_data = _load_npz(emb_file)
    embeddings = emb_data["embeddings"]
    doc_ids = emb_data["doc_ids"].tolist()
    splits = emb_data["splits"].tolist()

    oracle_file = os.path.join(oracle_path, "quality_labeled.parquet")
    oracle_docs = list(load_parquet(oracle_file))
    oracle_by_id = {d["doc_id"]: d for d in oracle_docs}

    train_emb, train_scores = [], []
    test_emb, test_scores = [], []
    test_buckets: list[str] = []

    for i, doc_id in enumerate(doc_ids):
        oracle = oracle_by_id.get(doc_id)
        if oracle is None or oracle.get("oracle_quality_score", -1) == -1:
            continue

        score = oracle["oracle_quality_score"]
        bucket = oracle.get("quality_bucket", "unknown")

        if splits[i] == "train":
            train_emb.append(embeddings[i])
            train_scores.append(score)
        else:
            test_emb.append(embeddings[i])
            test_scores.append(score)
            test_buckets.append(bucket)

    if not train_emb or not test_emb:
        logger.error("Insufficient data: %d train, %d test", len(train_emb), len(test_emb))
        return None

    X_train = np.array(train_emb)
    y_train = np.array(train_scores, dtype=float)
    X_test = np.array(test_emb)
    y_test = np.array(test_scores, dtype=float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, test_buckets


def evaluate_quality_probe(
    output_path: str,
    embeddings_path: str,
    oracle_path: str,
) -> None:
    """Train a linear probe on embeddings and evaluate against oracle quality scores.

    Reports Spearman/Kendall correlation, Ridge R^2, and bucket-ordinal Spearman.
    """
    from scipy import stats
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_squared_error, r2_score

    loaded = _load_quality_data(embeddings_path, oracle_path)
    if loaded is None:
        return
    X_train, y_train, X_test, y_test, test_buckets = loaded

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
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
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

    _write_json(results, os.path.join(output_path, "quality_results.json"))


MLP_ARCHITECTURES = [(128,), (128, 64), (256, 128)]


def evaluate_quality_mlp(
    output_path: str,
    embeddings_path: str,
    oracle_path: str,
) -> None:
    """Train MLP probes on embeddings and evaluate against oracle quality scores.

    Tries multiple architectures and reports the best by Spearman rho.
    """
    from scipy import stats
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.neural_network import MLPRegressor

    loaded = _load_quality_data(embeddings_path, oracle_path)
    if loaded is None:
        return
    X_train, y_train, X_test, y_test, test_buckets = loaded

    test_ordinals = np.array([QUALITY_ORDINAL.get(b, 2) for b in test_buckets], dtype=float)

    arch_results = []
    for arch in MLP_ARCHITECTURES:
        model = MLPRegressor(
            hidden_layer_sizes=arch,
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=500,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        spearman_r, spearman_p = stats.spearmanr(y_test, y_pred)
        kendall_tau, kendall_p = stats.kendalltau(y_test, y_pred)
        bucket_spearman, _ = stats.spearmanr(test_ordinals, y_pred)

        result = {
            "architecture": str(arch),
            "spearman_rho": float(spearman_r),
            "spearman_p": float(spearman_p),
            "kendall_tau": float(kendall_tau),
            "kendall_p": float(kendall_p),
            "bucket_ordinal_spearman": float(bucket_spearman),
            "r2": float(r2_score(y_test, y_pred)),
            "mse": float(mean_squared_error(y_test, y_pred)),
            "n_iter": int(model.n_iter_),
        }
        arch_results.append(result)
        logger.info(
            "MLP %s: Spearman=%.4f, Kendall=%.4f, R2=%.4f",
            arch,
            spearman_r,
            kendall_tau,
            result["r2"],
        )

    best = max(arch_results, key=lambda r: r["spearman_rho"])

    output = {
        "task": "quality_mlp",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "embedding_dim": int(X_train.shape[1]),
        "best_architecture": best["architecture"],
        "best_spearman_rho": best["spearman_rho"],
        "architectures": arch_results,
    }

    logger.info("Best MLP: %s with Spearman=%.4f", best["architecture"], best["spearman_rho"])

    _write_json(output, os.path.join(output_path, "quality_mlp_results.json"))


def _load_topic_data(
    embeddings_path: str,
    oracle_path: str,
) -> tuple[np.ndarray, np.ndarray, int, list[str]] | None:
    """Load and align topic embeddings with oracle labels, returning scaled features and encoded labels."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    emb_file = os.path.join(embeddings_path, "topic_embeddings.npz")
    emb_data = _load_npz(emb_file)
    embeddings = emb_data["embeddings"]
    doc_ids = emb_data["doc_ids"].tolist()

    oracle_file = os.path.join(oracle_path, "topic_labeled.parquet")
    oracle_docs = list(load_parquet(oracle_file))
    oracle_by_id = {d["doc_id"]: d for d in oracle_docs}

    valid_emb = []
    valid_labels = []
    for i, doc_id in enumerate(doc_ids):
        oracle = oracle_by_id.get(doc_id)
        if oracle is None or oracle.get("oracle_topic") == "labeling_failed":
            continue
        valid_emb.append(embeddings[i])
        valid_labels.append(oracle["oracle_topic"])

    if len(valid_emb) < 2:
        logger.error("Too few valid documents (%d)", len(valid_emb))
        return None

    X = np.array(valid_emb)
    le = LabelEncoder()
    y_true = le.fit_transform(valid_labels)
    n_true_clusters = len(le.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_true, n_true_clusters, valid_labels


def evaluate_topic_clusters(
    output_path: str,
    embeddings_path: str,
    oracle_path: str,
    n_clusters: int = 15,
) -> None:
    """Run K-Means on embeddings and evaluate against oracle topic labels.

    Reports ARI, NMI, homogeneity, completeness, and V-measure.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
        v_measure_score,
    )

    loaded = _load_topic_data(embeddings_path, oracle_path)
    if loaded is None:
        return
    X_scaled, y_true, n_true_clusters, true_labels = loaded

    if len(true_labels) < n_clusters:
        logger.error("Too few valid documents (%d) for %d clusters", len(true_labels), n_clusters)
        return

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_scaled)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    homogeneity = homogeneity_score(y_true, y_pred)
    completeness = completeness_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)

    unique_labels, label_counts = np.unique(true_labels, return_counts=True)
    results = {
        "task": "topic_clustering",
        "n_documents": len(true_labels),
        "n_true_clusters": int(n_true_clusters),
        "n_kmeans_clusters": n_clusters,
        "true_label_distribution": {label: int(count) for label, count in zip(unique_labels, label_counts, strict=True)},
        "ari": float(ari),
        "nmi": float(nmi),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "embedding_dim": int(X_scaled.shape[1]),
    }

    logger.info("Topic clustering results: ARI=%.4f, NMI=%.4f, V-measure=%.4f", ari, nmi, v_measure)

    _write_json(results, os.path.join(output_path, "topic_results.json"))


PCA_COMPONENTS = [8, 16, 32, 64, 128]
UMAP_COMPONENTS = [8, 16, 32]


def evaluate_topic_reduced(
    output_path: str,
    embeddings_path: str,
    oracle_path: str,
    n_clusters: int = 15,
) -> None:
    """Run K-Means on dimensionality-reduced embeddings (PCA and UMAP).

    Sweeps over multiple n_components values and reports ARI/NMI for each.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    loaded = _load_topic_data(embeddings_path, oracle_path)
    if loaded is None:
        return
    X_scaled, y_true, n_true_clusters, true_labels = loaded

    if len(true_labels) < n_clusters:
        logger.error("Too few valid documents (%d) for %d clusters", len(true_labels), n_clusters)
        return

    def _cluster_and_score(X_reduced: np.ndarray) -> tuple[float, float]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X_reduced)
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        return float(ari), float(nmi)

    configs = []

    # Baseline: no reduction
    ari, nmi = _cluster_and_score(X_scaled)
    configs.append({"method": "baseline", "n_components": int(X_scaled.shape[1]), "ari": ari, "nmi": nmi})
    logger.info("Baseline (d=%d): ARI=%.4f, NMI=%.4f", X_scaled.shape[1], ari, nmi)

    # PCA sweep
    for n_comp in PCA_COMPONENTS:
        if n_comp >= X_scaled.shape[1]:
            continue
        pca = PCA(n_components=n_comp, random_state=42)
        X_reduced = pca.fit_transform(X_scaled)
        ari, nmi = _cluster_and_score(X_reduced)
        explained_var = float(np.sum(pca.explained_variance_ratio_))
        configs.append(
            {
                "method": "pca",
                "n_components": n_comp,
                "ari": ari,
                "nmi": nmi,
                "explained_variance_ratio": explained_var,
            }
        )
        logger.info("PCA (d=%d, var=%.3f): ARI=%.4f, NMI=%.4f", n_comp, explained_var, ari, nmi)

    # UMAP sweep
    try:
        import umap

        for n_comp in UMAP_COMPONENTS:
            reducer = umap.UMAP(n_components=n_comp, random_state=42)
            X_reduced = reducer.fit_transform(X_scaled)
            ari, nmi = _cluster_and_score(X_reduced)
            configs.append({"method": "umap", "n_components": n_comp, "ari": ari, "nmi": nmi})
            logger.info("UMAP (d=%d): ARI=%.4f, NMI=%.4f", n_comp, ari, nmi)
    except ImportError:
        logger.warning("umap-learn not installed, skipping UMAP configurations")

    best = max(configs, key=lambda c: c["nmi"])

    output = {
        "task": "topic_reduced",
        "n_documents": len(true_labels),
        "n_true_clusters": int(n_true_clusters),
        "n_kmeans_clusters": n_clusters,
        "best_method": best["method"],
        "best_n_components": best["n_components"],
        "best_nmi": best["nmi"],
        "best_ari": best["ari"],
        "configs": configs,
    }

    logger.info(
        "Best config: %s (d=%d) NMI=%.4f, ARI=%.4f", best["method"], best["n_components"], best["nmi"], best["ari"]
    )

    _write_json(output, os.path.join(output_path, "topic_reduced_results.json"))
