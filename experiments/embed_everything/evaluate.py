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
import re
import tempfile

import numpy as np
from rigging.filesystem import open_url
from zephyr.readers import load_parquet

from experiments.embed_everything.oracle import TOPIC_TAXONOMY

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
    """Load and align topic embeddings with oracle labels, returning L2-normalized features.

    Embeddings come from ``normalize_embeddings=True`` in ``embed.py`` and are
    already unit-norm; we re-normalize defensively so downstream Euclidean
    K-Means is equivalent to cosine K-Means: for unit vectors
    ``||a-b||^2 = 2 - 2*cos(a,b)``, so argmin-Euclidean = argmax-cosine.
    Previous revisions applied StandardScaler here, which stripped the
    spherical geometry and degraded topic NMI.
    """
    from sklearn.preprocessing import LabelEncoder

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

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_normalized = X / norms

    return X_normalized, y_true, n_true_clusters, valid_labels


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
    X_norm, y_true, n_true_clusters, true_labels = loaded

    if len(true_labels) < n_clusters:
        logger.error("Too few valid documents (%d) for %d clusters", len(true_labels), n_clusters)
        return

    # K-Means on unit vectors == cosine K-Means for assignment purposes.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_norm)

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
        "embedding_dim": int(X_norm.shape[1]),
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
    X_norm, y_true, n_true_clusters, true_labels = loaded

    if len(true_labels) < n_clusters:
        logger.error("Too few valid documents (%d) for %d clusters", len(true_labels), n_clusters)
        return

    def _cluster_and_score(X_reduced: np.ndarray) -> tuple[float, float]:
        # Re-normalize after reduction so K-Means on the reduced space remains
        # cosine-equivalent (PCA/UMAP outputs are not unit-norm in general).
        reduced_norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
        reduced_norms[reduced_norms == 0] = 1.0
        X_unit = X_reduced / reduced_norms
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X_unit)
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        return float(ari), float(nmi)

    configs = []

    # Baseline: no reduction (already unit-norm from _load_topic_data).
    ari, nmi = _cluster_and_score(X_norm)
    configs.append({"method": "baseline", "n_components": int(X_norm.shape[1]), "ari": ari, "nmi": nmi})
    logger.info("Baseline (d=%d): ARI=%.4f, NMI=%.4f", X_norm.shape[1], ari, nmi)

    # PCA sweep
    for n_comp in PCA_COMPONENTS:
        if n_comp >= X_norm.shape[1]:
            continue
        pca = PCA(n_components=n_comp, random_state=42)
        X_reduced = pca.fit_transform(X_norm)
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
            X_reduced = reducer.fit_transform(X_norm)
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


# ---------------------------------------------------------------------------
# Fasttext baselines
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9 ]+")

# The Dolma-3 fasttext topic classifier emits snake_case labels that don't
# reduce to a few WebOrganizer display names by canonicalization alone — the
# classifier's training-label form expanded three buckets. Verified empirically
# against smoke output + the WebOrganizer project docs (24 topics total).
_FASTTEXT_TO_DISPLAY_ALIASES: dict[str, str] = {
    "software_development": "Software Dev.",
    "history_and_geography": "History",
    "science_math_and_technology": "Science & Tech.",
    # ``hardare`` is a literal typo in the Dolma-3 fasttext model's labels.
    "electronics_and_hardare": "Hardware",
    "travel_and_tourism": "Travel",
}


def _canonical_topic(label: str) -> str:
    """Collapse a topic label to a form that matches across '&'/'and' and punctuation."""
    lowered = label.lower().replace("&", " and ")
    alnum = _NON_ALNUM_SPACE_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", alnum).strip()


def _build_topic_canonical_map() -> dict[str, str]:
    """Map canonicalized WebOrganizer labels back to their display form."""
    return {_canonical_topic(name): name for name in TOPIC_TAXONOMY}


def _match_fasttext_to_display(raw_label: str, canonical_to_display: dict[str, str]) -> str | None:
    """Map a raw fasttext topic label to its WebOrganizer display form, or None if unknown."""
    if raw_label in _FASTTEXT_TO_DISPLAY_ALIASES:
        return _FASTTEXT_TO_DISPLAY_ALIASES[raw_label]
    return canonical_to_display.get(_canonical_topic(raw_label))


def evaluate_fasttext_quality(
    output_path: str,
    fasttext_path: str,
    oracle_path: str,
) -> None:
    """Correlate fasttext quality scores with oracle scores on the test split.

    No probe is trained: fasttext's score is already a per-doc prediction, so
    Spearman/Kendall between that score and the oracle 0-5 rubric is the
    head-to-head comparison with ``evaluate_quality_probe``.
    """
    from scipy import stats

    fasttext_file = os.path.join(fasttext_path, "quality_fasttext_scores.parquet")
    oracle_file = os.path.join(oracle_path, "quality_labeled.parquet")

    ft_rows = list(load_parquet(fasttext_file))
    oracle_docs = list(load_parquet(oracle_file))
    oracle_by_id = {d["doc_id"]: d for d in oracle_docs}

    scores: list[float] = []
    oracle_scores: list[float] = []
    test_buckets: list[str] = []

    for row in ft_rows:
        if row.get("split") != "test":
            continue
        oracle = oracle_by_id.get(row["doc_id"])
        if oracle is None or oracle.get("oracle_quality_score", -1) == -1:
            continue
        scores.append(float(row["fasttext_quality_score"]))
        oracle_scores.append(float(oracle["oracle_quality_score"]))
        test_buckets.append(oracle.get("quality_bucket", "unknown"))

    if len(scores) < 2:
        logger.error("Not enough test-split fasttext predictions to evaluate (%d)", len(scores))
        return

    spearman_r, spearman_p = stats.spearmanr(oracle_scores, scores)
    kendall_tau, kendall_p = stats.kendalltau(oracle_scores, scores)

    test_ordinals = np.array([QUALITY_ORDINAL.get(b, 2) for b in test_buckets], dtype=float)
    bucket_spearman, _ = stats.spearmanr(test_ordinals, scores)

    results = {
        "task": "fasttext_quality",
        "n_test": len(scores),
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_tau),
        "kendall_p": float(kendall_p),
        "bucket_ordinal_spearman": float(bucket_spearman),
    }
    logger.info(
        "Fasttext quality baseline: Spearman=%.4f, Kendall=%.4f (n_test=%d)",
        spearman_r,
        kendall_tau,
        len(scores),
    )
    _write_json(results, os.path.join(output_path, "fasttext_quality_results.json"))


def evaluate_fasttext_topic(
    output_path: str,
    fasttext_path: str,
    oracle_path: str,
) -> None:
    """Compare fasttext topic predictions to oracle topic labels on test split.

    Reports top-1 accuracy, macro-F1, per-class F1, and a confusion matrix over
    the WebOrganizer 24-class taxonomy. Both sides are normalized by
    ``_canonical_topic`` before comparison so punctuation/whitespace variants
    (e.g., fasttext's ``art_design`` vs oracle's ``Art & Design``) align.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    fasttext_file = os.path.join(fasttext_path, "topic_fasttext_predictions.parquet")
    oracle_file = os.path.join(oracle_path, "topic_labeled.parquet")

    ft_rows = list(load_parquet(fasttext_file))
    oracle_docs = list(load_parquet(oracle_file))
    oracle_by_id = {d["doc_id"]: d for d in oracle_docs}

    canonical_to_display = _build_topic_canonical_map()
    unmatched_fasttext_labels: set[str] = set()

    y_true: list[str] = []
    y_pred: list[str] = []
    for row in ft_rows:
        if row.get("split") != "test":
            continue
        oracle = oracle_by_id.get(row["doc_id"])
        if oracle is None or oracle.get("oracle_topic") == "labeling_failed":
            continue

        ft_raw = str(row["fasttext_topic"])
        ft_display = _match_fasttext_to_display(ft_raw, canonical_to_display)
        if ft_display is None:
            unmatched_fasttext_labels.add(ft_raw)
            ft_display = "unknown"

        y_true.append(oracle["oracle_topic"])
        y_pred.append(ft_display)

    if len(y_true) < 2:
        logger.error("Not enough test-split fasttext predictions to evaluate (%d)", len(y_true))
        return

    label_order = list(TOPIC_TAXONOMY) + (["unknown"] if "unknown" in y_pred else [])
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=list(TOPIC_TAXONOMY), average="macro", zero_division=0)
    per_class = classification_report(y_true, y_pred, labels=list(TOPIC_TAXONOMY), output_dict=True, zero_division=0)
    conf = confusion_matrix(y_true, y_pred, labels=label_order).tolist()

    results = {
        "task": "fasttext_topic",
        "n_test": len(y_true),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": label_order,
            "matrix": conf,
        },
        "unmatched_fasttext_labels": sorted(unmatched_fasttext_labels),
    }
    logger.info(
        "Fasttext topic baseline: accuracy=%.4f, macro-F1=%.4f (n_test=%d, unmatched=%d)",
        acc,
        macro_f1,
        len(y_true),
        len(unmatched_fasttext_labels),
    )
    if unmatched_fasttext_labels:
        logger.warning(
            "Fasttext labels not in WebOrganizer taxonomy (after normalization): %s", unmatched_fasttext_labels
        )
    _write_json(results, os.path.join(output_path, "fasttext_topic_results.json"))


def evaluate_oracle_retest(
    output_path: str,
    run1_path: str,
    run2_path: str,
) -> None:
    """Measure the oracle's test-retest noise on the same subset of docs.

    ``run1_path`` is the original oracle_quality output; ``run2_path`` is the
    relabel_quality_subset output, which carries ``oracle_quality_score``
    (fresh call) plus ``oracle_quality_score_prev`` (copied from run1 at
    relabel time, redundant with run1 but kept for sanity).

    The Spearman between the two runs is the **noise ceiling** on any probe
    evaluated against run1: a probe can't rank-correlate with run1 better
    than run1 rank-correlates with itself under re-sampling.
    """
    from scipy import stats

    run1_file = os.path.join(run1_path, "quality_labeled.parquet")
    run2_file = os.path.join(run2_path, "quality_retest.parquet")

    run1 = {d["doc_id"]: d for d in load_parquet(run1_file)}
    run2_rows = list(load_parquet(run2_file))

    s1: list[int] = []
    s2: list[int] = []
    buckets: list[str] = []
    for r2 in run2_rows:
        r1 = run1.get(r2["doc_id"])
        if r1 is None:
            continue
        if r1.get("oracle_quality_score", -1) == -1 or r2.get("oracle_quality_score", -1) == -1:
            continue
        s1.append(int(r1["oracle_quality_score"]))
        s2.append(int(r2["oracle_quality_score"]))
        buckets.append(r1.get("quality_bucket", "unknown"))

    if len(s1) < 2:
        logger.error("Too few retest pairs to evaluate (%d)", len(s1))
        return

    arr1 = np.array(s1, dtype=float)
    arr2 = np.array(s2, dtype=float)
    diff = arr2 - arr1

    spearman_r, spearman_p = stats.spearmanr(arr1, arr2)
    kendall_tau, kendall_p = stats.kendalltau(arr1, arr2)
    mae = float(np.mean(np.abs(diff)))
    exact = float(np.mean(diff == 0))
    off_by_one = float(np.mean(np.abs(diff) <= 1))

    per_bucket: dict[str, dict[str, float]] = {}
    for b in sorted(set(buckets)):
        mask = [bk == b for bk in buckets]
        n = sum(mask)
        if n == 0:
            continue
        d_sub = diff[mask]
        per_bucket[b] = {
            "n": int(n),
            "exact_agreement": float(np.mean(d_sub == 0)),
            "off_by_one": float(np.mean(np.abs(d_sub) <= 1)),
            "mean_drift_run2_minus_run1": float(np.mean(d_sub)),
            "mae": float(np.mean(np.abs(d_sub))),
        }

    results = {
        "task": "oracle_retest",
        "n_pairs": len(s1),
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_tau),
        "kendall_p": float(kendall_p),
        "mae": mae,
        "exact_agreement_rate": exact,
        "off_by_one_rate": off_by_one,
        "mean_drift_run2_minus_run1": float(np.mean(diff)),
        "run1_score_mean": float(arr1.mean()),
        "run2_score_mean": float(arr2.mean()),
        "run1_score_std": float(arr1.std()),
        "run2_score_std": float(arr2.std()),
        "per_bucket": per_bucket,
    }
    logger.info(
        "Oracle retest: Spearman=%.4f Kendall=%.4f exact=%.3f off_by_one=%.3f MAE=%.3f (n=%d)",
        spearman_r,
        kendall_tau,
        exact,
        off_by_one,
        mae,
        len(s1),
    )
    _write_json(results, os.path.join(output_path, "oracle_retest_results.json"))


def evaluate_topic_supervised(
    output_path: str,
    embeddings_path: str,
    oracle_path: str,
) -> None:
    """Train a supervised logistic-regression topic classifier on embeddings.

    Counterpart to ``evaluate_topic_clusters``'s unsupervised K-Means eval: this
    is an apples-to-apples comparison against the fasttext topic classifier,
    which is itself a supervised classifier.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.preprocessing import LabelEncoder

    emb_file = os.path.join(embeddings_path, "topic_embeddings.npz")
    emb_data = _load_npz(emb_file)
    embeddings = emb_data["embeddings"]
    doc_ids = emb_data["doc_ids"].tolist()
    splits = emb_data["splits"].tolist()

    oracle_file = os.path.join(oracle_path, "topic_labeled.parquet")
    oracle_docs = list(load_parquet(oracle_file))
    oracle_by_id = {d["doc_id"]: d for d in oracle_docs}

    train_emb, train_labels = [], []
    test_emb, test_labels = [], []
    for i, doc_id in enumerate(doc_ids):
        oracle = oracle_by_id.get(doc_id)
        if oracle is None or oracle.get("oracle_topic") == "labeling_failed":
            continue
        label = oracle["oracle_topic"]
        if splits[i] == "train":
            train_emb.append(embeddings[i])
            train_labels.append(label)
        else:
            test_emb.append(embeddings[i])
            test_labels.append(label)

    if not train_emb or not test_emb:
        logger.error("Insufficient topic data: %d train, %d test", len(train_emb), len(test_emb))
        return

    # Encode labels from the full taxonomy so held-out classes get a stable index.
    le = LabelEncoder().fit(list(TOPIC_TAXONOMY))
    # Any labels not in the taxonomy will break transform — surface them loudly.
    unseen = sorted(set(train_labels + test_labels) - set(TOPIC_TAXONOMY))
    if unseen:
        raise ValueError(f"Oracle produced topic labels outside the taxonomy: {unseen}")

    X_train = np.array(train_emb)
    X_test = np.array(test_emb)
    y_train = le.transform(train_labels)
    y_test = le.transform(test_labels)

    clf = LogisticRegression(max_iter=5000, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    per_class = classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        labels=list(TOPIC_TAXONOMY),
        output_dict=True,
        zero_division=0,
    )
    conf = confusion_matrix(y_test, y_pred, labels=list(range(len(TOPIC_TAXONOMY)))).tolist()

    results = {
        "task": "topic_supervised",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "embedding_dim": int(X_train.shape[1]),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(TOPIC_TAXONOMY),
            "matrix": conf,
        },
    }
    logger.info(
        "Supervised topic probe: accuracy=%.4f, macro-F1=%.4f (n_train=%d, n_test=%d, d=%d)",
        acc,
        macro_f1,
        X_train.shape[0],
        X_test.shape[0],
        X_train.shape[1],
    )
    _write_json(results, os.path.join(output_path, "topic_supervised_results.json"))
