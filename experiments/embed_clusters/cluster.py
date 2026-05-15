# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cosine K-means over precomputed embeddings, plus per-cluster summaries.

For each cluster we emit: size, mean cosine to centroid, top-N c-TF-IDF
(class-based TF-IDF, BERTopic-style) bigrams, and ``n_reps`` representative
documents (the closest to the centroid). We also report a silhouette score
computed on a 2k-doc subsample (full silhouette is O(N²)).

Outputs:
- ``cluster_stats.json`` — headline metrics + per-cluster summaries
- ``assignments.parquet`` — ``id, cluster_id`` so we can post-hoc filter
"""

import json
import logging
import os
import tempfile

import numpy as np
from rigging.filesystem import open_url
from zephyr.readers import load_parquet
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)

DEFAULT_K = 40
DEFAULT_N_REPS = 5
DEFAULT_N_TERMS = 10
DEFAULT_EXCERPT_CHARS = 500
SILHOUETTE_SAMPLE = 2000
CTFIDF_MAX_FEATURES = 20_000
RANDOM_STATE = 42


def _load_embeddings(samples_path: str, embeddings_path: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Return ``(X_unit, ids, texts)`` aligned by index, with X L2-normalized."""
    docs_by_id = {d["id"]: d["text"] for d in load_parquet(os.path.join(samples_path, "samples.parquet"))}

    local = os.path.join(tempfile.gettempdir(), "embeddings.npz")
    with open_url(os.path.join(embeddings_path, "embeddings.npz"), "rb") as src, open(local, "wb") as dst:
        dst.write(src.read())
    data = dict(np.load(local, allow_pickle=True))
    os.remove(local)

    embeddings = data["embeddings"].astype(np.float32, copy=False)
    ids = data["ids"].tolist()
    texts = [docs_by_id[i] for i in ids]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms, ids, texts


def cluster_and_summarize(
    output_path: str,
    samples_path: str,
    embeddings_path: str,
    k: int = DEFAULT_K,
    n_reps: int = DEFAULT_N_REPS,
    n_terms: int = DEFAULT_N_TERMS,
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
) -> None:
    """Run cosine K-means, summarize each cluster, write JSON + assignments parquet."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score

    X, ids, texts = _load_embeddings(samples_path, embeddings_path)
    logger.info("Loaded embeddings: %s; running K-means k=%d", X.shape, k)

    # Cosine K-means = Euclidean K-means on unit vectors.
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
    labels = km.labels_

    # c-TF-IDF: concatenate each cluster's texts into one "document" and
    # TF-IDF over the resulting K-document corpus. Top features per row are
    # the cluster's discriminating terms vs the rest of the corpus.
    cluster_corpus = [" ".join(t for t, lab in zip(texts, labels, strict=True) if lab == c) for c in range(k)]
    vec = TfidfVectorizer(
        max_features=CTFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf = vec.fit_transform(cluster_corpus)
    vocab = np.array(vec.get_feature_names_out())

    clusters = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        centroid = km.cluster_centers_[c]
        # cosine(centroid, x) for unit x = 1 - 0.5 * ||centroid - x||^2 only when
        # centroid is itself unit. KMeans centroids aren't unit-norm; compute
        # cosine directly to avoid that pitfall.
        cos_to_centroid = (X[idx] @ centroid) / (np.linalg.norm(centroid) + 1e-12)
        order = np.argsort(-cos_to_centroid)
        reps = idx[order[:n_reps]]

        row = tfidf[c].toarray().ravel()
        top_terms = vocab[np.argsort(-row)[:n_terms]].tolist()

        clusters.append(
            {
                "cluster_id": int(c),
                "size": len(idx),
                "mean_cos_to_centroid": float(cos_to_centroid.mean()),
                "top_terms": top_terms,
                "representatives": [{"id": ids[i], "excerpt": texts[i][:excerpt_chars]} for i in reps],
            }
        )

    sizes = np.bincount(labels, minlength=k)
    sub_size = min(SILHOUETTE_SAMPLE, len(X))
    sub = np.random.default_rng(RANDOM_STATE).choice(len(X), size=sub_size, replace=False)
    silhouette = float(silhouette_score(X[sub], labels[sub], metric="cosine"))

    summary = {
        "k": int(k),
        "n_docs": len(X),
        "embedding_dim": int(X.shape[1]),
        "silhouette_sample": silhouette,
        "silhouette_n": int(sub_size),
        "size_min": int(sizes.min()),
        "size_max": int(sizes.max()),
        "size_mean": float(sizes.mean()),
        "clusters": clusters,
    }

    with open_url(os.path.join(output_path, "cluster_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_parquet_file(
        [{"id": i, "cluster_id": int(lab)} for i, lab in zip(ids, labels, strict=True)],
        os.path.join(output_path, "assignments.parquet"),
    )

    logger.info(
        "Wrote cluster_stats.json (silhouette=%.3f, size min/max=%d/%d)",
        silhouette,
        sizes.min(),
        sizes.max(),
    )
