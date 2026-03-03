# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os

import numpy as np

from experiments.embed_everything.evaluate import evaluate_quality_probe, evaluate_topic_clusters


def _write_embeddings(path: str, embeddings: np.ndarray, ids: list[str], labels: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, embeddings=embeddings, ids=np.array(ids), labels=np.array(labels))


def test_evaluate_quality_probe_perfect_signal(tmp_path):
    """A linearly separable quality signal should yield high Spearman/Kendall."""
    rng = np.random.default_rng(42)
    n = 200
    # Quality labels 0-4, embeddings are just the label value + noise
    labels = np.repeat(np.arange(5), n // 5).astype(float)
    embeddings = np.column_stack([labels + rng.normal(0, 0.1, n), rng.normal(0, 0.1, n)])
    ids = [f"doc-{i}" for i in range(n)]
    label_strs = [str(int(l)) for l in labels]

    emb_path = str(tmp_path / "embeddings.npz")
    _write_embeddings(emb_path, embeddings, ids, label_strs)

    output_path = str(tmp_path / "output")
    result = evaluate_quality_probe(output_path=output_path, embeddings_path=emb_path, test_size=0.2, seed=42)

    assert result.spearman_rho > 0.8
    assert result.kendall_tau > 0.7


def test_evaluate_quality_probe_random_signal(tmp_path):
    """Random embeddings should yield low correlation."""
    rng = np.random.default_rng(42)
    n = 200
    labels = np.repeat(np.arange(5), n // 5).astype(float)
    embeddings = rng.normal(0, 1, (n, 32))
    ids = [f"doc-{i}" for i in range(n)]
    label_strs = [str(int(l)) for l in labels]

    emb_path = str(tmp_path / "embeddings.npz")
    _write_embeddings(emb_path, embeddings, ids, label_strs)

    output_path = str(tmp_path / "output")
    result = evaluate_quality_probe(output_path=output_path, embeddings_path=emb_path, test_size=0.2, seed=42)

    assert abs(result.spearman_rho) < 0.5


def test_evaluate_topic_clusters_separable(tmp_path):
    """Well-separated clusters should yield high ARI/NMI."""
    rng = np.random.default_rng(42)
    n_per_topic = 40
    n_topics = 5
    n = n_per_topic * n_topics

    labels = []
    embeddings_list = []
    for i in range(n_topics):
        center = np.zeros(10)
        center[i] = 5.0
        embeddings_list.append(center + rng.normal(0, 0.3, (n_per_topic, 10)))
        labels.extend([f"topic_{i}"] * n_per_topic)

    embeddings = np.vstack(embeddings_list)
    ids = [f"doc-{i}" for i in range(n)]

    emb_path = str(tmp_path / "embeddings.npz")
    _write_embeddings(emb_path, embeddings, ids, labels)

    output_path = str(tmp_path / "output")
    result = evaluate_topic_clusters(output_path=output_path, embeddings_path=emb_path, n_clusters=n_topics, seed=42)

    assert result.ari > 0.7
    assert result.nmi > 0.7


def test_evaluate_topic_clusters_writes_results(tmp_path):
    """Results are persisted as JSON."""
    rng = np.random.default_rng(42)
    n = 60
    embeddings = rng.normal(0, 1, (n, 5))
    labels = [f"t{i % 3}" for i in range(n)]
    ids = [f"doc-{i}" for i in range(n)]

    emb_path = str(tmp_path / "embeddings.npz")
    _write_embeddings(emb_path, embeddings, ids, labels)

    output_path = str(tmp_path / "output")
    evaluate_topic_clusters(output_path=output_path, embeddings_path=emb_path, n_clusters=3, seed=42)

    results_file = os.path.join(output_path, "results.json")
    assert os.path.exists(results_file)
    with open(results_file) as f:
        saved = json.load(f)
    assert "ari" in saved
    assert "nmi" in saved
