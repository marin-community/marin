# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the embedding-based domain typer."""

import numpy as np

from experiments.datakit.cluster.quality.fast_transformer.domain_mlp import fit, load, predict, predict_indices, save
from experiments.datakit.cluster.quality.fast_transformer.joined_labels import embedding_matrix

LABELS = ["prose", "code", "math"]


def _separable_embeddings(n_per_class: int, dim: int = 64, seed: int = 0):
    """int8 embedding rows in three well-separated directions, like the stored ones."""
    rng = np.random.default_rng(seed)
    rows, y = [], []
    for cls in range(len(LABELS)):
        center = np.zeros(dim)
        center[cls] = 1.0
        directions = center + 0.1 * rng.normal(size=(n_per_class, dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        rows.append(np.clip(directions * 127, -127, 127).astype(np.int8))
        y.extend([cls] * n_per_class)
    return np.concatenate(rows), np.array(y)


def test_fit_separates_types_and_survives_save_load(tmp_path):
    raw, y = _separable_embeddings(700)
    x = embedding_matrix(raw)
    model = fit(x, y, len(LABELS))
    accuracy = float((predict_indices(model, x) == y).mean())
    assert accuracy > 0.95, f"separable classes should be learned, got {accuracy:.3f}"

    path = str(tmp_path / "domain_mlp.npz")
    save(model, LABELS, path)
    loaded, labels = load(path)
    assert labels == LABELS
    predicted = predict(loaded, labels, raw)
    np.testing.assert_array_equal(predicted, np.array(LABELS)[predict_indices(model, x)])
