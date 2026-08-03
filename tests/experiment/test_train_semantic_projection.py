# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from train_semantic_projection import (  # noqa: E402
    fold_embedding_projection,
    retained_train_validation_indices,
    supervised_contrastive_loss,
    validation_decision,
)

from experiments.datakit.cluster.quality.fast_transformer.model import (  # noqa: E402
    FastEmbeddingTransformer,
    FastTransformerConfig,
)


def test_retained_train_validation_indices_drop_low_confidence_and_do_not_overlap() -> None:
    confidences = np.asarray([0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    leaves = ["A"] * 5 + ["B"] * 5

    training, validation, cutoff = retained_train_validation_indices(
        confidences,
        leaves,
        validation_fraction=0.2,
        drop_fraction=0.1,
    )

    assert 0 not in set(training) | set(validation)
    assert set(training).isdisjoint(validation)
    assert sorted(np.concatenate([training, validation]).tolist()) == list(range(1, 10))
    assert cutoff == 0.2


def test_retained_train_validation_indices_drop_exact_count_when_confidences_tie() -> None:
    confidences = np.asarray([0.1] * 5 + [0.9] * 5)
    leaves = ["A"] * 5 + ["B"] * 5

    training, validation, _ = retained_train_validation_indices(
        confidences,
        leaves,
        validation_fraction=0.2,
        drop_fraction=0.2,
    )

    assert len(set(range(10)) - set(training) - set(validation)) == 2


def test_supervised_contrastive_loss_prefers_cross_source_label_neighbors() -> None:
    labels = jnp.asarray([0, 1, 0, 1])
    sources = jnp.asarray([0, 0, 1, 1])
    aligned = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])
    reversed_neighbors = aligned[jnp.asarray([0, 1, 3, 2])]

    aligned_loss = supervised_contrastive_loss(aligned, labels, sources, temperature=0.1)
    reversed_loss = supervised_contrastive_loss(reversed_neighbors, labels, sources, temperature=0.1)

    assert float(aligned_loss) < float(reversed_loss)


def test_validation_decision_rejects_low_rank_vectors_despite_semantic_gain() -> None:
    base = {"parent_macro_f1": 0.4, "leaf_macro_f1": 0.4, "form_macro_f1": 0.4}
    projected = {
        "parent_macro_f1": 0.5,
        "leaf_macro_f1": 0.5,
        "form_macro_f1": 0.5,
        "geometry": {
            "finite_fraction": 1.0,
            "unique_fraction_4dp": 1.0,
            "effective_rank_fraction": 0.1,
            "total_variance": 0.8,
        },
    }
    decision = validation_decision(base, projected, folded_cosine_minimum=1.0)

    assert decision["semantic_mean_delta"] > 0
    assert not decision["gates"]["effective_rank_fraction"]
    assert not decision["passed"]


def test_fold_embedding_projection_preserves_projected_directions() -> None:
    config = FastTransformerConfig(
        vocab_size=16,
        max_tokens=8,
        pool_window=2,
        embed_dim=8,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2,
        dropout=0.0,
    )
    model = FastEmbeddingTransformer(config, output_dim=4, key=jax.random.PRNGKey(0))
    projection = np.eye(4) + 0.01 * np.asarray(jax.random.normal(jax.random.PRNGKey(1), (4, 4)))
    ids = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])

    projected = np.asarray(model(ids)) @ projection
    projected /= np.linalg.norm(projected, axis=1, keepdims=True)
    folded_model = fold_embedding_projection(model, projection)
    folded = np.asarray(folded_model(ids))
    cosines = np.sum(projected * folded, axis=1)

    assert float(cosines.min()) >= 0.999
    assert eqx.tree_equal(model.backbone, folded_model.backbone)
