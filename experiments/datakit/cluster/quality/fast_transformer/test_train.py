# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for train_regressor's trainable-leaf filtering."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformer, FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.train import TrainHParams, train_regressor


def test_params_filter_freezes_the_donor_table_while_the_projection_trains():
    """A deselected leaf must receive neither gradients nor weight decay —
    plain AdamW weight decay would silently erode a merely stop-gradiented
    donor table."""
    config = FastTransformerConfig(
        vocab_size=64,
        max_tokens=32,
        pool_window=8,
        pool_kind="mean",
        embed_dim=8,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        frozen_donor_dim=12,
    )
    model = FastTransformer(config, key=jr.PRNGKey(0))
    donor = jr.normal(jr.PRNGKey(1), (config.vocab_size, config.frozen_donor_dim))
    model = eqx.tree_at(lambda m: m.donor_embed, model, donor)
    params_filter = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    params_filter = eqx.tree_at(lambda m: m.donor_embed, params_filter, replace=False)

    rng = np.random.default_rng(0)
    ids = rng.integers(1, config.vocab_size, size=(64, config.max_tokens)).astype(np.int32)
    scores = rng.uniform(size=64).astype(np.float32)
    hp = TrainHParams(epochs=3, batch_size=16, patience=10)
    best, _, _ = train_regressor(model, ids, scores, ids[:16], scores[:16], hp, params_filter=params_filter)

    np.testing.assert_array_equal(np.asarray(best.donor_embed), np.asarray(donor))
    assert not np.allclose(np.asarray(best.donor_proj), np.asarray(model.donor_proj)), (
        "the projection is trainable and three epochs of updates must move it"
    )
    assert best.embed is None
    preds = best(jnp.asarray(ids[:4]))
    assert preds.shape == (4,) and bool(np.isfinite(np.asarray(preds)).all())
