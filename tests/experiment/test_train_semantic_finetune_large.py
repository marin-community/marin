# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from train_semantic_finetune_large import interpolate_models  # noqa: E402

from experiments.datakit.cluster.quality.fast_transformer.model import (  # noqa: E402
    FastEmbeddingTransformer,
    FastTransformerConfig,
)


def small_model(seed: int) -> FastEmbeddingTransformer:
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
    return FastEmbeddingTransformer(config, output_dim=4, key=jax.random.PRNGKey(seed))


def test_interpolate_models_keeps_exact_endpoint_behavior() -> None:
    base = small_model(0)
    fine_tuned = small_model(1)
    ids = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])

    base_endpoint = interpolate_models(base, fine_tuned, 0.0)
    tuned_endpoint = interpolate_models(base, fine_tuned, 1.0)
    midpoint = interpolate_models(base, fine_tuned, 0.5)

    np.testing.assert_allclose(np.asarray(base_endpoint(ids)), np.asarray(base(ids)), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(tuned_endpoint(ids)), np.asarray(fine_tuned(ids)), rtol=0, atol=0)
    assert np.isfinite(np.asarray(midpoint(ids))).all()
