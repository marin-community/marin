# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.grug_moe import MoEExpertMlp

from experiments.grug.moe.model import GrugModelConfig, paired_residual_expert_mlp


def test_paired_residual_experts_materialize_base_and_scaled_outer_weights() -> None:
    expert_mlp = MoEExpertMlp.init(
        num_experts=4,
        hidden_dim=2,
        intermediate_dim=3,
        initializer_std=0.02,
        key=jax.random.PRNGKey(0),
    )
    transformed = paired_residual_expert_mlp(expert_mlp)

    for original, effective in (
        (expert_mlp.w_gate, transformed.w_gate),
        (expert_mlp.w_up, transformed.w_up),
        (expert_mlp.w_down, transformed.w_down),
    ):
        assert original is not None
        assert effective is not None
        base, residual = jnp.split(original, 2, axis=0)
        expected = jnp.concatenate([base, (base + residual) / np.sqrt(2.0)], axis=0)
        np.testing.assert_allclose(effective, expected, rtol=1e-6, atol=1e-7)


def test_paired_residual_experts_require_even_expert_count() -> None:
    with pytest.raises(ValueError, match="even number of experts"):
        dataclasses.replace(
            GrugModelConfig(vocab_size=128, num_experts=4),
            num_experts=5,
            paired_expert_residuals=True,
        )
