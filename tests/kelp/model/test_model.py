# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Kelp tree diffusion model."""

import jax.numpy as jnp
import pytest
from jax import random

from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.model.model import (
    TreeDiffusionModel,
    forward,
    init_parameters,
)


@pytest.fixture
def tiny_config():
    """Tiny config for fast testing."""
    return TreeDiffusionConfig(
        vocab_size=100,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=32,
        num_diffusion_steps=10,
    )


@pytest.fixture
def tiny_params(tiny_config):
    """Initialize tiny parameters."""
    key = random.PRNGKey(42)
    return init_parameters(tiny_config, key=key)


class TestTreeDiffusionConfig:
    def test_config_validation(self):
        config = TreeDiffusionConfig(vocab_size=100, hidden_dim=32, num_heads=4, num_kv_heads=4)
        assert config.inferred_head_dim == 8

    def test_config_invalid_heads(self):
        with pytest.raises(ValueError):
            TreeDiffusionConfig(vocab_size=100, hidden_dim=32, num_heads=5, num_kv_heads=5)  # 32 not divisible by 5


class TestInitParameters:
    def test_init_creates_all_params(self, tiny_config):
        key = random.PRNGKey(0)
        params = init_parameters(tiny_config, key=key)

        assert params.token_embed.shape == (tiny_config.vocab_size, tiny_config.hidden_dim)
        assert params.timestep_embed.shape == (tiny_config.num_diffusion_steps, tiny_config.hidden_dim)
        assert params.output_proj.shape == (tiny_config.hidden_dim, tiny_config.vocab_size)
        assert len(params.blocks) == tiny_config.num_layers

    def test_block_shapes(self, tiny_config):
        key = random.PRNGKey(0)
        params = init_parameters(tiny_config, key=key)
        block = params.blocks[0]

        head_dim = tiny_config.inferred_head_dim
        assert block.attn.w_q.shape == (tiny_config.hidden_dim, tiny_config.num_heads * head_dim)
        assert block.attn.w_k.shape == (tiny_config.hidden_dim, tiny_config.num_kv_heads * head_dim)
        assert block.mlp_gate.shape == (tiny_config.hidden_dim, tiny_config.intermediate_dim)


class TestForward:
    def test_forward_shape(self, tiny_config, tiny_params):
        batch_size = 2
        seq_len = 16
        tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        timesteps = jnp.array([5, 3])

        logits = forward(tiny_params, tokens, timesteps, tiny_config)

        assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)

    def test_forward_different_timesteps(self, tiny_config, tiny_params):
        batch_size = 4
        seq_len = 8
        tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        timesteps = jnp.array([0, 3, 6, 9])

        logits = forward(tiny_params, tokens, timesteps, tiny_config)

        assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)
        assert not jnp.allclose(logits[0], logits[1])


class TestTreeDiffusionModel:
    def test_model_init(self, tiny_config):
        key = random.PRNGKey(0)
        model = TreeDiffusionModel.init(tiny_config, key)

        assert model.config == tiny_config
        assert model.params is not None
        assert model.schedule is not None

    def test_model_call(self, tiny_config):
        key = random.PRNGKey(0)
        model = TreeDiffusionModel.init(tiny_config, key)

        tokens = jnp.zeros((2, 16), dtype=jnp.int32)
        timesteps = jnp.array([5, 3])

        logits = model(tokens, timesteps)

        assert logits.shape == (2, 16, tiny_config.vocab_size)
