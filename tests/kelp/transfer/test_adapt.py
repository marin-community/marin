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

"""Tests for Kelp AR-to-tree-diffusion transfer."""

import jax.numpy as jnp
import pytest
from jax import random

from experiments.kelp.transfer.adapt import (
    init_timestep_embeddings,
    map_llama_key,
)


class TestInitTimestepEmbeddings:
    def test_zeros_init(self):
        embed = init_timestep_embeddings(100, 256, method="zeros")
        assert embed.shape == (100, 256)
        assert jnp.allclose(embed, 0.0)

    def test_random_init(self):
        key = random.PRNGKey(0)
        embed = init_timestep_embeddings(100, 256, method="random", key=key)
        assert embed.shape == (100, 256)
        assert not jnp.allclose(embed, 0.0)

    def test_sinusoidal_init(self):
        embed = init_timestep_embeddings(100, 256, method="sinusoidal")
        assert embed.shape == (100, 256)
        assert jnp.all(jnp.abs(embed) <= 1.0)


class TestMapLlamaKey:
    def test_embed_tokens(self):
        component, sub, idx = map_llama_key("model.embed_tokens.weight")
        assert component == "embed"
        assert sub == "tokens"
        assert idx is None

    def test_lm_head(self):
        component, sub, idx = map_llama_key("lm_head.weight")
        assert component == "lm_head"
        assert idx is None

    def test_layer_attn(self):
        component, sub, idx = map_llama_key("model.layers.5.self_attn.q_proj.weight")
        assert component == "layers"
        assert idx == 5
        assert "q_proj" in sub

    def test_layer_mlp(self):
        component, sub, idx = map_llama_key("model.layers.10.mlp.gate_proj.weight")
        assert component == "layers"
        assert idx == 10
        assert "gate_proj" in sub

    def test_norm(self):
        component, sub, idx = map_llama_key("model.norm.weight")
        assert component == "norm"
        assert idx is None
