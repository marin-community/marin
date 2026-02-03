# Copyright 2026 The Marin Authors
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
import jax.random as jrandom
import pytest

import haliax as hax

from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab
from experiments.kelp.tree_diffusion import (
    EditPredictor,
    TreeAttention,
    TreeDiffusionConfig,
    TreeDiffusionModel,
    TreeEmbedding,
    TreeEncoder,
    TreeMLP,
    TreeTransformerLayer,
)


@pytest.fixture
def config():
    node_vocab = PythonNodeVocab()
    value_vocab = PythonValueVocab()
    return TreeDiffusionConfig(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        max_nodes=32,
        max_depth=16,
        max_children=8,
        max_value_len=16,
        node_vocab_size=node_vocab.vocab_size,
        value_vocab_size=value_vocab.vocab_size,
    )


@pytest.fixture
def key():
    return jrandom.PRNGKey(42)


class TestTreeDiffusionConfig:
    def test_config_axes(self, config):
        assert config.Nodes.size == config.max_nodes
        assert config.Embed.size == config.hidden_dim
        assert config.Heads.size == config.num_heads
        assert config.HeadDim.size == config.hidden_dim // config.num_heads

    def test_config_vocab_axes(self, config):
        assert config.NodeVocab.size == config.node_vocab_size
        assert config.ValueVocab.size == config.value_vocab_size


class TestTreeEmbedding:
    def test_init(self, config, key):
        embedding = TreeEmbedding.init(config, key=key)
        assert embedding is not None

    def test_forward(self, config, key):
        embedding = TreeEmbedding.init(config, key=key)

        # Create dummy inputs
        node_types = hax.zeros(config.Nodes, dtype=jnp.int32)
        node_values = hax.zeros((config.Nodes, config.ValueLen), dtype=jnp.int32)
        depth = hax.zeros(config.Nodes, dtype=jnp.int32)

        output = embedding(node_types, node_values, depth, key=key)

        assert output.axes == (config.Nodes, config.Embed)


class TestTreeAttention:
    def test_init(self, config, key):
        attn = TreeAttention.init(config, key=key)
        assert attn is not None

    def test_forward(self, config, key):
        attn = TreeAttention.init(config, key=key)

        x = hax.zeros((config.Nodes, config.Embed), dtype=jnp.float32)
        mask = hax.ones(config.Nodes, dtype=jnp.int32)

        output = attn(x, mask=mask, key=key)

        assert output.axes == (config.Nodes, config.Embed)

    def test_forward_no_mask(self, config, key):
        attn = TreeAttention.init(config, key=key)

        x = hax.zeros((config.Nodes, config.Embed), dtype=jnp.float32)

        output = attn(x, mask=None, key=key)

        assert output.axes == (config.Nodes, config.Embed)


class TestTreeMLP:
    def test_init(self, config, key):
        mlp = TreeMLP.init(config, key=key)
        assert mlp is not None

    def test_forward(self, config, key):
        mlp = TreeMLP.init(config, key=key)

        x = hax.zeros((config.Nodes, config.Embed), dtype=jnp.float32)

        output = mlp(x, key=key)

        assert output.axes == (config.Nodes, config.Embed)


class TestTreeTransformerLayer:
    def test_init(self, config, key):
        layer = TreeTransformerLayer.init(config, key=key)
        assert layer is not None

    def test_forward(self, config, key):
        layer = TreeTransformerLayer.init(config, key=key)

        x = hax.zeros((config.Nodes, config.Embed), dtype=jnp.float32)
        mask = hax.ones(config.Nodes, dtype=jnp.int32)

        output = layer(x, mask=mask, key=key)

        assert output.axes == (config.Nodes, config.Embed)


class TestTreeEncoder:
    def test_init(self, config, key):
        encoder = TreeEncoder.init(config, key=key)
        assert encoder is not None
        assert len(encoder.layers) == config.num_layers

    def test_forward(self, config, key):
        encoder = TreeEncoder.init(config, key=key)

        node_types = hax.zeros(config.Nodes, dtype=jnp.int32)
        node_values = hax.zeros((config.Nodes, config.ValueLen), dtype=jnp.int32)
        depth = hax.zeros(config.Nodes, dtype=jnp.int32)
        mask = hax.ones(config.Nodes, dtype=jnp.int32)

        output = encoder(node_types, node_values, depth, mask=mask, key=key)

        assert output.axes == (config.Nodes, config.Embed)


class TestEditPredictor:
    def test_init(self, config, key):
        predictor = EditPredictor.init(config, key=key)
        assert predictor is not None

    def test_forward(self, config, key):
        predictor = EditPredictor.init(config, key=key)

        encoded = hax.zeros((config.Nodes, config.Embed), dtype=jnp.float32)
        mask = hax.ones(config.Nodes, dtype=jnp.int32)

        location_logits, type_logits, value_logits = predictor(encoded, mask=mask)

        assert location_logits.axes == (config.Nodes,)
        assert type_logits.axes == (config.NodeVocab,)
        assert value_logits.axes == (config.ValueLen, config.ValueVocab)


class TestTreeDiffusionModel:
    def test_init(self, config, key):
        model = TreeDiffusionModel.init(config, key=key)
        assert model is not None

    def test_forward(self, config, key):
        model = TreeDiffusionModel.init(config, key=key)

        node_types = hax.zeros(config.Nodes, dtype=jnp.int32)
        node_values = hax.zeros((config.Nodes, config.ValueLen), dtype=jnp.int32)
        depth = hax.zeros(config.Nodes, dtype=jnp.int32)
        mask = hax.ones(config.Nodes, dtype=jnp.int32)

        location_logits, type_logits, value_logits = model(node_types, node_values, depth, mask=mask, key=key)

        assert location_logits.axes == (config.Nodes,)
        assert type_logits.axes == (config.NodeVocab,)
        assert value_logits.axes == (config.ValueLen, config.ValueVocab)

    def test_forward_batched(self, config, key):
        """Test that model can handle batched inputs via vmap."""
        model = TreeDiffusionModel.init(config, key=key)

        Batch = hax.Axis("batch", 4)
        node_types = hax.zeros((Batch, config.Nodes), dtype=jnp.int32)
        node_values = hax.zeros((Batch, config.Nodes, config.ValueLen), dtype=jnp.int32)
        depth = hax.zeros((Batch, config.Nodes), dtype=jnp.int32)
        mask = hax.ones((Batch, config.Nodes), dtype=jnp.int32)

        # Don't pass key through vmap - use None for deterministic forward pass
        def single_forward(nt, nv, d, m):
            return model(nt, nv, d, mask=m, key=None)

        location_logits, type_logits, value_logits = hax.vmap(single_forward, Batch)(
            node_types, node_values, depth, mask
        )

        assert location_logits.axes == (Batch, config.Nodes)
        assert type_logits.axes == (Batch, config.NodeVocab)
        assert value_logits.axes == (Batch, config.ValueLen, config.ValueVocab)

    def test_gradients_flow(self, config, key):
        """Test that gradients can be computed through the model."""
        import equinox as eqx

        model = TreeDiffusionModel.init(config, key=key)

        node_types = hax.zeros(config.Nodes, dtype=jnp.int32)
        node_values = hax.zeros((config.Nodes, config.ValueLen), dtype=jnp.int32)
        depth = hax.zeros(config.Nodes, dtype=jnp.int32)
        mask = hax.ones(config.Nodes, dtype=jnp.int32)

        def loss_fn(model):
            loc, typ, val = model(node_types, node_values, depth, mask=mask, key=key)
            return hax.mean(loc).scalar() + hax.mean(typ).scalar() + hax.mean(val).scalar()

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

        # Loss should be a scalar
        assert loss.shape == ()

        # Gradients should exist
        assert grads is not None
