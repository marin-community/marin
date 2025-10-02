# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import jax.random as jrandom
import pytest

import haliax as hax
from haliax.nn import Embedding
from haliax.nn.mup import MupEmbedding
from haliax.state_dict import from_state_dict


@pytest.mark.parametrize("init_scale", [0.5, 1.0])
def test_mup_embedding_init_matches_embedding(init_scale: float):
    Vocab = hax.Axis("V", 8)
    Embed = (hax.Axis("E", 4),)

    key = jrandom.PRNGKey(0)

    baseline = Embedding.init(Vocab, Embed, key=key, init_scale=init_scale)
    mup = MupEmbedding.init(Vocab, Embed, key=key, init_scale=init_scale)

    assert mup.weight.axes == baseline.weight.axes
    assert jnp.allclose(mup.weight.array, baseline.weight.array)


def test_mup_embedding_unembedding_scale():
    Vocab = hax.Axis("V", 6)
    Embed = (hax.Axis("E", 3),)

    weight = hax.ones(hax.concat_axis_specs(Vocab, Embed))
    layer = MupEmbedding(weight=weight, Vocab=Vocab, Embed=Embed)

    assert layer.unembedding_mup_active_scale == pytest.approx(1.0 / Embed[0].size)
    assert jnp.allclose(layer.unembedding_weight.array, weight.array * layer.unembedding_mup_active_scale)

    Batch = hax.Axis("B", 2)
    inputs = hax.ones((Batch, *Embed))

    logits = layer.unembed(inputs)
    expected = hax.ones((Batch, Vocab))

    assert logits.axes == expected.axes
    assert jnp.allclose(logits.array, expected.array)


def test_mup_embedding_state_dict_adds_unembedding_weight():
    Vocab = hax.Axis("V", 5)
    Embed = (hax.Axis("E", 2),)

    weight = hax.random.normal(jrandom.PRNGKey(1), hax.concat_axis_specs(Vocab, Embed))
    layer = MupEmbedding(weight=weight, Vocab=Vocab, Embed=Embed)

    state = layer.to_state_dict()
    assert "unembedding_weight" in state
    assert jnp.allclose(state["unembedding_weight"], layer.unembedding_weight.array)

    template = MupEmbedding.init(Vocab, Embed, key=jrandom.PRNGKey(2))
    restored = from_state_dict(template, state)

    assert jnp.allclose(restored.weight.array, layer.weight.array)


def test_mup_embedding_initializer_range_warns():
    Vocab = hax.Axis("V", 3)
    Embed = (hax.Axis("E", 2),)

    with pytest.warns(DeprecationWarning):
        MupEmbedding.init(Vocab, Embed, key=jrandom.PRNGKey(3), initializer_range=0.5)
