# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax import Axis

from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.testing.toy_lm import ToyLmConfig, ToyLmHeadModel


def _toy_example(Batch: Axis, Pos: Axis, Vocab: Axis, *, key: PRNGKeyArray) -> LmExample:
    tokens = hax.random.randint(key, (Batch, Pos), 0, Vocab.size)
    loss_weight = hax.ones((Batch, Pos), dtype=jnp.float32).at[Pos, Pos.size - 1].set(0.0)
    return LmExample(tokens=tokens, loss_weight=loss_weight, attn_mask=AttentionMask.causal())


def test_compute_next_token_loss_reduction_returns_scalar():
    Vocab = Axis("vocab", 32)
    cfg = ToyLmConfig(max_seq_len=8, embed_dim=16)
    model = ToyLmHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))

    Batch = Axis("batch", 4)
    Pos = cfg.max_Pos.resize(8)
    example = _toy_example(Batch, Pos, Vocab, key=jax.random.PRNGKey(1))

    loss = model.compute_next_token_loss(example)
    assert loss.axes == ()
    assert jnp.shape(loss.array) == ()


def test_compute_next_token_loss_unreduced_has_expected_axes():
    Vocab = Axis("vocab", 32)
    cfg = ToyLmConfig(max_seq_len=8, embed_dim=16)
    model = ToyLmHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))

    Batch = Axis("batch", 4)
    Pos = cfg.max_Pos.resize(8)
    example = _toy_example(Batch, Pos, Vocab, key=jax.random.PRNGKey(1))

    loss = model.compute_next_token_loss(example, reduction=None, reduction_axis=())
    assert isinstance(loss, hax.NamedArray)
    assert loss.resolve_axis("batch").size == Batch.size
    assert loss.resolve_axis("position").size == Pos.size


def test_compute_next_token_loss_includes_aux_loss():
    Vocab = Axis("vocab", 32)
    cfg = ToyLmConfig(max_seq_len=8, embed_dim=16)
    model = ToyLmHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))

    Batch = Axis("batch", 4)
    Pos = cfg.max_Pos.resize(8)
    example = _toy_example(Batch, Pos, Vocab, key=jax.random.PRNGKey(1))

    base = model.compute_next_token_loss(example)
    model_with_aux = eqx.tree_at(lambda m: m.aux_loss, model, jnp.array(0.25, dtype=jnp.float32))
    with_aux = model_with_aux.compute_next_token_loss(example)

    assert pytest.approx(float(base) + 0.25, rel=1e-6, abs=1e-6) == float(with_aux)
