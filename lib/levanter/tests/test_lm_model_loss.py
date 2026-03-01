# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax import Axis, NamedArray

from levanter.data.text.examples import grug_lm_example_from_named
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel


@dataclass(frozen=True)
class ToyLmConfig(LmConfig["ToyLmHeadModel"]):
    max_seq_len: int = 8
    embed_dim: int = 16

    @property
    def model_type(self) -> Type["ToyLmHeadModel"]:
        return ToyLmHeadModel

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.embed_dim)


class ToyLmHeadModel(LmHeadModel[ToyLmConfig]):
    _config: ToyLmConfig = eqx.field(static=True)
    _Vocab: Axis = eqx.field(static=True)
    embed_weight: NamedArray
    lm_head: NamedArray
    aux_loss: jax.Array

    @property
    def config(self) -> ToyLmConfig:
        return self._config

    @property
    def Vocab(self) -> Axis:
        return self._Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: ToyLmConfig, *, key: PRNGKeyArray) -> "ToyLmHeadModel":
        k_embed, k_head = jax.random.split(key, 2)
        embed_weight = hax.random.normal(k_embed, (Vocab, config.Embed), dtype=jnp.float32)
        lm_head = hax.random.normal(k_head, (config.Embed, Vocab), dtype=jnp.float32)
        return cls(config, Vocab, embed_weight, lm_head, jnp.array(0.0, dtype=jnp.float32))

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray | tuple[NamedArray, NamedArray | float]:
        del attn_mask, key, pos_ids
        hidden = self.embed_weight.take(self.Vocab, input_ids)
        return hidden, self.aux_loss

    def get_lm_head(self) -> NamedArray:
        return self.lm_head

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "ToyLmHeadModel":
        del key
        if new_size != self.Vocab.size:
            raise NotImplementedError("ToyLmHeadModel.resize_vocab only supports a no-op resize.")
        return self


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


def test_compute_next_token_loss_array_matches_named_for_grug_example():
    Vocab = Axis("vocab", 32)
    cfg = ToyLmConfig(max_seq_len=8, embed_dim=16)
    model = ToyLmHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))

    Batch = Axis("batch", 4)
    Pos = cfg.max_Pos.resize(8)
    example = _toy_example(Batch, Pos, Vocab, key=jax.random.PRNGKey(1))
    grug_example = grug_lm_example_from_named(example)

    named_loss = model.compute_next_token_loss(example, reduction=None, reduction_axis=()).array
    grug_loss = model.compute_next_token_loss_array(grug_example, batch_axis=Batch, reduction=None, reduction_axis=())

    np.testing.assert_allclose(grug_loss, named_loss, rtol=1e-5, atol=1e-6)


def test_logits_from_token_ids_array_matches_named_logits():
    Vocab = Axis("vocab", 32)
    cfg = ToyLmConfig(max_seq_len=8, embed_dim=16)
    model = ToyLmHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))

    Batch = Axis("batch", 4)
    Pos = cfg.max_Pos.resize(8)
    example = _toy_example(Batch, Pos, Vocab, key=jax.random.PRNGKey(1))

    activations = model.activations(example.tokens)
    if isinstance(activations, tuple):
        activations = activations[0]
    named_logits = hax.dot(activations, model.get_lm_head(), axis=model.Embed).array
    array_logits = model.logits_from_token_ids_array(example.tokens.array, batch_axis=Batch)

    np.testing.assert_allclose(array_logits, named_logits, rtol=1e-5, atol=1e-6)
