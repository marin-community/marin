# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis, NamedArray, NamedOrNumeric
from jaxtyping import PRNGKeyArray

from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmConfig, LmHeadModel


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
    ) -> NamedArray | tuple[NamedArray, NamedOrNumeric]:
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
