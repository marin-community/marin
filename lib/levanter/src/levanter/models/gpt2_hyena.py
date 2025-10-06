# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""The GPT2 architecture, but with Hyena instead of Attention / Transformer."""

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Type

import equinox as eqx
import jax
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.jax_utils
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.layers.attention import AttentionMask
from levanter.models.gpt2 import Gpt2Embeddings, Gpt2Mlp
from levanter.models.hyena import HyenaConfig, HyenaOperator
from levanter.models.lm_model import LmConfig, LmHeadModel


@LmConfig.register_subclass("gpt2_hyena")
@dataclass(frozen=True)
class Gpt2HyenaConfig(LmConfig):
    num_layers: int = 12

    # how much to scale the embedding dim for the mlp layer
    mlp_scale: int = 4

    initializer_range: float = 0.02
    # dropout doesn't really help so we 0 it out by default
    embed_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5

    gradient_checkpointing: bool = True  # better to just always use this
    gradient_checkpointing_block_size: int = 5

    # NOTE: a bunch of things from here are used for the outer Gpt2 part of the architecture
    # as well (e.g. use_bias, layer_norm_epsilon, etc.). Some of these could in theory be
    # different between hyena and the outer Gpt2 part of the architecture, but keeping it simple
    # by having a single value for now.
    hyena: HyenaConfig = HyenaConfig()

    # Axes
    Pos = property(lambda self: self.hyena.Pos)
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: self.hyena.Embed)
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.Embed.size * self.mlp_scale))

    @property
    def model_type(cls) -> Type["Gpt2HyenaModel"]:
        return Gpt2HyenaModel

    def flops_per_token(self, vocab_size: int) -> Optional[float]:
        # TODO: implement
        return None


class Gpt2HyenaBlock(eqx.Module):
    config: Gpt2HyenaConfig = eqx.field(static=True)
    ln_1: hnn.LayerNorm
    hyena_operator: HyenaOperator
    ln_2: hnn.LayerNorm
    mlp: Gpt2Mlp
    resid_dropout: hnn.Dropout

    @staticmethod
    def init(config: Gpt2HyenaConfig, *, key) -> "Gpt2HyenaBlock":
        k_hyena, k_mlp = jrandom.split(key, 2)

        ln_1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.hyena.use_bias)

        hyena_operator = HyenaOperator.init(config.hyena, key=k_hyena)

        ln_2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.hyena.use_bias)
        mlp = Gpt2Mlp.init(
            config.Embed, config.Mlp, config.hyena.activation.to_fn(), key=k_mlp, use_bias=config.hyena.use_bias
        )
        resid_dropout = hnn.Dropout(pdrop=config.hyena.resid_pdrop)

        return Gpt2HyenaBlock(config, ln_1, hyena_operator, ln_2, mlp, resid_dropout)

    @named_call
    def __call__(self, x: NamedArray, *, key):
        k1, k2, k3, k4 = haliax.jax_utils.maybe_rng_split(key, 4)

        x_for_hyena = self.ln_1(x)
        hyena_output = self.hyena_operator(x_for_hyena, key=k1)
        hyena_output = self.resid_dropout(hyena_output, key=k2)
        x = x + hyena_output

        ff_output = self.mlp(self.ln_2(x), key=k3)
        ff_output = self.resid_dropout(ff_output, key=k4)
        x = x + ff_output

        return x


class Gpt2HyenaBackbone(ModuleWithStateDictSerialization):
    config: Gpt2HyenaConfig = eqx.field(static=True)
    blocks: Stacked[Gpt2HyenaBlock]
    ln_f: hnn.LayerNorm

    @staticmethod
    def init(config: Gpt2HyenaConfig, *, key):
        # vectorize the blocks
        blocks = Stacked.init(config.Layers, Gpt2HyenaBlock, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.hyena.use_bias)

        return Gpt2HyenaBackbone(config, blocks, ln_f)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        keys = hax.jax_utils.maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.blocks.fold(x, key=keys)
        x = self.ln_f(x)

        return x

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"blocks": "h"}


class Gpt2HyenaModel(LmHeadModel[Gpt2HyenaConfig]):
    backbone: Gpt2HyenaBackbone
    embeddings: Gpt2Embeddings

    @property
    def config(self):
        return self.backbone.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @classmethod
    def init(cls, Vocab: Axis, config: Gpt2HyenaConfig, *, key) -> "Gpt2HyenaModel":
        k_t, k_embeddings = jrandom.split(key, 2)
        backbone = Gpt2HyenaBackbone.init(config, key=k_t)
        embeddings = Gpt2Embeddings.init(
            Vocab,
            # Our config type has everything it needs, but is not a subclass of Gpt2Config
            config,  # type: ignore
            key=k_embeddings,
        )

        return Gpt2HyenaModel(backbone, embeddings)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        # NOTE: attn_mask not used since we use the Hyena operator instead of attention.
        k_embed, k_backbone = haliax.jax_utils.maybe_rng_split(key, 2)
        if pos_ids is None:
            pos_ids = hax.arange(input_ids.resolve_axis("position"), dtype=jax.numpy.int32)
        x = self.embeddings.embed(input_ids, pos_ids=pos_ids, key=k_embed)

        x = self.backbone(x, key=k_backbone)

        return x

    def get_lm_head(self) -> hax.NamedArray:
        return self.embeddings.token_embeddings.weight

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "Gpt2HyenaModel":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"backbone": None, "embeddings": None}
