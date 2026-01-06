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

"""
Minimal Gated DeltaNet language model for data efficiency experiments.

This file mirrors the lightweight transformer skeleton used in speedrun templates,
but swaps attention for the Gated DeltaNet token mixer implemented in
`levanter.layers.gated_deltanet`.
"""

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import ScanCheckpointPolicy, Stacked
from haliax.state_dict import ModuleWithStateDictSerialization
from jaxtyping import PRNGKeyArray

from levanter.layers import RmsNormConfig
from levanter.layers.attention import AttentionMask
from levanter.layers.gated_deltanet import GatedDeltaNet, GatedDeltaNetConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.types import BlockFoldable


def _to_padding_mask(attn_mask: AttentionMask | NamedArray | None, pos_axis: Axis) -> NamedArray | None:
    """Convert an AttentionMask to a simple [Batch?, Pos] padding mask for GDN."""
    if attn_mask is None:
        return None

    if isinstance(attn_mask, AttentionMask):
        materialized = attn_mask.materialize(pos_axis, pos_axis)
    else:
        materialized = attn_mask

    if materialized is None or isinstance(materialized, AttentionMask):
        return None

    mask = materialized
    try:
        key_axis = mask.resolve_axis("key_position")
    except ValueError:
        key_axis = None

    if key_axis is not None:
        mask = hax.any(mask, axis=key_axis)

    return mask.astype(jnp.float32)


@LmConfig.register_subclass("gated_deltanet")
@dataclass(frozen=True)
class GatedDeltaNetTransformerConfig(LmConfig["GatedDeltaNetLMHeadModel"]):
    max_seq_len: int = 2048
    hidden_dim: int = 2048
    intermediate_dim: int = 8192
    num_layers: int = 24
    num_heads: int = 16
    head_dim: int | None = None
    num_k_heads: int | None = None
    num_v_heads: int | None = None

    conv_kernel_size: int = 4
    chunk_size: int = 64

    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    use_bias: bool = False
    use_layer_norm_weight: bool = True
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False
    gradient_checkpointing: bool | ScanCheckpointPolicy | str = True
    cross_entropy_block_size: int | None = 4096

    def __post_init__(self):
        k_heads = self.num_k_heads or self.num_heads
        v_heads = self.num_v_heads or self.num_heads
        if self.head_dim is None:
            assert self.hidden_dim % self.num_heads == 0, "hidden_dim % num_heads must be 0 when head_dim=None"
        assert v_heads % k_heads == 0, "num_v_heads must be divisible by num_k_heads"

    @property
    def model_type(self) -> type["GatedDeltaNetLMHeadModel"]:
        return GatedDeltaNetLMHeadModel

    Embed = property(lambda self: Axis("embed", self.hidden_dim))
    Layers = property(lambda self: Axis("layers", self.num_layers))
    Mlp = property(lambda self: Axis("mlp", self.intermediate_dim))

    @property
    def norm_config(self) -> RmsNormConfig:
        return RmsNormConfig(
            use_weight=self.use_layer_norm_weight, use_bias=self.use_bias, eps=self.layer_norm_epsilon
        )

    def mk_LayerNorm(self, axis: AxisSpec):
        return self.norm_config.build(axis)

    @property
    def head_k_dim(self) -> int:
        return self.head_dim or (self.hidden_dim // self.num_heads)

    @property
    def head_v_dim(self) -> int:
        return self.head_dim or (self.hidden_dim // self.num_heads)

    @property
    def k_heads(self) -> int:
        return self.num_k_heads or self.num_heads

    @property
    def v_heads(self) -> int:
        return self.num_v_heads or self.num_heads

    def gdn_config(self) -> GatedDeltaNetConfig:
        return GatedDeltaNetConfig(
            Embed=self.Embed,
            num_k_heads=self.k_heads,
            num_v_heads=self.v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_kernel_size=self.conv_kernel_size,
            rms_norm_eps=self.layer_norm_epsilon,
        )

    def total_trainable_params(self, vocab_size: int) -> int:
        key_dim = self.k_heads * self.head_k_dim
        value_dim = self.v_heads * self.head_v_dim

        qkvz_proj = self.hidden_dim * (2 * key_dim + 2 * value_dim)
        ba_proj = self.hidden_dim * (2 * self.v_heads)
        conv = (2 * key_dim + value_dim) * self.conv_kernel_size
        out_proj = value_dim * self.hidden_dim
        o_norm = self.head_v_dim
        discretization = 2 * self.v_heads  # A_log + dt_bias
        norms = 2 * self.hidden_dim  # pre-gdn and post-gdn
        mlp = 3 * self.hidden_dim * self.intermediate_dim

        per_layer = qkvz_proj + ba_proj + conv + out_proj + o_norm + discretization + norms + mlp
        transformer = self.num_layers * per_layer + self.hidden_dim  # final norm

        token_embedding = vocab_size * self.hidden_dim
        head = 0 if self.tie_word_embeddings else token_embedding

        return int(transformer + token_embedding + head)


class GatedDeltaNetMlp(eqx.Module):
    gate_proj: hnn.Linear
    up_proj: hnn.Linear
    down_proj: hnn.Linear
    act: ActivationFunctionEnum | Callable = eqx.field(static=True)

    @staticmethod
    def init(Embed: AxisSpec, Mlp: AxisSpec, activation_fn: ActivationFunctionEnum, *, key, use_bias: bool = False):
        k_fc, k_up, k_down = jrandom.split(key, 3)
        gate_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias, out_first=True)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up, use_bias=use_bias, out_first=True)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down, use_bias=use_bias, out_first=True)
        return GatedDeltaNetMlp(gate_proj, up_proj, down_proj, activation_fn)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        act_fn = self.act.to_fn() if isinstance(self.act, ActivationFunctionEnum) else self.act
        gated = act_fn(self.gate_proj(x, key=k_gate)) * self.up_proj(x, key=k_up)
        return self.down_proj(gated, key=k_down)


class GatedDeltaNetDecoderLayer(eqx.Module):
    config: GatedDeltaNetTransformerConfig = eqx.field(static=True)
    token_mixer: GatedDeltaNet
    mlp: GatedDeltaNetMlp
    input_layernorm: hnn.RmsNorm
    post_mixer_layernorm: hnn.RmsNorm

    @staticmethod
    def init(config: GatedDeltaNetTransformerConfig, *, key) -> "GatedDeltaNetDecoderLayer":
        k_gdn, k_mlp = jrandom.split(key, 2)
        token_mixer = GatedDeltaNet.init(config.gdn_config(), key=k_gdn)
        mlp = GatedDeltaNetMlp.init(
            config.Embed, config.Mlp, config.activation_function, key=k_mlp, use_bias=config.use_bias
        )
        ln1 = config.mk_LayerNorm(config.Embed)
        ln2 = config.mk_LayerNorm(config.Embed)
        return GatedDeltaNetDecoderLayer(config, token_mixer, mlp, ln1, ln2)

    @named_call
    def __call__(self, x: NamedArray, mask: NamedArray | None, *, key=None) -> NamedArray:
        k_gdn, k_mlp = maybe_rng_split(key, 2)

        residual = x
        gdn_out, _ = self.token_mixer(
            self.input_layernorm(x),
            inference=False,
            chunk_size=self.config.chunk_size,
            attention_mask=mask,
            decode_state=None,
        )
        x = residual + gdn_out

        residual = x
        mlp_out = self.mlp(self.post_mixer_layernorm(x), key=k_mlp)
        return residual + mlp_out


class GatedDeltaNetTransformer(eqx.Module):
    config: GatedDeltaNetTransformerConfig = eqx.field(static=True)
    layers: BlockFoldable[GatedDeltaNetDecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def init(config: GatedDeltaNetTransformerConfig, *, key):
        S = Stacked
        layers = S.init(config.Layers, GatedDeltaNetDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config, key=shaped_rng_split(key, config.num_layers)
        )
        return GatedDeltaNetTransformer(config, layers, config.mk_LayerNorm(config.Embed))

    @named_call
    def __call__(self, x: NamedArray, attn_mask: NamedArray | None, *, key=None) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys)
        return self.norm(x)


class GatedDeltaNetEmbedding(ModuleWithStateDictSerialization, eqx.Module):
    token_embeddings: hnn.Embedding
    norm: hnn.RmsNorm | None = None

    @staticmethod
    def init(Vocab: Axis, config: GatedDeltaNetTransformerConfig, *, key):
        embedding = hnn.Embedding.init(Vocab, config.Embed, key=key)
        return GatedDeltaNetEmbedding(embedding, None)

    @property
    def Vocab(self) -> Axis:
        return self.token_embeddings.Vocab

    @named_call
    def embed(self, input_ids: NamedArray):
        return self.token_embeddings(input_ids)


class GatedDeltaNetLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[GatedDeltaNetTransformerConfig]):
    transformer: GatedDeltaNetTransformer
    embeddings: GatedDeltaNetEmbedding
    lm_head: hnn.Linear | None

    @property
    def config(self) -> GatedDeltaNetTransformerConfig:
        return self.transformer.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: GatedDeltaNetTransformerConfig, *, key) -> "GatedDeltaNetLMHeadModel":
        k_t, k_e = jrandom.split(key, 2)
        transformer = GatedDeltaNetTransformer.init(config, key=k_t)
        embeddings = GatedDeltaNetEmbedding.init(Vocab, config, key=k_e)
        lm_head = (
            None
            if config.tie_word_embeddings
            else hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_e, use_bias=False, out_first=True)
        )
        return GatedDeltaNetLMHeadModel(transformer, embeddings, lm_head)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        del pos_ids  # GDN does not use explicit position IDs
        mask = _to_padding_mask(attn_mask, input_ids.resolve_axis("position"))
        hidden = self.transformer(self.embeddings.embed(input_ids), attn_mask=mask, key=key)
        return hidden

    def get_lm_head(self) -> hax.NamedArray:
        return self.embeddings.token_embeddings.weight if self.lm_head is None else self.lm_head.weight

    def resize_vocab(self, new_size: int, key: PRNGKeyArray | None = None) -> "GatedDeltaNetLMHeadModel":
        raise NotImplementedError("resize_vocab is not implemented for GatedDeltaNetLMHeadModel")
