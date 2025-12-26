# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.jax_utils
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.nn import RmsNorm
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig, LmWithHfSerializationMixin
from levanter.layers.attention import AttentionBackend, AttentionMask, dot_product_attention
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig, RotaryEmbeddings
from levanter.models.lm_model import LmConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()
from transformers import GPT2Config as HfGpt2Config  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@LmConfig.register_subclass("gpt2")
@dataclass(frozen=True)
class Gpt2Config(HFCompatConfig):
    max_seq_len: int = 1024
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12

    # how much to scale the embedding dim for the mlp layer
    mlp_scale: int = 4

    initializer_range: float = 0.02
    # dropout doesn't really help so we 0 it out by default
    embed_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-2
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.gelu_new

    # mistral tweaks:
    scale_attn_by_inverse_layer_idx: bool = False
    upcast_attn: bool = True

    gradient_checkpointing: bool = True  # better to just always use this

    use_bias: bool = False
    qk_norm: bool = True  # whether to apply layer norm to query and key vectors
    use_rms_norm: bool = True  # whether to use RMSNorm instead of LayerNorm for non-QK norms
    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)

    use_flash_attention: Optional[bool] = None
    attn_backend: Optional[AttentionBackend] = AttentionBackend.VANILLA
    flash_attention_block_size: Optional[int] = None

    # Axes
    @property
    def Embed(self) -> Axis:
        return Axis(name="embed", size=self.hidden_dim)

    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.hidden_dim * self.mlp_scale))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_dim // self.num_heads))

    @property
    def model_type(self) -> Type["Gpt2LMHeadModel"]:
        return Gpt2LMHeadModel

    def hf_checkpoint_converter(self, ref_checkpoint: Optional[str] = None) -> HFCheckpointConverter["Gpt2Config"]:  # type: ignore
        # We trust this code because it's in our hub repo
        return HFCheckpointConverter(self.__class__, reference_checkpoint="gpt2", ignore_prefix="transformer")

    def to_hf_config(self, vocab_size, config_overrides=None) -> HfGpt2Config:
        if config_overrides is None:
            config_overrides = {}

        return HfGpt2Config(
            vocab_size=vocab_size,
            n_positions=self.max_seq_len,
            n_layer=self.num_layers,
            n_head=self.num_heads,
            n_embd=self.hidden_dim,
            initializer_range=self.initializer_range,
            attn_pdrop=self.attn_pdrop,
            embd_pdrop=self.embed_pdrop,
            layer_norm_epsilon=self.layer_norm_epsilon,
            activation_function=self.activation_function,
            scale_attn_by_inverse_layer_idx=self.scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=self.upcast_attn,
            **config_overrides,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        return Gpt2Config(
            max_seq_len=hf_config.n_positions,
            # vocab_size=config.vocab_size,
            num_layers=hf_config.n_layer,
            num_heads=hf_config.n_head,
            hidden_dim=hf_config.n_embd,
            initializer_range=hf_config.initializer_range,
            attn_pdrop=hf_config.attn_pdrop,
            embed_pdrop=hf_config.embd_pdrop,
            layer_norm_epsilon=hf_config.layer_norm_epsilon,
            activation_function=ActivationFunctionEnum(hf_config.activation_function),
            scale_attn_by_inverse_layer_idx=hf_config.scale_attn_by_inverse_layer_idx,
            upcast_attn=hf_config.reorder_and_upcast_attn,
        )

    def flops_per_token(self, vocab_size: int, context_length: int) -> Optional[float]:
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.hidden_dim * self.mlp_scale,
            num_layers=self.num_layers,
            num_kv_heads=self.num_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=False,
        )

    def make_norm_layer(self, axis: Axis) -> Union[hnn.LayerNorm, RmsNorm]:
        """Create a normalization layer (LayerNorm or RmsNorm) based on config."""
        if self.use_rms_norm:
            return RmsNorm.init(axis, eps=self.layer_norm_epsilon, use_weight=True, use_bias=False)
        else:
            return hnn.LayerNorm.init(axis, eps=self.layer_norm_epsilon, use_bias=self.use_bias)


class Gpt2Mlp(eqx.Module):
    c_fc: hnn.Linear  # projection from Embed to Intermediate (typically 4x Embed)
    c_proj: hnn.Linear  # projection from Intermediate to Embed
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(
        Embed: Axis, Mlp: Axis, activation_fn: Union[ActivationFunctionEnum, Callable], *, key, use_bias: bool = False, initializer_range: float = 0.02
    ) -> "Gpt2Mlp":
        k_fc, k_proj = jrandom.split(key, 2)
        c_fc = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias, out_first=False, init_scale=math.sqrt(Embed.size) * initializer_range)
        c_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_proj, use_bias=use_bias, out_first=False, init_scale=math.sqrt(Mlp.size) * initializer_range)
        if isinstance(activation_fn, ActivationFunctionEnum):
            activation_fn = activation_fn.to_fn()

        return Gpt2Mlp(c_fc, c_proj, activation_fn)

    @named_call
    def __call__(self, x: NamedArray, *, key=None):
        k1, k2 = haliax.jax_utils.maybe_rng_split(key, 2)
        x = self.c_fc(x, key=k1)
        x = self.act(x)
        x = self.c_proj(x, key=k2)
        return x


class Gpt2Attention(eqx.Module):
    config: Gpt2Config = eqx.field(static=True)

    c_attn: hnn.Linear  # input projection from [embed] -> [(q, k, v), heads, head_dim]
    c_proj: hnn.Linear  # output projection from [heads, head_dim] -> [embed]
    q_norm: Optional[hnn.LayerNorm]  # query normalization
    k_norm: Optional[hnn.LayerNorm]  # key normalization
    rot_embs: Optional[RotaryEmbeddings]  # rotary position embeddings
    inference: bool

    @staticmethod
    def init(config: Gpt2Config, *, key) -> "Gpt2Attention":
        Qkv = Axis("qkv", size=3)
        use_bias = config.use_bias
        Embed = config.Embed

        k_c, k_proj = jrandom.split(key, 2)
        c_attn = hnn.Linear.init(
            In=Embed, Out=(Qkv, config.Heads, config.HeadSize), key=k_c, use_bias=use_bias, out_first=False, init_scale=math.sqrt(Embed.size) * config.initializer_range
        )
        c_proj = hnn.Linear.init(
            In=(config.Heads, config.HeadSize), Out=Embed, key=k_proj, use_bias=use_bias, out_first=False, init_scale=math.sqrt(config.Heads.size * config.HeadSize.size) * config.initializer_range
        )

        # Initialize QK normalization layers if enabled
        q_norm = None
        k_norm = None
        if config.qk_norm:
            q_norm = hnn.LayerNorm.init(config.HeadSize, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
            k_norm = hnn.LayerNorm.init(config.HeadSize, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        # Initialize RoPE embeddings
        rot_embs = config.rope.build(config.HeadSize) if config.rope is not None else None

        return Gpt2Attention(config, c_attn, c_proj, q_norm, k_norm, rot_embs, inference=False)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[AttentionMask | NamedArray], layer_idx, *, key, pos_ids: NamedArray | None = None):
        k_drop, k_attn, k_out = hax.jax_utils.maybe_rng_split(key, 3)
        qkv_out = self.c_attn(x, key=k_attn).rearrange((..., "qkv", "heads", "position", "head_size"))
        q, k, v = qkv_out.unbind("qkv")

        # Apply QK normalization if enabled
        if self.config.qk_norm and self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply rotary position embeddings if configured (before renaming axes)
        if self.rot_embs is not None:
            if pos_ids is None:
                pos_ids = hax.arange(x.resolve_axis("position"), dtype=jnp.int32)
            q = self.rot_embs(q, pos_ids)
            k = self.rot_embs(k, pos_ids)

        # Rename k and v's Pos as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        # mistral tweak: attention scores can overflow FP16, or just be too imprecise, so upcast to FP32
        if self.config.scale_attn_by_inverse_layer_idx:
            q = q / (layer_idx + 1.0)

        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask=mask,
            inference=self.inference,
            use_flash=self.config.use_flash_attention,
            attn_backend=self.config.attn_backend,
            flash_block_size=self.config.flash_attention_block_size,
            prng=k_drop,
            attention_dtype=jnp.float32 if self.config.upcast_attn else None,
        )

        attn_output = attn_output.astype(x.dtype)
        attn_output = self.c_proj(attn_output, key=k_out)

        return attn_output


class Gpt2Block(eqx.Module):
    ln_1: Union[hnn.LayerNorm, RmsNorm]
    attn: Gpt2Attention
    ln_2: Union[hnn.LayerNorm, RmsNorm]
    mlp: Gpt2Mlp
    resid_dropout: hnn.Dropout

    @staticmethod
    def init(config: Gpt2Config, *, key) -> "Gpt2Block":
        k_attn, k_mlp = jrandom.split(key, 2)

        ln_1 = config.make_norm_layer(config.Embed)
        attn = Gpt2Attention.init(config, key=k_attn)
        ln_2 = config.make_norm_layer(config.Embed)
        mlp = Gpt2Mlp.init(config.Embed, config.Mlp, config.activation_function, key=k_mlp, use_bias=config.use_bias, initializer_range=config.initializer_range)
        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)

        return Gpt2Block(ln_1, attn, ln_2, mlp, resid_dropout)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[AttentionMask | NamedArray], layer_idx, *, key, pos_ids: NamedArray | None = None):
        k1, k2, k3, k4 = haliax.jax_utils.maybe_rng_split(key, 4)

        attn_output = self.attn(self.ln_1(x), mask=mask, layer_idx=layer_idx, key=k1, pos_ids=pos_ids)
        attn_output = self.resid_dropout(attn_output, key=k2)
        x = x + attn_output

        ff_output = self.mlp(self.ln_2(x), key=k3)
        ff_output = self.resid_dropout(ff_output, key=k4)
        x = x + ff_output

        return x


class Gpt2Transformer(ModuleWithStateDictSerialization):
    config: Gpt2Config = eqx.field(static=True)
    blocks: Stacked[Gpt2Block]
    ln_f: Union[hnn.LayerNorm, RmsNorm]

    @staticmethod
    def init(config: Gpt2Config, *, key):
        # vectorize the blocks
        blocks = Stacked.init(config.Layers, Gpt2Block, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.make_norm_layer(config.Embed)

        return Gpt2Transformer(config, blocks, ln_f)

    @named_call
    def __call__(self, x: NamedArray, attn_mask: Optional[AttentionMask | NamedArray], *, key=None, pos_ids: NamedArray | None = None) -> NamedArray:
        keys = hax.jax_utils.maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.blocks.fold(x, attn_mask, hax.arange(self.config.Layers), key=keys, pos_ids=pos_ids)
        x = self.ln_f(x)

        return x

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"blocks": "h"}


class Gpt2Embeddings(ModuleWithStateDictSerialization, eqx.Module):
    Vocab: Axis = eqx.field(static=True)
    config: Gpt2Config = eqx.field(static=True)

    token_embeddings: hnn.Embedding
    dropout: hnn.Dropout

    @staticmethod
    def init(Vocab: Axis, config: Gpt2Config, *, key) -> "Gpt2Embeddings":
        token_embeddings = hnn.Embedding.init(
            Vocab, config.Embed, key=key, initializer_range=config.initializer_range
        )
        dropout = hnn.Dropout(pdrop=config.embed_pdrop)

        return Gpt2Embeddings(Vocab, config, token_embeddings, dropout)

    @named_call
    def embed(self, input_ids, *, key, pos_ids: NamedArray | None = None):
        input_embeds = self.token_embeddings(input_ids)
        x = self.dropout(input_embeds, key=key)
        return x

    def unembed(self, x: NamedArray):
        return hax.dot(x, self.token_embeddings.weight, axis="embed")

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "wte"}

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_token_embeddings = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, Vocab=self.Vocab.resize(new_size), token_embeddings=new_token_embeddings)


class Gpt2LMHeadModel(LmWithHfSerializationMixin[Gpt2Config]):
    transformer: Gpt2Transformer
    embeddings: Gpt2Embeddings

    @property
    def config(self):
        return self.transformer.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def Pos(self) -> Axis:
        return self.config.max_Pos

    @classmethod
    def init(cls, Vocab: Axis, config: Gpt2Config, *, key) -> "Gpt2LMHeadModel":
        k_t, k_embeddings = jrandom.split(key, 2)
        transformer = Gpt2Transformer.init(config, key=k_t)
        embeddings = Gpt2Embeddings.init(Vocab, config, key=k_embeddings)

        return Gpt2LMHeadModel(transformer, embeddings)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        k_embed, k_transformer = haliax.jax_utils.maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids, key=k_embed, pos_ids=pos_ids)
        x = self.transformer(x, attn_mask, key=k_transformer, pos_ids=pos_ids)

        return x

    def get_lm_head(self) -> hax.NamedArray:
        return self.embeddings.token_embeddings.weight

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "Gpt2LMHeadModel":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": None, "embeddings": None}
