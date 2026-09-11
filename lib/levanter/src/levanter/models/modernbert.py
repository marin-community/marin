# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""ModernBERT: a bidirectional encoder with alternating global/local attention.

This mirrors HuggingFace's ``ModernBertModel``/``ModernBertForMaskedLM`` closely
enough to round-trip state dicts. Notable architectural points handled here:

- Combined ``Wqkv`` attention projection and gated (GeGLU) ``Wi`` MLP.
- LayerNorm (no bias by default) rather than RMSNorm; an embedding LayerNorm and
  an ``Identity`` attention norm on the first layer.
- Alternating attention: every ``global_attn_every_n_layers``-th layer is global
  (rope theta ``global_rope_theta``, full attention); the rest are local with a
  symmetric sliding window of ``local_attention // 2`` and rope theta
  ``local_rope_theta``.
"""

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, AxisSpec, NamedArray
from haliax._src.state_dict import default_eqx_module_from_state_dict, default_eqx_module_to_state_dict, with_prefix
from haliax.jax_utils import maybe_rng_split, named_call
from haliax.state_dict import ModuleWithStateDictSerialization, StateDict

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.layers.attention import AttentionBackend, AttentionMask, dot_product_attention
from levanter.layers.normalization import LayerNormConfig
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddings
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag

silence_transformer_nag()
from transformers import ModernBertConfig as HfModernBertConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402

# transformers>=5 stores ModernBERT's per-layer attention pattern and RoPE thetas structurally
# (``layer_types`` + nested ``rope_parameters``) instead of the flat ``global_attn_every_n_layers`` /
# ``global_rope_theta`` / ``local_rope_theta`` kwargs. Bridge through the structural fields so the
# config round-trips on transformers 5 without relying on popped legacy attributes.
_HF_GLOBAL_LAYER = "full_attention"
_HF_LOCAL_LAYER = "sliding_attention"


def _hf_layer_types(num_layers: int, global_attn_every_n_layers: int) -> list[str]:
    """HF ``layer_types`` list: global (full) attention every ``global_attn_every_n_layers`` layers."""
    return [
        _HF_GLOBAL_LAYER if idx % global_attn_every_n_layers == 0 else _HF_LOCAL_LAYER for idx in range(num_layers)
    ]


def _hf_rope_parameters(global_rope_theta: float, local_rope_theta: float) -> dict[str, dict]:
    """HF ``rope_parameters`` mapping each layer type to its RoPE base wavelength."""
    return {
        _HF_GLOBAL_LAYER: {"rope_type": "default", "rope_theta": global_rope_theta},
        _HF_LOCAL_LAYER: {"rope_type": "default", "rope_theta": local_rope_theta},
    }


def _global_attn_every_n_layers(layer_types: list[str]) -> int:
    """Recover the global-attention period from HF ``layer_types`` (spacing between full-attention layers)."""
    global_indices = [idx for idx, layer_type in enumerate(layer_types) if layer_type == _HF_GLOBAL_LAYER]
    if len(global_indices) >= 2:
        return global_indices[1] - global_indices[0]
    return len(layer_types)


@LmConfig.register_subclass("modernbert")
@dataclass(frozen=True)
class ModernBertConfig(HFCompatConfig):
    """Config for ModernBERT.

    Defaults match ``answerdotai/ModernBERT-base``.
    """

    max_seq_len: int = 8192
    hidden_dim: int = 768
    intermediate_dim: int = 1152
    num_layers: int = 22
    num_heads: int = 12
    head_dim: int | None = None

    layer_norm_epsilon: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    decoder_bias: bool = True
    classifier_bias: bool = False

    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.gelu
    classifier_activation: ActivationFunctionEnum = ActivationFunctionEnum.gelu

    global_attn_every_n_layers: int = 3
    local_attention: int = 128  # full local-window width; the symmetric radius is half of this
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0

    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    pad_token_id: int = 50283

    # Attention is materialized with explicit masks (the local window is symmetric, which the
    # shared sliding-window backends do not express), so attention runs through the vanilla kernel.
    attn_backend: AttentionBackend = AttentionBackend.VANILLA
    upcast_attn: bool = False

    reference_checkpoint: str = "answerdotai/ModernBERT-base"
    tokenizer: Optional[str] = None

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def Heads(self) -> Axis:
        return Axis("heads", self.num_heads)

    @property
    def HeadSize(self) -> Axis:
        return Axis("head_size", self.head_dim if self.head_dim is not None else self.hidden_dim // self.num_heads)

    @property
    def Mlp(self) -> Axis:
        return Axis("mlp", self.intermediate_dim)

    def __post_init__(self):
        if self.hidden_dim % self.num_heads != 0 and self.head_dim is None:
            raise ValueError(f"hidden_dim={self.hidden_dim} not divisible by num_heads={self.num_heads}")
        if self.local_attention % 2 != 0:
            raise ValueError(f"local_attention must be even, got {self.local_attention}")

    def is_global_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.global_attn_every_n_layers == 0

    def rope_for_layer(self, layer_idx: int) -> DefaultRotaryEmbeddingsConfig:
        theta = self.global_rope_theta if self.is_global_layer(layer_idx) else self.local_rope_theta
        return DefaultRotaryEmbeddingsConfig(theta=theta)

    def sliding_window_for_layer(self, layer_idx: int) -> int | None:
        """Symmetric window radius for a layer, or None for global layers."""
        if self.is_global_layer(layer_idx):
            return None
        return self.local_attention // 2

    @property
    def norm_config(self) -> LayerNormConfig:
        return LayerNormConfig(eps=self.layer_norm_epsilon, use_weight=True, use_bias=self.norm_bias)

    def mk_LayerNorm(self, axis: AxisSpec) -> hnn.LayerNorm:
        return self.norm_config.build(axis)

    @property
    def model_type(self) -> Type["ModernBertForMaskedLM"]:  # pyrefly: ignore[bad-override]
        return ModernBertForMaskedLM

    def hf_checkpoint_converter(  # pyrefly: ignore[bad-override]
        self, ref_checkpoint: Optional[str] = None
    ) -> HFCheckpointConverter["ModernBertConfig"]:
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=False,
            tokenizer=ref_checkpoint if self.tokenizer is None else self.tokenizer,
            HfConfigClass=HfModernBertConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "ModernBertConfig":
        return ModernBertConfig(
            max_seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            layer_norm_epsilon=hf_config.norm_eps,
            norm_bias=hf_config.norm_bias,
            attention_bias=hf_config.attention_bias,
            mlp_bias=hf_config.mlp_bias,
            decoder_bias=hf_config.decoder_bias,
            classifier_bias=hf_config.classifier_bias,
            activation_function=ActivationFunctionEnum(hf_config.hidden_activation),
            classifier_activation=ActivationFunctionEnum(hf_config.classifier_activation),
            global_attn_every_n_layers=_global_attn_every_n_layers(hf_config.layer_types),
            local_attention=hf_config.local_attention,
            global_rope_theta=hf_config.rope_parameters[_HF_GLOBAL_LAYER]["rope_theta"],
            local_rope_theta=hf_config.rope_parameters[_HF_LOCAL_LAYER]["rope_theta"],
            initializer_range=hf_config.initializer_range,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            pad_token_id=hf_config.pad_token_id,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfModernBertConfig:
        if config_overrides is None:
            config_overrides = {}
        return HfModernBertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=self.max_seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            norm_eps=self.layer_norm_epsilon,
            norm_bias=self.norm_bias,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            decoder_bias=self.decoder_bias,
            classifier_bias=self.classifier_bias,
            hidden_activation=self.activation_function.value,
            classifier_activation=self.classifier_activation.value,
            layer_types=_hf_layer_types(self.num_layers, self.global_attn_every_n_layers),
            local_attention=self.local_attention,
            rope_parameters=_hf_rope_parameters(self.global_rope_theta, self.local_rope_theta),
            initializer_range=self.initializer_range,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
            **config_overrides,
        )

    def flops_per_token(self, vocab_size: int, context_length: int) -> Optional[float]:
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=True,
        )


def _resolve_mask(mask: AttentionMask | NamedArray | None, radius: int | None) -> AttentionMask | NamedArray | None:
    """Combine the incoming mask with a layer's symmetric local window (None for global layers).

    Using the structured ``bidirectional_window`` (rather than a materialized mask) lets the same
    code run on the vanilla kernel and the TPU splash kernel, which turns the local window into an
    O(seq * window) operation instead of O(seq^2).
    """
    if radius is None:
        return mask
    window = AttentionMask.bidirectional_sliding_window(radius)
    if mask is None:
        return window
    if isinstance(mask, AttentionMask):
        return mask & window
    return AttentionMask.explicit(mask) & window


class ModernBertEmbeddings(ModuleWithStateDictSerialization):
    tok_embeddings: hnn.Embedding
    norm: hnn.LayerNorm

    @staticmethod
    def init(Vocab: Axis, config: ModernBertConfig, *, key) -> "ModernBertEmbeddings":
        tok_embeddings = hnn.Embedding.init(Vocab, config.Embed, key=key)
        norm = config.mk_LayerNorm(config.Embed)
        return ModernBertEmbeddings(tok_embeddings, norm)

    @property
    def Vocab(self) -> Axis:
        return self.tok_embeddings.Vocab

    @named_call
    def embed(self, input_ids: NamedArray) -> NamedArray:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertAttention(ModuleWithStateDictSerialization):
    config: ModernBertConfig = eqx.field(static=True)
    sliding_window: int | None = eqx.field(static=True)
    Wqkv: hnn.Linear
    Wo: hnn.Linear
    rot_embs: RotaryEmbeddings

    @staticmethod
    def init(config: ModernBertConfig, layer_idx: int, *, key) -> "ModernBertAttention":
        k_qkv, k_o = jrandom.split(key, 2)
        Qkv = Axis("qkv", 3)
        Wqkv = hnn.Linear.init(
            In=config.Embed,
            Out=(Qkv, config.Heads, config.HeadSize),
            key=k_qkv,
            use_bias=config.attention_bias,
            out_first=True,
        )
        Wo = hnn.Linear.init(
            In=(config.Heads, config.HeadSize),
            Out=config.Embed,
            key=k_o,
            use_bias=config.attention_bias,
            out_first=True,
        )
        rot_embs = config.rope_for_layer(layer_idx).build(config.HeadSize)
        return ModernBertAttention(config, config.sliding_window_for_layer(layer_idx), Wqkv, Wo, rot_embs)

    @named_call
    def __call__(
        self, x: NamedArray, mask: AttentionMask | NamedArray | None, *, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        Pos = x.resolve_axis("position")
        qkv = self.Wqkv(x)
        q = qkv["qkv", 0]
        k = qkv["qkv", 1]
        v = qkv["qkv", 2]

        if pos_ids is None:
            pos_ids = hax.arange(Pos)
        q = self.rot_embs(q, pos_ids).astype(q.dtype)
        k = self.rot_embs(k, pos_ids).astype(k.dtype)

        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        resolved = _resolve_mask(mask, self.sliding_window)
        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            resolved,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            attn_backend=self.config.attn_backend,
        )
        attn_output = attn_output.astype(x.dtype)
        return self.Wo(attn_output)


class ModernBertMlp(ModuleWithStateDictSerialization):
    Wi: hnn.Linear
    Wo: hnn.Linear
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(config: ModernBertConfig, *, key) -> "ModernBertMlp":
        k_in, k_out = jrandom.split(key, 2)
        Glu = Axis("glu", 2)
        Wi = hnn.Linear.init(
            In=config.Embed, Out=(Glu, config.Mlp), key=k_in, use_bias=config.mlp_bias, out_first=True
        )
        Wo = hnn.Linear.init(In=config.Mlp, Out=config.Embed, key=k_out, use_bias=config.mlp_bias, out_first=True)
        return ModernBertMlp(Wi, Wo, config.activation_function.to_fn())

    @named_call
    def __call__(self, x: NamedArray) -> NamedArray:
        h = self.Wi(x)
        return self.Wo(self.act(h["glu", 0]) * h["glu", 1])


class ModernBertEncoderLayer(ModuleWithStateDictSerialization):
    attn_norm: Optional[hnn.LayerNorm]
    attn: ModernBertAttention
    mlp_norm: hnn.LayerNorm
    mlp: ModernBertMlp

    @staticmethod
    def init(config: ModernBertConfig, layer_idx: int, *, key) -> "ModernBertEncoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)
        # The first layer's input is already normalized by the embedding LayerNorm (HF uses Identity here).
        attn_norm = None if layer_idx == 0 else config.mk_LayerNorm(config.Embed)
        attn = ModernBertAttention.init(config, layer_idx, key=k_attn)
        mlp_norm = config.mk_LayerNorm(config.Embed)
        mlp = ModernBertMlp.init(config, key=k_mlp)
        return ModernBertEncoderLayer(attn_norm, attn, mlp_norm, mlp)

    @named_call
    def __call__(
        self, x: NamedArray, mask: AttentionMask | NamedArray | None, *, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        attn_in = x if self.attn_norm is None else self.attn_norm(x)
        x = x + self.attn(attn_in, mask, pos_ids=pos_ids)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ModernBertEncoder(ModuleWithStateDictSerialization):
    config: ModernBertConfig = eqx.field(static=True)
    embeddings: ModernBertEmbeddings
    layers: list  # list[ModernBertEncoderLayer]; a list serializes as layers.{i}
    final_norm: hnn.LayerNorm

    @staticmethod
    def init(Vocab: Axis, config: ModernBertConfig, *, key) -> "ModernBertEncoder":
        k_emb, k_layers = jrandom.split(key, 2)
        embeddings = ModernBertEmbeddings.init(Vocab, config, key=k_emb)
        layer_keys = jrandom.split(k_layers, config.num_layers)
        layers = [ModernBertEncoderLayer.init(config, i, key=layer_keys[i]) for i in range(config.num_layers)]
        final_norm = config.mk_LayerNorm(config.Embed)
        return ModernBertEncoder(config, embeddings, layers, final_norm)

    @named_call
    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attn_mask, pos_ids=pos_ids)
        return self.final_norm(x)


class ModernBertPredictionHead(ModuleWithStateDictSerialization):
    dense: hnn.Linear
    norm: hnn.LayerNorm
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(config: ModernBertConfig, *, key) -> "ModernBertPredictionHead":
        dense = hnn.Linear.init(
            In=config.Embed,
            Out=config.Embed.alias("head_embed"),
            key=key,
            use_bias=config.classifier_bias,
            out_first=True,
        )
        norm = config.mk_LayerNorm(config.Embed)
        return ModernBertPredictionHead(dense, norm, config.classifier_activation.to_fn())

    @named_call
    def __call__(self, x: NamedArray) -> NamedArray:
        h = self.act(self.dense(x)).rename({"head_embed": "embed"})
        return self.norm(h)


class ModernBertForMaskedLM(ModuleWithStateDictSerialization, LmHeadModel[ModernBertConfig]):
    model: ModernBertEncoder
    head: ModernBertPredictionHead
    decoder: hnn.Linear

    @property
    def config(self) -> ModernBertConfig:
        return self.model.config

    @property
    def Vocab(self) -> Axis:
        return self.model.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: ModernBertConfig, *, key) -> "ModernBertForMaskedLM":
        k_model, k_head, k_dec = jrandom.split(key, 3)
        model = ModernBertEncoder.init(Vocab, config, key=k_model)
        head = ModernBertPredictionHead.init(config, key=k_head)
        decoder = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_dec, use_bias=config.decoder_bias, out_first=True)
        return ModernBertForMaskedLM(model, head, decoder)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        return self.model(input_ids, attn_mask, pos_ids=pos_ids)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        x = self.activations(input_ids, attn_mask, pos_ids=pos_ids)
        return self.decoder(self.head(x))

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "ModernBertForMaskedLM":
        # ModernBERT ties the decoder weight to the token embeddings; tied checkpoints omit
        # ``decoder.weight`` (but still carry ``decoder.bias``), so reconstruct it from the embeddings.
        decoder_weight_key = with_prefix(prefix, "decoder.weight")
        if self.config.tie_word_embeddings and decoder_weight_key not in state_dict:
            embed_key = with_prefix(prefix, "model.embeddings.tok_embeddings.weight")
            state_dict = {**state_dict, decoder_weight_key: state_dict[embed_key]}
        return default_eqx_module_from_state_dict(self, state_dict, prefix)

    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        state_dict = default_eqx_module_to_state_dict(self, prefix)
        if self.config.tie_word_embeddings:
            state_dict.pop(with_prefix(prefix, "decoder.weight"), None)
        return state_dict

    def get_lm_head(self) -> NamedArray:
        return self.decoder.weight

    def resize_vocab(self, new_size: int, key=None) -> "ModernBertForMaskedLM":
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = dataclasses.replace(
            self.model.embeddings,
            tok_embeddings=self.model.embeddings.tok_embeddings.resize_embeddings(new_size, key=k1),
        )
        new_model = dataclasses.replace(self.model, embeddings=new_embeddings)
        new_Vocab = self.Vocab.resize(new_size)
        new_weight = hax.tree_util.resize_axis(self.decoder.weight, self.Vocab, new_size, key=k2)
        new_decoder = dataclasses.replace(self.decoder, Out=new_Vocab, weight=new_weight)
        if self.decoder.bias is not None:
            new_bias = hax.tree_util.resize_axis(self.decoder.bias, self.Vocab, new_size, key=k2)
            new_decoder = dataclasses.replace(new_decoder, bias=new_bias)
        return dataclasses.replace(self, model=new_model, decoder=new_decoder)


__all__ = [
    "ModernBertConfig",
    "ModernBertEncoder",
    "ModernBertForMaskedLM",
    "ModernBertPredictionHead",
]
