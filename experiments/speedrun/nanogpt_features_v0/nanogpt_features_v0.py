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
Hackable transformer training speedrun sweep (template)

This file is intentionally self-contained:
- Defines a compact, Llama-ish transformer that implements Levanter's LmHeadModel
- Provides a ready-to-run speedrun sweep across multiple model sizes

How to run:
  1) Set env vars (WANDB_API_KEY, HF_TOKEN, etc.) as in the tutorial:
     https://marin.readthedocs.io/en/latest/tutorials/submitting-speedrun/
  2) From repo root:
       python marin/run/ray_run.py -- python -m experiments.speedrun.run1.main --force_run_failed true
  3) Optional: SR_USE_TPU=1 to use TPU resource presets (default is GPU).
"""

# =========================
# Submission metadata
# TODO: fill out your information when you start
# =========================

SUBMISSION_BRANCH = "nanogpt_features_v0"
SUBMISSION_DESCRIPTION = "Includes subset of features from Modded-NanoGPT: Partial RoPE, QK Norm, 2.5 TPP, Relu^2 MLP, X0 Skip, exponential decay of resid, backout lambda, reduced head counts, rms_norm, 0 init out projections, boosted attn scale"
SUBMISSION_AUTHOR_NAME = "Larry Dial"
SUBMISSION_AUTHOR_AFFILIATION = "Independent"
SUBMISSION_AUTHOR_URL = "https://github.com/ClassicLarry"

# ruff: noqa: E402
# nodryrun
import dataclasses
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union, cast, overload

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from fray.cluster import ResourceConfig
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.normalization import LayerNormBase
from haliax.nn.scan import ScanCheckpointPolicy, Stacked
from haliax.state_dict import ModuleWithStateDictSerialization
from jaxtyping import PRNGKeyArray
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.layers import LayerNormConfigBase, RmsNormConfig
from levanter.layers.attention import Attention, AttentionBackend, AttentionConfig, AttentionMask, AttentionWithSink, dot_product_attention
from levanter.layers.kv_cache import KvPageCache
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig, _rotate_half
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.optim import MuonConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag
from levanter.utils.types import BlockFoldable
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger("ray")

_IMPORT_PATH = getattr(__spec__, "name", __name__)

silence_transformer_nag()


# =========================
# Hackable config & modules
# TODO: make any model architecture changes
# =========================


@LmConfig.register_subclass("hackable_transformer")
@dataclass(frozen=True)
class HackableTransformerConfig(LmConfig["HackableLMHeadModel"]):
    # Core dims
    seq_len: int = 2048
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int | None = None

    # activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    activation_function = 'relu squared'
    use_bias: bool = False
    use_layer_norm_weight: bool = False # True -> False
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False
    input_embedding_norm: bool = True # False -> True

    # Attention
    use_attention_sink: bool = False
    upcast_attn: bool = False
    attn_backend: AttentionBackend | None = None
    flash_attention_block_size: int | None = None
    qk_norm: LayerNormConfigBase | None = RmsNormConfig(use_weight=use_layer_norm_weight, use_bias=use_bias, eps=layer_norm_epsilon) # None -> RmsNormConfig  # set to RmsNormConfig(...) to enable
    
    gradient_checkpointing: bool | ScanCheckpointPolicy | str = True
    initializer_range: float = 0.02
    reference_checkpoint: str = "NousResearch/Llama-2-7b-hf"
    tokenizer: str | None = None

    def __post_init__(self):
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        if self.head_dim is None:
            assert self.hidden_dim % self.num_heads == 0, "hidden_dim % num_heads must be 0 when head_dim=None"

    @property
    def num_context_layers(self) -> int:
        return 2 * self.num_layers // 3

    @property
    def scaling_factor(self) -> int:
        # boost attn scale factor by 1.35 to match nanogpt
        return 1.35 / (self.hidden_dim//self.num_heads) ** 0.5

    @property
    def num_prediction_layers(self) -> int:
        return self.num_layers - self.num_context_layers
    
    # ---- LmConfig API ----
    @property
    def model_type(self) -> type["HackableLMHeadModel"]:
        return HackableLMHeadModel

    Pos = property(lambda self: Axis("position", self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis("embed", self.hidden_dim))
    AttnGate = property(lambda self: Axis("attn_gate", 12))
    Layers = property(lambda self: Axis("layers", self.num_layers))
    ContextLayers = property(lambda self: Axis("context_layers", self.num_context_layers))
    PredictionLayers = property(lambda self: Axis("prediction_layers", self.num_prediction_layers))
    Mlp = property(lambda self: Axis("mlp", self.intermediate_dim))

    @property
    def norm_config(self) -> LayerNormConfigBase:
        return RmsNormConfig(use_weight=self.use_layer_norm_weight, use_bias=self.use_bias, eps=self.layer_norm_epsilon)

    def mk_LayerNorm(self, axis: AxisSpec):
        return self.norm_config.build(axis)

    def attention_config(self) -> AttentionConfig:
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            qk_norm=self.qk_norm,
            scaling_factor = self.scaling_factor
        )

    @property
    def actual_head_size(self) -> int:
        return self.head_dim or (self.hidden_dim // self.num_heads)

    def flops_per_token(self, vocab_size: int) -> float | None:
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            vocab_size=vocab_size,
            glu=False,
        )

    def total_trainable_params(self, vocab_size: int) -> int:
        token_embedding = vocab_size * self.hidden_dim
        hs = self.actual_head_size
        attn = (
            self.hidden_dim * hs * self.num_heads
            + 2 * self.hidden_dim * hs * self.num_kv_heads
            + hs * self.num_heads * self.hidden_dim
        )
        mlp = 2 * self.hidden_dim * self.intermediate_dim
        transformer = self.num_layers * (attn + mlp)

        head = 0 if self.tie_word_embeddings else token_embedding
        return int(transformer + token_embedding + head)


class HackableRope(eqx.Module):
    """Partial RoPE over 50% of head dims"""
    HeadDim: Axis = eqx.field(static=True)
    def __call__(self, q: NamedArray, position_ids: NamedArray) -> NamedArray:
        with jax.ensure_compile_time_eval():
            theta = 1024
            factor = 1.0
            # fraction of head dimensions to apply RoPE 
            partial_rotary_factor = 0.5
            
            rotated_size = int(self.HeadDim.size * partial_rotary_factor)
            HeadHalfSize = self.HeadDim.resize(self.HeadDim.size // 2)
            inv_freq = 1.0 / (theta ** (hax.arange(HeadHalfSize, step=2) / rotated_size))
            rotated_half_size = rotated_size // 2
            inv_freq = hax.where(hax.arange(HeadHalfSize) < rotated_half_size,inv_freq,0.0)
            inv_freq = inv_freq / factor

        freqs = inv_freq.broadcast_axis(position_ids.axes) * position_ids
        emb = hax.concatenate(self.HeadDim, (freqs, freqs))
        cos = hax.cos(emb).astype(q.dtype)
        sin = hax.sin(emb).astype(q.dtype)

        q_embed = q * cos + _rotate_half(q, self.HeadDim) * sin
        return q_embed


class HackableAttention(eqx.Module):
    """Standard Attn for now. Calls HackableRope"""

    config: AttentionConfig = eqx.field(static=True)
    q_proj: hnn.Linear
    k_proj: hnn.Linear
    v_proj: hnn.Linear
    o_proj: hnn.Linear
    q_norm: Optional[LayerNormBase] = None
    k_norm: Optional[LayerNormBase] = None
    rot_embs: Optional[HackableRope] = None

    @staticmethod
    def init(config: AttentionConfig, *, key) -> "Attention":
        use_bias = config.use_bias
        use_output_bias = config.use_output_bias if config.use_output_bias is not None else use_bias
        k_q, k_k, k_v, k_o, k_g = jrandom.split(key, 5)
        q_proj = hnn.Linear.init(
            In=config.Embed,
            Out=(config.KVHeads, config.QHeadsPerGroup, config.HeadSize),
            key=k_q,
            use_bias=use_bias,
            out_first=True,
        )
        k_proj = hnn.Linear.init(
            In=config.Embed,
            Out=(config.KVHeads, config.HeadSize),
            key=k_k,
            use_bias=use_bias,
            out_first=True,
        )
        v_proj = hnn.Linear.init(
            In=(config.Embed),
            Out=(config.KVHeads, config.HeadSize),
            key=k_v,
            use_bias=use_bias,
            out_first=True,
        )
        o_proj = hnn.Linear.init(
            In=(config.Heads, config.HeadSize),
            Out=config.Embed,
            key=k_o,
            use_bias=use_output_bias,
            out_first=True,
        )
        o_proj = eqx.tree_at(lambda m: m.weight, o_proj, o_proj.weight * 0)

        q_norm = None
        k_norm = None
        if config.qk_norm is not None:
            q_norm = config.qk_norm.build(config.HeadSize)
            k_norm = config.qk_norm.build(config.HeadSize)

        # Build rotary embeddings once during initialization if configured
        rot_embs = HackableRope(config.HeadSize)

        return HackableAttention(config, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rot_embs)

    def empty_page_cache(self, spec: PageTableSpec, *, dtype) -> "KvPageCache":
        return KvPageCache.init(spec, self.config.KVHeads, self.config.HeadSize, dtype=dtype)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[NamedArray | AttentionMask],
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        key_proj, key_o, key_gate = maybe_rng_split(key, 3)

        # Shared computation of q, k, v
        q, k, v = self._compute_qkv(x, key=key_proj, pos_ids=pos_ids)

        # Reshape for attention kernels (convert embed → heads/head_size)
        q = q.rearrange((..., "kv_head", "q_heads_per_group", "position", "head_size"))
        k = k.rearrange((..., "kv_head", "position", "head_size"))
        v = v.rearrange((..., "kv_head", "position", "head_size"))

        # Distinguish key sequence axis for attention
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        # Apply attention
        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            attn_backend=self.config.attn_backend,
            flash_block_size=self.config.flash_attention_block_size,
            scaling_factor=self.config.scaling_factor,
            logits_soft_cap=self.config.logits_soft_cap,
            inference=True,
            prng=key,
        )
        # Flatten heads and apply output projection
        attn_output = attn_output.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)
        attn_output = self.o_proj(attn_output, key=key_o)

        return attn_output

    # Note: the non-paged decode path has been removed. Use paged_decode.

    @named_call
    @jax.profiler.annotate_function
    def paged_decode(
        self,
        x: NamedArray,
        kv_cache: "KvPageCache",
        batch_info: PageBatchInfo,
        *,
        pos_ids: NamedArray,
        key=None,
    ) -> tuple[NamedArray, "KvPageCache"]:
        """Decode-time forward pass using a paged KV cache.

        This method is intended for autoregressive decoding and prefill.  ``batch_info``
        describes where the new keys and values should be written in ``kv_cache``.
        Currently only causal masks are supported.
        """

        key_proj, key_o, key_gate = maybe_rng_split(key, 3)

        q, k, v = self._compute_qkv(x, key=key_proj, pos_ids=pos_ids)

        kv_cache = kv_cache.update(batch_info, k, v)

        sm_scale = (
            self.config.scaling_factor
            if self.config.scaling_factor is not None
            else 1.0 / math.sqrt(self.config.HeadSize.size)
        )

        attn_tokens = ragged_paged_attention(
            q,
            kv_cache.kv_pages,
            batch_info.seq_lens,
            batch_info.page_indices,
            batch_info.cu_q_lens,
            batch_info.num_seqs,
            sm_scale=sm_scale,
            soft_cap=self.config.logits_soft_cap,
        )
        attn_output = attn_tokens.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)
        attn_output = self.o_proj(attn_output, key=key_o)

        return attn_output, kv_cache

    @named_call
    def _compute_qkv(
        self,
        x: NamedArray,
        *,
        key,
        pos_ids: NamedArray | None = None,
    ) -> tuple[NamedArray, NamedArray, NamedArray]:
        """Project *x* to Q, K and V and apply all per-head processing."""

        # Split the projection key into three – one for each of Q, K, V
        key_q, key_k, key_v = maybe_rng_split(key, 3)

        # Linear projections
        q = self.q_proj(x, key=key_q)
        k = self.k_proj(x, key=key_k)
        v = self.v_proj(x, key=key_v)

        # Optional QK layer-norm
        if self.config.qk_norm is not None:
            q = self.q_norm(q)  # type: ignore[misc]
            k = self.k_norm(k)  # type: ignore[misc]

        # Apply rotary embeddings if configured
        if self.rot_embs is not None:
            if pos_ids is None:
                pos_ids = hax.arange(x.resolve_axis("position"))
            q = self.rot_embs(q, pos_ids).astype(q.dtype)
            k = self.rot_embs(k, pos_ids).astype(k.dtype)

        return q, k, v

class HackableMlp(eqx.Module):
    """MLP RELU^2"""
    up_proj: hnn.Linear
    down_proj: hnn.Linear

    @staticmethod
    def init(Embed: AxisSpec, Mlp: AxisSpec, *, key, use_bias=False):
        k_up_proj, k_down_proj = jrandom.split(key, 2)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up_proj, use_bias=use_bias, out_first=True)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down_proj, use_bias=use_bias, out_first=True)
        down_proj = eqx.tree_at(lambda m: m.weight, down_proj, down_proj.weight * 0)
        return HackableMlp(up_proj, down_proj)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_up, k_down = maybe_rng_split(key, 2)
        h = hax.square(hax.nn.relu(self.up_proj(x, key=k_up)))
        return self.down_proj(h, key=k_down)


class HackableDecoderLayer(eqx.Module):
    """One transformer block."""

    config: HackableTransformerConfig = eqx.field(static=True)
    self_attn: Attention | AttentionWithSink
    mlp: HackableMlp
    input_layernorm: hnn.RmsNorm
    post_attention_layernorm: hnn.RmsNorm
    x_lambda: NamedArray
    x0_lambda: NamedArray
    post_attn_layernorm: hnn.RmsNorm | None = None
    post_mlp_layernorm: hnn.RmsNorm | None = None

    @staticmethod
    def init(config: HackableTransformerConfig, *, key) -> "HackableDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)
        attn_cfg = config.attention_config()
        attn = HackableAttention.init(attn_cfg, key=k_attn)
        mlp = HackableMlp.init(config.Embed, config.Mlp, key=k_mlp, use_bias=config.use_bias)
        input_layernorm = config.mk_LayerNorm(config.Embed)
        post_attention_layernorm = config.mk_LayerNorm(config.Embed)

        x_lambda = hax.ones(()) * 1.1 # init for this param may want to be based on num_layers
        x0_lambda = hax.zeros(())
        return HackableDecoderLayer(config, attn, mlp, input_layernorm, post_attention_layernorm, x_lambda, x0_lambda)

    @named_call
    def __call__(
        self, x: NamedArray, x0: NamedArray, mask: NamedArray | AttentionMask | None, *, key=None, pos_ids: NamedArray | None = None
    ):
        k_attn, k_mlp = maybe_rng_split(key, 2)
        x = self.x_lambda * x + self.x0_lambda * x0
        x = x + self.self_attn(x=self.input_layernorm(x), mask=mask, key=k_attn, pos_ids=pos_ids)
        x = x + self.mlp(self.post_attention_layernorm(x), key=k_mlp)
        return x


class HackableTransformer(eqx.Module):
    config: HackableTransformerConfig = eqx.field(static=True)
    context_layers: BlockFoldable[HackableDecoderLayer]
    prediction_layers: BlockFoldable[HackableDecoderLayer]
    norm: hnn.RmsNorm
    backout_lambda: NamedArray

    @staticmethod
    def init(config: HackableTransformerConfig, *, key):
        S = Stacked  # use BlockSeq for non-homogeneous layers
        context_layers = S.init(config.ContextLayers, HackableDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config, key=shaped_rng_split(key, config.num_context_layers)
        )
        prediction_layers = S.init(config.PredictionLayers, HackableDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config, key=shaped_rng_split(key, config.num_prediction_layers)
        )
        backout_lambda = hax.ones(()) * 0.5
        return HackableTransformer(config, context_layers, prediction_layers, config.mk_LayerNorm(config.Embed), backout_lambda)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: NamedArray | AttentionMask | None, *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys_context = maybe_rng_split(key, self.config.num_context_layers) if key is not None else None
        keys_prediction = maybe_rng_split(key, self.config.num_prediction_layers) if key is not None else None
        x0 = x
        x_context = self.context_layers.fold(x, x0, mask=attn_mask, key=keys_context, pos_ids=pos_ids)
        x_prediction = self.prediction_layers.fold(x_context, x0, mask=attn_mask, key=keys_prediction, pos_ids=pos_ids)
        x_prediction = x_prediction - self.backout_lambda * x_context
        return self.norm(x_prediction)


class HackableEmbedding(ModuleWithStateDictSerialization, eqx.Module):
    token_embeddings: hnn.Embedding
    norm: hnn.RmsNorm | None = None

    @staticmethod
    def init(Vocab: Axis, config: HackableTransformerConfig, *, key):
        emb = hnn.Embedding.init(Vocab, config.Embed, key=key)
        ln = config.mk_LayerNorm(config.Embed) if config.input_embedding_norm else None
        return HackableEmbedding(emb, ln)

    @property
    def Vocab(self) -> Axis:
        return self.token_embeddings.Vocab

    @named_call
    def embed(self, input_ids: NamedArray):
        x = self.token_embeddings(input_ids)
        return self.norm(x) if self.norm is not None else x


class HackableLMHeadModel(
    ModuleWithStateDictSerialization,
    LmHeadModel[HackableTransformerConfig],
):
    """Minimal Llama-like implementation of LmHeadModel"""

    transformer: HackableTransformer
    embeddings: HackableEmbedding
    lm_head: hnn.Linear | None

    @property
    def config(self) -> HackableTransformerConfig:
        return self.transformer.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: HackableTransformerConfig, *, key) -> "HackableLMHeadModel":
        k_t, k_e = jrandom.split(key, 2)
        transformer = HackableTransformer.init(config, key=k_t)
        embeddings = HackableEmbedding.init(Vocab, config, key=k_e)
        lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_e, use_bias=False, out_first=True)
        lm_head = eqx.tree_at(lambda m: m.weight, lm_head, lm_head.weight * 0)
        return HackableLMHeadModel(transformer, embeddings, lm_head)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        return self.transformer(self.embeddings.embed(input_ids), attn_mask=attn_mask, key=key, pos_ids=pos_ids)

    def get_lm_head(self) -> hax.NamedArray:
        return self.embeddings.token_embeddings.weight if self.lm_head is None else self.lm_head.weight

    def resize_vocab(self, new_size: int, key: PRNGKeyArray | None = None) -> "HackableLMHeadModel":
        raise NotImplementedError("resize_vocab is not implemented for HackableLMHeadModel")


# =========================
# Speedrun sweep definition
# =========================

AUTHOR = Author(
    name=SUBMISSION_AUTHOR_NAME,
    affiliation=SUBMISSION_AUTHOR_AFFILIATION,
    url=(SUBMISSION_AUTHOR_URL or None),
)


def _get_num_train_steps(param_count: int, batch_size: int, seq_len: int, tpp: int = 20) -> int:
    total_tokens = int(param_count * tpp)
    return max(1, total_tokens // (batch_size * seq_len))


# =========================
# Model configuration presets
# TODO: make any model configuration changes
# =========================


def _size_presets() -> dict[str, HackableTransformerConfig]:
    base = dict(
        seq_len=2048,
        attn_backend=AttentionBackend.JAX_FLASH,
        qk_norm=RmsNormConfig(use_weight=False, use_bias=False, eps=1e-5),
        tie_word_embeddings=False,
        cross_entropy_block_size=2048, # avoid materializing full logits (batch*seq*vocab)
    )
    return {
        "150m": HackableTransformerConfig(
            hidden_dim=512, intermediate_dim=512*4, num_layers=6, num_heads=4, num_kv_heads=4, **base
        ),
        "270m": HackableTransformerConfig(
            hidden_dim=768, intermediate_dim=768*4, num_layers=11, num_heads=6, num_kv_heads=6, **base
        ),
        "460m": HackableTransformerConfig(
            hidden_dim=1024, intermediate_dim=1024*4, num_layers=16, num_heads=8, num_kv_heads=8, **base
        )
    }


# =========================
# Muon optimizer presets
# See https://wandb.ai/marin-community/marin/reports/Fantastic-Optimizers-and-Where-to-Find-Them--VmlldzoxMjgzMzQ2NQ
# TODO: make any optimizer changes. You can use different optimizers: e.g.,
# "130m": AdamHConfig(
#             learning_rate=0.02,
#             adam_lr=0.008,
#             min_lr_ratio=0,
#             warmup=1000,
#             beta1=0.9,
#             beta2=0.98,
#             epsilon=1e-20,
#             max_grad_norm=1,
#             nesterov=False,
#         ),
# see available optimizers in lib/levanter/src/levanter/optim
# =========================


def _muon_presets() -> dict[str, MuonConfig]:
    return {
        "150m": MuonConfig(
            learning_rate=0.02,
            adam_lr=0.0064,
            weight_decay=0,
            min_lr_ratio=0.1,
            warmup=0,
            momentum=0.95,
            beta1=0.8,
            beta2=0.95,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=0.5,
        ),
        "270m": MuonConfig(
            learning_rate=0.02,
            adam_lr=0.0064,
            weight_decay=0,
            min_lr_ratio=0.1,
            warmup=0,
            momentum=0.95,
            beta1=0.8,
            beta2=0.95,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=0.5,
        ),
        "460m": MuonConfig(
            learning_rate=0.02,
            adam_lr=0.0064,
            weight_decay=0.01,
            min_lr_ratio=0.1,
            warmup=0,
            momentum=0.95,
            beta1=0.8,
            beta2=0.95,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=0.5,
        ),
    }


# =========================
# Resource presets (IMPORTANT!)
# TODO: edit tpu_type or accelerator_type to match what you have available on your hardware
# e.g., GpuConfig(gpu_count=8, accelerator_type="H100"),
# If you ignore this and there is a mismatch, training cannot start if an unavailable resource is requested!
# =========================


def _resource_presets(use_tpu: bool = False):
    if use_tpu:
        return {
            "150m": ResourceConfig.with_tpu("v5p-32"),
            "270m": ResourceConfig.with_tpu("v5p-32"),
            "460m": ResourceConfig.with_tpu("v5p-32"),
            "1_2b": ResourceConfig.with_tpu("v5p-32"),
        }
    return {
        "150m": ResourceConfig.with_gpu("H100", count=1),
        "270m": ResourceConfig.with_gpu("H100", count=1),
        "460m": ResourceConfig.with_gpu("H100", count=2),
        "1_2b": ResourceConfig.with_gpu("H100", count=4),
    }


# =========================
# Batch size presets
# TODO: edit to adjust for your hardware
# =========================


def _batch_sizes() -> dict[str, int]:
    return {"150m": 192, "270m": 192, "460m": 192}


def build_run(size: str, *, use_tpu: bool = False) -> tuple[str, SpeedrunConfig]:
    sizes = _size_presets()
    if size not in sizes:
        raise ValueError(f"Unknown size: {size}")
    model_cfg = sizes[size]

    batch = _batch_sizes()[size]
    seq_len = model_cfg.seq_len
    params = int(model_cfg.total_trainable_params(llama3_tokenizer_vocab_size))
    steps = _get_num_train_steps(params, batch, seq_len, tpp=2.5)

    muon = _muon_presets()[size]
    resources = _resource_presets(use_tpu=use_tpu)[size]

    train = SimpleTrainConfig(
        resources=resources,
        train_batch_size=batch,
        num_train_steps=steps,
        learning_rate=muon.learning_rate,
        optimizer_config=muon,
        steps_per_hf_export=-1,  # disable checkpointing
    )

    run_name = f"{SUBMISSION_BRANCH}_{size}"
    desc = f"{SUBMISSION_DESCRIPTION} ({size})"
    cfg = SpeedrunConfig(author=AUTHOR, description=desc, model_config=model_cfg, train_config=train)
    return run_name, cfg


if __name__ == "__main__":
    ###
    # make the current __main__ module importable under its canonical name
    sys.modules[_IMPORT_PATH] = sys.modules[__name__]
    # allow the workers to import the classes
    for _cls in (
        HackableTransformerConfig,
        HackableMlp,
        HackableDecoderLayer,
        HackableTransformer,
        HackableEmbedding,
        HackableLMHeadModel,
    ):
        _cls.__module__ = _IMPORT_PATH
    ###

    sizes = ["150m", "270m", "460m"]
    #sizes = ["270m"]
    use_tpu = bool(int(os.environ.get("SR_USE_TPU", "0")))
    steps = []
    for s in sizes:
        name, cfg = build_run(s, use_tpu=use_tpu)
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))
    executor_main(steps=steps, description=SUBMISSION_DESCRIPTION)
