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
mHC (Manifold-Constrained Hyper-Connections) speedrun sweep.

This file is intentionally self-contained:
- Defines a compact, Llama-ish transformer with mHC residual stream mixing
- Provides a ready-to-run speedrun sweep across multiple model sizes

Based on https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections

How to run:
  1) Set env vars (WANDB_API_KEY, HF_TOKEN, etc.) as in the tutorial:
     https://marin.readthedocs.io/en/latest/tutorials/submitting-speedrun/
  2) From repo root:
       python marin/run/ray_run.py -- python -m experiments.speedrun.mhc_hyper_connections.main --force_run_failed true
  3) Optional: SR_USE_TPU=1 to use TPU resource presets (default is GPU).
"""

# =========================
# Submission metadata
# TODO: fill out your information when you start
# =========================

SUBMISSION_BRANCH = "mhc_hyper_connections"
SUBMISSION_DESCRIPTION = "mHC (manifold-constrained hyper-connections) for a Llama-ish transformer"
SUBMISSION_AUTHOR_NAME = "TODO"
SUBMISSION_AUTHOR_AFFILIATION = "TODO"
SUBMISSION_AUTHOR_URL = ""

# ruff: noqa: E402
# nodryrun
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from fray.cluster import ResourceConfig
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization
from jaxtyping import PRNGKeyArray
from levanter.layers import LayerNormConfigBase, RmsNormConfig
from levanter.layers.attention import Attention, AttentionBackend, AttentionConfig, AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
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
# mHC config & modules
# TODO: make any model architecture changes
# =========================


@LmConfig.register_subclass("mhc_hyper_connections")
@dataclass(frozen=True)
class HackableTransformerConfig(LmConfig["HackableLMHeadModel"]):
    # Core dims
    max_seq_len: int = 2048
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int | None = None

    # mHC (manifold-constrained hyper-connections) - fixed per paper
    mhc_num_streams: int = 4  # n=4 per paper
    mhc_num_iters: int = 20  # Sinkhorn iterations (20 per paper)
    mhc_tau: float = 0.05  # Sinkhorn temperature

    def __post_init__(self):
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        if self.head_dim is None:
            assert self.hidden_dim % self.num_heads == 0, "hidden_dim % num_heads must be 0 when head_dim=None"

    # ---- LmConfig API ----
    @property
    def model_type(self) -> type["HackableLMHeadModel"]:
        return HackableLMHeadModel

    Embed = property(lambda self: Axis("embed", self.hidden_dim))
    Layers = property(lambda self: Axis("layers", self.num_layers))
    Mlp = property(lambda self: Axis("mlp", self.intermediate_dim))
    Streams = property(lambda self: Axis("stream", self.mhc_num_streams))
    StreamIn = property(lambda self: Axis("stream_in", self.mhc_num_streams))
    StreamOut = property(lambda self: Axis("stream_out", self.mhc_num_streams))

    @property
    def norm_config(self) -> LayerNormConfigBase:
        return RmsNormConfig(use_weight=True, use_bias=False, eps=1e-5)

    def mk_LayerNorm(self, axis: AxisSpec):
        return self.norm_config.build(axis)

    def attention_config(self) -> AttentionConfig:
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            use_bias=False,
            upcast_attn=False,
            attn_backend=AttentionBackend.JAX_FLASH,
            flash_attention_block_size=None,
            rope=DefaultRotaryEmbeddingsConfig(),
            qk_norm=None,
        )

    @property
    def actual_head_size(self) -> int:
        return self.head_dim or (self.hidden_dim // self.num_heads)

    def flops_per_token(self, vocab_size: int, context_length: int) -> float | None:
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=True,
        )

    def total_trainable_params(self, vocab_size: int) -> int:
        token_embedding = vocab_size * self.hidden_dim
        hs = self.actual_head_size
        attn = (
            self.hidden_dim * hs * self.num_heads
            + 2 * self.hidden_dim * hs * self.num_kv_heads
            + hs * self.num_heads * self.hidden_dim
        )
        mlp = 3 * self.hidden_dim * self.intermediate_dim
        transformer = self.num_layers * (attn + mlp + 2 * self.hidden_dim) + self.hidden_dim
        head = token_embedding
        # mHC params: Static: h_res (n^2), h_pre (n), h_post (n)
        n = self.mhc_num_streams
        per_branch_static = n**2 + 2 * n
        # Dynamic per paper: norm (d), theta_pre (d*n), alpha_pre (1), theta_post (d*n), alpha_post (1)
        per_branch_dynamic = (
            self.hidden_dim  # norm weight
            + 2 * self.hidden_dim * n  # theta_pre + theta_post
            + 2  # alpha_pre + alpha_post (scalars)
        )
        mhc_params = self.num_layers * 2 * (per_branch_static + per_branch_dynamic)
        return int(transformer + token_embedding + head + mhc_params)


# =========================
# mHC helpers
# =========================


def _sinkhorn_log(logits: jnp.ndarray, num_iters: int, tau: float) -> jnp.ndarray:
    n = logits.shape[-1]
    z = logits / tau
    # Reference: log_marginal = -log(n) for uniform distribution
    log_marginal = jnp.full((n,), -jnp.log(n), dtype=logits.dtype)
    u = jnp.zeros((logits.shape[0],), dtype=logits.dtype)
    v = jnp.zeros((logits.shape[1],), dtype=logits.dtype)

    for _ in range(num_iters):
        u = log_marginal - jnn.logsumexp(z + v[None, :], axis=1)
        v = log_marginal - jnn.logsumexp(z + u[:, None], axis=0)

    # Reference: scale output by n
    return jnp.exp(z + u[:, None] + v[None, :]) * n


class HackableMlp(eqx.Module):
    """GLU MLP"""

    gate_proj: hnn.Linear
    up_proj: hnn.Linear
    down_proj: hnn.Linear
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(Embed: AxisSpec, Mlp: AxisSpec, activation_fn: ActivationFunctionEnum | Callable, *, key, use_bias=False):
        k_fc, k_up_proj, k_down_proj = jrandom.split(key, 3)
        gate_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias, out_first=True)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up_proj, use_bias=use_bias, out_first=True)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down_proj, use_bias=use_bias, out_first=True)
        if isinstance(activation_fn, ActivationFunctionEnum):
            activation_fn = activation_fn.to_fn()
        elif isinstance(activation_fn, str):
            activation_fn = ActivationFunctionEnum(activation_fn).to_fn()
        return HackableMlp(gate_proj, up_proj, down_proj, activation_fn)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        h = self.act(self.gate_proj(x, key=k_gate)) * self.up_proj(x, key=k_up)
        return self.down_proj(h, key=k_down)


class MhcStreamExpander(eqx.Module):
    """Expands residual stream from C to n×C dimensions (n=4 per paper)."""

    config: HackableTransformerConfig = eqx.field(static=True)

    @staticmethod
    def init(config: HackableTransformerConfig, *, key) -> "MhcStreamExpander":
        _ = key
        return MhcStreamExpander(config)

    def __call__(self, x: NamedArray) -> NamedArray:
        return hax.broadcast_axis(x, self.config.Streams)


class MhcStreamReducer(eqx.Module):
    """Reduces n residual streams back to single stream via summation."""

    config: HackableTransformerConfig = eqx.field(static=True)

    def __call__(self, x: NamedArray) -> NamedArray:
        return hax.sum(x, axis=self.config.Streams)


class MhcHyperConnections(eqx.Module):
    """mHC width/depth connections for a single branch.

    Per the paper (arxiv 2512.24880v2), mHC uses both static and dynamic components:
        ℋ_pre = α_pre · tanh(θ_pre · x̃) + b_pre
    Where α is a learnable scalar gate, θ is a projection, and b is static bias.
    """

    config: HackableTransformerConfig = eqx.field(static=True)
    h_res_logits: NamedArray  # static bias for h_res
    h_pre_logits: NamedArray  # static bias for h_pre
    h_post_logits: NamedArray  # static bias for h_post
    # Dynamic components per paper: α · tanh(θ · x̃)
    norm: hnn.RmsNorm
    theta_pre: NamedArray  # (Embed, Streams) projection for h_pre
    alpha_pre: jnp.ndarray  # scalar gate for h_pre
    theta_post: NamedArray  # (Embed, Streams) projection for h_post
    alpha_post: jnp.ndarray  # scalar gate for h_post

    @staticmethod
    def init(config: HackableTransformerConfig, *, key) -> "MhcHyperConnections":
        k_init, k_pre, k_post = jrandom.split(key, 3)
        # Static biases (b in the paper)
        # Use one-hot to avoid int() which fails under vmap/jit tracing
        init_index = jrandom.randint(k_init, (), 0, config.mhc_num_streams)
        h_res = jnp.full((config.mhc_num_streams, config.mhc_num_streams), -8.0)
        h_res = h_res + jnp.eye(config.mhc_num_streams) * 8.0
        one_hot = jnn.one_hot(init_index, config.mhc_num_streams)
        h_pre = jnp.full((config.mhc_num_streams,), -8.0) + one_hot * 8.0
        h_post = jnp.zeros((config.mhc_num_streams,))
        # Dynamic components per paper: α · tanh(θ · x̃) + b
        # θ initialized to zero, α initialized to small value (1e-2)
        norm = config.mk_LayerNorm(config.Embed)
        theta_pre = hax.zeros((config.Embed, config.Streams))
        alpha_pre = jnp.array(1e-2)
        theta_post = hax.zeros((config.Embed, config.Streams))
        alpha_post = jnp.array(1e-2)
        return MhcHyperConnections(
            config,
            hax.named(h_res, (config.StreamOut, config.StreamIn)),
            hax.named(h_pre, (config.Streams,)),
            hax.named(h_post, (config.Streams,)),
            norm,
            theta_pre,
            alpha_pre,
            theta_post,
            alpha_post,
        )

    def _project_h_res(self) -> NamedArray:
        logits = self.h_res_logits.array
        proj = _sinkhorn_log(logits, num_iters=self.config.mhc_num_iters, tau=self.config.mhc_tau)
        return hax.named(proj, (self.config.StreamOut, self.config.StreamIn))

    def _width_connection(self, residuals: NamedArray) -> tuple[NamedArray, NamedArray, NamedArray]:
        # h_res via Sinkhorn projection (doubly stochastic)
        h_res = self._project_h_res()

        # Per paper: ℋ_pre = α · tanh(θ · x̃) + b
        # where x̃ is normalized input, b is static bias (sigmoid for non-negativity)
        # Aggregate across input streams first, then project to output stream weights
        normed = self.norm(residuals)  # (Streams, Batch, Pos, Embed)
        normed_agg = hax.mean(normed, axis=self.config.Streams)  # (Batch, Pos, Embed)
        # Dynamic: α · tanh(θ · x̃) -> (Batch, Pos, Streams)
        dynamic_pre = hax.dot(normed_agg, self.theta_pre, axis=self.config.Embed)
        dynamic_pre = hax.tanh(dynamic_pre) * self.alpha_pre
        # Static: sigmoid(b_pre) -> (Streams,) for non-negativity per paper
        static_pre = jnn.sigmoid(self.h_pre_logits)
        h_pre = static_pre + dynamic_pre  # broadcasts to (Batch, Pos, Streams)

        # Same pattern for h_post
        dynamic_post = hax.dot(normed_agg, self.theta_post, axis=self.config.Embed)
        dynamic_post = hax.tanh(dynamic_post) * self.alpha_post
        static_post = jnn.sigmoid(self.h_post_logits)
        h_post = static_post + dynamic_post

        # Mix residuals via h_res (width connection)
        residuals_in = residuals.rename({self.config.Streams.name: self.config.StreamIn.name})
        residuals_mixed = hax.dot(h_res, residuals_in, axis=self.config.StreamIn)
        residuals_mixed = residuals_mixed.rename({self.config.StreamOut.name: self.config.Streams.name})
        residuals_mixed = residuals_mixed.rearrange(residuals.axes)

        # Compute branch input via h_pre
        branch_input = hax.dot(residuals, h_pre, axis=self.config.Streams)
        return branch_input, residuals_mixed, h_post

    def _depth_connection(self, branch_output: NamedArray, residuals: NamedArray, h_post: NamedArray) -> NamedArray:
        branch_to_streams = branch_output.broadcast_axis(self.config.Streams) * h_post
        return residuals + branch_to_streams

    def __call__(self, residuals: NamedArray, *, branch: Callable, key=None, **branch_kwargs) -> NamedArray:
        branch_input, residuals_mixed, h_post = self._width_connection(residuals)
        branch_output = branch(branch_input, key=key, **branch_kwargs)
        return self._depth_connection(branch_output, residuals_mixed, h_post)


class HackableDecoderLayer(eqx.Module):
    """One transformer block with mHC connections."""

    config: HackableTransformerConfig = eqx.field(static=True)
    self_attn: Attention
    mlp: HackableMlp
    input_layernorm: hnn.RmsNorm
    post_attention_layernorm: hnn.RmsNorm
    mhc_attn: MhcHyperConnections
    mhc_mlp: MhcHyperConnections

    @staticmethod
    def init(config: HackableTransformerConfig, *, key) -> "HackableDecoderLayer":
        k_attn, k_mlp, k_mhc_attn, k_mhc_mlp = jrandom.split(key, 4)
        attn_cfg = config.attention_config()
        attn = Attention.init(attn_cfg, key=k_attn)
        mlp = HackableMlp.init(config.Embed, config.Mlp, ActivationFunctionEnum.silu, key=k_mlp, use_bias=False)
        ln1 = config.mk_LayerNorm(config.Embed)
        ln2 = config.mk_LayerNorm(config.Embed)
        mhc_attn = MhcHyperConnections.init(config, key=k_mhc_attn)
        mhc_mlp = MhcHyperConnections.init(config, key=k_mhc_mlp)
        return HackableDecoderLayer(
            config,
            attn,
            mlp,
            ln1,
            ln2,
            mhc_attn,
            mhc_mlp,
        )

    def _attn_branch(
        self,
        x: NamedArray,
        *,
        mask: NamedArray | AttentionMask | None,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        x = self.input_layernorm(x)
        return self.self_attn(x=x, mask=mask, key=key, pos_ids=pos_ids)

    def _mlp_branch(self, x: NamedArray, *, key=None) -> NamedArray:
        x = self.post_attention_layernorm(x)
        return self.mlp(x, key=key)

    @named_call
    def __call__(
        self, x: NamedArray, mask: NamedArray | AttentionMask | None, *, key=None, pos_ids: NamedArray | None = None
    ):
        k_attn, k_mlp = maybe_rng_split(key, 2)
        x = self.mhc_attn(x, branch=self._attn_branch, mask=mask, key=k_attn, pos_ids=pos_ids)
        return self.mhc_mlp(x, branch=self._mlp_branch, key=k_mlp)


class HackableTransformer(eqx.Module):
    config: HackableTransformerConfig = eqx.field(static=True)
    layers: BlockFoldable[HackableDecoderLayer]
    norm: hnn.RmsNorm
    stream_expander: MhcStreamExpander
    stream_reducer: MhcStreamReducer

    @staticmethod
    def init(config: HackableTransformerConfig, *, key):
        S = Stacked  # use BlockSeq for non-homogeneous layers
        k_layers, k_stream = jrandom.split(key, 2)
        layers = S.init(config.Layers, HackableDecoderLayer, gradient_checkpointing=True)(
            config, key=shaped_rng_split(k_layers, config.num_layers)
        )
        stream_expander = MhcStreamExpander.init(config, key=k_stream)
        stream_reducer = MhcStreamReducer(config)
        return HackableTransformer(config, layers, config.mk_LayerNorm(config.Embed), stream_expander, stream_reducer)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: NamedArray | AttentionMask | None, *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.stream_expander(x)
        x = self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids)
        x = self.norm(x)
        return self.stream_reducer(x)


class HackableEmbedding(ModuleWithStateDictSerialization, eqx.Module):
    token_embeddings: hnn.Embedding
    norm: hnn.RmsNorm | None = None

    @staticmethod
    def init(Vocab: Axis, config: HackableTransformerConfig, *, key):
        emb = hnn.Embedding.init(Vocab, config.Embed, key=key)
        return HackableEmbedding(emb, None)

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
    total_tokens = param_count * tpp
    return max(1, total_tokens // (batch_size * seq_len))


# =========================
# Model configuration presets
# TODO: make any model configuration changes
# =========================


def _size_presets() -> dict[str, HackableTransformerConfig]:
    base = dict(
        max_seq_len=4096,
        cross_entropy_block_size=4096,  # avoid materializing full logits (batch*seq*vocab)
    )
    return {
        "130m": HackableTransformerConfig(
            hidden_dim=512, intermediate_dim=1792, num_layers=6, num_heads=8, num_kv_heads=8, **base
        ),
        "300m": HackableTransformerConfig(
            hidden_dim=768, intermediate_dim=2688, num_layers=12, num_heads=12, num_kv_heads=12, **base
        ),
        "520m": HackableTransformerConfig(
            hidden_dim=1024, intermediate_dim=3584, num_layers=24, num_heads=16, num_kv_heads=8, **base
        ),
        "1_2b": HackableTransformerConfig(
            hidden_dim=2048, intermediate_dim=7168, num_layers=16, num_heads=16, num_kv_heads=8, **base
        ),
    }


# =========================
# Muon optimizer presets
# See https://wandb.ai/marin-community/marin/reports/Fantastic-Optimizers-and-Where-to-Find-Them--VmlldzoxMjgzMzQ2NQ
# =========================


def _muon_presets() -> dict[str, MuonConfig]:
    return {
        "130m": MuonConfig(
            learning_rate=0.016,
            adam_lr=0.0032,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.95,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=0.8,
        ),
        "300m": MuonConfig(
            learning_rate=0.008,
            adam_lr=0.0024,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=0.8,
        ),
        "520m": MuonConfig(
            learning_rate=0.008,
            adam_lr=0.0024,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-25,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=1,
        ),
        "1_2b": MuonConfig(
            learning_rate=0.004,
            adam_lr=0.0012,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=2,
            lr_schedule="linear",
            decay=1,
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
            "130m": ResourceConfig.with_tpu("v5p-32"),
            "300m": ResourceConfig.with_tpu("v5p-32"),
            "520m": ResourceConfig.with_tpu("v5p-32"),
            "1_2b": ResourceConfig.with_tpu("v5p-32"),
        }
    return {
        "130m": ResourceConfig.with_gpu("A100-80G", count=1),
        "300m": ResourceConfig.with_gpu("A100-80G", count=1),
        "520m": ResourceConfig.with_gpu("A100-80G", count=2),
        "1_2b": ResourceConfig.with_gpu("A100-80G", count=4),
    }


# =========================
# Batch size presets
# TODO: edit to adjust for your hardware
# =========================


def _batch_sizes() -> dict[str, int]:
    return {"130m": 128, "300m": 128, "520m": 128, "1_2b": 256}


def build_run(size: str, *, use_tpu: bool = False) -> tuple[str, SpeedrunConfig]:
    sizes = _size_presets()
    if size not in sizes:
        raise ValueError(f"Unknown size: {size}")
    model_cfg = sizes[size]

    batch = _batch_sizes()[size]
    # train on same seq_len as model max seq len
    train_seq_len = model_cfg.max_seq_len
    params = int(model_cfg.total_trainable_params(llama3_tokenizer_vocab_size))
    steps = _get_num_train_steps(params, batch, train_seq_len, tpp=20)

    muon = _muon_presets()[size]
    resources = _resource_presets(use_tpu=use_tpu)[size]

    train = SimpleTrainConfig(
        resources=resources,
        train_seq_len=train_seq_len,
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
        MhcStreamExpander,
        MhcStreamReducer,
        MhcHyperConnections,
        HackableDecoderLayer,
        HackableTransformer,
        HackableEmbedding,
        HackableLMHeadModel,
    ):
        _cls.__module__ = _IMPORT_PATH
    ###

    sizes = [
        "130m",
    ]
    # TODO: uncomment to run all sizes
    # sizes = ["130m", "300m", "520m", "1_2b"]
    use_tpu = bool(int(os.environ.get("SR_USE_TPU", "0")))
    steps = []
    for s in sizes:
        name, cfg = build_run(s, use_tpu=use_tpu)
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))
    executor_main(steps=steps, description=SUBMISSION_DESCRIPTION)
