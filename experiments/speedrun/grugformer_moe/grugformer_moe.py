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
Grugformer MoE experiment (router + ragged expert MLP).

This is an experiment-only implementation: it keeps the MoE logic local to this entrypoint
and does not modify the canonical `levanter.grug` core.

Design goals:
- "Grug simple": explicit tensor shapes, minimal abstractions.
- "Vanilla custom_mixtral logic": top-k routing + sort/permute dispatch + ragged_dot_general expert MLP,
  with load-balancing loss and router z-loss.
- Replicated experts (no expert-parallel all-to-all).
"""

# nodryrun

import dataclasses
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import TypeVar

import haliax as hax
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from einops import rearrange
from fray.cluster import ResourceConfig
from haliax import Axis, NamedArray
from haliax.partitioning import _get_mesh
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P, reshard
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from levanter.grug.attention import AttentionMask, RotaryConfig, apply_rotary_embedding, attention
from levanter.grug.sharding import Pbatch, Pvocab, unshard
from levanter.layers.attention import AttentionMask as LevanterAttentionMask
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.utils.flop_utils import lm_flops_per_token
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger("ray")

# Ruff/Pyflakes treats string literals in annotations as forward refs and checks that bare
# identifiers (e.g. "D") are defined. These are jaxtyping dimension labels, not runtime symbols.
D = TypeVar("D")


#### Conventions
#
# Mesh meanings:
# - "data": data parallel sharding axis. We also shard parameters over this axis (ZeRO-ish).
# - "model": model parallel sharding axis (TP).
#
# Dim names used in comments:
# - B = batch
# - S = sequence length
# - D = hidden dim
# - I = intermediate dim (per-expert)
# - E = number of routed experts
# - K = experts per token (top-k)
# - T = flattened tokens (= B*S)
# - TR = token-repeat (= T*K)


@dataclass(frozen=True)
class GrugMoeModelConfig:
    # Core grug hyperparams
    vocab_size: int
    hidden_dim: int = 2048
    intermediate_dim: int = 5632
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None
    max_seq_len: int = 4096
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)
    cross_entropy_block_size: int | None = 32768
    cross_entropy_implementation: str | None = "xla"

    # MoE hyperparams (vanilla Mixtral-ish)
    n_routed_experts: int = 8  # E
    num_experts_per_tok: int = 2  # K
    lbl_coef: float | None = 0.01
    rzl_coef: float | None = 0.001
    router_fp32: bool = False
    router_topk_then_softmax: bool = False

    def __post_init__(self) -> None:
        _ = self.inferred_head_dim
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_experts_per_tok <= 0:
            raise ValueError("num_experts_per_tok must be positive")
        if self.n_routed_experts <= 0:
            raise ValueError("n_routed_experts must be positive")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok cannot exceed n_routed_experts")

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads


@register_dataclass
@dataclass(frozen=True)
class GrugAttentionParams:
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array


@register_dataclass
@dataclass(frozen=True)
class GrugMoeBlockParams:
    attn: GrugAttentionParams
    rms_attn: jax.Array
    rms_mlp: jax.Array
    router_w: jax.Array  # [D, E]
    w1: jax.Array  # [E, D, I] (gate_proj)
    w3: jax.Array  # [E, D, I] (up_proj)
    w2: jax.Array  # [E, I, D] (down_proj)


@register_dataclass
@dataclass(frozen=True)
class GrugMoeParameters:
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[GrugMoeBlockParams, ...]
    final_norm: jax.Array


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * jax.random.truncated_normal(key, -3, 3, shape)


@partial(jax.jit, static_argnames=("cfg",))
def init_parameters(cfg: GrugMoeModelConfig, *, key: PRNGKeyArray) -> GrugMoeParameters:
    head_dim = cfg.inferred_head_dim
    key, embed_key, out_key = jax.random.split(key, 3)
    layer_keys = jax.random.split(key, cfg.num_layers)

    token_embed = reshard(_init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pvocab)
    output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Pvocab)
    final_norm = reshard(jnp.ones((cfg.hidden_dim,), dtype=jnp.float32), P(None))

    blocks: list[GrugMoeBlockParams] = []
    # extract shape sizes for brevity and consistency
    hidden_dim = cfg.hidden_dim
    num_heads = cfg.num_heads
    num_kv_heads = cfg.num_kv_heads
    intermediate_dim = cfg.intermediate_dim
    num_experts = cfg.n_routed_experts
    for i in range(cfg.num_layers):
        (
            k_q,
            k_k,
            k_v,
            k_o,
            k_router,
            k_w1,
            k_w2,
            k_w3,
        ) = jax.random.split(layer_keys[i], 8)

        attn = GrugAttentionParams(
            w_q=reshard(_init_weight(k_q, (hidden_dim, num_heads * head_dim), cfg.initializer_std), P("data", "model")),
            w_k=reshard(
                _init_weight(k_k, (hidden_dim, num_kv_heads * head_dim), cfg.initializer_std), P("data", "model")
            ),
            w_v=reshard(
                _init_weight(k_v, (hidden_dim, num_kv_heads * head_dim), cfg.initializer_std), P("data", "model")
            ),
            w_o=reshard(_init_weight(k_o, (num_heads * head_dim, hidden_dim), cfg.initializer_std), P("model", "data")),
        )

        # Router maps D -> E. Keep the expert axis replicated (no expert-parallel sharding).
        router_w = reshard(_init_weight(k_router, (hidden_dim, num_experts), cfg.initializer_std), P("data", None))

        # Expert weights are replicated over E and follow the same (data, model) sharding pattern as Grug MLP.
        w1 = reshard(
            _init_weight(k_w1, (num_experts, hidden_dim, intermediate_dim), cfg.initializer_std),
            P(None, "data", "model"),
        )
        w3 = reshard(
            _init_weight(k_w3, (num_experts, hidden_dim, intermediate_dim), cfg.initializer_std),
            P(None, "data", "model"),
        )
        w2 = reshard(
            _init_weight(k_w2, (num_experts, intermediate_dim, hidden_dim), cfg.initializer_std),
            P(None, "model", "data"),
        )

        # keep rms replicated
        rms_attn = jnp.ones((hidden_dim,), dtype=jnp.float32)
        rms_mlp = jnp.ones((hidden_dim,), dtype=jnp.float32)

        blocks.append(
            GrugMoeBlockParams(
                attn=attn,
                rms_attn=rms_attn,
                rms_mlp=rms_mlp,
                router_w=router_w,
                w1=w1,
                w3=w3,
                w2=w2,
            )
        )

    return GrugMoeParameters(
        token_embed=token_embed,
        output_proj=output_proj,
        blocks=tuple(blocks),
        final_norm=final_norm,
    )


def rms_norm(x: Float[Array, "... D"], weight: Float[Array, "D"], eps: float) -> Float[Array, "... D"]:
    weight = unshard(weight)
    # Levanter runs with mixed precision (bf16 compute, fp32 params) + strict dtype promotion.
    # Do RMSNorm math in fp32, then cast back to the input dtype.
    out_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    w_f32 = weight.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(variance + eps)
    y = (x_f32 * inv) * w_f32
    return y.astype(out_dtype)


def _ragged_moe_linear(
    x: jax.Array,
    w: jax.Array,
    group_sizes: jax.Array,
) -> jax.Array:
    """Ragged MoE linear: (TR, In) x (E, In, Out) with groups along TR.

    Shapes:
      - x: [TR, In]
      - w: [E, In, Out]
      - group_sizes: [E] (sum == TR)
      - out: [TR, Out]
    """
    # Everything other than the contracting dimension is treated as ragged.
    dim_numbers = jax.lax.RaggedDotDimensionNumbers(
        dot_dimension_numbers=(((1,), (1,)), ((), ())),
        lhs_ragged_dimensions=(0,),
        rhs_group_dimensions=(0,),
    )
    # `ragged_dot_general` doesn't yet have a built-in sharding rule. On SPMD runs we
    # drop this op into full auto-sharding mode.
    #
    # This MoE is still *not* expert-parallel: expert weights are replicated and all routing
    # / dispatch happens per-device for that device's local tokens.
    mesh = _get_mesh()
    if mesh is not None and not getattr(mesh, "empty", False):
        w_sharding = getattr(w, "sharding", None)
        w_spec = getattr(w_sharding, "spec", None)
        out_axis = w_spec[-1] if w_spec is not None and len(w_spec) == w.ndim else None
        out_sharding = NamedSharding(mesh, P(Pbatch[0], out_axis))

        ragged = jax.sharding.auto_axes(
            lambda lhs, rhs, gs: jax.lax.ragged_dot_general(
                lhs=lhs,
                rhs=rhs,
                group_sizes=gs,
                ragged_dot_dimension_numbers=dim_numbers,
            )
        )
        try:
            return ragged(x, w, group_sizes, out_sharding=out_sharding)
        except TypeError:
            # Some JAX builds spell this kwarg differently.
            return ragged(x, w, group_sizes, out_shardings=out_sharding)  # type: ignore[call-arg]

    return jax.lax.ragged_dot_general(
        lhs=x,
        rhs=w,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dim_numbers,
    )


def _route(
    selection_logits: jax.Array,
    router_logits: jax.Array,
    *,
    top_k: int,
    topk_then_softmax: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Top-k route tokens to experts.

    Shapes:
      - selection_logits: [T, E]
      - router_logits: [T, E]
      - topk_weights: [T, K]
      - topk_idx: [T, K] (int32 expert ids)
      - router_probs: [T, E]
    """
    router_probs = jax.nn.softmax(router_logits, axis=-1)
    _scores, topk_idx = jax.lax.top_k(selection_logits, top_k)

    if topk_then_softmax:
        selected_logits = jnp.take_along_axis(router_logits, topk_idx, axis=-1)
        topk_weights = jax.nn.softmax(selected_logits, axis=-1)
    else:
        topk_weights = jnp.take_along_axis(router_probs, topk_idx, axis=-1)
        topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)

    return topk_weights, topk_idx.astype(jnp.int32), router_probs


def _permute(
    x_flat: jax.Array,
    topk_idx_flat: jax.Array,
    *,
    num_experts_per_tok: int,
    n_routed_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sort token-repeat stream by expert id.

    Shapes:
      - x_flat: [T, D]
      - topk_idx_flat: [TR] where TR = T*K
      - x_repeat_sort: [TR, D]
      - group_sizes: [E]
      - sort_idx: [TR]
    """
    sort_idx = jnp.argsort(topk_idx_flat, axis=-1)
    x_repeat_sort = jnp.take(x_flat, sort_idx // num_experts_per_tok, axis=0)
    group_sizes = jnp.bincount(topk_idx_flat, length=n_routed_experts).astype(jnp.int32)
    return x_repeat_sort, group_sizes, sort_idx.astype(jnp.int32)


def _unpermute(
    out_repeat_sort: jax.Array,
    sort_idx: jax.Array,
    *,
    num_experts_per_tok: int,
    hidden_dim: int,
) -> jax.Array:
    """Invert expert sort and unflatten token-repeat back to [T, K, D]."""
    inv_sort_idx = jnp.argsort(sort_idx, axis=-1)
    out_repeat = jnp.take(out_repeat_sort, inv_sort_idx, axis=0)
    return jnp.reshape(out_repeat, (-1, num_experts_per_tok, hidden_dim))


def moe_mlp(block: GrugMoeBlockParams, x: Float[Array, "B S D"], cfg: GrugMoeModelConfig) -> tuple[jax.Array, jax.Array]:
    """MoE MLP with Mixtral-style routing/dispatch and auxiliary router losses."""
    B, S, D = x.shape
    E = cfg.n_routed_experts
    K = cfg.num_experts_per_tok
    T = B * S
    TR = T * K

    x_flat = jnp.reshape(x, (T, D))  # [B, S, D] -> [T, D]

    x_for_gate = x_flat.astype(jnp.float32) if cfg.router_fp32 else x_flat
    router_logits = jnp.einsum("td,de->te", x_for_gate, block.router_w)  # [T, D] @ [D, E] -> [T, E]
    if cfg.router_fp32 and router_logits.dtype != jnp.float32:
        router_logits = router_logits.astype(jnp.float32)

    selection_logits = router_logits

    mesh = _get_mesh()
    if mesh is not None and not getattr(mesh, "empty", False):
        route = shard_map(
            lambda sel, rlog: _route(
                sel,
                rlog,
                top_k=K,
                topk_then_softmax=cfg.router_topk_then_softmax,
            ),
            mesh=mesh,
            in_specs=(Pbatch, Pbatch),
            out_specs=(Pbatch, Pbatch, Pbatch),
            check_rep=False,
        )
        topk_weights, topk_idx, router_probs = route(selection_logits, router_logits)
    else:
        topk_weights, topk_idx, router_probs = _route(
            selection_logits,
            router_logits,
            top_k=K,
            topk_then_softmax=cfg.router_topk_then_softmax,
        )

    topk_idx_flat = jnp.reshape(topk_idx, (TR,))  # [T, K] -> [TR]

    if mesh is not None and not getattr(mesh, "empty", False):
        permute = shard_map(
            lambda x_t, idx_tr: _permute(
                x_t,
                idx_tr,
                num_experts_per_tok=K,
                n_routed_experts=E,
            ),
            mesh=mesh,
            in_specs=(Pbatch, Pbatch),
            out_specs=(Pbatch, P(None), Pbatch),
            check_rep=False,
        )
        x_repeat_sort, group_sizes, sort_idx = permute(x_flat, topk_idx_flat)
    else:
        x_repeat_sort, group_sizes, sort_idx = _permute(
            x_flat,
            topk_idx_flat,
            num_experts_per_tok=K,
            n_routed_experts=E,
        )

    # Expert MLP on the sorted token-repeat stream. All expert math is per-shard (replicated across E).
    #
    # Shapes:
    #   - x_repeat_sort: [TR, D]
    #   - group_sizes: [E], sum(group_sizes) == TR
    #   - w1/w3: [E, D, I], w2: [E, I, D]
    w1_out = _ragged_moe_linear(x_repeat_sort, block.w1, group_sizes)  # [TR, I]
    w3_out = _ragged_moe_linear(x_repeat_sort, block.w3, group_sizes)  # [TR, I]
    gated = jax.nn.silu(w1_out) * w3_out  # [TR, I]
    out_repeat_sort = _ragged_moe_linear(gated, block.w2, group_sizes)  # [TR, D]

    if mesh is not None and not getattr(mesh, "empty", False):
        unpermute = shard_map(
            lambda out_tr_d, sidx_tr: _unpermute(
                out_tr_d,
                sidx_tr,
                num_experts_per_tok=K,
                hidden_dim=D,
            ),
            mesh=mesh,
            in_specs=(Pbatch, Pbatch),
            out_specs=Pbatch,
            check_rep=False,
        )
        out_repeat_unflat = unpermute(out_repeat_sort, sort_idx)  # [T, K, D]
    else:
        out_repeat_unflat = _unpermute(out_repeat_sort, sort_idx, num_experts_per_tok=K, hidden_dim=D)

    out_flat = jnp.sum(out_repeat_unflat * topk_weights[..., None], axis=1)  # [T, D]
    out = jnp.reshape(out_flat, (B, S, D))  # [T, D] -> [B, S, D]

    # --- Aux router losses (vanilla Mixtral-ish) ---
    aux = jnp.array(0.0, dtype=jnp.float32)

    if cfg.lbl_coef is not None:
        # group_sizes: [E] counts assignments over token-repeat stream (TR = T*K)
        # expert_loads: [E] sums to 1
        group_sizes_f = group_sizes.astype(jnp.float32)
        expert_loads = group_sizes_f / jnp.sum(group_sizes_f)
        f = expert_loads * (E / K)  # [E]
        p = jnp.mean(router_probs.astype(jnp.float32), axis=0)  # [T, E] -> [E]
        aux = aux + jnp.asarray(cfg.lbl_coef, dtype=jnp.float32) * jnp.sum(f * p)  # scalar

    if cfg.rzl_coef is not None:
        z = jsp.special.logsumexp(router_logits.astype(jnp.float32), axis=-1)  # [T]
        aux = aux + jnp.asarray(cfg.rzl_coef, dtype=jnp.float32) * jnp.mean(z**2)  # scalar

    return out, aux


def _transformer_hidden(
    params: GrugMoeParameters,
    token_ids: Int[Array, "B S"],
    cfg: GrugMoeModelConfig,
    *,
    mask: AttentionMask | jax.Array | None,
) -> tuple[Float[Array, "B S D"], jax.Array]:
    head_dim = cfg.inferred_head_dim
    seq_len = token_ids.shape[1]

    if mask is None:
        mask = AttentionMask.causal()

    hidden = params.token_embed.at[token_ids].get(out_sharding=Pbatch)  # [B, S, D]

    aux_total = jnp.array(0.0, dtype=jnp.float32)

    for block in params.blocks:
        attn_in = rms_norm(hidden, block.rms_attn, cfg.layer_norm_eps)
        q = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_v), "... (m d) -> ... m d", d=head_dim)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=cfg.rope)
        attn_out = attention(q, k, v, mask)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        attn_out = jnp.einsum("bsh,hd->bsd", attn_out, block.attn.w_o, out_sharding=Pbatch)

        hidden = hidden + attn_out
        mlp_in = rms_norm(hidden, block.rms_mlp, cfg.layer_norm_eps)
        mlp_out, aux = moe_mlp(block, mlp_in, cfg)
        hidden = hidden + mlp_out
        aux_total = aux_total + aux

    hidden = rms_norm(hidden, params.final_norm, cfg.layer_norm_eps)
    return hidden, aux_total


def activations(
    params: GrugMoeParameters,
    token_ids: Int[Array, "B S"],
    cfg: GrugMoeModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
) -> tuple[Float[Array, "B S D"], jax.Array]:
    """Return final hidden states (and aux loss scalar)."""
    return _transformer_hidden(params, token_ids, cfg, mask=mask)


class GrugMoeWrapper(LmHeadModel[PyTree]):
    """Minimal LmHeadModel wrapper around this experiment-local Grug+MoE implementation."""

    params: GrugMoeParameters
    grug_config: GrugMoeModelConfig

    @property
    def config(self) -> GrugMoeModelConfig:
        return self.grug_config

    @property
    def Pos(self) -> Axis:
        return Axis("position", self.grug_config.max_seq_len)

    @property
    def KeyPos(self) -> Axis:
        return self.Pos.alias("key_position")

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.grug_config.vocab_size)

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.grug_config.hidden_dim)

    @classmethod
    def init(cls, Vocab: Axis, config: GrugMoeModelConfig, *, key: PRNGKeyArray) -> "GrugMoeWrapper":
        cfg = dataclasses.replace(config, vocab_size=Vocab.size)
        params = init_parameters(cfg, key=key)
        return cls(params=params, grug_config=cfg)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: LevanterAttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> tuple[NamedArray, jax.Array]:
        del key, pos_ids  # grug core doesn't use PRNGs/pos_ids yet

        mask = _mask_from_levanter(attn_mask)
        hidden, aux = activations(self.params, input_ids.array, self.grug_config, mask=mask)
        return hax.named(hidden, (*input_ids.axes, self.Embed)), aux

    def get_lm_head(self) -> NamedArray:
        return hax.named(self.params.output_proj, (self.Embed, self.Vocab))

    def compute_next_token_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction: hax.ReductionFunction | None = hax.mean,
        reduction_axis: hax.AxisSelection | None = None,
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype | None = jnp.float32,
        logit_soft_cap: float | None = None,
    ) -> jnp.ndarray | NamedArray:
        activations = self.activations(example.tokens, example.attn_mask, key=key)

        aux_loss = 0
        if isinstance(activations, tuple):
            activations, aux_loss = activations

        loss = maybe_fused_next_token_loss(
            self.Pos,
            self.Embed,
            self.Vocab,
            activations,
            self.get_lm_head(),
            example.tokens,
            loss_weight=example.loss_weight,
            reduction=reduction,
            reduction_axis=reduction_axis,
            logsumexp_weight=logsumexp_weight,
            block_size=self.grug_config.cross_entropy_block_size,
            dtype=loss_dtype,
            logit_soft_cap=logit_soft_cap,
            implementation=self.grug_config.cross_entropy_implementation,
        )
        return loss + aux_loss

    def resize_vocab(self, new_size: int, key: PRNGKeyArray | None = None) -> "GrugMoeWrapper":
        raise NotImplementedError("GrugMoeWrapper does not yet support resizing the vocabulary.")


def _mask_from_levanter(attn_mask: LevanterAttentionMask | NamedArray | None) -> AttentionMask | jax.Array | None:
    mask: AttentionMask | jax.Array | None = None
    if isinstance(attn_mask, LevanterAttentionMask):
        if attn_mask.explicit_mask is not None:
            raise NotImplementedError("Grug does not support explicit masks yet.")
        if attn_mask.causal_offset is not None:
            raise NotImplementedError("Grug does not support causal offsets yet.")
        segment_ids = None
        if attn_mask.segment_ids is not None:
            q_seg, kv_seg = attn_mask.segment_ids
            segment_ids = (q_seg.array, kv_seg.array)
        mask = AttentionMask(
            is_causal=attn_mask.is_causal,
            segment_ids=segment_ids,
            sliding_window=attn_mask.sliding_window,
        )
    elif isinstance(attn_mask, NamedArray):
        raise NotImplementedError(
            "NamedArray attention masks are not supported by Grug (pass a Levanter AttentionMask)."
        )
    return mask


def _get_num_train_steps(param_count: int, batch_size: int, max_seq_len: int, tpp: int = 20) -> int:
    total_tokens = param_count * tpp
    return max(1, total_tokens // (batch_size * max_seq_len))


def _size_presets() -> dict[str, "GrugformerMoeConfig"]:
    base = dict(max_seq_len=2048, head_dim=None, n_routed_experts=8, num_experts_per_tok=2)
    return {
        "130m": GrugformerMoeConfig(
            hidden_dim=512, intermediate_dim=1792, num_layers=6, num_heads=8, num_kv_heads=8, **base
        ),
        "300m": GrugformerMoeConfig(
            hidden_dim=768, intermediate_dim=2688, num_layers=12, num_heads=12, num_kv_heads=12, **base
        ),
        "520m": GrugformerMoeConfig(
            hidden_dim=1024, intermediate_dim=3584, num_layers=24, num_heads=16, num_kv_heads=16, **base
        ),
    }


def _resource_presets(use_tpu: bool = False):
    if use_tpu:
        return {
            "130m": ResourceConfig.with_tpu("v5p-8"),
            "300m": ResourceConfig.with_tpu("v5p-8"),
            "520m": ResourceConfig.with_tpu("v5p-8"),
        }
    return {
        "130m": ResourceConfig.with_gpu("A100-80G", count=1),
        "300m": ResourceConfig.with_gpu("A100-80G", count=1),
        "520m": ResourceConfig.with_gpu("A100-80G", count=2),
    }


def _batch_sizes() -> dict[str, int]:
    return {"130m": 128, "300m": 128, "520m": 128}


@LmConfig.register_subclass("grugformer_moe")
@dataclass(frozen=True)
class GrugformerMoeConfig(LmConfig[GrugMoeWrapper]):
    """Speedrun LmConfig wrapper around an experiment-local Grug+MoE transformer."""

    # LmConfig field
    max_seq_len: int = 2048

    # Core hyperparams
    hidden_dim: int = 1024
    intermediate_dim: int = 2752
    num_layers: int = 12
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None

    # MoE hyperparams
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    lbl_coef: float | None = 0.01
    rzl_coef: float | None = 0.001
    router_fp32: bool = False
    router_topk_then_softmax: bool = False
    cross_entropy_block_size: int | None = 32768
    cross_entropy_implementation: str | None = "xla"

    # ---- LmConfig API ----
    @property
    def model_type(self) -> type[GrugMoeWrapper]:
        return GrugMoeWrapper

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> GrugMoeWrapper:
        cfg = GrugMoeModelConfig(
            vocab_size=Vocab.size,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            n_routed_experts=self.n_routed_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            lbl_coef=self.lbl_coef,
            rzl_coef=self.rzl_coef,
            router_fp32=self.router_fp32,
            router_topk_then_softmax=self.router_topk_then_softmax,
            cross_entropy_block_size=self.cross_entropy_block_size,
            cross_entropy_implementation=self.cross_entropy_implementation,
        )
        return GrugMoeWrapper.init(Vocab, cfg, key=key)

    def flops_per_token(self, vocab_size: int, context_length: int) -> float | None:
        # Rough FLOP estimate: attention + (MoE MLP per-token uses K experts).
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
        head_dim = self.head_dim or (self.hidden_dim // self.num_heads)
        token_embedding = vocab_size * self.hidden_dim
        attn = (
            self.hidden_dim * head_dim * self.num_heads
            + 2 * self.hidden_dim * head_dim * self.num_kv_heads
            + head_dim * self.num_heads * self.hidden_dim
        )
        router = self.hidden_dim * self.n_routed_experts
        experts = 3 * self.n_routed_experts * self.hidden_dim * self.intermediate_dim
        moe = router + experts
        transformer = self.num_layers * (attn + moe + 2 * self.hidden_dim) + self.hidden_dim
        return int(transformer + 2 * token_embedding)


def build_run(size: str, *, use_tpu: bool = False) -> tuple[str, SpeedrunConfig]:
    sizes = _size_presets()
    if size not in sizes:
        raise ValueError(f"Unknown size: {size}")
    model_cfg = sizes[size]

    batch = _batch_sizes()[size]
    max_seq_len = model_cfg.max_seq_len
    params = int(model_cfg.total_trainable_params(llama3_tokenizer_vocab_size))
    steps = _get_num_train_steps(params, batch, max_seq_len, tpp=20)
    resources = _resource_presets(use_tpu=use_tpu)[size]

    train = SimpleTrainConfig(
        resources,
        train_seq_len=max_seq_len,
        train_batch_size=batch,
        num_train_steps=steps,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
        steps_per_hf_export=-1,
        explicit_mesh_axes=True,
    )

    run_name = f"grugformer_moe_{size}"
    desc = f"Grugformer MoE experiment (Mixtral-style router/dispatch) ({size})."
    cfg = SpeedrunConfig(
        author=Author(
            name="__YOUR_NAME__",
            affiliation="__YOUR_AFFILIATION__",
            url="__YOUR_URL__",
        ),
        description=desc,
        model_config=model_cfg,
        train_config=train,
    )
    return run_name, cfg


def main() -> None:
    sizes = ["130m", "300m", "520m"]
    use_tpu = bool(int(os.environ.get("SR_USE_TPU", "0")))

    steps = []
    for s in sizes:
        name, cfg = build_run(s, use_tpu=use_tpu)
        if cfg.vocab_size != llama3_tokenizer_vocab_size:
            raise AssertionError("Speedrun vocab_size mismatch; expected llama3_tokenizer_vocab_size")
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="Grugformer MoE experiment (Mixtral-style router/dispatch).")


if __name__ == "__main__":
    main()
