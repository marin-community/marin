# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import levanter.tracker
from einops import rearrange
from haliax.jax_utils import named_call
from haliax.partitioning import _get_mesh
from jax import random
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .attention import AttentionMask, RotaryConfig, apply_rotary_embedding, attention
from .loss import fused_linear_softmax_cross_entropy_loss
from .sharding import Pbatch, unshard


#### Conventions

# Mesh meanings:
# - "data": data parallel sharding axis.
# All model weights (including expert weights) are fully replicated across chips.

# Dim names:
# - B = batch
# - D = embedding / hidden dim
# - S = sequence length
# - N = num heads
# - M = num kv heads
# - H = head dim
# - I = intermediate dim
# - T = tokens (B * S, flattened batch)
# - K = num_experts_per_tok
# - TR = T * K (tokens repeated per expert, sorted by expert)
# - E = n_routed_experts


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the Grug Mixtral MoE style transformer."""

    vocab_size: int
    hidden_dim: int = 1536
    intermediate_dim: int = 4608
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 12
    head_dim: int | None = None
    max_seq_len: int = 2048
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02

    num_experts_per_tok: int = 2
    n_routed_experts: int = 8

    lbl_coef: float | None = 0.01
    rzl_coef: float | None = 0.001

    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)

    def __post_init__(self) -> None:
        _ = self.inferred_head_dim
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads


class CausalSelfAttention(eqx.Module):
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_o = random.split(key, 4)
        D, N, M, H = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
        return CausalSelfAttention(
            w_q=_init_weight(k_q, (D, N * H), cfg.initializer_std),
            w_k=_init_weight(k_k, (D, M * H), cfg.initializer_std),
            w_v=_init_weight(k_v, (D, M * H), cfg.initializer_std),
            w_o=_init_weight(k_o, (N * H, D), cfg.initializer_std),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]

        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (m d) -> ... m d", d=head_dim)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=self.cfg.rope)
        attn_out = attention(q, k, v, mask)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=Pbatch)


class MOE(eqx.Module):
    router_w: jax.Array
    w1: jax.Array
    w2: jax.Array
    w3: jax.Array
    cfg: GrugModelConfig = eqx.field(static=True)

    _ragged_dim_numbers = jax.lax.RaggedDotDimensionNumbers(
        dot_dimension_numbers=(((1,), (1,)), ((), ())),
        lhs_ragged_dimensions=(0,),
        rhs_group_dimensions=(0,),
    )

    @staticmethod
    def _ragged_linear(x: jax.Array, w: jax.Array, group_sizes: jax.Array) -> jax.Array:
        """Ragged MoE linear: (TR, In) x (E, In, Out) with groups along TR."""
        return jax.lax.ragged_dot_general(
            lhs=x,
            rhs=w,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=MOE._ragged_dim_numbers,
        )

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MOE":
        k_router_w, k_w1, k_w2, k_w3 = random.split(key, 4)
        E, D, I = cfg.n_routed_experts, cfg.hidden_dim, cfg.intermediate_dim
        router_w = _init_weight(k_router_w, (D, E), cfg.initializer_std)
        w1 = _init_weight(k_w1, (E, D, I), cfg.initializer_std)
        w2 = _init_weight(k_w2, (E, D, I), cfg.initializer_std)
        w3 = _init_weight(k_w3, (E, I, D), cfg.initializer_std)
        return MOE(router_w, w1, w2, w3, cfg)

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> tuple[Float[Array, "B S D"], dict]:
        B, S, D = x.shape
        x_flat = jnp.reshape(x, (B * S, D))
        router_logits = jnp.einsum("td,de->te", x_flat, self.router_w)
        topk_weights, topk_idx, router_probs = self._route(router_logits)
        topk_idx_flat = jnp.reshape(topk_idx, (B * S * self.cfg.num_experts_per_tok,))
        mesh = _get_mesh()

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(Pbatch, Pbatch, Pbatch, P(), P(), P()),
            out_specs=(Pbatch, P()),
        )
        def _moe_block(x_flat, topk_idx_flat, topk_weights, w1, w2, w3):
            x_repeat_sort, group_sizes, sort_idx = self._permute(x_flat, topk_idx_flat)
            w1_out = MOE._ragged_linear(x_repeat_sort, w1, group_sizes)  # [TR, I]
            w2_out = MOE._ragged_linear(x_repeat_sort, w2, group_sizes)  # [TR, I]
            gated = jax.nn.silu(w1_out) * w2_out  # [TR, I]
            out_repeat_sort = MOE._ragged_linear(gated, w3, group_sizes)  # [TR, D]
            out_repeat_unflat = self._unpermute(out_repeat_sort, sort_idx)
            out_flat = jnp.sum(out_repeat_unflat * topk_weights[..., None], axis=1)  # [T, D]

            # compute statistics and aux loss over global batch
            global_group_sizes = jax.lax.psum(group_sizes, "data")
            return out_flat, global_group_sizes

        out_flat, group_sizes = _moe_block(x_flat, topk_idx_flat, topk_weights, self.w1, self.w2, self.w3)
        out = jnp.reshape(out_flat, (B, S, D))

        extras = {}
        if self.cfg.lbl_coef is not None:
            group_sizes_f = group_sizes.astype(jnp.float32)
            expert_loads = group_sizes_f / jnp.sum(group_sizes_f)
            extras["expert_loads"] = expert_loads
            f = expert_loads * (self.cfg.n_routed_experts / self.cfg.num_experts_per_tok)
            p = jnp.mean(router_probs.astype(jnp.float32), axis=0)  # [T, E] -> [E]
            extras["load_balancing_loss"] = jnp.asarray(self.cfg.lbl_coef, dtype=jnp.float32) * jnp.sum(f * p)

        if self.cfg.rzl_coef is not None:
            z = jsp.special.logsumexp(router_logits.astype(jnp.float32), axis=-1)
            extras["router_z_loss"] = jnp.asarray(self.cfg.rzl_coef, dtype=jnp.float32) * jnp.mean(z**2)

        return out, extras

    def _route(
        self, router_logits: Float[Array, "T E"]
    ) -> tuple[Float[Array, "T K"], Int[Array, "T K"], Float[Array, "T E"]]:
        """Select top-k experts per token and compute normalized routing weights."""
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        _scores, topk_idx = jax.lax.top_k(router_logits, self.cfg.num_experts_per_tok)
        topk_weights = jnp.take_along_axis(router_probs, topk_idx, axis=-1)
        topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
        return topk_weights, topk_idx.astype(jnp.int32), router_probs

    def _permute(
        self, x_flat: jax.Array, topk_idx_flat: jax.Array
    ) -> tuple[Float[Array, "TR D"], Int[Array, "E"], Int[Array, "TR"]]:
        """Sort tokens by assigned expert and compute per-expert group sizes for ragged_dot."""
        sort_idx = jnp.argsort(topk_idx_flat, axis=-1)
        x_repeat_sort = jnp.take(x_flat, sort_idx // self.cfg.num_experts_per_tok, axis=0)
        group_sizes = jnp.bincount(topk_idx_flat, length=self.cfg.n_routed_experts).astype(jnp.int32)
        return x_repeat_sort, group_sizes, sort_idx.astype(jnp.int32)

    def _unpermute(self, out_repeat_sort: jax.Array, sort_idx: jax.Array) -> Float[Array, "T K D"]:
        """Reverse the expert-sorted order back to the original token layout."""
        inv_sort_idx = jnp.argsort(sort_idx, axis=-1)
        out_repeat = jnp.take(out_repeat_sort, inv_sort_idx, axis=0)
        return jnp.reshape(out_repeat, (-1, self.cfg.num_experts_per_tok, self.cfg.hidden_dim))


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float) -> "RMSNorm":
        return RMSNorm(weight=jnp.ones((dim,), dtype=jnp.float32), eps=eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        weight = unshard(self.weight)
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * weight).astype(dtype)


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    moe: MOE

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Block":
        attn_key, moe_key = random.split(key, 2)
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            moe=MOE.init(cfg, key=moe_key),
        )

    @named_call
    def __call__(
        self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array
    ) -> tuple[Float[Array, "B S D"], dict]:
        x = x + self.attn(self.rms_attn(x), mask)
        moe_out, extras = self.moe(self.rms_mlp(x))
        x = x + moe_out
        return x, extras


class Transformer(eqx.Module):
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, out_key, *block_keys = random.split(key, cfg.num_layers + 2)
        token_embed = _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std)
        output_proj = _init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std)
        blocks = tuple(Block.init(cfg, key=layer_key) for layer_key in block_keys)
        final_norm = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)
        return Transformer(
            token_embed=token_embed,
            output_proj=output_proj,
            blocks=blocks,
            final_norm=final_norm,
            config=cfg,
        )

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S D"]:
        if mask is None:
            mask = AttentionMask.causal()

        hidden = self.token_embed.at[token_ids].get(out_sharding=Pbatch)
        all_extras = []
        for block in self.blocks:
            hidden, extras = eqx.filter_checkpoint(block)(hidden, mask)
            all_extras.append(extras)
        aux_loss = self.parse_aux_loss(all_extras)
        return self.final_norm(hidden), aux_loss

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        hidden, _ = self(token_ids, mask=mask)
        return jnp.einsum("bsh,hd->bsd", hidden, self.output_proj, out_sharding=Pbatch)

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
    ) -> jax.Array:
        """Compute next-token cross-entropy loss for a batch."""
        hidden, aux_loss = self(token_ids, mask=mask)
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        return (
            fused_linear_softmax_cross_entropy_loss(
                hidden,
                self.output_proj,
                labels,
                weight=loss_weight,
                reduction=reduction,
                logsumexp_weight=logsumexp_weight,
                dtype=loss_dtype,
            )
            + aux_loss
        )

    def parse_aux_loss(self, all_extras) -> Float[Array, ""]:
        load_balancing_loss = 0
        router_z_loss = 0
        stats = {}
        for i, extras in enumerate(all_extras):
            if "load_balancing_loss" in extras:
                stats[f"train/layer_{i}/load_balancing_loss"] = jax.lax.stop_gradient(extras["load_balancing_loss"])
                load_balancing_loss += extras["load_balancing_loss"]
            if "router_z_loss" in extras:
                stats[f"train/layer_{i}/router_z_loss"] = jax.lax.stop_gradient(extras["router_z_loss"])
                router_z_loss += extras["router_z_loss"]
            if "expert_loads" in extras:
                expert_loads = extras["expert_loads"]  # [E], sums to 1
                n_experts = self.config.n_routed_experts

                entropy = -jnp.sum(expert_loads * jnp.log(expert_loads + 1e-6))
                load_violation_max = jnp.max(expert_loads) * n_experts

                stats[f"train/layer_{i}/routing_entropy"] = jax.lax.stop_gradient(entropy)
                stats[f"train/layer_{i}/load_violation_max"] = jax.lax.stop_gradient(load_violation_max)
                for j in range(n_experts):
                    stats[f"train/layer_{i}/expert_{j}/load"] = jax.lax.stop_gradient(expert_loads[j])

        stats["train/load_balancing_loss"] = jax.lax.stop_gradient(load_balancing_loss)
        stats["train/router_z_loss"] = jax.lax.stop_gradient(router_z_loss)
        levanter.tracker.jit_log(stats)
        aux_loss = load_balancing_loss + router_z_loss
        return aux_loss


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


__all__ = [
    "CausalSelfAttention",
    "MOE",
    "RMSNorm",
    "Block",
    "Transformer",
    "GrugModelConfig",
]
