# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model.

This variant intentionally mirrors `experiments/grug/base/model.py` and applies
MoE-specific changes inline. Keeping the file largely self-contained follows the
grug copy-first workflow in `.agents/skills/change-grug/`.
"""

import dataclasses
import math

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray

from levanter.grug.attention import AttentionMask, RotaryConfig, apply_rotary_embedding, attention
from levanter.grug.grug_moe import MoeActivation, moe_mlp
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pvocab, unshard
from levanter.tracker.histogram import Histogram
from levanter.utils.activation import ActivationFunctionEnum

_DEFAULT_EP_CAPACITY_FACTOR = 1.25


def _mesh_axis_size(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty or axis_name not in mesh.shape:
        raise ValueError(f"grug/moe requires an abstract mesh with axis '{axis_name}'")
    return int(mesh.shape[axis_name])


def _batch_spec() -> P:
    return P(("data", "expert"))


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the compact grug MoE transformer.

    Default architecture:
    - Head dim 128, num_heads = hidden_dim // 128.
    - QK norm on attention queries and keys.
    - RMS norm after embedding and before lm_head.
    - 2 leading dense layers with intermediate_dim = 3 * hidden_dim (6144).
    - MoE in all subsequent layers: E=64 experts, K=4 active per token,
      expert intermediate_dim = hidden_dim // 2 (1024).
    - 1 shared expert with intermediate_dim = hidden_dim (2048) per MoE layer.
    - Router z-loss 0.001 averaged over layers; no aux load-balancing loss
      (use bias-based balancing instead).
    - Standard RoPE with theta=10000.
    - SiGLU activations on all MLPs.
    - Z-loss on final logits (logsumexp_weight=1e-4) configured in trainer.
    """

    vocab_size: int
    hidden_dim: int = 512
    # Expert MLP intermediate dim (hidden_dim // 2).
    intermediate_dim: int = 256
    # Shared expert intermediate dim per MoE layer (1x hidden_dim).
    shared_expert_intermediate_dim: int = 512
    num_experts: int = 64
    num_experts_per_token: int = 4
    num_layers: int = 24
    num_heads: int = 4
    num_kv_heads: int = 4
    head_dim: int | None = None
    max_seq_len: int = 4096
    layer_norm_eps: float = 1e-5
    qk_mult: float = 1.0
    residual_t: float = 0.3  # Residual mixing: sqrt(1-t)*x + sqrt(t)*f(x)
    num_dense_layers: int = 2
    # Dense layer intermediate dim (3x hidden_dim).
    dense_intermediate_dim: int = 1536
    sliding_window: int | None = None
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)

    def __post_init__(self) -> None:
        _ = self.inferred_head_dim
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.num_experts_per_token <= 0:
            raise ValueError("num_experts_per_token must be positive")
        if self.num_experts_per_token > self.num_experts:
            raise ValueError("num_experts_per_token must be <= num_experts")
        if self.shared_expert_intermediate_dim < 0:
            raise ValueError("shared_expert_intermediate_dim must be non-negative")

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads


def rms_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Non-parametric RMS norm over the last dimension."""
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + eps)).astype(x.dtype)


class CausalSelfAttention(eqx.Module):
    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]
    attn_gate: Float[Array, "D N"]
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), d), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (d, m * h), d), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (d, m * h), d), P("data", "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), n * h), P("model", "data")),
            attn_gate=reshard(jnp.zeros((d, n)), P(None, None)),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array, use_rope: bool = True
    ) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        batch_spec = _batch_spec()

        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (m d) -> ... m d", d=head_dim)
        q = rms_norm(q)
        k = rms_norm(k)
        if use_rope:
            q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=self.cfg.rope)
        q = q * self.cfg.qk_mult
        attn_out = attention(q, k, v, mask)
        gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.attn_gate))[..., None]
        attn_out = gate * attn_out
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=batch_spec)


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


class DenseMLP(eqx.Module):
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array

    @staticmethod
    def init(hidden_dim: int, intermediate_dim: int, *, key: PRNGKeyArray) -> "DenseMLP":
        k_gate, k_up, k_down = random.split(key, 3)
        alpha_ffn = intermediate_dim / hidden_dim
        return DenseMLP(
            w_gate=reshard(_init_weight(k_gate, (hidden_dim, intermediate_dim), hidden_dim), P("data", "model")),
            w_up=reshard(_init_weight(k_up, (hidden_dim, intermediate_dim), hidden_dim), P("data", "model")),
            w_down=reshard(
                _init_down_proj(k_down, (intermediate_dim, hidden_dim), hidden_dim, alpha_ffn), P("model", "data")
            ),
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        *,
        activation: MoeActivation = ActivationFunctionEnum.silu,
    ) -> Float[Array, "B S D"]:
        if isinstance(activation, ActivationFunctionEnum):
            activation_fn = activation.to_jax_fn()
        else:
            activation_fn = activation

        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        gate = jnp.einsum("td,dm->tm", x_flat, self.w_gate)
        up = jnp.einsum("td,dm->tm", x_flat, self.w_up)
        out_flat = jnp.einsum("tm,md->td", activation_fn(gate) * up, self.w_down, out_sharding=_batch_spec())
        return rearrange(out_flat, "(b s) d -> b s d", b=b, s=s)


def _routing_stats(
    selected_experts: Int[Array, "T K"],
    *,
    num_experts: int,
) -> dict[str, jax.Array]:
    expert_counts = jnp.sum(jax.nn.one_hot(selected_experts, num_experts, dtype=jnp.float32), axis=(0, 1))
    total_assignments = jnp.maximum(jnp.sum(expert_counts), 1.0)
    assignment_fraction = expert_counts / total_assignments
    routing_entropy = -jnp.sum(assignment_fraction * jnp.log(assignment_fraction + 1e-6))

    return {
        "routing_counts": expert_counts,
        "routing_entropy": routing_entropy,
    }


def _summarize_router_metrics(router_metrics: dict[str, jax.Array]) -> dict[str, jax.Array | Histogram]:
    routing_entropy = router_metrics["routing_entropy_per_layer"]
    routing_counts = router_metrics["routing_counts_per_layer"]
    num_layers = int(routing_entropy.shape[0])

    out: dict[str, jax.Array | Histogram] = {
        "train/router/routing_entropy_mean": jnp.mean(routing_entropy),
        # Raw routing counts for bias updates (not logged to tracker).
        "train/router/routing_counts_per_layer": routing_counts,
    }
    for i in range(num_layers):
        out[f"train/router/layer_{i}/routing_entropy"] = routing_entropy[i]
        out[f"train/router/layer_{i}/routing_hist"] = _histogram_from_expert_counts(routing_counts[i])
    return out


def _histogram_from_expert_counts(expert_counts: jax.Array) -> Histogram:
    counts = jnp.asarray(expert_counts, dtype=jnp.float32)
    num_experts = counts.shape[0]
    expert_ids = jnp.arange(num_experts, dtype=jnp.float32)
    num = jnp.sum(counts)
    sum_values = jnp.sum(counts * expert_ids)
    sum_squares = jnp.sum(counts * expert_ids * expert_ids)
    nonzero = counts > 0
    min_value = jnp.where(nonzero, expert_ids, jnp.inf).min()
    max_value = jnp.where(nonzero, expert_ids, -jnp.inf).max()
    min_value = jnp.where(num > 0, min_value, 0.0)
    max_value = jnp.where(num > 0, max_value, 0.0)
    bucket_limits = jnp.arange(num_experts + 1, dtype=jnp.float32)
    return Histogram(
        min=min_value,
        max=max_value,
        num=num,
        sum=sum_values,
        sum_squares=sum_squares,
        bucket_limits=bucket_limits,
        bucket_counts=counts,
    )


class MoEMLP(eqx.Module):
    router: jax.Array
    router_bias: jax.Array
    w_gate_up: jax.Array  # merged gate+up: (E, D, 2*I)
    w_down: jax.Array
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MoEMLP":
        k_router, k_gate, k_up, k_down = random.split(key, 4)
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        d, e, i = (
            cfg.hidden_dim,
            cfg.num_experts,
            cfg.intermediate_dim,
        )

        w_gate = _init_weight(k_gate, (e, d, i), d)
        w_up = _init_weight(k_up, (e, d, i), d)
        w_gate_up = jnp.concatenate([w_gate, w_up], axis=-1)  # (E, D, 2*I)

        return MoEMLP(
            router=reshard(_init_weight(k_router, (d, e), d), P(None, None)),
            router_bias=jnp.zeros((e,)),
            w_gate_up=reshard(w_gate_up, P("expert", "data", "model")),
            w_down=reshard(_init_down_proj(k_down, (e, i, d), d, i / d), P("expert", "model", "data")),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        router_logits = jnp.einsum("td,de->te", x_flat, reshard(self.router, P(None, None)))
        # Bias added for selection only (stop_gradient so it doesn't affect router training)
        biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
        _topk_logits, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token)
        # Combine weights: sigmoid on selected experts, normalized to sum to 1.
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        topk_weights = jax.nn.sigmoid(unbiased_topk)
        combine_weights = (topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)).astype(x.dtype)
        router_stats = _routing_stats(
            selected_experts,
            num_experts=self.cfg.num_experts,
        )

        routed_flat = moe_mlp(
            x_flat,
            selected_experts.astype(jnp.int32),
            combine_weights,
            self.w_gate_up,
            self.w_down,
            activation=ActivationFunctionEnum.silu,
            mesh=get_abstract_mesh(),
            capacity_factor=_DEFAULT_EP_CAPACITY_FACTOR,
        )
        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        routed = reshard(routed, _batch_spec())
        return routed, router_stats


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp: MoEMLP | None
    shared: DenseMLP | None
    dense_mlp: DenseMLP | None
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray, dense_only: bool = False) -> "Block":
        attn_key, mlp_key, shared_key, dense_key = random.split(key, 4)
        shared = None
        moe = None
        dense_mlp = None
        if dense_only:
            dense_mlp = DenseMLP.init(
                cfg.hidden_dim,
                cfg.dense_intermediate_dim,
                key=dense_key,
            )
        else:
            moe = MoEMLP.init(cfg, key=mlp_key)
            if cfg.shared_expert_intermediate_dim > 0:
                shared = DenseMLP.init(
                    cfg.hidden_dim,
                    cfg.shared_expert_intermediate_dim,
                    key=shared_key,
                )
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp=moe,
            shared=shared,
            dense_mlp=dense_mlp,
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        use_rope: bool = True,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        t = jnp.array(self.cfg.residual_t, dtype=x.dtype)
        x = jnp.sqrt(1.0 - t) * x + jnp.sqrt(t) * self.attn(self.rms_attn(x), mask, use_rope=use_rope)
        mlp_in = self.rms_mlp(x)
        if self.dense_mlp is not None:
            mlp_out = self.dense_mlp(mlp_in, activation=ActivationFunctionEnum.silu)
            router_stats = {}
        else:
            assert self.mlp is not None
            mlp_out, router_stats = self.mlp(mlp_in)
            if self.shared is not None:
                mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        x = jnp.sqrt(1.0 - t) * x + jnp.sqrt(t) * mlp_out
        return x, router_stats


class Transformer(eqx.Module):
    token_embed: jax.Array
    embed_gain: jax.Array
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, out_key, *block_keys = random.split(key, cfg.num_layers + 2)
        raw_embed = _init_unit(embed_key, (cfg.vocab_size, cfg.hidden_dim))
        row_norms = jnp.sqrt(jnp.sum(jnp.square(raw_embed), axis=1, keepdims=True))
        token_embed = reshard(raw_embed / row_norms * math.sqrt(cfg.hidden_dim), Pvocab)
        embed_gain = jnp.array(1.0, dtype=jnp.float32)
        output_proj = reshard(_init_unit(out_key, (cfg.hidden_dim, cfg.vocab_size)), Pvocab)
        blocks = tuple(
            Block.init(cfg, key=block_keys[i], dense_only=(i < cfg.num_dense_layers)) for i in range(cfg.num_layers)
        )
        final_norm = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)
        return Transformer(
            token_embed=token_embed,
            embed_gain=embed_gain,
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
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        if mask is None:
            mask = AttentionMask.causal()

        batch_spec = _batch_spec()
        hidden = self.token_embed.at[token_ids].get(out_sharding=batch_spec) * self.embed_gain

        cfg = self.config
        if cfg.sliding_window is not None:
            segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
            short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window // 2, segment_ids=segment_ids)
            long_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)

        moe_router_stats: list[dict[str, jax.Array]] = []
        for i, block in enumerate(self.blocks):
            if cfg.sliding_window is not None:
                layer_mask = long_mask if i % 4 == 3 else short_mask
            else:
                layer_mask = mask
            hidden, router_stats = eqx.filter_checkpoint(block)(hidden, layer_mask)
            if router_stats:
                moe_router_stats.append(router_stats)

        router_metrics = {
            "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
            "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
        }
        return self.final_norm(hidden), router_metrics

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        batch_spec = _batch_spec()
        hidden, _ = self(token_ids, mask=mask)
        scaled_output_proj = self.output_proj * (0.5 / math.sqrt(self.config.hidden_dim))
        return jnp.einsum("bsh,hd->bsd", hidden, scaled_output_proj, out_sharding=batch_spec)

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
        return_router_metrics: bool = False,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | Histogram]]:
        hidden, router_metrics = self(token_ids, mask=mask)
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)
        scaled_output_proj = self.output_proj * (0.5 / math.sqrt(self.config.hidden_dim))

        cross_entropy_loss = fused_linear_softmax_cross_entropy_loss(
            hidden,
            scaled_output_proj,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
        )
        if return_router_metrics:
            summarized_metrics = _summarize_router_metrics(router_metrics)
            summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
            return cross_entropy_loss, summarized_metrics
        return cross_entropy_loss


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], fan_in: int) -> Float[Array, "..."]:
    """Initialize with variance 1/fan_in (truncated normal)."""
    std = 1.0 / math.sqrt(fan_in)
    return std * random.truncated_normal(key, -3, 3, shape)


def _init_down_proj(key: PRNGKeyArray, shape: tuple[int, ...], hidden_dim: int, alpha_ffn: float) -> Float[Array, "..."]:
    """Initialize down-projection with std = 1/(alpha_ffn * sqrt(hidden_dim)).

    Follows CompleteP for MoE (arxiv 2601.20205): treats alpha_ffn as intermediate
    width in a mean-field two-layer MLP rather than using standard fan_in init.
    """
    std = 1.0 / (alpha_ffn * math.sqrt(hidden_dim))
    return std * random.truncated_normal(key, -3, 3, shape)


def _init_unit(key: PRNGKeyArray, shape: tuple[int, ...]) -> Float[Array, "..."]:
    """Initialize with variance 1 (truncated normal)."""
    return random.truncated_normal(key, -3, 3, shape)


def debug_mesh_and_token_pspec(num_devices: int) -> tuple[jax.sharding.AbstractMesh, P]:
    """Return a small abstract mesh and token sharding for lowering contract tests."""
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}")
    # Keep expert axis at 2 when possible to exercise EP lowering, otherwise
    # fall back to expert=1.
    expert = 2 if num_devices % 2 == 0 else 1
    data = max(1, num_devices // expert)
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(data, expert, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    return mesh, P(("data", "expert"), None)


__all__ = [
    "Block",
    "CausalSelfAttention",
    "DenseMLP",
    "GrugModelConfig",
    "MoEMLP",
    "MoeActivation",
    "RMSNorm",
    "Transformer",
    "debug_mesh_and_token_pspec",
    "moe_mlp",
]
