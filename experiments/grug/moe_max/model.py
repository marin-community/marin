# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model.

This variant intentionally mirrors `experiments/grug/base/model.py` and applies
MoE-specific changes inline. Keeping the file largely self-contained follows the
grug copy-first workflow in `docs/recipes/change_grug.md`.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray

from levanter.grug.attention import AttentionMask, attention
from levanter.grug.grug_moe import MoeActivation, moe_mlp
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pvocab
from levanter.tracker.histogram import Histogram
from levanter.utils.activation import ActivationFunctionEnum

_DEFAULT_EP_CAPACITY_FACTOR = 1.25


def _rotary_cache(seq_len: int, rotary_dim: int, rope_theta: float) -> tuple[Float[Array, "S D"], Float[Array, "S D"]]:
    half_dim = rotary_dim // 2
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]
    return jnp.cos(angles), jnp.sin(angles)


@named_call
def apply_rotary_embedding(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S H D"],
    *,
    seq_len: int,
    head_dim: int,
    rope_theta: float,
    partial_rotary_factor: float = 0.5,
) -> tuple[Float[Array, "B S H D"], Float[Array, "B S H D"]]:
    """Partial rotary embedding: only the first ``partial_rotary_factor`` of the head dim is rotated."""
    rotary_dim = int(head_dim * partial_rotary_factor)
    cos, sin = _rotary_cache(seq_len, rotary_dim, rope_theta)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def _apply(x: Float[Array, "B S H D"]) -> Float[Array, "B S H D"]:
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x1, x2 = jnp.split(x_rot, 2, axis=-1)
        x_rot = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
        return jnp.concatenate([x_rot, x_pass], axis=-1)

    return _apply(q), _apply(k)


@named_call
def qk_norm(x: Float[Array, "B S H D"], eps: float = 1e-6) -> Float[Array, "B S H D"]:
    """Non-parametric RMS norm over the head dimension."""
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + eps)).astype(x.dtype)


def _mesh_axis_size(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty or axis_name not in mesh.shape:
        raise ValueError(f"grug/moe requires an abstract mesh with axis '{axis_name}'")
    return int(mesh.shape[axis_name])


def _batch_spec() -> P:
    return P(("data", "expert"))


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the compact grug MoE transformer."""

    vocab_size: int
    hidden_dim: int = 2048
    intermediate_dim: int = 5632
    shared_expert_intermediate_dim: int = 5632
    num_experts: int = 8
    num_experts_per_token: int = 2
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None
    max_seq_len: int = 4096
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    load_balancing_loss_coef: float | None = 0.01
    router_z_loss_coef: float | None = 0.001
    rope_theta: float = 1024.0
    partial_rotary_factor: float = 0.5
    gate_input_dim: int = 12
    num_ve_layers: int = 0
    sliding_window: int | None = None
    ep_capacity_factor: float = 1.25
    backout_layer: int | None = None

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
        if self.load_balancing_loss_coef is not None and self.load_balancing_loss_coef < 0:
            raise ValueError("load_balancing_loss_coef must be non-negative when set")
        if self.router_z_loss_coef is not None and self.router_z_loss_coef < 0:
            raise ValueError("router_z_loss_coef must be non-negative when set")

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
    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]
    ve_embed: jax.Array | None
    value_lambda: jax.Array
    ve_lambda: jax.Array
    ve_gate: jax.Array
    attn_gate: jax.Array
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray, has_ve: bool = False) -> "CausalSelfAttention":
        k_q, k_k, k_v, _k_o, k_ve = random.split(key, 5)
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
        g = cfg.gate_input_dim

        ve_embed = None
        if has_ve:
            ve_dim = m * h
            ve_embed = reshard(_init_weight(k_ve, (cfg.vocab_size, ve_dim), cfg.initializer_std), Pvocab)

        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), cfg.initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_o=reshard(jnp.zeros((n * h, d)), P("model", "data")),
            ve_embed=ve_embed,
            value_lambda=jnp.full((), 0.5, dtype=jnp.float32),
            ve_lambda=jnp.full((), 0.5, dtype=jnp.float32),
            ve_gate=reshard(jnp.zeros((g, m), dtype=jnp.float32), P(None, None)),
            attn_gate=reshard(jnp.zeros((g, n), dtype=jnp.float32), P(None, None)),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        token_ids: Int[Array, "B S"] | None = None,
    ) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        batch_spec = _batch_spec()
        g = self.cfg.gate_input_dim

        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (m d) -> ... m d", d=head_dim)

        q = qk_norm(q)
        k = qk_norm(k)

        q, k = apply_rotary_embedding(
            q, k, seq_len=seq_len, head_dim=head_dim,
            rope_theta=self.cfg.rope_theta, partial_rotary_factor=self.cfg.partial_rotary_factor,
        )

        if self.ve_embed is not None and token_ids is not None:
            ve = self.ve_embed.at[token_ids].get(out_sharding=batch_spec)
            ve_heads = rearrange(ve, "... (m d) -> ... m d", d=head_dim)
            gate_out = 2 * jax.nn.sigmoid(x[..., :g] @ self.ve_gate)
            v = self.value_lambda * v + self.ve_lambda * gate_out[..., None] * ve_heads

        attn_out = attention(q, k, v, mask)

        attn_gate_out = 2 * jax.nn.sigmoid(x[..., :g] @ self.attn_gate)
        attn_out = attn_gate_out[..., None] * attn_out

        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=batch_spec)


class RMSNorm(eqx.Module):
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float) -> "RMSNorm":
        return RMSNorm(eps=eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return normed.astype(dtype)


class DenseMLP(eqx.Module):
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array

    @staticmethod
    def init(hidden_dim: int, intermediate_dim: int, initializer_std: float, *, key: PRNGKeyArray) -> "DenseMLP":
        k_gate, k_up = random.split(key, 2)
        return DenseMLP(
            w_gate=reshard(_init_weight(k_gate, (hidden_dim, intermediate_dim), initializer_std), P("data", "model")),
            w_up=reshard(_init_weight(k_up, (hidden_dim, intermediate_dim), initializer_std), P("data", "model")),
            w_down=reshard(jnp.zeros((intermediate_dim, hidden_dim)), P("model", "data")),
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
    router_probs: Float[Array, "T E"],
    router_logits: Float[Array, "T E"],
    *,
    num_experts: int,
    num_experts_per_token: int,
) -> dict[str, jax.Array]:
    router_probs_f = router_probs.astype(jnp.float32)
    router_logits_f = router_logits.astype(jnp.float32)
    expert_counts = jnp.sum(jax.nn.one_hot(selected_experts, num_experts, dtype=jnp.float32), axis=(0, 1))
    total_assignments = jnp.maximum(jnp.sum(expert_counts), 1.0)
    assignment_fraction = expert_counts / total_assignments
    routing_entropy = -jnp.sum(assignment_fraction * jnp.log(assignment_fraction + 1e-6))
    # Match the Switch/OLMoE-style scaling: E * sum_i(f_i * p_i), where
    # f_i is token fraction for expert i (counts per token, not per assignment).
    # assignment_fraction sums to 1 over assignments, so convert with top-k.
    token_fraction = assignment_fraction * num_experts_per_token
    p = jnp.mean(router_probs_f, axis=0)
    load_balancing_loss = num_experts * jnp.sum(token_fraction * p)
    z = jsp.special.logsumexp(router_logits_f, axis=-1)
    router_z_loss = jnp.mean(z**2)

    return {
        "routing_counts": expert_counts,
        "routing_entropy": routing_entropy,
        "load_balancing_loss": load_balancing_loss,
        "router_z_loss": router_z_loss,
    }


def _summarize_router_metrics(router_metrics: dict[str, jax.Array]) -> dict[str, jax.Array | Histogram]:
    routing_entropy = router_metrics["routing_entropy_per_layer"]
    routing_counts = router_metrics["routing_counts_per_layer"]
    load_balancing_loss = router_metrics["load_balancing_loss_per_layer"]
    router_z_loss = router_metrics["router_z_loss_per_layer"]
    num_layers = int(routing_entropy.shape[0])
    aux_loss_per_layer = load_balancing_loss + router_z_loss

    out: dict[str, jax.Array | Histogram] = {
        "train/router/routing_entropy_mean": jnp.mean(routing_entropy),
        # Match MaxText + Megatron/Nemotron practice: log layer-mean raw
        # router terms for comparability across depth.
        "train/router/load_balancing_loss": jnp.mean(load_balancing_loss),
        "train/router/router_z_loss": jnp.mean(router_z_loss),
        # Keep aux loss as a per-step aggregate while exposing mean terms above.
        "train/router/aux_loss": jnp.sum(aux_loss_per_layer),
    }
    for i in range(num_layers):
        out[f"train/router/layer_{i}/routing_entropy"] = routing_entropy[i]
        out[f"train/router/layer_{i}/load_balancing_loss"] = load_balancing_loss[i]
        out[f"train/router/layer_{i}/router_z_loss"] = router_z_loss[i]
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
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MoEMLP":
        k_router, k_gate, k_up = random.split(key, 3)
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        d, e, i = (
            cfg.hidden_dim,
            cfg.num_experts,
            cfg.intermediate_dim,
        )

        return MoEMLP(
            router=reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None)),
            w_gate=reshard(_init_weight(k_gate, (e, d, i), cfg.initializer_std), P("expert", "data", "model")),
            w_up=reshard(_init_weight(k_up, (e, d, i), cfg.initializer_std), P("expert", "data", "model")),
            w_down=reshard(jnp.zeros((e, i, d)), P("expert", "model", "data")),
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
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        topk_logits, selected_experts = jax.lax.top_k(router_logits, self.cfg.num_experts_per_token)
        combine_weights = jax.nn.softmax(topk_logits, axis=-1).astype(x.dtype)
        router_stats = _routing_stats(
            selected_experts,
            router_probs,
            router_logits,
            num_experts=self.cfg.num_experts,
            num_experts_per_token=self.cfg.num_experts_per_token,
        )

        w_up_gate = jnp.concatenate([self.w_gate, self.w_up], axis=-1)
        routed_flat = moe_mlp(
            x_flat,
            selected_experts.astype(jnp.int32),
            combine_weights,
            w_up_gate,
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

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray, has_ve: bool = False, dense_only: bool = False) -> "Block":
        attn_key, mlp_key, shared_key, dense_key = random.split(key, 4)
        shared = None
        moe = None
        dense_mlp = None
        if dense_only:
            dense_mlp = DenseMLP.init(
                cfg.hidden_dim,
                cfg.intermediate_dim,
                cfg.initializer_std,
                key=dense_key,
            )
        else:
            moe = MoEMLP.init(cfg, key=mlp_key)
            if cfg.shared_expert_intermediate_dim > 0:
                shared = DenseMLP.init(
                    cfg.hidden_dim,
                    cfg.shared_expert_intermediate_dim,
                    cfg.initializer_std,
                    key=shared_key,
                )
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn=CausalSelfAttention.init(cfg, key=attn_key, has_ve=has_ve),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp=moe,
            shared=shared,
            dense_mlp=dense_mlp,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        token_ids: Int[Array, "B S"] | None = None,
        x0: Float[Array, "B S D"] | None = None,
        resid_lambda: jax.Array | None = None,
        x0_lambda: jax.Array | None = None,
        attn_input: Float[Array, "B S D"] | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        # Backout: use attn_input for attention if provided, but residual still uses x.
        attn_in = attn_input if attn_input is not None else x
        x = x + self.attn(self.rms_attn(attn_in), mask, token_ids=token_ids)
        if resid_lambda is not None and x0 is not None:
            x = resid_lambda * x + x0_lambda * x0
        mlp_in = self.rms_mlp(x)
        if self.dense_mlp is not None:
            mlp_out = self.dense_mlp(mlp_in, activation=ActivationFunctionEnum.silu)
            router_stats = {}
        else:
            mlp_out, router_stats = self.mlp(mlp_in)
            if self.shared is not None:
                mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        x = x + mlp_out
        return x, router_stats


class Transformer(eqx.Module):
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    embed_norm: RMSNorm
    final_norm: RMSNorm
    resid_lambdas: tuple[jax.Array, ...]
    x0_lambdas: tuple[jax.Array, ...]
    backout_lambda: jax.Array
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, _out_key, *block_keys = random.split(key, cfg.num_layers + 2)
        token_embed = reshard(_init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pvocab)
        output_proj = reshard(jnp.zeros((cfg.hidden_dim, cfg.vocab_size)), Pvocab)
        blocks = tuple(
            Block.init(
                cfg, key=block_keys[i],
                has_ve=(i >= cfg.num_layers - cfg.num_ve_layers),
                dense_only=(i == 0),
            )
            for i in range(cfg.num_layers)
        )
        embed_norm = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)
        final_norm = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)
        resid_lambdas = tuple(jnp.ones((), dtype=jnp.float32) for _ in range(cfg.num_layers))
        x0_lambdas = tuple(jnp.zeros((), dtype=jnp.float32) for _ in range(cfg.num_layers))
        backout_lambda = jnp.full((), 0.5, dtype=jnp.float32)

        return Transformer(
            token_embed=token_embed,
            output_proj=output_proj,
            blocks=blocks,
            embed_norm=embed_norm,
            final_norm=final_norm,
            resid_lambdas=resid_lambdas,
            x0_lambdas=x0_lambdas,
            backout_lambda=backout_lambda,
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
        hidden = self.token_embed.at[token_ids].get(out_sharding=batch_spec)
        hidden = self.embed_norm(hidden)
        x0 = hidden

        cfg = self.config
        if cfg.sliding_window is not None:
            segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
            short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window // 2, segment_ids=segment_ids)
            long_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)

        moe_router_stats: list[dict[str, jax.Array]] = []
        x_backout = None
        for i, (resid_lambda, x0_lambda, block) in enumerate(
            zip(self.resid_lambdas, self.x0_lambdas, self.blocks, strict=True)
        ):
            if cfg.sliding_window is not None:
                layer_mask = long_mask if i % 2 == 1 else short_mask
            else:
                layer_mask = mask
            hidden, router_stats = eqx.filter_checkpoint(block)(
                hidden, layer_mask, token_ids=token_ids, x0=x0,
                resid_lambda=resid_lambda, x0_lambda=x0_lambda,
                attn_input=x_backout,
            )
            if cfg.backout_layer is not None and i == cfg.backout_layer:
                x_backout = hidden
            if router_stats:
                moe_router_stats.append(router_stats)

        router_metrics = {
            "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
            "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
            "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in moe_router_stats], axis=0),
            "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in moe_router_stats], axis=0),
        }
        if x_backout is not None:
            hidden = hidden - self.backout_lambda * x_backout
        return self.final_norm(hidden), router_metrics

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        batch_spec = _batch_spec()
        hidden, _ = self(token_ids, mask=mask)
        return jnp.einsum("bsh,hd->bsd", hidden, self.output_proj, out_sharding=batch_spec)

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

        cross_entropy_loss = fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
        )
        # Keep router metrics raw and apply coefficients only at the final
        # objective composition step (same separation as MaxText/Megatron).
        load_balancing_loss_coef = (
            0.0 if self.config.load_balancing_loss_coef is None else self.config.load_balancing_loss_coef
        )
        router_z_loss_coef = 0.0 if self.config.router_z_loss_coef is None else self.config.router_z_loss_coef
        num_moe_layers = router_metrics["load_balancing_loss_per_layer"].shape[0]
        lbl = jnp.sum(router_metrics["load_balancing_loss_per_layer"]) / num_moe_layers
        rzl = jnp.sum(router_metrics["router_z_loss_per_layer"]) / num_moe_layers
        aux_loss = load_balancing_loss_coef * lbl + router_z_loss_coef * rzl
        include_aux_in_loss = reduction != "none" and (load_balancing_loss_coef != 0.0 or router_z_loss_coef != 0.0)
        loss = cross_entropy_loss + aux_loss if include_aux_in_loss else cross_entropy_loss
        if return_router_metrics:
            summarized_metrics = _summarize_router_metrics(router_metrics)
            summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
            summarized_metrics["train/router/aux_loss_weighted"] = aux_loss
            return loss, summarized_metrics
        return loss


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


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
