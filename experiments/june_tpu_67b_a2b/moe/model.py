# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model.

Architecture: QB-routed MoE with GatedNorm, XSA, sigmoid combine weights.
No load-balancing loss; router z-loss only. All layers are MoE (no dense layers).
"""

import dataclasses
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from einops import rearrange
from haliax.jax_utils import named_call
from haliax.nn import ArrayStacked
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard

try:
    from jax.shard_map import shard_map  # pyrefly: ignore[missing-import]
except ModuleNotFoundError:
    from jax.experimental.shard_map import shard_map
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from levanter.grug.attention import (
    AttentionMask,
    GrugAttentionImplementation,
    RotaryConfig,
    align_kv_heads,
    attention,
)
from levanter.grug.grug_moe import (
    MOE_REMAT_SAVE_NAMES,
    MoeActivation,
    MoEExpertMlp,
    MoeImplementation,
    resolve_moe_implementation,
)
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pembed_vocab, Plm_head, unshard
from levanter.tracker.histogram import Histogram, SummaryStats
from levanter.utils.activation import ActivationFunctionEnum

_DEFAULT_EP_CAPACITY_FACTOR = 1.0
_GATED_NORM_RANK = 128
_ROUTING_RENORM_SUM = 2.5


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


def _mesh_axis_size(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty:
        raise ValueError("grug/moe requires a non-empty abstract mesh")
    if axis_name not in mesh.shape:
        # compact_grug_mesh standardizes on (replica_dcn, data, expert, model) with length-1
        # axes kept, so any missing axis is a caller bug rather than a "size 1" shortcut.
        raise ValueError(f"grug/moe requires an abstract mesh with axis '{axis_name}'")
    return int(mesh.shape[axis_name])


RematMode = Literal["recompute_all", "save_moe"]


def _batch_spec() -> P:
    return P(_BATCH_AXES)


def _batch_reshard(x: jax.Array) -> jax.Array:
    return reshard(x, _batch_spec())


def _layer_attention_masks(mask: AttentionMask, *, sliding_window: int) -> tuple[AttentionMask, AttentionMask]:
    return mask.with_sliding_window(sliding_window // 2), mask.with_sliding_window(sliding_window)


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the grug MoE transformer.

    Architecture choices (GatedNorm, XSA, QB routing) are hardcoded.
    Only shape/size knobs live here. All layers are MoE.
    """

    vocab_size: int
    hidden_dim: int = 512
    intermediate_dim: int = 256
    shared_expert_intermediate_dim: int = 512
    num_experts: int = 256
    num_experts_per_token: int = 4
    num_layers: int = 6
    num_heads: int = 4
    num_kv_heads: int = 1
    head_dim: int | None = None
    max_seq_len: int = 4096
    sliding_window: int = 2048
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.3
    qk_mult_long_scale: float = 1.0
    """Extra multiplier on ``qk_mult`` applied only on the long-attention branch
    (every-4th-and-last layer, full causal). Short (sliding-window) layers are
    unaffected. Intended for YaRN-style attention-temperature scaling at long
    context extension, where softmax logits sharpen with sequence length but
    sliding-window layers stay bounded by their window."""
    router_z_loss_coef: float = 0.0
    # When True, the every-4th-and-last "long" layers skip the Partial Key
    # Offset (no shift of the second half of K, no doc-start zeroing). They
    # still run full causal attention (no sliding window); only the PKO step
    # is bypassed. Short layers are unaffected (PKO never ran on them).
    disable_pko: bool = False
    # When True, the long layers skip rotary embedding entirely (Q and K go
    # into attention un-rotated). Short layers still apply half-RoPE.
    disable_long_rope: bool = False
    attention_implementation: GrugAttentionImplementation | None = None
    moe_implementation: MoeImplementation | None = None
    ce_implementation: str | None = None
    """Fused cross-entropy backend selection (levanter fused_cross_entropy_loss). None keeps the
    backend default (GPU: full-logits ``xla`` path). Set ``"batched_xla"`` to use the blocked-vocab
    (cut) CE that avoids materializing the [tokens, vocab] logits tile -- large HBM saving at long
    context. None is byte-identical to the prior behaviour."""
    remat_mode: RematMode = "recompute_all"
    """Per-block gradient checkpointing. "recompute_all" reruns the whole block in
    backward (lowest memory); "save_moe" keeps the tagged MoE dispatch tensors so
    backward skips re-running expert dispatch and its EP collectives."""
    replicate_attn_weights: bool = False
    """If True, store w_q/w_k/w_v/w_o fully replicated (P(None, None)) instead
    of FSDP-sharded across the data axis. Attention weights are small (~128 KB
    per layer per chip after FSDP), so replicating them costs little HBM but
    eliminates the per-layer FSDP all-gather for the QKVO matmuls. MFU probe
    knob to isolate whether attention collectives matter."""
    split_w_gate_up: bool = True
    """If True (default), store MoE w_gate and w_up as separate pytree leaves
    so MuonH's Newton-Schulz orthogonalises each half independently. If False,
    fuse them into a single (E, D, 2I) tensor at init time, eliminating the
    per-forward concat temp (~6.7 GB at d=2560, BS=4096) and running one
    bigger NS call per expert. Changes NS semantics; use only for throughput
    probes unless you're OK with the different optimization target."""
    use_array_stacked_blocks: bool = False
    """Stack all transformer blocks into a single ``ArrayStacked[Block]`` and run
    them through one ``jax.lax.scan``. Collapses N per-layer subgraphs into one
    scan body so XLA only plans HBM for one iteration's intermediates — needed
    at scale where the unrolled program OOMs at compile time."""
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
        resolve_moe_implementation(self.moe_implementation)

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


def _rope_inv_freq(half_dim: int, theta: float) -> jax.Array:
    """RoPE inv_freq for half_dim channels."""
    return 1.0 / (theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))


def _apply_half_rope(
    q: jax.Array,
    k: jax.Array,
    *,
    seq_len: int,
    head_dim: int,
    rope: RotaryConfig,
) -> tuple[jax.Array, jax.Array]:
    """RoPE applied to the first ``head_dim`` channels of q/k.

    Matches ``levanter.grug.attention.apply_rotary_embedding``. Convention:
    ``x1, x2 = split(x, 2, axis=-1)``; rotation is
    ``[x1*cos - x2*sin, x2*cos + x1*sin]``.
    """
    half_dim = head_dim // 2
    inv_freq = _rope_inv_freq(half_dim, rope.theta)
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]

    def _apply(x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(dtype)

    return _apply(q), _apply(k)


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
        in_spec = P(None, None) if cfg.replicate_attn_weights else P("data", "model")
        out_spec = P(None, None) if cfg.replicate_attn_weights else P("model", "data")
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), cfg.initializer_std), in_spec),
            w_k=reshard(_init_weight(k_k, (d, m * h), cfg.initializer_std), in_spec),
            w_v=reshard(_init_weight(k_v, (d, m * h), cfg.initializer_std), in_spec),
            w_o=reshard(_init_weight(k_o, (n * h, d), cfg.initializer_std), out_spec),
            attn_gate=reshard(jnp.zeros((d, n)), P(None, None)),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        use_pko: bool = False,
        disable_rope: bool = False,
        qk_mult_scale: float = 1.0,
    ) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        batch_spec = _batch_spec()

        # Split the flattened head dim (num_heads * head_dim) into (heads, head_dim).
        # Under tensor/model parallelism the projection output's last dim is sharded over the
        # ``model`` axis, and JAX's explicit-mesh reshape cannot infer which output axis carries
        # that sharding on a split -> pass out_sharding explicitly (model on the head axis,
        # head_dim replicated). At model_axis==1 this is byte-identical to the old einops rearrange.
        _qkv_head_spec = P(_BATCH_AXES, None, "model", None)
        q = jnp.einsum("bsh,hd->bsd", x, self.w_q).reshape(
            (x.shape[0], seq_len, -1, head_dim), out_sharding=_qkv_head_spec
        )
        k = jnp.einsum("bsh,hd->bsd", x, self.w_k).reshape(
            (x.shape[0], seq_len, -1, head_dim), out_sharding=_qkv_head_spec
        )
        v = jnp.einsum("bsh,hd->bsd", x, self.w_v).reshape(
            (x.shape[0], seq_len, -1, head_dim), out_sharding=_qkv_head_spec
        )

        # Shift the second half of K's head_dim back by one position so the
        # query at position i sees K[i] on head_dim[:half] but K[i-1] on
        # head_dim[half:]. Zero the shifted half at document starts so the
        # cross-half look-back does not leak across docs. Runs before the
        # rms_norm on Q/K below.
        if use_pko:
            half = head_dim // 2
            k_stationary = k[..., half:]
            k_shifted = jnp.concatenate([k_stationary[:, :1, :, :], k_stationary[:, :-1, :, :]], axis=1)
            segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
            if segment_ids is None:
                # No segment info (raw-mask or unsegmented eval path): only position 0 is a doc start.
                is_doc_start_seq = jnp.zeros((seq_len,), dtype=bool).at[0].set(True)
                is_doc_start = jnp.broadcast_to(is_doc_start_seq, k_shifted.shape[:2])
            else:
                q_seg = segment_ids[0]
                if q_seg.ndim == 1:
                    is_doc_start_seq = jnp.concatenate([jnp.ones((1,), dtype=bool), q_seg[1:] != q_seg[:-1]])
                    is_doc_start = jnp.broadcast_to(is_doc_start_seq, k_shifted.shape[:2])
                else:
                    is_doc_start = jnp.concatenate(
                        [jnp.ones_like(q_seg[:, :1], dtype=bool), q_seg[:, 1:] != q_seg[:, :-1]],
                        axis=1,
                    )
            k_shifted = jnp.where(is_doc_start[..., None, None], jnp.zeros_like(k_shifted), k_shifted)
            k = jnp.concatenate([k[..., :half], k_shifted], axis=-1)

        q = rms_norm(q)
        k = rms_norm(k)
        cfg = self.cfg
        if not disable_rope:
            # Half-RoPE: apply rotary embedding only to first half of Q/K head_dim.
            # Second half is rope-free on every layer.
            half = head_dim // 2
            q_rot, k_rot = _apply_half_rope(
                q[..., :half],
                k[..., :half],
                seq_len=seq_len,
                head_dim=half,
                rope=cfg.rope,
            )
            q = jnp.concatenate([q_rot, q[..., half:]], axis=-1)
            k = jnp.concatenate([k_rot, k[..., half:]], axis=-1)
        q = q * cfg.qk_mult * qk_mult_scale
        attn_out = attention(q, k, v, mask, implementation=self.cfg.attention_implementation)
        # Half-RoPE's slice+concat on the head_dim axis can leave the explicit-mesh
        # propagator with ``model`` annotated on ``head_dim`` rather than
        # ``num_q_heads``; force the canonical TP layout so it matches ``aligned_v``.
        attn_out = reshard(attn_out, P(_BATCH_AXES, None, "model", None))
        aligned_v = align_kv_heads(v, num_q_heads=attn_out.shape[2])
        aligned_v = reshard(aligned_v, P(_BATCH_AXES, None, "model", None))
        # Exclusive Self Attention: subtract the component of yᵢ parallel to vᵢ.
        # zᵢ = yᵢ - (yᵢᵀvᵢ / ‖vᵢ‖²) vᵢ, per head.
        dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
        v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
        attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v
        # Headwise gating: sigmoid(x @ attn_gate) produces one scalar per head.
        gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.attn_gate))[..., None]
        attn_out = gate * attn_out
        # Merge (heads, head_dim) back to a flat head dim. The head axis is sharded over ``model``
        # (tensor parallel), so pin the merged dim's sharding explicitly rather than relying on
        # explicit-mesh reshape inference. At model_axis==1 this equals the old einops rearrange.
        attn_out = attn_out.reshape(
            (attn_out.shape[0], attn_out.shape[1], -1), out_sharding=P(_BATCH_AXES, None, "model")
        )
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


class GatedNorm(eqx.Module):
    """Learnable per-dimension gating. Compensates for AdamH's bounded activation norms.
    See https://arxiv.org/abs/2601.22966v1"""

    w_down: jax.Array
    w_up: jax.Array

    @staticmethod
    def init(hidden_dim: int, initializer_std: float, *, key: PRNGKeyArray) -> "GatedNorm":
        k_down, k_up = random.split(key)
        return GatedNorm(
            w_down=reshard(_init_weight(k_down, (hidden_dim, _GATED_NORM_RANK), initializer_std), P(None, None)),
            w_up=reshard(_init_weight(k_up, (_GATED_NORM_RANK, hidden_dim), initializer_std), P(None, None)),
        )

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        gate_hidden = jnp.einsum("...d,dr->...r", x, self.w_down)
        # TODO: silu activation here isn't explored, just cargo-culted from Qwen. Likely low-hanging ablation fruit
        # (e.g. compare no activation, relu, etc.).
        gate_hidden = jax.nn.silu(gate_hidden)
        gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, self.w_up))
        return x * gate.astype(x.dtype)


class DenseMLP(eqx.Module):
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array

    @staticmethod
    def init(hidden_dim: int, intermediate_dim: int, initializer_std: float, *, key: PRNGKeyArray) -> "DenseMLP":
        k_gate, k_up, k_down = random.split(key, 3)
        return DenseMLP(
            w_gate=reshard(_init_weight(k_gate, (hidden_dim, intermediate_dim), initializer_std), P("data", "model")),
            w_up=reshard(_init_weight(k_up, (hidden_dim, intermediate_dim), initializer_std), P("data", "model")),
            w_down=reshard(_init_weight(k_down, (intermediate_dim, hidden_dim), initializer_std), P("model", "data")),
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
        # Reshard after the reshape so the shared-expert output carries the same
        # canonical batch sharding as the routed MoE output (MoEMLP reshards its
        # routed result identically). Splitting the fused
        # ("replica_dcn", "data", "expert") token axis back into (b, s) otherwise
        # leaks the `expert` mesh axis onto the seq dim, so the shared+routed
        # residual add fails with a ShardingTypeError on a multi-node mesh.
        return _batch_reshard(rearrange(out_flat, "(b s) d -> b s d", b=b, s=s))


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


def _summarize_router_metrics(router_metrics: dict[str, jax.Array]) -> dict[str, jax.Array | SummaryStats]:
    routing_entropy = router_metrics["routing_entropy_per_layer"]
    routing_counts = router_metrics["routing_counts_per_layer"]
    load_balancing_loss = router_metrics["load_balancing_loss_per_layer"]
    router_z_loss = router_metrics["router_z_loss_per_layer"]
    capacity_overflow = router_metrics["capacity_overflow_per_layer"]
    num_layers = int(routing_entropy.shape[0])

    # Per-layer total assignments = sum of routing_counts over experts (= tokens * k).
    assignments_per_layer = jnp.sum(routing_counts.astype(jnp.float32), axis=-1)
    capacity_overflow_rate = capacity_overflow.astype(jnp.float32) / jnp.maximum(assignments_per_layer, 1.0)

    out: dict[str, jax.Array | SummaryStats] = {
        "train/router/routing_entropy_mean": jnp.mean(routing_entropy),
        "train/router/load_balancing_loss": jnp.mean(load_balancing_loss),
        "train/router/router_z_loss": jnp.mean(router_z_loss),
        "train/router/routing_counts_per_layer": routing_counts,
        "train/router/capacity_overflow_rate_mean": jnp.mean(capacity_overflow_rate),
        "qb_beta_per_layer": router_metrics["qb_beta_per_layer"],
    }
    for i in range(num_layers):
        out[f"train/router/layer_{i}/routing_entropy"] = routing_entropy[i]
        out[f"train/router/layer_{i}/load_balancing_loss"] = load_balancing_loss[i]
        out[f"train/router/layer_{i}/router_z_loss"] = router_z_loss[i]
        out[f"train/router/layer_{i}/routing_hist"] = _histogram_from_expert_counts(routing_counts[i])
        out[f"train/router/layer_{i}/capacity_overflow_rate"] = capacity_overflow_rate[i]
    return out


def _histogram_from_expert_counts(expert_counts: jax.Array) -> SummaryStats:
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
    histogram = Histogram(bucket_limits=bucket_limits, bucket_counts=counts)
    return SummaryStats.from_reduced_values(
        min=min_value,
        max=max_value,
        num=num,
        nonzero_count=jnp.sum(nonzero),
        sum=sum_values,
        sum_squares=sum_squares,
        histogram=histogram,
    )


class MoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights."""

    router: jax.Array
    router_bias: jax.Array
    expert_mlp: MoEExpertMlp
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MoEMLP":
        k_router, k_expert = random.split(key, 2)
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")
        if not cfg.split_w_gate_up:
            raise ValueError("the current Levanter MoE API only supports split w_gate and w_up weights")

        d, e = cfg.hidden_dim, cfg.num_experts
        return MoEMLP(
            router=reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None)),
            router_bias=jnp.zeros((e,)),
            expert_mlp=MoEExpertMlp.init(
                num_experts=cfg.num_experts,
                hidden_dim=cfg.hidden_dim,
                intermediate_dim=cfg.intermediate_dim,
                initializer_std=cfg.initializer_std,
                key=k_expert,
                implementation=cfg.moe_implementation,
                activation=ActivationFunctionEnum.silu,
                capacity_factor=_DEFAULT_EP_CAPACITY_FACTOR,
            ),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        # Keep the router path in fp32 before top-k, softmax, and QB statistics.
        router_logits = jnp.einsum("td,de->te", x_flat, reshard(self.router, P(None, None))).astype(jnp.float32)
        biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        # Select top-(K+1) on biased logits; the (K+1)-th is the QB threshold alpha.
        _topk_logits, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token + 1)
        qb_alpha = _topk_logits[:, -1:]
        selected_experts = selected_experts[:, :-1]
        # Sigmoid combine weights on unbiased logits for selected experts.
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        combine_weights_f = jax.nn.sigmoid(unbiased_topk)
        # Renormalize K combine weights to sum to ``_ROUTING_RENORM_SUM`` (baked in).
        denom = jnp.sum(combine_weights_f, axis=-1, keepdims=True)
        combine_weights_f = combine_weights_f * (_ROUTING_RENORM_SUM / (denom + 1e-9))
        combine_weights = combine_weights_f.astype(x.dtype)
        router_stats = _routing_stats(
            selected_experts,
            router_probs,
            router_logits,
            num_experts=self.cfg.num_experts,
            num_experts_per_token=self.cfg.num_experts_per_token,
        )
        # Sharded QB: compute beta locally per device, then average.
        mesh = get_abstract_mesh()
        s_minus_alpha = reshard(router_logits - qb_alpha, P(_BATCH_AXES, None))
        num_devices = 1
        for a in _BATCH_AXES:
            num_devices *= mesh.shape[a]
        local_tokens = s_minus_alpha.shape[0] // num_devices
        qb_count = max(1, local_tokens * self.cfg.num_experts_per_token // self.cfg.num_experts)

        def _local_qb_beta(s_ma):
            topk_vals, _ = jax.lax.top_k(s_ma.T, qb_count)
            beta = topk_vals[:, -1]
            return jax.lax.pmean(beta, axis_name=_BATCH_AXES)

        router_stats["qb_beta"] = shard_map(
            _local_qb_beta,
            mesh=mesh,
            in_specs=(P(_BATCH_AXES, None),),
            out_specs=P(),
        )(s_minus_alpha)

        routed_flat, dropped_assignments = self.expert_mlp(
            x_flat,
            selected_experts.astype(jnp.int32),
            combine_weights,
            mesh=get_abstract_mesh(),
            report_capacity_overflow=True,
        )
        router_stats["capacity_overflow"] = dropped_assignments.astype(jnp.float32)

        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        routed = reshard(routed, _batch_spec())
        return routed, router_stats


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    mlp: MoEMLP
    shared: DenseMLP | None

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Block":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        shared = None
        if cfg.shared_expert_intermediate_dim > 0:
            shared = DenseMLP.init(
                cfg.hidden_dim, cfg.shared_expert_intermediate_dim, cfg.initializer_std, key=shared_key
            )
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key),
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key),
            mlp=MoEMLP.init(cfg, key=mlp_key),
            shared=shared,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        short_mask: AttentionMask | jax.Array,
        long_mask: AttentionMask | jax.Array,
        use_long_mask: Bool[Array, ""] | bool,
        use_pko: bool = False,
        disable_long_rope: bool = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        # lax.cond so the body has a uniform shape across scan iterations:
        # long layers use the full causal mask (and may PKO / drop RoPE); short
        # layers use the sliding-window mask, never PKO, and always RoPE.
        attn_out = jax.lax.cond(
            jnp.asarray(use_long_mask, dtype=jnp.bool_),
            lambda _: self.attn(
                attn_in,
                long_mask,
                use_pko=use_pko,
                disable_rope=disable_long_rope,
                qk_mult_scale=self.attn.cfg.qk_mult_long_scale,
            ),
            lambda _: self.attn(attn_in, short_mask, use_pko=False, disable_rope=False, qk_mult_scale=1.0),
            operand=None,
        )
        x = x + attn_out
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        mlp_out, router_stats = self.mlp(mlp_in)
        if self.shared is not None:
            mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        x = x + mlp_out
        return x, router_stats


def _long_layer_schedule(num_layers: int) -> jax.Array:
    """Bool[num_layers] = True for every 4th layer and the last layer."""
    idx = jnp.arange(num_layers)
    return ((idx % 4) == 3) | (idx == num_layers - 1)


class Transformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm
    output_proj: jax.Array
    # Exactly one of ``blocks`` / ``stacked_blocks`` is populated:
    #   - ``blocks``: per-block path (use_array_stacked_blocks=False)
    #   - ``stacked_blocks``: stacked path (homogeneous Block, lax.scan)
    blocks: tuple[Block, ...] | None
    stacked_blocks: ArrayStacked[Block] | None
    final_norm: RMSNorm
    final_gated_norm: GatedNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        if cfg.use_array_stacked_blocks and not cfg.disable_pko:
            raise ValueError(
                "use_array_stacked_blocks=True currently requires disable_pko=True "
                "because CausalSelfAttention reads use_pko at trace time."
            )

        # 4 module-level keys + per-layer keys.
        keys = random.split(key, cfg.num_layers + 4)
        embed_key, out_key, embed_gn_key, final_gn_key = keys[:4]
        block_keys = keys[4:]
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)

        blocks: tuple[Block, ...] | None
        stacked_blocks: ArrayStacked[Block] | None
        if cfg.use_array_stacked_blocks:
            blocks = None
            stacked_blocks = ArrayStacked.init(cfg.num_layers, Block)(cfg=cfg, key=block_keys)
        else:
            blocks = tuple(Block.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers))
            stacked_blocks = None

        return Transformer(
            token_embed=token_embed,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key),
            output_proj=output_proj,
            blocks=blocks,
            stacked_blocks=stacked_blocks,
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            final_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key),
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
        cfg = self.config
        hidden = self.token_embed.at[token_ids].get(out_sharding=batch_spec)
        hidden = self.embed_norm(hidden)
        hidden = self.embed_gated_norm(hidden)

        # Short layers: sliding window. Long layers (every 4th + last): full causal.
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)

        if cfg.remat_mode == "save_moe":
            remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
        else:
            remat_policy = None

        if self.blocks is not None:
            num_blocks = len(self.blocks)
            moe_router_stats: list[dict[str, jax.Array]] = []
            for i, block in enumerate(self.blocks):
                is_last = i == num_blocks - 1
                is_long = i % 4 == 3 or is_last
                use_pko = is_long and not cfg.disable_pko
                hidden, router_stats = eqx.filter_checkpoint(block, policy=remat_policy)(
                    hidden, short_mask, long_mask, is_long, use_pko, cfg.disable_long_rope
                )
                moe_router_stats.append(router_stats)
            router_metrics = {
                "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
                "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
                "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in moe_router_stats], axis=0),
                "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in moe_router_stats], axis=0),
                "qb_beta_per_layer": jnp.stack([s["qb_beta"] for s in moe_router_stats], axis=0),
                "capacity_overflow_per_layer": jnp.stack([s["capacity_overflow"] for s in moe_router_stats], axis=0),
            }
        else:
            assert self.stacked_blocks is not None
            mask_schedule = _long_layer_schedule(cfg.num_layers)

            def _scan_layers(
                carry_hidden: Float[Array, "B S D"],
                scan_inputs: tuple[Block, Bool[Array, ""]],
            ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
                layer, layer_use_long_mask = scan_inputs
                return eqx.filter_checkpoint(layer, policy=remat_policy)(
                    carry_hidden, short_mask, long_mask, layer_use_long_mask, False, cfg.disable_long_rope
                )

            hidden, stacked_router_stats = jax.lax.scan(
                _scan_layers,
                hidden,
                xs=(self.stacked_blocks.stacked, mask_schedule),
            )
            router_metrics = {
                "routing_entropy_per_layer": stacked_router_stats["routing_entropy"],
                "routing_counts_per_layer": stacked_router_stats["routing_counts"],
                "load_balancing_loss_per_layer": stacked_router_stats["load_balancing_loss"],
                "router_z_loss_per_layer": stacked_router_stats["router_z_loss"],
                "qb_beta_per_layer": stacked_router_stats["qb_beta"],
                "capacity_overflow_per_layer": stacked_router_stats["capacity_overflow"],
            }
        hidden = self.final_gated_norm(self.final_norm(hidden))
        return hidden, router_metrics

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
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
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
            implementation=self.config.ce_implementation,
        )
        # No load-balancing loss; router z-loss only.
        num_moe_layers = router_metrics["router_z_loss_per_layer"].shape[0]
        rzl = jnp.sum(router_metrics["router_z_loss_per_layer"]) / num_moe_layers
        aux_loss = self.config.router_z_loss_coef * rzl
        loss = cross_entropy_loss + aux_loss if reduction != "none" else cross_entropy_loss
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
    expert = 2 if num_devices % 2 == 0 else 1
    data = max(1, num_devices // expert)
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1, data, expert, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    return mesh, P(("replica_dcn", "data", "expert"), None)


__all__ = [
    "Block",
    "CausalSelfAttention",
    "DenseMLP",
    "GatedNorm",
    "GrugModelConfig",
    "MoEMLP",
    "MoeActivation",
    "RMSNorm",
    "Transformer",
    "debug_mesh_and_token_pspec",
]
