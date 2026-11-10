# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model.

Architecture: QB-routed MoE with GatedNorm, XSA, sigmoid combine weights.
No load-balancing loss; router z-loss only. All layers are MoE (no dense layers).
"""

import dataclasses
import functools
import math
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from levanter.grug.attention import (
    AttentionMask,
    GrugAttentionImplementation,
    RotaryConfig,
    align_kv_heads,
    attention,
    token_validity_from_attention_mask,
)
from levanter.grug.grug_moe import (
    MOE_DROPPED_ASSIGNMENTS_METRIC,
    MOE_REMAT_SAVE_NAMES,
    MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC,
    MOE_VALID_ASSIGNMENTS_METRIC,
    MoeActivation,
    MoEExpertMlp,
    MoeImplementation,
    QBRoutedMoE,
    validate_moe_implementation,
)
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pembed_vocab, Plm_head, _mesh_axis_size, unshard
from levanter.kernels.pallas.add_rms_norm import AddRmsNormImpl, add_rms_norm_bwd, add_rms_norm_fwd
from levanter.kernels.pallas.gated_norm import GatedNormImpl, gated_norm_gate_bwd, gated_norm_gate_fwd, gated_norm_silu
from levanter.layers.stacked import ArrayStacked
from levanter.tracker.histogram import Histogram, SummaryStats
from levanter.utils.activation import ActivationFunctionEnum

_DEFAULT_EP_CAPACITY_FACTOR = 1.0
_GATED_NORM_RANK = 128


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


RematMode = Literal["recompute_all", "save_moe"]
FusedNormImpl = Literal["off", "xla", "pallas"]
# GatedNorm's rank-128 down projection ``n w_down``: a remat policy that saves it skips the down GEMM when the backward
# reruns the block (``GrugModelConfig.gated_norm_remat``).
GATED_NORM_REMAT_SAVE_NAME = "gated_norm_hidden"
GatedNormRemat = Literal["recompute", "save_hidden"]
# The attention's head-gate logits ``x attn_gate`` ([B, S, num_heads]; ``GrugModelConfig.head_gate_remat``).
HEAD_GATE_REMAT_SAVE_NAME = "attn_head_gate"
HeadGateRemat = Literal["recompute", "save"]


def _batch_spec() -> P:
    return P(_BATCH_AXES)


def _batch_reshard(x: jax.Array) -> jax.Array:
    return reshard(x, _batch_spec())


def _layer_attention_masks(mask: AttentionMask, *, sliding_window: int) -> tuple[AttentionMask, AttentionMask]:
    return mask.with_sliding_window(sliding_window // 2), mask.with_sliding_window(sliding_window)


LONG_ATTENTION_INTERVAL = 4


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
    capacity_factor: float = _DEFAULT_EP_CAPACITY_FACTOR
    pooled_transport_capacity_factor: float | None = None
    """Sender pool capacity per destination shard; required by ``ring_filtered`` and ``fixed_pooled_wave_all_to_all``."""
    num_expert_waves: int = 1
    """Static wave count for ``fixed_pooled_wave_all_to_all``; must divide the local expert count."""
    ce_implementation: str | None = None
    """Fused cross-entropy backend selection (levanter fused_cross_entropy_loss). None keeps the
    backend default (GPU: full-logits ``xla`` path). Set ``"batched_xla"`` to use the blocked-vocab
    (cut) CE that avoids materializing the [tokens, vocab] logits tile -- large HBM saving at long
    context. None is byte-identical to the prior behaviour."""
    remat_mode: RematMode = "recompute_all"
    """Per-block gradient checkpointing. "recompute_all" reruns the whole block in
    backward (lowest memory); "save_moe" keeps the tagged MoE dispatch tensors so
    backward skips re-running expert dispatch and its EP collectives."""
    fused_norm_impl: FusedNormImpl = "off"
    """How each ``RMSNorm`` followed by its ``GatedNorm`` runs (the two per block, and the embedding and final norms),
    together with the residual add before the block's MLP norm. "off": the separate modules and a plain add. "xla" and
    "pallas": ``fused_norm_and_gate``, one ``jax.custom_vjp`` per norm whose forward returns the residual sum ``h`` and
    the gated output and whose backward hands the GatedNorm's two cotangent terms and the residual's cotangent to a
    single RMS-norm backward, written in JAX ops ("xla") or as Pallas-Triton kernels for GPU, one per direction
    ("pallas", ``levanter.kernels.pallas.add_rms_norm``). Both compute the norm in float32 from the unrounded residual
    sum and round its outputs once, so they differ from "off" at bfloat16 rounding level (measured at [8, 4096, 2560]
    against a float32 run: the same error in ``h``, ``dx`` and the norm weight's gradient as "off", within 1.4x of it
    in the GatedNorm output and weight gradients)."""
    gated_norm_impl: GatedNormImpl = "xla"
    """How the fused norm (``fused_norm_impl`` "xla" or "pallas") runs GatedNorm's gate. "xla": JAX ops, the backward
    keeping the [B, S, D] gate. Under remat that gate has two users, so XLA writes it from the up-projection GEMM and
    multiplies it into the output in a separate kernel, and in the backward writes the up projection's cotangent in one
    kernel and reads it again in its GEMM. "pallas": the up projection, sigmoid and product in one Pallas-Triton kernel
    that writes the output and the gate, and a backward kernel that forms the up projection's cotangent and its
    rank-128 GEMM in one pass (``levanter.kernels.pallas.gated_norm``); the down projection and the weight-gradient
    GEMMs stay in XLA (deterministic). Float32 between roundings to the activation dtype, so the gate and its gradients
    differ from "xla" at bfloat16 rounding level. Needs ``fused_norm_impl`` "xla" or "pallas"."""
    gated_norm_remat: GatedNormRemat = "recompute"
    """Whether the per-block gradient checkpoint saves GatedNorm's rank-128 down projection ``n w_down`` (bfloat16
    [B, S, 128], 8 MiB per norm per layer at 8 x 4096 tokens per device): "recompute" (the remat reruns the down GEMM)
    or "save_hidden" (adds it to the ``remat_mode`` policy; the backward's rerun of the block skips the down GEMM).
    Same values either way."""
    head_gate_remat: HeadGateRemat = "recompute"
    """Whether the per-block gradient checkpoint saves the attention's head-gate logits ``x attn_gate`` ([B, S,
    num_heads], 1.3 MiB per layer at 8 x 4096 tokens per device in bfloat16): "recompute" (the backward's rerun of the
    block reruns that N=num_heads GEMM over the full hidden width) or "save" (adds them to the ``remat_mode``
    policy). Same values either way."""
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
        if not math.isfinite(self.capacity_factor) or self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be finite and positive")
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
        validate_moe_implementation(self.moe_implementation)
        if self.gated_norm_impl == "pallas" and self.fused_norm_impl == "off":
            raise ValueError('gated_norm_impl="pallas" needs fused_norm_impl "xla" or "pallas"')

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
    # AD retains different intermediates; pin the reduction to preserve forward rounding.
    variance = jax.lax.optimization_barrier(variance)
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
    position_ids: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """RoPE applied to the first ``head_dim`` channels of q/k.

    Matches ``levanter.grug.attention.apply_rotary_embedding``. Convention:
    ``x1, x2 = split(x, 2, axis=-1)``; rotation is
    ``[x1*cos - x2*sin, x2*cos + x1*sin]``.
    """
    half_dim = head_dim // 2
    inv_freq = _rope_inv_freq(half_dim, rope.theta)
    positions = (
        jnp.arange(seq_len, dtype=jnp.float32)[None, :] if position_ids is None else position_ids.astype(jnp.float32)
    )
    angles = positions[..., None] * inv_freq
    cos = jnp.cos(angles)[:, :, None, :]
    sin = jnp.sin(angles)[:, :, None, :]

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
        position_ids: jax.Array | None = None,
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
                position_ids=position_ids,
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
        gate_logits = checkpoint_name(jnp.einsum("bsd,dn->bsn", x, self.attn_gate), HEAD_GATE_REMAT_SAVE_NAME)
        gate = 2 * jax.nn.sigmoid(gate_logits)[..., None]
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
        # AD retains different intermediates; pin the reduction to preserve forward rounding.
        variance = jax.lax.optimization_barrier(variance)
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
        return x * _gated_norm_gate(x, self.w_down, self.w_up).astype(x.dtype)


def _gated_norm_gate(x: jax.Array, w_down: jax.Array, w_up: jax.Array) -> jax.Array:
    """``GatedNorm``'s gate ``sigmoid(silu(x w_down) w_up)``."""
    # TODO: silu activation here isn't explored, just cargo-culted from Qwen. Likely low-hanging ablation fruit
    # (e.g. compare no activation, relu, etc.).
    gate_hidden = jax.nn.silu(_gated_norm_hidden(x, w_down))
    return jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, w_up))


def _gated_norm_hidden(x: jax.Array, w_down: jax.Array) -> jax.Array:
    """GatedNorm's down projection, named for the remat policy (``GrugModelConfig.gated_norm_remat``)."""
    return checkpoint_name(jnp.einsum("...d,dr->...r", x, w_down), GATED_NORM_REMAT_SAVE_NAME)


def _per_device(fn, *args, out_specs):
    """``fn`` on each device's rows: ``args`` are activations [B, S, D] (batch-sharded; None passes through) or
    replicated vectors and matrices; ``fn`` sees activations as rows [b * s, D] and returns rows (or [n, D] partial
    sums, ``out_specs`` "partials", one per device in the result's leading dimension)."""
    mesh = get_abstract_mesh()
    first = next(a for a in args if a is not None and a.ndim == 3)
    act_spec = jax.typeof(first).sharding.spec
    s = first.shape[1]
    if len(act_spec) > 2 and act_spec[2] is not None:
        raise ValueError(f"the fused norm needs an unsharded hidden dimension, got {act_spec}")

    def local(*xs):
        rows = [x.reshape(-1, x.shape[-1]) if x is not None and x.ndim == 3 else x for x in xs]
        outs = fn(*rows)
        return tuple(
            o.reshape(-1, s, o.shape[-1]) if spec == "rows" else o[None] for o, spec in zip(outs, out_specs, strict=True)
        )

    if mesh.empty:
        outs = local(*args)
        return tuple(o if spec == "rows" else o[0] for o, spec in zip(outs, out_specs, strict=True))
    batch = act_spec[0] if len(act_spec) else None
    in_specs = tuple(None if a is None else act_spec if a.ndim == 3 else P(*(None,) * a.ndim) for a in args)
    specs = tuple(act_spec if spec == "rows" else P(batch, None) for spec in out_specs)
    return jax.shard_map(local, in_specs=in_specs, out_specs=specs, check_vma=False)(*args)


def _weight_gradient(partials: jax.Array, weight: jax.Array) -> jax.Array:
    """The weight gradient from per-device partial sums [n, D] (or the sum itself without a mesh)."""
    partials = partials.astype(jnp.float32)
    if partials.ndim == 1:  # no mesh
        return partials.astype(weight.dtype)
    return reshard(jnp.sum(partials, axis=0).astype(weight.dtype), jax.typeof(weight).sharding.spec)


def _norm_and_gate_fwd(x, residual, weight, w_down, w_up, eps, impl, gate_impl, return_h):
    def kernel(x, residual, weight):
        h, n, inv_rms = add_rms_norm_fwd(x, residual, weight, eps=eps, impl=impl)
        return (n, inv_rms[:, None]) if residual is None else (h, n, inv_rms[:, None])

    if residual is None:
        h = x
        n, inv_rms = _per_device(kernel, x, None, weight, out_specs=("rows", "rows"))
    else:
        h, n, inv_rms = _per_device(kernel, x, residual, weight, out_specs=("rows", "rows", "rows"))
    if gate_impl == "xla":
        gate, gate_vjp = jax.vjp(_gated_norm_gate, n, w_down, w_up)
        out = n * gate.astype(n.dtype)
        gate_res = (gate, gate_vjp)
    else:
        a = _gated_norm_hidden(n, w_down).astype(n.dtype)
        out, gate = _per_device(
            lambda n, a, w_up: gated_norm_gate_fwd(n, a, w_up, impl=gate_impl), n, a, w_up, out_specs=("rows", "rows")
        )
        gate_res = (gate, a, w_down, w_up)
    return (h, out) if return_h else out, (h, inv_rms, weight, n, gate_res)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _norm_and_gate(
    x, residual, weight, w_down, w_up, eps: float, impl: AddRmsNormImpl, gate_impl: GatedNormImpl, return_h: bool
):
    """``fused_norm_and_gate``'s custom VJP."""
    return _norm_and_gate_fwd(x, residual, weight, w_down, w_up, eps, impl, gate_impl, return_h)[0]


def _gate_bwd_pallas(n, d_out, a, gate, w_down, w_up, gate_impl):
    """``(d_n_from_gate, d_w_down, d_w_up)`` through the gate kernels' backward, then XLA GEMMs for the rest (the
    transposes of GatedNorm's own matmuls)."""
    dz, da = _per_device(
        lambda n, a, gate, w_up, d_out: gated_norm_gate_bwd(n, a, gate, w_up, d_out, impl=gate_impl),
        n,
        a,
        gate,
        w_up,
        d_out,
        out_specs=("rows", "rows"),
    )
    _, down_vjp = jax.vjp(lambda n, w: jnp.einsum("...d,dr->...r", n, w).astype(n.dtype), n, w_down)
    d_n_from_gate, d_w_down = down_vjp(da)
    _, up_vjp = jax.vjp(lambda w: jnp.einsum("...r,rd->...d", gated_norm_silu(a), w).astype(n.dtype), w_up)
    (d_w_up,) = up_vjp(dz)
    return d_n_from_gate, d_w_down, d_w_up


def _norm_and_gate_bwd(eps, impl, gate_impl, return_h, residuals, cotangents):
    del eps
    h, inv_rms, weight, n, gate_res = residuals
    gate = gate_res[0]
    d_h, d_out = cotangents if return_h else (None, cotangents)
    # out = n * gate: the gate's cotangent goes back through GatedNorm's matmuls (its d_n term), n's own term is
    # d_out * gate; the RMS-norm backward adds both, and the residual's cotangent, in one pass.
    if gate_impl == "xla":
        d_n_from_gate, d_w_down, d_w_up = gate_res[1]((d_out * n).astype(gate.dtype))
    else:
        _, a, w_down, w_up = gate_res
        d_n_from_gate, d_w_down, d_w_up = _gate_bwd_pallas(n, d_out, a, gate, w_down, w_up, gate_impl)

    def kernel(h, inv_rms, weight, d_out, gate, d_n_from_gate, d_h):
        return add_rms_norm_bwd(h, inv_rms[:, 0], weight, d_out, gate=gate, d_n_extra=d_n_from_gate, d_h=d_h, impl=impl)

    dx, dw = _per_device(
        kernel, h, inv_rms, weight, d_out, gate.astype(n.dtype), d_n_from_gate, d_h, out_specs=("rows", "partials")
    )
    # h = x + residual: both get h's cotangent (``None`` stands for the absent residual's).
    return dx, dx, _weight_gradient(dw, weight), d_w_down, d_w_up


_norm_and_gate.defvjp(_norm_and_gate_fwd, _norm_and_gate_bwd)


def fused_norm_and_gate(
    x: Float[Array, "B S D"],
    residual: Float[Array, "B S D"] | None,
    norm: RMSNorm,
    gated: GatedNorm,
    impl: AddRmsNormImpl,
    *,
    return_h: bool,
    gate_impl: GatedNormImpl = "xla",
):
    """``out = gated(norm(h))`` with ``h = x + residual`` (``h = x`` without a residual); ``(h, out)`` if
    ``return_h``, else ``out``.

    One ``jax.custom_vjp`` over the add, the RMS norm and the GatedNorm (``GrugModelConfig.fused_norm_impl``). Its
    backward adds ``h``'s own cotangent (from the caller's later uses of ``h``) inside the norm's backward kernel, so
    callers that use ``x`` again should use the returned ``h`` instead. The norm's kernels, and with ``gate_impl``
    "pallas" (``GrugModelConfig.gated_norm_impl``) the gate's, run per device under a ``shard_map`` over the
    activations' batch sharding; GatedNorm's other matmuls stay global, as in ``GatedNorm``."""
    return _norm_and_gate(
        x, residual, unshard(norm.weight), gated.w_down, gated.w_up, norm.eps, impl, gate_impl, return_h
    )


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


def _summarize_router_metrics(router_metrics: dict[str, jax.Array]) -> dict[str, jax.Array | SummaryStats]:
    routing_entropy = router_metrics["routing_entropy_per_layer"]
    routing_counts = router_metrics["routing_counts_per_layer"]
    load_balancing_loss = router_metrics["load_balancing_loss_per_layer"]
    router_z_loss = router_metrics["router_z_loss_per_layer"]
    capacity_overflow = router_metrics["capacity_overflow_per_layer"]
    skipped_assignments = router_metrics["skipped_assignments_per_layer"]
    num_layers = int(routing_entropy.shape[0])

    # Per-layer total assignments = sum of routing_counts over experts (= tokens * k).
    assignments_per_layer = jnp.sum(routing_counts.astype(jnp.float32), axis=-1)
    capacity_overflow_rate = capacity_overflow.astype(jnp.float32) / jnp.maximum(assignments_per_layer, 1.0)
    total_positions_per_layer = assignments_per_layer + skipped_assignments.astype(jnp.float32)
    skipped_fraction = skipped_assignments.astype(jnp.float32) / jnp.maximum(total_positions_per_layer, 1.0)

    out: dict[str, jax.Array | SummaryStats] = {
        "train/router/routing_entropy_mean": jnp.mean(routing_entropy),
        "train/router/load_balancing_loss": jnp.mean(load_balancing_loss),
        "train/router/router_z_loss": jnp.mean(router_z_loss),
        "train/router/routing_counts_per_layer": routing_counts,
        "train/router/capacity_overflow_rate_mean": jnp.mean(capacity_overflow_rate),
        "train/router/skipped_padding_fraction_mean": jnp.mean(skipped_fraction),
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
    routed_moe: QBRoutedMoE
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

        return MoEMLP(
            router=reshard(
                _init_weight(k_router, (cfg.hidden_dim, cfg.num_experts), cfg.initializer_std), P(None, None)
            ),
            router_bias=jnp.zeros((cfg.num_experts,)),
            expert_mlp=MoEExpertMlp.init(
                num_experts=cfg.num_experts,
                hidden_dim=cfg.hidden_dim,
                intermediate_dim=cfg.intermediate_dim,
                initializer_std=cfg.initializer_std,
                key=k_expert,
                implementation=cfg.moe_implementation,
                activation=ActivationFunctionEnum.silu,
                capacity_factor=cfg.capacity_factor,
                pooled_transport_capacity_factor=cfg.pooled_transport_capacity_factor,
                num_expert_waves=cfg.num_expert_waves,
            ),
            routed_moe=QBRoutedMoE(
                num_experts_per_token=cfg.num_experts_per_token,
                batch_axes=_BATCH_AXES,
            ),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        token_valid: Bool[Array, "B S"],
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        token_valid_flat = rearrange(token_valid, "b s -> (b s)")
        routed_flat, router_stats = self.routed_moe(
            x_flat,
            token_valid_flat,
            router=self.router,
            router_bias=self.router_bias,
            expert_mlp=self.expert_mlp,
            mesh=get_abstract_mesh(),
        )

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
        position_ids: jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        fused = self.attn.cfg.fused_norm_impl
        gate_impl = self.attn.cfg.gated_norm_impl
        if fused == "off":
            attn_in = self.attn_gated_norm(self.rms_attn(x))
        else:
            # x comes back as h, so its cotangent from the residual path joins the norm's backward kernel.
            x, attn_in = fused_norm_and_gate(
                x, None, self.rms_attn, self.attn_gated_norm, fused, return_h=True, gate_impl=gate_impl
            )
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
                position_ids=position_ids,
            ),
            lambda _: self.attn(
                attn_in, short_mask, use_pko=False, disable_rope=False, qk_mult_scale=1.0, position_ids=position_ids
            ),
            operand=None,
        )
        if fused == "off":
            x = x + attn_out
            mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        else:
            x, mlp_in = fused_norm_and_gate(
                x, attn_out, self.rms_mlp, self.mlp_gated_norm, fused, return_h=True, gate_impl=gate_impl
            )
        token_valid = token_validity_from_attention_mask(
            short_mask,
            batch_size=x.shape[0],
            sequence_length=x.shape[1],
        )
        mlp_out, router_stats = self.mlp(mlp_in, token_valid)
        if self.shared is not None:
            mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        x = x + mlp_out
        return x, router_stats


def _long_layer_schedule(num_layers: int) -> jax.Array:
    """Bool[num_layers] = True for every 4th layer and the last layer."""
    idx = jnp.arange(num_layers)
    return ((idx % LONG_ATTENTION_INTERVAL) == LONG_ATTENTION_INTERVAL - 1) | (idx == num_layers - 1)


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
        *,
        position_ids: jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        if mask is None:
            mask = AttentionMask.causal()

        batch_spec = _batch_spec()
        cfg = self.config
        hidden = self.token_embed.at[token_ids].get(out_sharding=batch_spec)
        # Keep gather fusion from changing normalization rounding between scoring and AD.
        hidden = jax.lax.optimization_barrier(hidden)
        hidden = self._norm_and_gate(hidden, self.embed_norm, self.embed_gated_norm)

        # Short layers: sliding window. Long layers (every 4th + last): full causal.
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)

        saved_names = MOE_REMAT_SAVE_NAMES if cfg.remat_mode == "save_moe" else ()
        if cfg.gated_norm_remat == "save_hidden":
            saved_names = (*saved_names, GATED_NORM_REMAT_SAVE_NAME)
        if cfg.head_gate_remat == "save":
            saved_names = (*saved_names, HEAD_GATE_REMAT_SAVE_NAME)
        remat_policy = jax.checkpoint_policies.save_only_these_names(*saved_names) if saved_names else None

        if self.blocks is not None:
            num_blocks = len(self.blocks)
            moe_router_stats: list[dict[str, jax.Array]] = []
            for i, block in enumerate(self.blocks):
                is_last = i == num_blocks - 1
                is_long = i % LONG_ATTENTION_INTERVAL == LONG_ATTENTION_INTERVAL - 1 or is_last
                use_pko = is_long and not cfg.disable_pko
                hidden, router_stats = eqx.filter_checkpoint(block, policy=remat_policy)(
                    hidden, short_mask, long_mask, is_long, use_pko, cfg.disable_long_rope, position_ids
                )
                moe_router_stats.append(router_stats)
            router_metrics = {
                "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
                "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
                "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in moe_router_stats], axis=0),
                "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in moe_router_stats], axis=0),
                "qb_beta_per_layer": jnp.stack([s["qb_beta"] for s in moe_router_stats], axis=0),
                "capacity_overflow_per_layer": jnp.stack([s["capacity_overflow"] for s in moe_router_stats], axis=0),
                "skipped_assignments_per_layer": jnp.stack([s["skipped_assignments"] for s in moe_router_stats], axis=0),
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
                    carry_hidden, short_mask, long_mask, layer_use_long_mask, False, cfg.disable_long_rope, position_ids
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
                "skipped_assignments_per_layer": stacked_router_stats["skipped_assignments"],
            }
        hidden = self._norm_and_gate(hidden, self.final_norm, self.final_gated_norm)
        return hidden, router_metrics

    def _norm_and_gate(self, x: jax.Array, norm: RMSNorm, gated: GatedNorm) -> jax.Array:
        cfg = self.config
        if cfg.fused_norm_impl == "off":
            return gated(norm(x))
        return fused_norm_and_gate(
            x, None, norm, gated, cfg.fused_norm_impl, return_h=False, gate_impl=cfg.gated_norm_impl
        )

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
            summarized_metrics[MOE_DROPPED_ASSIGNMENTS_METRIC] = router_metrics["capacity_overflow_per_layer"]
            summarized_metrics[MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC] = router_metrics["skipped_assignments_per_layer"]
            summarized_metrics[MOE_VALID_ASSIGNMENTS_METRIC] = jnp.sum(
                router_metrics["routing_counts_per_layer"], axis=-1, dtype=jnp.int32
            )
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
    axis_names = ("replica_dcn", "data", "expert", "context", "model")
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1, data, expert, 1, 1),
        axis_names=axis_names,
        axis_types=(jax.sharding.AxisType.Explicit,) * len(axis_names),
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
