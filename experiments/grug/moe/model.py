# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model with Multi-head Latent Attention.

Architecture: QB-routed MoE with GatedNorm, sigmoid combine weights, Multi-head
Latent Attention (MLA, DeepSeek-V2 style) on the attention path. No load-balancing
loss; router z-loss only. All layers are MoE (no dense layers). All layers run
identical full causal attention (no sliding window, no PKO, no long/short split).

Simplifications vs the prior CausalSelfAttention:
  - Multi-head latent attention with a single ``kv_compression_dim`` knob:
    cache the compressed KV latent c_kv + a shared rope-bearing K (k_r broadcast
    across heads). Half-RoPE convention preserved (first head_dim/2 rope-bearing,
    rest no-rope).
  - Removed: attention gate, XSA (exclusive self-attention v-parallel subtract),
    PKO (partial key offset / doc-start zero), sliding window, GQA num_kv_heads,
    disable_long_rope, all long/short layer branching.
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
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard

try:
    from jax.shard_map import shard_map
except ModuleNotFoundError:
    from jax.experimental.shard_map import shard_map
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import (
    AttentionMask,
    GrugAttentionImplementation,
    RotaryConfig,
    apply_rotary_embedding,
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


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the grug MoE transformer.

    Architecture choices (GatedNorm, MLA, QB routing) are hardcoded.
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
    head_dim: int | None = None
    kv_compression_dim: int | None = None
    """MLA KV-latent dim ``d_c``. None defaults to ``hidden_dim // 4``."""
    mla_norm_compressed: bool = False
    """When True, apply RMSNorm to the compressed KV latent ``c_kv`` (DeepSeek-V2
    recipe) and skip the post-up-projection norm on ``k_nr``. When False, apply
    RMSNorm to ``k_nr`` after up-projection (legacy code path). The compressed
    placement preserves the matrix-absorption inference optimization that lets
    MLA cache only ``c_kv + k_r`` at decode -- ``rms_norm(k_nr)`` would force a
    per-token reconstruction step at decode."""
    qk_rope_head_dim: int | None = None
    """When None (default), use the legacy half-RoPE convention: per head
    head_dim is shared between Q/K/V, rope is applied to the first head_dim//2
    channels of Q and to the shared rope-bearing K (also head_dim//2 wide).
    When set, switch to DeepSeek-V2 additive-rope: per head Q and K have
    ``head_dim + qk_rope_head_dim`` channels total (no-rope = head_dim,
    additional rope = qk_rope_head_dim shared across heads); V stays at
    head_dim. Activating this flag also switches normalisation to the
    DeepSeek-strict scheme: ``rms_norm(c_kv)`` only, no QK norm on q/k/v."""
    mla_norm_compressed_learnable: bool = False
    """When True, the RMSNorm applied to ``c_kv`` uses a learnable per-channel
    weight (DeepSeek-V2's ``kv_a_layernorm``) instead of the gainless
    ``rms_norm()`` function. Only takes effect when c_kv is actually being
    normalized (i.e., ``mla_norm_compressed=True`` or ``qk_rope_head_dim`` is
    set). Adds ``(d_c,)`` learnable params per layer."""
    xsa: bool = False
    """When True, apply exclusive self-attention (XSA) after the attention call:
    subtract the per-head component of ``attn_out`` that is parallel to ``v`` at
    the same query position. ``z = y - (y·v / ‖v‖²) v``. In MLA, ``v`` is per-head
    by construction (up-projected from c_kv via w_uv), so no align_kv_heads is
    needed. Cost is 3 small per-head ops; absorption-compatible at decode via
    a precomputed M[h] = w_uv[h].T @ w_uv[h] matrix."""
    xsa_alternate_skip_first: bool = False
    """When True (and ``xsa=True``), XSA fires only on odd-indexed layers
    (i % 2 == 1) -- skipping layer 0, applying on layer 1, skipping layer 2, etc.
    With an even ``num_layers`` this lands XSA on the final layer."""
    pko_last_layer: bool = False
    """When True, apply Partial Key Offset (PKO) on the final transformer layer
    only. PKO shifts the no-rope K signal back by one position (with doc-start
    zero) so each query at position i attends to K's no-rope content from
    position i-1 instead of i. For MLA the shift is applied directly to
    ``c_kv`` before the no-rope K up-projection (path-agnostic for half-RoPE
    vs additive rope; V's c_kv path is left unshifted). Absorption-compatible:
    at decode you just index c_kv[j-1] instead of c_kv[j] for the no-rope dot
    product."""
    pko_half_k_dims: bool = False
    """When True (and PKO is being applied via ``pko_last_layer``), only half of
    the per-head k_nope channels come from the PKO-shifted c_kv; the other half
    come from the unshifted c_kv. Mimics the original PKO's "shift only half
    head_dim" behavior. Costs one extra c_kv up-projection on the PKO layer."""
    attn_gate: bool = False
    """When True, add a per-head sigmoid attention gate
    ``gate = 2 * sigmoid(x @ w_attn_gate)`` with ``w_attn_gate: (D, NH)``,
    applied as a scalar per (token, head) multiplier on the attention output
    before the output projection. Zero-init means gate=1.0 at step 0 (matches
    no-gate model); learns to gate down from pass-through. Absorption-friendly:
    the gate depends only on the current query, so at decode it adds one
    (D, NH) matmul per step plus a per-head scalar multiply on the absorbed
    output -- no cache interaction."""
    max_seq_len: int = 4096
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.3
    router_z_loss_coef: float = 0.0
    attention_implementation: GrugAttentionImplementation | None = None
    moe_implementation: MoeImplementation | None = None
    remat_mode: RematMode = "recompute_all"
    """Per-block gradient checkpointing. "recompute_all" reruns the whole block in
    backward (lowest memory); "save_moe" keeps the tagged MoE dispatch tensors so
    backward skips re-running expert dispatch and its EP collectives."""
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)

    def __post_init__(self) -> None:
        hd = self.inferred_head_dim
        # Half-RoPE requires even head_dim; additive-rope just requires a positive rope dim.
        if self.qk_rope_head_dim is None and hd % 2 != 0:
            raise ValueError(f"head_dim must be even for half-RoPE; got {hd}")
        if self.qk_rope_head_dim is not None and self.qk_rope_head_dim <= 0:
            raise ValueError(f"qk_rope_head_dim must be positive when set; got {self.qk_rope_head_dim}")
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
        if self.inferred_kv_compression_dim <= 0:
            raise ValueError("kv_compression_dim must be positive")
        resolve_moe_implementation(self.moe_implementation)

    @property
    def inferred_kv_compression_dim(self) -> int:
        return self.kv_compression_dim if self.kv_compression_dim is not None else max(1, self.hidden_dim // 4)

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


class CausalSelfAttention(eqx.Module):
    """Multi-head Latent Attention (DeepSeek-V2 style).

    Two architectural paths depending on ``cfg.qk_rope_head_dim``:

    Legacy half-RoPE (``qk_rope_head_dim is None``):
      Per head: Q/K/V each have width ``head_dim``. RoPE rotates the first
      ``head_dim // 2`` channels of Q and the shared ``k_r`` (``half`` wide).
      ``k_nr = c_kv @ w_uk_nr`` is ``half`` wide per head. K concat: ``[k_r, k_nr]``.

    Additive rope (``qk_rope_head_dim`` set; DeepSeek-V2):
      Per head: V is ``head_dim`` wide; Q and K are ``head_dim + rope_dim`` wide
      (no-rope = ``head_dim``, rope = ``rope_dim`` added on top, shared across heads).
      ``k_nr = c_kv @ w_uk_nope`` is ``head_dim`` wide per head. K concat:
      ``[k_nope, k_rope_shared]``. Normalisation is DeepSeek-strict: rms_norm on
      ``c_kv`` only, no QK norm on q/k/v.

    Per-token KV state factorises as ``c_kv (d_c)`` + ``k_r (rope width)``.
    """

    w_q: jax.Array  # legacy (D, NH*HD); additive (D, NH * (HD + rope))
    w_dkv: jax.Array  # (D, d_c)
    w_uk_nr: jax.Array  # legacy (d_c, NH*half); additive (d_c, NH*HD)  -- no-rope K up-proj
    w_uv: jax.Array  # (d_c, NH * HD)
    w_kr: jax.Array  # legacy (D, half); additive (D, rope)
    w_o: jax.Array  # (NH * HD, D)
    w_attn_gate: jax.Array | None  # (D, NH) when cfg.attn_gate else None
    c_kv_norm: RMSNorm | None  # learnable per-channel RMSNorm on c_kv (DeepSeek's kv_a_layernorm)
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_dkv, k_uk, k_uv, k_kr, k_o = random.split(key, 6)
        d = cfg.hidden_dim
        n = cfg.num_heads
        hd = cfg.inferred_head_dim
        d_c = cfg.inferred_kv_compression_dim

        if cfg.qk_rope_head_dim is None:
            # Legacy half-RoPE: rope is the first half of head_dim, baked in.
            half = hd // 2
            w_q_cols = n * hd
            w_uk_nr_cols = n * half
            w_kr_cols = half
        else:
            # Additive rope: extra rope channels added on top of head_dim per head.
            rope_dim = cfg.qk_rope_head_dim
            w_q_cols = n * (hd + rope_dim)
            w_uk_nr_cols = n * hd
            w_kr_cols = rope_dim

        # Per-head sigmoid attention gate. Zero-init -> sigmoid(0)*2 = 1.0 pass-through
        # at step 0, so the model is identical to the no-gate baseline before any training.
        w_attn_gate = reshard(jnp.zeros((d, n)), P(None, None)) if cfg.attn_gate else None

        # Learnable RMSNorm on c_kv (DeepSeek's kv_a_layernorm). Only instantiate when
        # c_kv is actually being normalized AND the learnable flag is on.
        c_kv_will_be_normed = cfg.mla_norm_compressed or cfg.qk_rope_head_dim is not None
        c_kv_norm = (
            RMSNorm.init(d_c, cfg.layer_norm_eps)
            if (c_kv_will_be_normed and cfg.mla_norm_compressed_learnable)
            else None
        )

        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, w_q_cols), cfg.initializer_std), P("data", "model")),
            w_dkv=reshard(_init_weight(k_dkv, (d, d_c), cfg.initializer_std), P("data", None)),
            w_uk_nr=reshard(_init_weight(k_uk, (d_c, w_uk_nr_cols), cfg.initializer_std), P(None, "model")),
            w_uv=reshard(_init_weight(k_uv, (d_c, n * hd), cfg.initializer_std), P(None, "model")),
            w_kr=reshard(_init_weight(k_kr, (d, w_kr_cols), cfg.initializer_std), P("data", None)),
            w_o=reshard(_init_weight(k_o, (n * hd, d), cfg.initializer_std), P("model", "data")),
            w_attn_gate=w_attn_gate,
            c_kv_norm=c_kv_norm,
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        use_pko: bool = False,
        use_xsa: bool = False,
    ) -> Float[Array, "B S D"]:
        hd = self.cfg.inferred_head_dim
        nh = self.cfg.num_heads
        seq_len = x.shape[1]
        batch_spec = _batch_spec()
        rope_dim = self.cfg.qk_rope_head_dim
        is_additive = rope_dim is not None

        # Q projection. Legacy: per-head HD. Additive: per-head (HD + rope_dim).
        q_per_head = hd + rope_dim if is_additive else hd
        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=q_per_head)

        # Compressed KV latent.
        c_kv = jnp.einsum("bsh,hc->bsc", x, self.w_dkv)
        # DeepSeek-V2 places RMSNorm on c_kv (the cached tensor) so that decode
        # absorption (q_nope @ w_uk_nope.T into a fixed per-head matrix) still works.
        if self.cfg.mla_norm_compressed or is_additive:
            if self.c_kv_norm is not None:
                # Learnable per-channel RMSNorm (DeepSeek's kv_a_layernorm).
                c_kv = self.c_kv_norm(c_kv)
            else:
                c_kv = rms_norm(c_kv)

        # Optional PKO: shift c_kv back by one position (with doc-start zero) for
        # the K no-rope up-projection only. V is up-projected from the unshifted
        # c_kv. Absorption-compatible: at decode you read c_kv[j-1] instead of
        # c_kv[j] for the no-rope dot product; V still reads c_kv[j].
        if use_pko:
            c_kv_shifted = jnp.concatenate([c_kv[:, :1, :], c_kv[:, :-1, :]], axis=1)
            segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
            if segment_ids is None:
                is_doc_start_seq = jnp.zeros((seq_len,), dtype=bool).at[0].set(True)
                is_doc_start = jnp.broadcast_to(is_doc_start_seq, c_kv_shifted.shape[:2])
            else:
                q_seg = segment_ids[0]
                if q_seg.ndim == 1:
                    is_doc_start_seq = jnp.concatenate([jnp.ones((1,), dtype=bool), q_seg[1:] != q_seg[:-1]])
                    is_doc_start = jnp.broadcast_to(is_doc_start_seq, c_kv_shifted.shape[:2])
                else:
                    is_doc_start = jnp.concatenate(
                        [jnp.ones_like(q_seg[:, :1], dtype=bool), q_seg[:, 1:] != q_seg[:, :-1]],
                        axis=1,
                    )
            c_kv_shifted = jnp.where(is_doc_start[..., None], jnp.zeros_like(c_kv_shifted), c_kv_shifted)
        else:
            c_kv_shifted = None

        # K no-rope half (per head) up-projected from c_kv. Legacy: per-head
        # half. Additive: per-head HD.
        k_nr_per_head = hd if is_additive else hd // 2
        if use_pko and self.cfg.pko_half_k_dims:
            # Half-shift PKO: first half of per-head k_nope from shifted c_kv,
            # second half from unshifted. Mimics original PKO's "shift only half
            # the head_dim" behavior. Costs one extra c_kv up-projection.
            k_nope_shifted = rearrange(
                jnp.einsum("bsc,cd->bsd", c_kv_shifted, self.w_uk_nr),
                "... (n d) -> ... n d",
                d=k_nr_per_head,
            )
            k_nope_unshifted = rearrange(
                jnp.einsum("bsc,cd->bsd", c_kv, self.w_uk_nr),
                "... (n d) -> ... n d",
                d=k_nr_per_head,
            )
            half = k_nr_per_head // 2
            k_nr = jnp.concatenate(
                [k_nope_shifted[..., :half], k_nope_unshifted[..., half:]],
                axis=-1,
            )
        elif use_pko:
            # Full shift: all per-head k_nope channels come from shifted c_kv.
            k_nr = rearrange(
                jnp.einsum("bsc,cd->bsd", c_kv_shifted, self.w_uk_nr),
                "... (n d) -> ... n d",
                d=k_nr_per_head,
            )
        else:
            k_nr = rearrange(
                jnp.einsum("bsc,cd->bsd", c_kv, self.w_uk_nr),
                "... (n d) -> ... n d",
                d=k_nr_per_head,
            )
        # V (per head) up-projected from the unshifted c_kv.
        v = rearrange(
            jnp.einsum("bsc,cd->bsd", c_kv, self.w_uv),
            "... (n d) -> ... n d",
            d=hd,
        )
        # Rope-bearing K (shared across heads): width is half (legacy) or rope_dim (additive).
        # Broadcast to (B, S, NH, rope_width). w_kr is P("data", None) so its output lacks
        # ``model`` on the heads dim; k_nr has ``model`` on heads -> reshard k_r to match
        # before the post-rope concat with k_nr.
        rope_width = rope_dim if is_additive else hd // 2
        k_r = jnp.einsum("bsh,hd->bsd", x, self.w_kr)
        k_r = jnp.broadcast_to(k_r[:, :, None, :], (k_r.shape[0], seq_len, nh, rope_width))
        k_r = reshard(k_r, P(_BATCH_AXES, None, "model", None))

        # Normalisation:
        #   - Additive (DeepSeek-strict): only c_kv was normed above. No QK norm.
        #   - Legacy with mla_norm_compressed=True: rms_norm(c_kv) ran above; rms_norm
        #     on q and k_r still apply, skip k_nr to preserve absorption.
        #   - Legacy with mla_norm_compressed=False: full per-tensor norms on q/k_nr/k_r.
        if not is_additive:
            q = rms_norm(q)
            if not self.cfg.mla_norm_compressed:
                k_nr = rms_norm(k_nr)
            k_r = rms_norm(k_r)

        # Apply RoPE.
        if is_additive:
            # Split Q into nope (HD) and rope (rope_dim) halves.
            q_nope, q_r = q[..., :hd], q[..., hd:]
            q_r, k_r = apply_rotary_embedding(q_r, k_r, seq_len=seq_len, head_dim=rope_dim, rope=self.cfg.rope)
            q = jnp.concatenate([q_nope, q_r], axis=-1)
            q = reshard(q, P(_BATCH_AXES, None, "model", None))
            k_r = reshard(k_r, P(_BATCH_AXES, None, "model", None))
            k_nr = reshard(k_nr, P(_BATCH_AXES, None, "model", None))
            k = jnp.concatenate([k_nr, k_r], axis=-1)
        else:
            # Half-RoPE on the first head_dim//2 channels of Q and on the shared k_r.
            half = hd // 2
            q_r, q_nr = q[..., :half], q[..., half:]
            q_r, k_r = apply_rotary_embedding(q_r, k_r, seq_len=seq_len, head_dim=half, rope=self.cfg.rope)
            # The slice + rope + concat path can drop the ``model`` annotation off the
            # heads dim. Force the canonical layout on both Q and K so the K concat below
            # sees matching shardings on ``k_r`` and ``k_nr``.
            q = jnp.concatenate([q_r, q_nr], axis=-1)
            q = reshard(q, P(_BATCH_AXES, None, "model", None))
            k_r = reshard(k_r, P(_BATCH_AXES, None, "model", None))
            k_nr = reshard(k_nr, P(_BATCH_AXES, None, "model", None))
            k = jnp.concatenate([k_r, k_nr], axis=-1)

        q = q * self.cfg.qk_mult
        attn_out = attention(q, k, v, mask, implementation=self.cfg.attention_implementation)
        # Half-RoPE's slice+concat can leave the propagator with ``model`` annotated on
        # head_dim rather than num_q_heads; force the canonical TP layout.
        attn_out = reshard(attn_out, P(_BATCH_AXES, None, "model", None))
        # Exclusive Self Attention: subtract the per-head component of attn_out
        # parallel to v at the same query position. MLA's v is already per-head,
        # so no align_kv_heads call is needed.
        if use_xsa:
            v_aligned = reshard(v, P(_BATCH_AXES, None, "model", None))
            dot = jnp.sum(attn_out * v_aligned, axis=-1, keepdims=True)
            v_norm_sq = jnp.sum(v_aligned * v_aligned, axis=-1, keepdims=True)
            attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * v_aligned
        # Per-head sigmoid attention gate (optional). Scalar per (token, head),
        # broadcast over head_dim. Zero-init -> gate=1.0 at step 0.
        if self.w_attn_gate is not None:
            gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.w_attn_gate))
            attn_out = attn_out * gate[..., None]
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=batch_spec)


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
        "qb_beta_per_layer": router_metrics.get("qb_beta_per_layer"),
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
                split_w_gate_up=True,  # store w_gate / w_up separately so MuonH NS sees each half as its own leaf
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
        mask: AttentionMask | jax.Array,
        use_pko: bool = False,
        use_xsa: bool = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        x = x + self.attn(attn_in, mask, use_pko=use_pko, use_xsa=use_xsa)
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        mlp_out, router_stats = self.mlp(mlp_in)
        if self.shared is not None:
            mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        x = x + mlp_out
        return x, router_stats


class Transformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    final_gated_norm: GatedNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, out_key, embed_gn_key, final_gn_key, *block_keys = random.split(key, cfg.num_layers + 4)
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)
        blocks = tuple(Block.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers))
        return Transformer(
            token_embed=token_embed,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key),
            output_proj=output_proj,
            blocks=blocks,
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
        hidden = self.embed_gated_norm(self.embed_norm(hidden))

        # All layers run identical full causal attention -- no sliding window, no
        # long/short branching.
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        full_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)

        if cfg.remat_mode == "save_moe":
            remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
        else:
            remat_policy = None

        moe_router_stats: list[dict[str, jax.Array]] = []
        num_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            # PKO on the final layer only when cfg.pko_last_layer is set.
            is_last = i == num_blocks - 1
            use_pko = cfg.pko_last_layer and is_last
            # XSA fires on every layer when cfg.xsa, OR only on odd-indexed
            # layers when xsa_alternate_skip_first is also set (so layer 0 is
            # skipped; with even num_layers this lands XSA on the last layer).
            if cfg.xsa and cfg.xsa_alternate_skip_first:
                use_xsa = i % 2 == 1
            else:
                use_xsa = cfg.xsa
            hidden, router_stats = eqx.filter_checkpoint(block, policy=remat_policy)(hidden, full_mask, use_pko, use_xsa)
            moe_router_stats.append(router_stats)

        router_metrics = {
            "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
            "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
            "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in moe_router_stats], axis=0),
            "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in moe_router_stats], axis=0),
            "qb_beta_per_layer": jnp.stack([s["qb_beta"] for s in moe_router_stats], axis=0),
            "capacity_overflow_per_layer": jnp.stack([s["capacity_overflow"] for s in moe_router_stats], axis=0),
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
