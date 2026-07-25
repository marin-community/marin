# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model.

Barebones MoE transformer: token embedding, learnable RMSNorm, grouped-query
causal attention with RoPE, a sigmoid-combine top-k MoE (routed experts plus an
optional shared dense expert), and an lm_head. All layers are identical
full-causal MoE blocks run through a single ``lax.scan`` over stacked blocks.
"""

import dataclasses
import os
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax import Axis
from haliax.jax_utils import named_call
from haliax.nn import ArrayStacked
from jax import random, shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import (
    AttentionMask,
    GrugAttentionImplementation,
    RotaryConfig,
    align_kv_heads,
    apply_rotary_embedding,
    apply_rotary_embedding_fused,
    attention,
    fa4_cute_segment_bounds,
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
from levanter.utils.activation import ActivationFunctionEnum

_DEFAULT_EP_CAPACITY_FACTOR = 1.0
_ROUTING_RENORM_SUM = 2.5

_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")

RematMode = Literal["recompute_all", "save_moe"]


def _mesh_axis_size(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty:
        raise ValueError("grug/moe requires a non-empty abstract mesh")
    if axis_name not in mesh.shape:
        # compact_grug_mesh standardizes on (replica_dcn, data, expert, model) with length-1
        # axes kept, so any missing axis is a caller bug rather than a "size 1" shortcut.
        raise ValueError(f"grug/moe requires an abstract mesh with axis '{axis_name}'")
    return int(mesh.shape[axis_name])


def _batch_spec() -> P:
    return P(_BATCH_AXES)


def _batch_reshard(x: jax.Array) -> jax.Array:
    return reshard(x, _batch_spec())


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


def _embedding_gather(token_embed: jax.Array, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
    """Replica-local embedding lookup.

    The naive ``token_embed.at[token_ids].get(out_sharding=...)`` emits an all-to-all to lay the
    gathered rows onto the batch-sharded token axis; because that axis spans ``replica_dcn`` it runs
    cross-rack and its NCCL first-call rendezvous wedges at 8+ racks. Instead, run the gather under a
    ``shard_map`` over the batch axes so each shard looks up its own tokens in its (fully replicated)
    copy of the table -- a purely local op, no collective. Relies on ``Pembed_vocab`` replicating the
    table across all devices; output hidden is replicated, matching the downstream FFN/attention.
    """

    def _local(te: jax.Array, ids: jax.Array) -> jax.Array:
        return te[ids]

    # Pin the (tiny int32) indices to the batch sharding so they match the shard_map in_spec.
    token_ids = reshard(token_ids, P(_BATCH_AXES, None))
    return shard_map(
        _local,
        mesh=get_abstract_mesh(),
        in_specs=(P(None, None), P(_BATCH_AXES, None)),
        out_specs=P(_BATCH_AXES, None, None),
    )(token_embed, token_ids)


@dataclass(frozen=True)
class GrugModelConfig:
    """Shape/size hyperparameters for the grug MoE transformer. All layers are MoE."""

    vocab_size: int
    hidden_dim: int = 512
    intermediate_dim: int = 256
    shared_expert_intermediate_dim: int = 512
    # Split the shared expert into this many independent DenseMLPs, each of size
    # shared_expert_intermediate_dim // num_shared_experts (same total params).
    num_shared_experts: int = 1
    num_experts: int = 256
    num_experts_per_token: int = 4
    num_layers: int = 6
    num_heads: int = 4
    num_kv_heads: int = 1
    head_dim: int | None = None
    max_seq_len: int = 8192
    # Sliding-window attention applied to every layer. 0 (default) = full causal.
    sliding_window: int = 0
    # Interleave full-causal ("global") attention layers into an otherwise sliding-window stack:
    # every ``global_every``-th layer (1-indexed, so the last layer of each group and the final
    # layer) attends the whole document instead of the ``sliding_window``. 0 (default) = disabled
    # (pure sliding window). No-op unless ``sliding_window`` is also set.
    global_every: int = 0
    # NoPE on the global layers: skip RoPE entirely on the full-causal (global) layers. RoPE's
    # positional decay hurts a full-context layer, so global layers often generalize better without
    # it (Llama-4 style). No-op unless ``global_every`` interleaving is active.
    disable_long_rope: bool = False
    # Partial RoPE: rotate only the first ``rope_fraction`` of each head's dimension (GPT-NeoX
    # ``partial_rotary_factor``); the rest passes through unrotated, giving the head some
    # position-independent capacity. 1.0 (default) = full RoPE. 0.5 = half-RoPE (reference default).
    rope_fraction: float = 1.0
    # Apply RoPE as a single fused ``f1*x + f2*flip(x)`` (interleaved pairs) instead of the split-half
    # multiply+concat. Drops the concat on the (large) q/k activation. Different pair convention --
    # fine from scratch, not checkpoint-compatible with the split-half path.
    rope_fused: bool = False
    # Apply a learnable GatedNorm after each RMSNorm (attn + mlp inputs). Off in the barebones default.
    gated_norm: bool = False
    # Per-head sigmoid attention gate: gate = 2*sigmoid(x @ attn_gate), a scalar per (token, head)
    # multiplier on the attention output before w_o. Zero-init -> gate==1 at step 0 (identical to the
    # no-gate model); training learns to gate down from pass-through. Absorption-friendly (depends
    # only on the current query). Off in the barebones default (SCALE_ATTN_GATE=1).
    attn_gate: bool = False
    # Exclusive Self Attention: per head, subtract the component of the attention output parallel to
    # vᵢ, zᵢ = yᵢ - (yᵢ·vᵢ / ‖vᵢ‖²) vᵢ. Off in the barebones default (SCALE_XSA=1).
    xsa: bool = False
    # QB aux-loss-free load balancing (DeepSeek-V3 style): bias the top-k expert *selection* with a
    # per-expert router_bias while forming combine weights from the unbiased logits, and feed the
    # resulting qb_beta stat into the next step's router_bias. Off in the barebones default
    # (SCALE_MOE_QB=1); when off the router is the plain top-k path (byte-for-byte unchanged).
    qb_routing: bool = False
    # Inkling-style short causal convolutions (SConv): a depthwise kernel-W causal 1-D conv over the
    # sequence, applied after the K and V projections and on the attention/MoE branch outputs. Cheap
    # local token mixing / short-range inductive bias that frees attention + experts from carrying it.
    # Identity-init -> inert at step 0. Off in the barebones default (SCALE_SCONV=1).
    sconv: bool = False
    sconv_kernel: int = 4
    # Which of the 4 SConv sites are active (subset of "k","v","attn","mlp"); default all four.
    sconv_sites: tuple[str, ...] = ("k", "v", "attn", "mlp")
    # ``unroll`` for the layer scan. >1 unrolls that many layers per scan step, letting XLA overlap
    # layer N's weight all-gather with layer N-1's compute (cross-iteration) -- at the risk of the
    # #7407 CUBIN-load bug that scan-collective pipelining hit on the grug model at d6144.
    scan_unroll: int = 1
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.3
    attention_implementation: GrugAttentionImplementation | None = None
    moe_implementation: MoeImplementation | None = None
    remat_mode: RematMode = "recompute_all"
    """Per-block gradient checkpointing. "recompute_all" reruns the whole block in
    backward (lowest memory); "save_moe" keeps the tagged MoE dispatch tensors so
    backward skips re-running expert dispatch and its EP collectives."""
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
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def model_type(self) -> type["Transformer"]:
        return Transformer

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads

    def rotary_dim(self, head_dim: int) -> int:
        """Leading head dims that get rotary (partial RoPE); the rest pass through. Even, <= head_dim."""
        if self.rope_fraction >= 1.0:
            return head_dim
        return (int(head_dim * self.rope_fraction) // 2) * 2

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> "Transformer":
        cfg = self if Vocab.size == self.vocab_size else dataclasses.replace(self, vocab_size=Vocab.size)
        return Transformer.init(cfg, key=key)


# Attention-projection shardings. The FSDP variant shards the contraction ("data") axis, so XLA
# inserts a per-layer weight all-gather before the attention matmuls.
_QKV_SPEC = P("data", "model")
_O_SPEC = P("model", "data")


def _qkv_projection_pipelined(
    x: Float[Array, "B S D"],
    w_q: Float[Array, "D NH"],
    w_k: Float[Array, "D MH"],
    w_v: Float[Array, "D MH"],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gather each QKV weight immediately before its projection (SCALE_ATTN_PIPELINE=1).

    The default path emits q/k/v as three independent einsums; XLA front-loads all three FSDP
    weight all-gathers (serial on one comm stream) before any projection, so the projections stall
    at the layer opening. Here the gather and its consuming matmul live in one shard_map body, in
    sequence, so the k/v gathers can overlap the q/k projection GEMMs instead of all being exposed.
    """
    mesh = get_abstract_mesh()
    x_spec = _batch_spec()
    out_spec = P(_BATCH_AXES, None, "model")

    def local(x_l, wq_l, wk_l, wv_l):
        wq = jax.lax.all_gather(wq_l, "data", axis=0, tiled=True)
        q = jnp.einsum("bsh,hd->bsd", x_l, wq)
        wk = jax.lax.all_gather(wk_l, "data", axis=0, tiled=True)
        k = jnp.einsum("bsh,hd->bsd", x_l, wk)
        wv = jax.lax.all_gather(wv_l, "data", axis=0, tiled=True)
        v = jnp.einsum("bsh,hd->bsd", x_l, wv)
        return q, k, v

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(x_spec, _QKV_SPEC, _QKV_SPEC, _QKV_SPEC),
        out_specs=(out_spec, out_spec, out_spec),
        check_vma=False,
    )(reshard(x, x_spec), w_q, w_k, w_v)


class ShortConv(eqx.Module):
    """Depthwise causal 1-D convolution over the sequence axis (Inkling-style SConv).

    A kernel of ``W`` taps mixes each channel with its own ``W-1`` causal predecessors:
    ``out[t] = sum_{lag=0..W-1} weight[lag] * x[t-lag]`` independently per channel. Identity-init
    (``weight[0]=1``, later taps 0) makes it a pass-through at step 0, so bolting it onto a tuned
    baseline changes nothing until training learns the local mixing. Weights are tiny (``W*C``) and
    routed to Adam (never Muon-orthogonalized).

    Implemented as a pad-and-shift weighted sum rather than ``lax.conv_general_dilated``: for this
    shape (kernel 4, per-channel over 1024/6144 channels, seq 4096) a fused depthwise conv fell into
    cuDNN's slow generic grouped-conv path and measured ~4 MFU points *worse* than this elementwise
    version, which XLA fuses well. It is also trivially shard-local -- pure elementwise plus a
    sequence-axis shift, no cross-channel or cross-shard dependency, so no collectives.
    """

    weight: Float[Array, "W C"]
    kernel_size: int = eqx.field(static=True)

    @staticmethod
    def init(channels: int, kernel_size: int) -> "ShortConv":
        weight = jnp.zeros((kernel_size, channels)).at[0].set(1.0)
        # SCALE_SCONV_SHARD: FSDP-shard the channel dim over data so the grad reduce-scatters
        # (coalesced) instead of a standalone replicated all-reduce; forward gathers the weight.
        spec = P(None, "data") if os.environ.get("SCALE_SCONV_SHARD") == "1" else P(None, None)
        return ShortConv(weight=reshard(weight, spec), kernel_size=kernel_size)

    def __call__(self, x: Float[Array, "B S C"], segment_ids: Int[Array, "B S"] | None = None) -> Float[Array, "B S C"]:
        # Depthwise causal shift-and-sum. When segment_ids are given (packed documents), a tap that
        # reaches back into a previous document is zeroed, so the conv never mixes across a document
        # boundary -- matching the segment-masked attention. The lag-0 (current-token) tap is always kept.
        seq_len = x.shape[1]
        # Gather the (possibly FSDP-sharded) weight to replicated for the per-channel multiply; a
        # no-op when SCALE_SCONV_SHARD is off. Sharded storage -> grad reduce-scatters; here we all-gather.
        weight = reshard(self.weight, P(None, None))
        out = weight[0] * x
        for lag in range(1, self.kernel_size):
            shifted = jnp.pad(x, ((0, 0), (lag, 0), (0, 0)))[:, :seq_len, :]
            if segment_ids is not None:
                # -1 sentinel in the pad never equals a real (>=0) segment id, so leading positions
                # with no in-document history are zeroed too.
                seg_shifted = jnp.pad(segment_ids, ((0, 0), (lag, 0)), constant_values=-1)[:, :seq_len]
                shifted = jnp.where((seg_shifted == segment_ids)[..., None], shifted, 0.0)
            out = out + weight[lag] * shifted
        return out


class CausalSelfAttention(eqx.Module):
    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]
    attn_gate: Float[Array, "D N"] | None  # per-head sigmoid gate (cfg.attn_gate); routed to adam
    sconv_k: ShortConv | None  # SConv after the K projection (cfg.sconv)
    sconv_v: ShortConv | None  # SConv after the V projection (cfg.sconv)
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), cfg.initializer_std), _QKV_SPEC),
            w_k=reshard(_init_weight(k_k, (d, m * h), cfg.initializer_std), _QKV_SPEC),
            w_v=reshard(_init_weight(k_v, (d, m * h), cfg.initializer_std), _QKV_SPEC),
            w_o=reshard(_init_weight(k_o, (n * h, d), cfg.initializer_std), _O_SPEC),
            # Zero-init -> 2*sigmoid(0) == 1 pass-through at step 0, identical to the no-gate model.
            attn_gate=(reshard(jnp.zeros((d, n)), P(None, None)) if cfg.attn_gate else None),
            sconv_k=(ShortConv.init(m * h, cfg.sconv_kernel) if cfg.sconv and "k" in cfg.sconv_sites else None),
            sconv_v=(ShortConv.init(m * h, cfg.sconv_kernel) if cfg.sconv and "v" in cfg.sconv_sites else None),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        disable_rope: jax.Array | None = None,
    ) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]

        if os.environ.get("SCALE_ATTN_PIPELINE") == "1":
            q_flat, k_flat, v_flat = _qkv_projection_pipelined(x, self.w_q, self.w_k, self.w_v)
        else:
            q_flat = jnp.einsum("bsh,hd->bsd", x, self.w_q)
            k_flat = jnp.einsum("bsh,hd->bsd", x, self.w_k)
            v_flat = jnp.einsum("bsh,hd->bsd", x, self.w_v)
        # SConv: depthwise causal conv after the K and V projections (before head-split/RMS).
        # Guard each site independently -- K-only leaves sconv_v None. Segment ids (packed-doc
        # boundaries) come from the mask so the conv never mixes across a document boundary.
        # AttentionMask.segment_ids is a (q, kv) tuple; use the q (per-position) ids -- for
        # self-attention q == kv, and the conv indexes positions within the one sequence.
        _seg = mask.segment_ids if isinstance(mask, AttentionMask) else None
        sconv_segment_ids = _seg[0] if _seg is not None else None
        if self.sconv_k is not None:
            k_flat = self.sconv_k(k_flat, sconv_segment_ids)
        if self.sconv_v is not None:
            v_flat = self.sconv_v(v_flat, sconv_segment_ids)
        q = rearrange(q_flat, "... (n d) -> ... n d", d=head_dim)
        k = rearrange(k_flat, "... (m d) -> ... m d", d=head_dim)
        v = rearrange(v_flat, "... (m d) -> ... m d", d=head_dim)

        q = rms_norm(q)
        k = rms_norm(k)
        # Partial RoPE: rotate only the leading rotary_dim of each head, pass the tail through. When
        # rotary_dim == head_dim (rope_fraction=1) this is plain full RoPE with no slice/concat. XLA
        # fuses the contiguous slice+concat, and the rotary math runs on rotary_dim (not head_dim).
        # NoPE (disable_rope on the flagged global layers) is expressed as angle_scale=0 -> RoPE becomes
        # the identity, so there is a single rope pass and no jnp.where doubling the live q,k.
        rotary_dim = self.cfg.rotary_dim(head_dim)
        angle_scale = None if disable_rope is None else (~disable_rope).astype(jnp.float32)

        if self.cfg.rope_fused:
            # Single fused affine form. Partial rope (rotary_dim) and NoPE (per-layer factor select) are
            # handled inside via constant-folded factor caches -- no split/concat on q,k, and the trig
            # stays a hoisted constant (NoPE selects identity factors rather than scaling angles).
            q, k = apply_rotary_embedding_fused(
                q,
                k,
                seq_len=seq_len,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                rope=self.cfg.rope,
                disable_rope=disable_rope,
            )
        else:

            def _rope(qh: jax.Array, kh: jax.Array) -> tuple[jax.Array, jax.Array]:
                if rotary_dim >= head_dim:
                    return apply_rotary_embedding(
                        qh, kh, seq_len=seq_len, head_dim=head_dim, rope=self.cfg.rope, angle_scale=angle_scale
                    )
                q_rot, k_rot = apply_rotary_embedding(
                    qh[..., :rotary_dim],
                    kh[..., :rotary_dim],
                    seq_len=seq_len,
                    head_dim=rotary_dim,
                    rope=self.cfg.rope,
                    angle_scale=angle_scale,
                )
                return (
                    jnp.concatenate([q_rot, qh[..., rotary_dim:]], axis=-1),
                    jnp.concatenate([k_rot, kh[..., rotary_dim:]], axis=-1),
                )

            q, k = _rope(q, k)
        q = q * self.cfg.qk_mult

        attn_out = attention(q, k, v, mask, implementation=self.cfg.attention_implementation)
        # Exclusive Self Attention (optional): per head, subtract the component of yᵢ parallel to vᵢ,
        # zᵢ = yᵢ - (yᵢ·vᵢ / ‖vᵢ‖²) vᵢ. align_kv_heads broadcasts the GQA KV heads up to the Q heads;
        # reshard both to the canonical TP layout (model on num_q_heads) so they align per head.
        if self.cfg.xsa:
            attn_out = reshard(attn_out, P(_BATCH_AXES, None, "model", None))
            aligned_v = reshard(align_kv_heads(v, num_q_heads=attn_out.shape[2]), P(_BATCH_AXES, None, "model", None))
            dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
            v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
            attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v
        # Per-head sigmoid attention gate (optional): scalar per (token, head), broadcast over
        # head_dim, applied before merging heads. Zero-init -> gate==1 at step 0.
        if self.attn_gate is not None:
            gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.attn_gate))
            attn_out = attn_out * gate[..., None]
        # Merge heads into hidden dim while keeping model-axis sharding for w_o.
        attn_out = jnp.reshape(
            attn_out,
            (*attn_out.shape[:-2], attn_out.shape[-2] * attn_out.shape[-1]),
            out_sharding=P(_BATCH_AXES, None, "model"),
        )
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=_batch_spec())


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


_GATED_NORM_RANK = 128


class GatedNorm(eqx.Module):
    """Learnable low-rank per-dimension gating applied after RMSNorm (optional; SCALE_GATED_NORM=1)."""

    w_down: jax.Array
    w_up: jax.Array

    @staticmethod
    def init(hidden_dim: int, initializer_std: float, *, key: PRNGKeyArray) -> "GatedNorm":
        k_down, k_up = random.split(key)
        # SCALE_GATED_NORM_SHARD: FSDP-shard the hidden dim (D=6144) over data so the weight grads
        # reduce-scatter (coalesced) instead of standalone replicated all-reduces; forward gathers.
        # Under scan these stack to 3D [L, D, r] / [L, r, D] and route to muonh's padded-stack-sharded
        # Newton-Schulz, which reshards to its own layout -- storage sharding is transparent to NS.
        shard = os.environ.get("SCALE_GATED_NORM_SHARD") == "1"
        down_spec = P("data", None) if shard else P(None, None)
        up_spec = P(None, "data") if shard else P(None, None)
        return GatedNorm(
            w_down=reshard(_init_weight(k_down, (hidden_dim, _GATED_NORM_RANK), initializer_std), down_spec),
            w_up=reshard(_init_weight(k_up, (_GATED_NORM_RANK, hidden_dim), initializer_std), up_spec),
        )

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        # Gather the (possibly FSDP-sharded) weights to replicated for the local low-rank gating;
        # a no-op when SCALE_GATED_NORM_SHARD is off. Sharded storage -> grad reduce-scatters.
        w_down = reshard(self.w_down, P(None, None))
        w_up = reshard(self.w_up, P(None, None))
        gate_hidden = jax.nn.silu(jnp.einsum("...d,dr->...r", x, w_down))
        gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, w_up))
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


def _compute_qb_beta(router_logits: jax.Array, qb_alpha: jax.Array, cfg: "GrugModelConfig") -> jax.Array:
    """Sharded QB beta (DeepSeek-V3 aux-loss-free balancing).

    Per device, take the ``qb_count``-th largest ``(logit - alpha)`` gap over the local tokens, then
    average across the batch axes. The qb_beta pmean is an f32 all-reduce whose result only feeds the
    NEXT step's router_bias (train.py:_apply_qb_betas), never this forward -- yet XLA schedules it as a
    synchronous per-layer collective in the forward critical path, which can stall the compute stream
    and block the weight-gather overlap. ``SCALE_MOE_SKIP_QB`` drops it entirely (router_bias stops
    updating, so loss quality is invalid under this flag -- throughput diagnostic only).
    """
    if os.environ.get("SCALE_MOE_SKIP_QB") == "1":
        return jnp.zeros((cfg.num_experts,), dtype=jnp.float32)
    mesh = get_abstract_mesh()
    s_minus_alpha = reshard(router_logits - qb_alpha, P(_BATCH_AXES, None))
    num_devices = 1
    for a in _BATCH_AXES:
        num_devices *= mesh.shape[a]
    local_tokens = s_minus_alpha.shape[0] // num_devices
    qb_count = max(1, local_tokens * cfg.num_experts_per_token // cfg.num_experts)

    def _local_qb_beta(s_ma):
        topk_vals, _ = jax.lax.top_k(s_ma.T, qb_count)
        beta = topk_vals[:, -1]
        return jax.lax.pmean(beta, axis_name=_BATCH_AXES)

    return shard_map(
        _local_qb_beta,
        mesh=mesh,
        in_specs=(P(_BATCH_AXES, None),),
        out_specs=P(),
    )(s_minus_alpha)


class MoEMLP(eqx.Module):
    """Top-k routed MoE with sigmoid combine weights (optional QB balancing behind cfg.qb_routing)."""

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
            ),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        w13_pre0: jax.Array | None = None,
        w2_pre0: jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], jax.Array]:
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        # Keep the router path in fp32 before top-k and the sigmoid combine.
        router_logits = jnp.einsum("td,de->te", x_flat, reshard(self.router, P(None, None))).astype(jnp.float32)
        if self.cfg.qb_routing:
            # QB aux-loss-free balancing: bias the top-k *selection* with the per-expert router_bias,
            # but form combine weights from the UNBIASED logits. Select top-(K+1); the (K+1)-th biased
            # logit is the QB threshold alpha, and qb_beta (returned up the stack) feeds NEXT step's
            # router_bias (train.py:_apply_qb_betas).
            biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
            topk_biased, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token + 1)
            qb_alpha = topk_biased[:, -1:]
            selected_experts = selected_experts[:, :-1]
            topk_logits = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
            qb_beta = _compute_qb_beta(router_logits, qb_alpha, self.cfg)
        else:
            topk_logits, selected_experts = jax.lax.top_k(router_logits, self.cfg.num_experts_per_token)
            qb_beta = jnp.zeros((self.cfg.num_experts,), dtype=jnp.float32)
        # Sigmoid combine weights on the selected logits, renormalized to sum to ``_ROUTING_RENORM_SUM``.
        combine_weights_f = jax.nn.sigmoid(topk_logits)
        denom = jnp.sum(combine_weights_f, axis=-1, keepdims=True)
        combine_weights_f = combine_weights_f * (_ROUTING_RENORM_SUM / (denom + 1e-9))
        combine_weights = combine_weights_f.astype(x.dtype)

        routed_flat = self.expert_mlp(
            x_flat,
            selected_experts.astype(jnp.int32),
            combine_weights,
            mesh=get_abstract_mesh(),
            report_capacity_overflow=False,
            w13_pre0=w13_pre0,
            w2_pre0=w2_pre0,
        )
        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        return reshard(routed, _batch_spec()), qb_beta


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm | None
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm | None
    mlp: MoEMLP
    shared: tuple[DenseMLP, ...] | None
    sconv_attn: ShortConv | None  # SConv on the attention branch output (cfg.sconv)
    sconv_mlp: ShortConv | None  # SConv on the MoE branch output (cfg.sconv)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Block":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        shared = None
        if cfg.shared_expert_intermediate_dim > 0:
            n = cfg.num_shared_experts
            per_expert_dim = cfg.shared_expert_intermediate_dim // n
            shared = tuple(
                DenseMLP.init(cfg.hidden_dim, per_expert_dim, cfg.initializer_std, key=k)
                for k in random.split(shared_key, n)
            )
        attn_gated_norm = (
            GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key) if cfg.gated_norm else None
        )
        mlp_gated_norm = GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key) if cfg.gated_norm else None
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn_gated_norm=attn_gated_norm,
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=mlp_gated_norm,
            mlp=MoEMLP.init(cfg, key=mlp_key),
            shared=shared,
            sconv_attn=(
                ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if cfg.sconv and "attn" in cfg.sconv_sites else None
            ),
            sconv_mlp=(
                ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if cfg.sconv and "mlp" in cfg.sconv_sites else None
            ),
        )

    @named_call
    def __call__(
        self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array, disable_rope: jax.Array | None = None
    ) -> tuple[Float[Array, "B S D"], jax.Array]:
        # SCALE_MOE_HOIST_CHUNK0: reshard chunk-0's routed-expert weights to replicated HERE, before the
        # attention call, so the all-gather is emitted ahead of attention and XLA overlaps it forward
        # (it will not hoist a collective backward across the whole attention block). Threaded into the
        # MoE, whose chunked path then skips chunk-0's in-region gather. Chunks 1+ still gather in-region.
        w13_pre0 = w2_pre0 = None
        if os.environ.get("SCALE_MOE_HOIST_CHUNK0") == "1":
            sizes_env = os.environ.get("SCALE_MOE_CHUNK_SIZES")
            if sizes_env:
                per = int(sizes_env.split(",")[0])
            else:
                per = self.mlp.cfg.num_experts // int(os.environ.get("SCALE_MOE_EXPERT_CHUNKS", "1"))
            em = self.mlp.expert_mlp
            w_up_gate = em.w_gate_up if em.w_gate_up is not None else jnp.concatenate([em.w_gate, em.w_up], axis=-1)
            w13_pre0 = reshard(w_up_gate[:per], P(None, None, None))
            w2_pre0 = reshard(em.w_down[:per], P(None, None, None))

        # Segment ids (packed-document boundaries) for the branch-output SConvs; None when unpacked.
        # AttentionMask.segment_ids is a (q, kv) tuple; the q (per-position) ids are what the conv needs.
        _seg = mask.segment_ids if isinstance(mask, AttentionMask) else None
        sconv_segment_ids = _seg[0] if _seg is not None else None

        attn_in = self.rms_attn(x)
        if self.attn_gated_norm is not None:
            attn_in = self.attn_gated_norm(attn_in)
        attn_out = self.attn(attn_in, mask, disable_rope)
        if self.sconv_attn is not None:
            attn_out = self.sconv_attn(attn_out, sconv_segment_ids)
        x = x + attn_out
        if os.environ.get("SCALE_ATTN_ONLY") == "1":
            # Isolation probe: attention block only, no MoE/MLP. Loss is meaningless; used to profile
            # the attention weight-gather behavior with the MoE (memory hog + competing gathers) removed.
            return x, jnp.zeros((self.mlp.cfg.num_experts,), jnp.float32)
        mlp_in = self.rms_mlp(x)
        if self.mlp_gated_norm is not None:
            mlp_in = self.mlp_gated_norm(mlp_in)
        mlp_out, qb_beta = self.mlp(mlp_in, w13_pre0, w2_pre0)
        if self.shared is not None:
            for shared_expert in self.shared:
                mlp_out = mlp_out + shared_expert(mlp_in, activation=ActivationFunctionEnum.silu)
        if self.sconv_mlp is not None:
            mlp_out = self.sconv_mlp(mlp_out, sconv_segment_ids)
        return x + mlp_out, qb_beta


class Transformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: RMSNorm
    output_proj: jax.Array
    stacked_blocks: ArrayStacked[Block]
    final_norm: RMSNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(
        cfg_or_vocab: GrugModelConfig | Axis,
        config: GrugModelConfig | None = None,
        *,
        key: PRNGKeyArray,
    ) -> "Transformer":
        if isinstance(cfg_or_vocab, Axis):
            if config is None:
                raise ValueError("config must be provided when initializing with a Vocab axis")
            cfg = (
                config
                if cfg_or_vocab.size == config.vocab_size
                else dataclasses.replace(config, vocab_size=cfg_or_vocab.size)
            )
        else:
            if config is not None:
                raise ValueError("config must not be provided when initializing directly from GrugModelConfig")
            cfg = cfg_or_vocab

        embed_key, out_key, *block_keys = random.split(key, cfg.num_layers + 2)
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)
        stacked_blocks = ArrayStacked.init(cfg.num_layers, Block)(cfg, key=jnp.stack(block_keys))

        return Transformer(
            token_embed=token_embed,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            output_proj=output_proj,
            stacked_blocks=stacked_blocks,
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            config=cfg,
        )

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.config.vocab_size)

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], Float[Array, "L E"]]:
        if mask is None:
            mask = AttentionMask.causal()

        cfg = self.config
        hidden = _embedding_gather(self.token_embed, token_ids)
        hidden = self.embed_norm(hidden)

        # Every layer is identical full-causal MoE, so the mask is a scan constant.
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        if segment_ids is not None:
            # Pin the per-token [B, S] segment ids batch-sharded before they enter the layer scan.
            # Otherwise they reach the FA4 callback as {maximal device=0} and the compiler falls back
            # to an involuntary full-remat scatter to [num_devices, 1], which serializes through
            # device 0 and can wedge the MoE all-to-all (collective rendezvous timeout at scale).
            segment_ids = _batch_reshard(segment_ids)
        sliding_window = cfg.sliding_window or None
        causal_mask = AttentionMask(is_causal=True, sliding_window=sliding_window, segment_ids=segment_ids)

        # Precompute the FA4 per-token metadata OUTSIDE the scan and attach it via ``with_fa4_bounds``,
        # keeping the FA4 pure_callback's device-0-pinned metadata out of the scan body (an in-body
        # callback forces an involuntary full rematerialization that serializes through device 0 and
        # wedges the MoE all-to-all at 8+ racks). For interleaved global/sliding stacks we precompute
        # BOTH window bound sets here and select per layer with a cheap in-body ``jnp.where`` on a
        # scalar flag -- the exact "precompute once, select per layer" path fa4_bounds was built for.
        batch_size, seq_len = hidden.shape[0], hidden.shape[1]
        sliding_lb, sliding_valid = fa4_cute_segment_bounds(
            causal_mask, batch_size=batch_size, seq_len=seq_len, sliding_window=sliding_window
        )
        sliding_lb, sliding_valid = _batch_reshard(sliding_lb), _batch_reshard(sliding_valid)

        if cfg.remat_mode == "save_moe":
            remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
        else:
            remat_policy = None

        interleave_global = bool(cfg.global_every) and sliding_window is not None
        if interleave_global:
            # Only the lower_bounds depend on the window; ``valid`` (segment_ids >= 0) does not, so we
            # reuse ``sliding_valid`` for both layer types. Global layers drop the window (full causal
            # within the document). Every ``global_every``-th layer (1-indexed) is global, and the final
            # layer is forced global so the last block always sees the whole document.
            global_lb, _ = fa4_cute_segment_bounds(
                causal_mask, batch_size=batch_size, seq_len=seq_len, sliding_window=None
            )
            global_lb = _batch_reshard(global_lb)
            is_global = ((jnp.arange(cfg.num_layers) + 1) % cfg.global_every) == 0
            is_global = is_global.at[-1].set(True)

            def _scan_layer(
                carry_hidden: Float[Array, "B S D"], layer_and_flag: tuple[Block, jax.Array]
            ) -> tuple[Float[Array, "B S D"], jax.Array]:
                layer, layer_is_global = layer_and_flag
                lower_bounds = jnp.where(layer_is_global, global_lb, sliding_lb)
                layer_mask = causal_mask.with_fa4_bounds(lower_bounds, sliding_valid)
                # NoPE on the global layers when enabled; otherwise rope on every layer (disable_rope=None).
                disable_rope = layer_is_global if cfg.disable_long_rope else None
                new_hidden, qb_beta = eqx.filter_checkpoint(layer, policy=remat_policy)(
                    carry_hidden, layer_mask, disable_rope
                )
                return new_hidden, qb_beta

            hidden, qb_beta_per_layer = jax.lax.scan(
                _scan_layer, hidden, xs=(self.stacked_blocks.stacked, is_global), unroll=cfg.scan_unroll
            )
            return self.final_norm(hidden), qb_beta_per_layer

        layer_mask = causal_mask.with_fa4_bounds(sliding_lb, sliding_valid)

        def _scan_layer(carry_hidden: Float[Array, "B S D"], layer: Block) -> tuple[Float[Array, "B S D"], jax.Array]:
            # Carry the hidden state; OUTPUT the per-layer qb_beta so scan stacks it to [L, E].
            new_hidden, qb_beta = eqx.filter_checkpoint(layer, policy=remat_policy)(carry_hidden, layer_mask)
            return new_hidden, qb_beta

        hidden, qb_beta_per_layer = jax.lax.scan(
            _scan_layer, hidden, xs=self.stacked_blocks.stacked, unroll=cfg.scan_unroll
        )
        return self.final_norm(hidden), qb_beta_per_layer

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        hidden, _ = self(token_ids, mask=mask)
        return jnp.einsum("bsh,hd->bsd", hidden, self.output_proj, out_sharding=_batch_spec())

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
    ) -> tuple[jax.Array, Float[Array, "L E"]]:
        hidden, qb_beta_per_layer = self(token_ids, mask=mask)
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)
        loss = fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
        )
        return loss, qb_beta_per_layer


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
    "GrugModelConfig",
    "MoEMLP",
    "MoeActivation",
    "RMSNorm",
    "Transformer",
    "debug_mesh_and_token_pspec",
]
