# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import functools
import math
from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int

from haliax.jax_utils import named_call
from haliax.partitioning import _get_mesh


@dataclass(frozen=True)
class RotaryConfig:
    """Lightweight rotary embedding configuration."""

    theta: float = 10000.0
    scaling_factor: float | None = None


@functools.partial(register_dataclass, data_fields=["segment_ids"], meta_fields=["is_causal", "sliding_window"])
@dataclass(frozen=True)
class AttentionMask:
    """Grug attention mask spec.

    This is deliberately simpler than `levanter.layers.attention.AttentionMask`:
    - Stores raw JAX arrays (no NamedArray fields).
    - Supports causal masking, sliding windows, and segment IDs.
    """

    is_causal: bool = False
    segment_ids: tuple[jax.Array, jax.Array] | None = None
    sliding_window: int | None = None

    @classmethod
    def causal(cls, *, sliding_window: int | None = None) -> "AttentionMask":
        return cls(is_causal=True, sliding_window=sliding_window)

    def with_segment_ids(
        self, q_segment_ids: Int[Array, "..."], kv_segment_ids: Int[Array, "..."] | None = None
    ) -> "AttentionMask":
        kv_ids = q_segment_ids if kv_segment_ids is None else kv_segment_ids
        return AttentionMask(
            is_causal=self.is_causal,
            segment_ids=(q_segment_ids, kv_ids),
            sliding_window=self.sliding_window,
        )

    def with_sliding_window(self, sliding_window: int | None) -> "AttentionMask":
        return AttentionMask(
            is_causal=self.is_causal,
            segment_ids=self.segment_ids,
            sliding_window=sliding_window,
        )

    def materialize_mask(self, q_len: int, k_len: int) -> Bool[Array, "..."] | None:
        """Return a boolean mask (True = allowed) or None.

        Shapes:
          - If `segment_ids` is unset, returns `(q_len, k_len)` (broadcastable across batch).
          - If `segment_ids` is set with per-batch IDs, returns `(batch, q_len, k_len)`.
        """
        mask = None

        if self.is_causal:
            q_idx = jnp.arange(q_len)[:, None]
            k_idx = jnp.arange(k_len)[None, :]
            allowed = k_idx <= q_idx
            mask = allowed

        if self.sliding_window is not None:
            if self.sliding_window <= 0:
                raise ValueError(f"sliding_window must be positive, got {self.sliding_window}")
            q_idx = jnp.arange(q_len)[:, None]
            k_idx = jnp.arange(k_len)[None, :]
            # Standard sliding-window semantics: `sliding_window=W` means "keep the last W tokens,
            # including self". Without causality, this is "don't look too far back":
            #   k >= q - (W - 1)
            allowed = k_idx >= q_idx - (self.sliding_window - 1)
            mask = allowed if mask is None else jnp.logical_and(mask, allowed)

        if self.segment_ids is not None:
            q_seg, k_seg = self.segment_ids
            if q_seg.ndim != k_seg.ndim:
                raise ValueError(f"segment_ids ndim mismatch: q={q_seg.ndim}, k={k_seg.ndim}")
            if q_seg.ndim == 1:
                allowed = q_seg[:, None] == k_seg[None, :]
            elif q_seg.ndim == 2:
                if q_seg.shape[0] != k_seg.shape[0]:
                    raise ValueError(f"segment_ids batch mismatch: q={q_seg.shape[0]}, k={k_seg.shape[0]}")
                allowed = q_seg[:, :, None] == k_seg[:, None, :]
            else:
                raise ValueError(f"segment_ids must be 1D or 2D, got ndim={q_seg.ndim}")
            mask = allowed if mask is None else jnp.logical_and(mask, allowed)

        return mask


def _rotary_cache(seq_len: int, head_dim: int, rope: RotaryConfig) -> tuple[Float[Array, "S D"], Float[Array, "S D"]]:
    half_dim = head_dim // 2
    inv_freq = 1.0 / (rope.theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    return cos, sin


@named_call
def apply_rotary_embedding(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S H D"],
    *,
    seq_len: int,
    head_dim: int,
    rope: RotaryConfig,
) -> tuple[Float[Array, "B S H D"], Float[Array, "B S H D"]]:
    cos, sin = _rotary_cache(seq_len, head_dim, rope)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def _apply(x: Float[Array, "B S H D"]) -> Float[Array, "B S H D"]:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    return _apply(q), _apply(k)


def reference_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    logits_dtype: jnp.dtype | None,
    sinks: Float[Array, "Hq"] | None = None,
) -> Float[Array, "B Q Hq D"]:
    head_dim = q.shape[-1]
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]

    if num_q_heads != num_kv_heads:
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        repeat = num_q_heads // num_kv_heads
        k = jnp.repeat(k, repeat, axis=2)
        v = jnp.repeat(v, repeat, axis=2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q * scale, k)

    # GPT-OSS-style attention sinks: add a per-head sink logit to the softmax
    # denominator (concat-and-drop), which reduces attention mass on real tokens.

    explicit = None
    if mask is None:
        explicit = None
    elif isinstance(mask, AttentionMask):
        explicit = mask.materialize_mask(scores.shape[-2], scores.shape[-1])
    else:
        explicit = mask

    if explicit is not None:
        # Standardize dense masks to [B, Q, K] (bool = allowed, float = additive bias).
        if explicit.ndim == 2:
            explicit = explicit[None, :, :]
        if explicit.ndim != 3:
            raise ValueError(f"explicit mask must have shape [batch, q, k], got shape={explicit.shape}")
        if explicit.shape[0] not in (1, q.shape[0]):
            raise ValueError(f"explicit mask batch dim must be 1 or {q.shape[0]}, got {explicit.shape[0]}")
        if explicit.shape[1] != scores.shape[-2] or explicit.shape[2] != scores.shape[-1]:
            raise ValueError(
                "explicit mask must match attention shapes: "
                f"got mask={explicit.shape}, expected [batch,{scores.shape[-2]},{scores.shape[-1]}]"
            )

        explicit = explicit[:, None, :, :]  # -> [B, 1, Q, K], broadcast across heads.
        if explicit.dtype == jnp.bool_:
            scores = jnp.where(explicit, scores, jnp.array(-1e9, dtype=scores.dtype))
        else:
            scores = scores + explicit
    if logits_dtype is not None:
        scores = scores.astype(logits_dtype)
    if sinks is not None:
        # Compute softmax with an extra sink position, but do not return sink weights.
        # scores: [B, H, Q, K], sinks: [H]
        max_scores = jnp.max(scores, axis=-1, keepdims=True)
        sinks_bh = sinks[None, :, None, None]
        max_or_sink = jnp.maximum(max_scores, sinks_bh)
        scores_exp = jnp.exp(scores - max_or_sink)
        sinks_exp = jnp.exp(sinks_bh - max_or_sink)
        denom = scores_exp.sum(axis=-1, keepdims=True) + sinks_exp
        weights = (scores_exp / denom).astype(v.dtype)
    else:
        weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    ctx = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return ctx.astype(v.dtype)


def _spec_shard_factor(entry: str | tuple[str, ...] | None, mesh) -> int:
    """
    Compute the size of the mesh axes associated with a PartitionSpec entry.

    Splash attention can handle various kinds of sequence parallelism but needs this to function
    """
    if entry is None or mesh is None:
        return 1
    if isinstance(entry, str):
        return mesh.shape[entry]
    if isinstance(entry, tuple):
        factor = 1
        for name in entry:
            factor *= mesh.shape[name]
        return factor
    raise TypeError(f"Unsupported PartitionSpec entry: {entry!r}")


def _tpu_splash_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | jax.Array | None,
    sinks: Float[Array, "Hq"] | None = None,
) -> Float[Array, "B Q Hq D"]:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
    from jax.experimental.pallas.ops.tpu.splash_attention import SegmentIds as SplashSegmentIds

    # Splash attention expects BHSD.
    q_ = jnp.transpose(q, (0, 2, 1, 3))
    k_ = jnp.transpose(k, (0, 2, 1, 3))
    v_ = jnp.transpose(v, (0, 2, 1, 3))

    B, Hq, Sq, D = q_.shape
    _, _, Sk, _ = k_.shape

    if Sk % 128 != 0:
        raise NotImplementedError("Splash attention requires key/value sequence length to be a multiple of 128.")

    q_ = q_ * (1.0 / math.sqrt(D))

    mesh = _get_mesh()
    if mesh is None or getattr(mesh, "empty", False):
        raise RuntimeError("TPU splash attention requires a JAX mesh context.")

    def _named_sharding_of(x: jax.Array, *, label: str) -> NamedSharding:
        """Extract NamedSharding from a JAX value or tracer.

        In JAX, `.sharding` is not available on tracers during staging; however the sharding
        is still available on the underlying abstract value in many cases.
        """
        sharding = None
        try:
            sharding = x.sharding  # type: ignore[attr-defined]
        except Exception:
            sharding = None
        if sharding is None:
            aval = getattr(x, "aval", None)
            sharding = getattr(aval, "sharding", None) if aval is not None else None
        if not isinstance(sharding, NamedSharding):
            raise TypeError(
                f"TPU splash attention expects NamedSharding on {label} under an explicit mesh; got {sharding!r}."
            )
        return sharding

    q_sharding = _named_sharding_of(q_, label="q")
    k_sharding = _named_sharding_of(k_, label="k")
    v_sharding = _named_sharding_of(v_, label="v")

    q_pspec = q_sharding.spec
    k_pspec = k_sharding.spec
    v_pspec = v_sharding.spec

    # KV sequence must not be sharded for splash attention.
    if k_pspec[2] is not None:
        raise NotImplementedError(
            "Splash attention does not support sharding the KV sequence dimension. "
            f"Got KV sequence spec: {k_pspec[2]}"
        )

    head_shards = _spec_shard_factor(q_pspec[1], mesh)
    q_seq_shards = _spec_shard_factor(q_pspec[2], mesh)
    kv_seq_shards = _spec_shard_factor(k_pspec[2], mesh)

    # MaxText uses a block size of 512. Pick per-shard blocks that evenly divide each shard length,
    # preferring multiples of 128 when possible.
    block_size = 512

    shard_Sq = max(1, Sq // max(1, q_seq_shards))
    shard_Sk = max(1, Sk // max(1, kv_seq_shards))

    def _compatible_block(shard_len: int, max_block: int) -> int:
        """Pick largest block <= max_block that divides shard_len; prefer multiples of 128."""
        if shard_len <= 0:
            return max_block
        cap = min(max_block, shard_len)
        for step in (128, 1):
            candidate = cap - (cap % step)
            while candidate >= step:
                if shard_len % candidate == 0:
                    return candidate
                candidate -= step
        return 1

    block_q = _compatible_block(shard_Sq, block_size)
    block_kv = _compatible_block(shard_Sk, block_size)

    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block_q,
        block_kv_compute=block_kv,
        block_kv=block_kv,
        block_q_dkv=block_q,
        block_kv_dkv=block_kv,
        block_kv_dkv_compute=block_q,
        block_q_dq=block_q,
        block_kv_dq=block_kv,
    )

    if mask is None:
        base_mask = splash_attention_mask.FullMask(_shape=(Sq, Sk))
        segment_ids = None
        segment_ids_axes = None
        segment_batch_axis = None
    elif isinstance(mask, AttentionMask):
        if mask.is_causal:
            base_mask = splash_attention_mask.CausalMask((Sq, Sk), offset=0, shard_count=q_seq_shards)
        else:
            base_mask = splash_attention_mask.FullMask(_shape=(Sq, Sk))

        if mask.sliding_window is not None:
            if mask.sliding_window <= 0:
                raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")
            local_mask = splash_attention_mask.LocalMask(
                shape=(Sq, Sk),
                # Grug's `sliding_window` matches the "lookback" semantics used in the
                # reference mask materialization: allow attending to keys with
                #   k >= q - (sliding_window - 1)
                # (and optionally combine with causal).
                window_size=(mask.sliding_window - 1, None),
                offset=0,
                shard_count=q_seq_shards,
            )
            base_mask = splash_attention_mask.LogicalAnd(base_mask, local_mask)

        if mask.segment_ids is not None:
            q_segment_ids, kv_segment_ids = mask.segment_ids
            q_seg_sharding = _named_sharding_of(q_segment_ids, label="segment_ids.q")
            kv_seg_sharding = _named_sharding_of(kv_segment_ids, label="segment_ids.kv")
            segment_ids = SplashSegmentIds(q_segment_ids, kv_segment_ids)
            segment_ids_axes = SplashSegmentIds(
                q_seg_sharding.spec,
                kv_seg_sharding.spec,
            )
            segment_batch_axis = SplashSegmentIds(
                0 if q_segment_ids.ndim == 2 else None,
                0 if kv_segment_ids.ndim == 2 else None,
            )
        else:
            segment_ids = None
            segment_ids_axes = None
            segment_batch_axis = None
    else:
        raise NotImplementedError("Dense masks are not supported for splash attention.")

    kernel_mask = splash_attention_mask.MultiHeadMask(masks=[base_mask for _ in range(Hq)])

    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=kernel_mask,
        block_sizes=block_sizes,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
    )

    kernel_sharding = NamedSharding(mesh, P(q_pspec[1], q_pspec[2]))
    kernel_specs = splash_kernel.manual_sharding_spec(kernel_sharding)

    # Sinks are replicated (per-head scalars)
    sinks_spec = P(None) if sinks is not None else None

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(q_pspec, k_pspec, v_pspec, segment_ids_axes, kernel_specs, sinks_spec),
        out_specs=q_pspec,
        check_rep=False,
    )
    def wrap(q_bhsd, k_bhsd, v_bhsd, seg_ids, kernel, sinks_):
        def call_kernel(q_b, k_b, v_b, seg_id):
            return kernel(q_b, k_b, v_b, segment_ids=seg_id, sinks=sinks_)

        return jax.vmap(call_kernel, in_axes=(0, 0, 0, segment_batch_axis))(q_bhsd, k_bhsd, v_bhsd, seg_ids)

    out = wrap(q_, k_, v_, segment_ids, splash_kernel, sinks)
    return jnp.transpose(out, (0, 2, 1, 3)).astype(v.dtype)


def attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    sinks: Float[Array, "Hq"] | None = None,
) -> Float[Array, "B Q Hq D"]:
    if jax.default_backend() == "tpu":
        if isinstance(mask, jax.Array):
            return reference_attention(q, k, v, mask, logits_dtype=jnp.float32, sinks=sinks)
        return _tpu_splash_attention(q, k, v, mask, sinks=sinks)
    return reference_attention(q, k, v, mask, logits_dtype=jnp.float32, sinks=sinks)


__all__ = [
    "AttentionMask",
    "RotaryConfig",
    "apply_rotary_embedding",
    "attention",
    "reference_attention",
]
