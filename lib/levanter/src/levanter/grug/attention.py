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

from haliax.partitioning import _get_mesh

from .config import RotaryConfig


def _positions_from_segment_ids_2d(segment_ids: jax.Array, *, pad_value: int) -> jax.Array:
    """Compute per-token positions that reset at segment boundaries.

    `segment_ids` uses -1 for padding. This helper is sharding-friendly: it keeps the batch
    dimension intact and scans over the sequence dimension, avoiding `vmap(scan)` patterns
    that can trigger sharding errors on TPU.
    """
    if segment_ids.ndim != 2:
        raise ValueError(f"segment_ids must be 2D (batch, seq), got shape={segment_ids.shape}")

    # Make sure the scan carry has the same sharding as the per-batch ids.
    # Using plain `jnp.full((batch,), ...)` can default to replicated sharding and then
    # JAX will error in `where/select` when mixing sharded and replicated values.
    init_last = jnp.full_like(segment_ids[:, 0], jnp.int32(-2))
    init_pos = jnp.full_like(segment_ids[:, 0], jnp.int32(-1))
    pad = jnp.int32(pad_value)

    # Scan over sequence; lax.scan is time-major, so swap to (seq, batch).
    segment_tm = jnp.swapaxes(segment_ids.astype(jnp.int32), 0, 1)

    def step(carry, seg_t):
        last_seg, pos = carry
        is_new = seg_t != last_seg
        pos = jnp.where(is_new, jnp.zeros_like(pos), pos + jnp.int32(1))
        last_seg = jnp.where(is_new, seg_t, last_seg)
        pos_out = jnp.where(seg_t < 0, jnp.full_like(pos, pad), pos)
        return (last_seg, pos), pos_out

    (_, _), pos_tm = jax.lax.scan(step, (init_last, init_pos), segment_tm)
    return jnp.swapaxes(pos_tm, 0, 1)


@functools.partial(
    register_dataclass, data_fields=["segment_ids"], meta_fields=["is_causal", "causal_offset", "sliding_window"]
)
@dataclass(frozen=True)
class AttentionMask:
    """Grug attention mask spec.

    This is deliberately simpler than `levanter.layers.attention.AttentionMask`:
    - Stores raw JAX arrays (no NamedArray fields).
    - Does not support explicit masks (for now).
    - Supports causal masking, sliding windows, and segment IDs.
    """

    is_causal: bool = False
    causal_offset: int = 0
    segment_ids: tuple[jax.Array, jax.Array] | None = None
    sliding_window: int | None = None

    @classmethod
    def causal(cls, *, offset: int = 0, sliding_window: int | None = None) -> "AttentionMask":
        return cls(is_causal=True, causal_offset=offset, sliding_window=sliding_window)

    def with_segment_ids(self, q_segment_ids: jax.Array, kv_segment_ids: jax.Array | None = None) -> "AttentionMask":
        kv_ids = q_segment_ids if kv_segment_ids is None else kv_segment_ids
        return AttentionMask(
            is_causal=self.is_causal,
            causal_offset=self.causal_offset,
            segment_ids=(q_segment_ids, kv_ids),
            sliding_window=self.sliding_window,
        )

    def with_sliding_window(self, sliding_window: int | None) -> "AttentionMask":
        return AttentionMask(
            is_causal=self.is_causal,
            causal_offset=self.causal_offset,
            segment_ids=self.segment_ids,
            sliding_window=sliding_window,
        )

    def materialize_mask(self, q_len: int, k_len: int) -> jax.Array | None:
        """Return a boolean mask (True = allowed) or None.

        Shapes:
          - If `segment_ids` is unset, returns `(q_len, k_len)` (broadcastable across batch).
          - If `segment_ids` is set with per-batch IDs, returns `(batch, q_len, k_len)`.
        """
        mask = None

        if self.is_causal:
            q_idx = jnp.arange(q_len)[:, None]
            k_idx = jnp.arange(k_len)[None, :]
            allowed = k_idx <= q_idx + self.causal_offset
            mask = allowed

        if self.sliding_window is not None:
            q_idx = jnp.arange(q_len)[:, None]
            k_idx = jnp.arange(k_len)[None, :]
            allowed = k_idx >= q_idx - self.sliding_window
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


def _rotary_cache(seq_len: int, head_dim: int, rope: RotaryConfig) -> tuple[jax.Array, jax.Array]:
    half_dim = head_dim // 2
    inv_freq = 1.0 / (rope.theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    return cos, sin


def apply_rotary_embedding(
    q: jax.Array,
    k: jax.Array,
    *,
    seq_len: int,
    head_dim: int,
    rope: RotaryConfig,
) -> tuple[jax.Array, jax.Array]:
    cos, sin = _rotary_cache(seq_len, head_dim, rope)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def _apply(x: jax.Array) -> jax.Array:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    return _apply(q), _apply(k)


def reference_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: AttentionMask | jax.Array | None,
    *,
    logits_dtype: jnp.dtype | None,
) -> jax.Array:
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
    if isinstance(mask, AttentionMask):
        mask = mask.materialize_mask(scores.shape[-2], scores.shape[-1])

    if mask is not None:
        if mask.dtype == jnp.bool_:
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            else:
                raise ValueError(f"mask must be 2D or 3D, got shape={mask.shape}")
            scores = jnp.where(mask, scores, jnp.array(-1e9, dtype=scores.dtype))
        else:
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            else:
                raise ValueError(f"mask must be 2D or 3D, got shape={mask.shape}")
            scores = scores + mask
    if logits_dtype is not None:
        scores = scores.astype(logits_dtype)
    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    ctx = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return ctx.astype(v.dtype)


def _pspec_from_sharding(x: jax.Array) -> P:
    sharding = getattr(x, "sharding", None)
    if isinstance(sharding, NamedSharding) and isinstance(sharding.spec, P):
        return sharding.spec
    # In tracing contexts we may not have a concrete sharding available.
    # Default to the common "data-parallel batch" convention used by grug.
    if x.ndim == 4:
        return P(("replica", "data"), None, None, None)
    if x.ndim == 3:
        return P(("replica", "data"), None, None)
    if x.ndim == 2:
        return P(("replica", "data"), None)
    return P(*(None,) * x.ndim)


def _spec_shard_factor(entry: object, mesh) -> int:
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
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: AttentionMask | jax.Array | None,
) -> jax.Array:
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

    q_pspec = _pspec_from_sharding(q_)
    k_pspec = _pspec_from_sharding(k_)
    v_pspec = _pspec_from_sharding(v_)

    # KV sequence must not be sharded for splash attention.
    if k_pspec[2] is not None:
        raise NotImplementedError(
            "Splash attention does not support sharding the KV sequence dimension. "
            f"Got KV sequence spec: {k_pspec[2]}"
        )

    head_shards = _spec_shard_factor(q_pspec[1], mesh)
    q_seq_shards = _spec_shard_factor(q_pspec[2], mesh)

    # Choose blocks per-shard so we avoid partial blocks.
    def _compatible_block(shard_len: int, max_block: int) -> int:
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

    block_size = 512
    shard_Sq = max(1, Sq // max(1, q_seq_shards))
    block_q = _compatible_block(shard_Sq, block_size)
    block_kv = _compatible_block(Sk, block_size)

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
        if mask.causal_offset != 0:
            raise NotImplementedError("Causal offsets are not supported for splash attention.")

        if mask.is_causal:
            base_mask = splash_attention_mask.CausalMask((Sq, Sk), offset=0, shard_count=q_seq_shards)
        else:
            base_mask = splash_attention_mask.FullMask(_shape=(Sq, Sk))

        if mask.sliding_window is not None:
            local_mask = splash_attention_mask.LocalMask(
                shape=(Sq, Sk),
                window_size=(mask.sliding_window - 1, None),
                offset=0,
                shard_count=q_seq_shards,
            )
            base_mask = splash_attention_mask.LogicalAnd(base_mask, local_mask)

        if mask.segment_ids is not None:
            q_segment_ids, kv_segment_ids = mask.segment_ids
            segment_ids = SplashSegmentIds(q_segment_ids, kv_segment_ids)
            segment_ids_axes = SplashSegmentIds(
                _pspec_from_sharding(q_segment_ids),
                _pspec_from_sharding(kv_segment_ids),
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
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
        block_sizes=block_sizes,
    )

    kernel_sharding = NamedSharding(mesh, P(q_pspec[1], q_pspec[2]))
    kernel_specs = splash_kernel.manual_sharding_spec(kernel_sharding)

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(q_pspec, k_pspec, v_pspec, segment_ids_axes, kernel_specs),
        out_specs=q_pspec,
        check_rep=False,
    )
    def wrap(q_bhsd, k_bhsd, v_bhsd, seg_ids, kernel):
        def call_kernel(q_b, k_b, v_b, si):
            if si is None:
                return kernel(q_b, k_b, v_b)
            return kernel(q_b, k_b, v_b, segment_ids=si)

        return jax.vmap(call_kernel, in_axes=(0, 0, 0, segment_batch_axis))(q_bhsd, k_bhsd, v_bhsd, seg_ids)

    out = wrap(q_, k_, v_, segment_ids, splash_kernel)
    return jnp.transpose(out, (0, 2, 1, 3)).astype(v.dtype)


def attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: AttentionMask | jax.Array | None,
) -> jax.Array:
    if jax.default_backend() == "tpu":
        return _tpu_splash_attention(q, k, v, mask)
    return reference_attention(q, k, v, mask, logits_dtype=jnp.float32)


__all__ = [
    "AttentionMask",
    "apply_rotary_embedding",
    "attention",
    "reference_attention",
]
