# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import functools
import inspect
import math
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding
from jaxtyping import Array, Bool, Float, Int

from haliax.jax_utils import named_call
from haliax.partitioning import _get_mesh
from levanter.grug.fa4_cute_backend import fa4_cute_attention_forward

_SHARD_MAP_CHECK_KWARG = "check_vma" if "check_vma" in inspect.signature(shard_map).parameters else "check_rep"
_SHARD_MAP_CHECK_KWARGS = {_SHARD_MAP_CHECK_KWARG: False}
GrugAttentionImplementation = Literal[
    "reference",
    "tpu_splash",
    "gpu_xla",
    "gpu_cudnn",
    "gpu_te",
    "gpu_fa4_cute",
]
DEFAULT_MAX_PACKED_SEGMENTS = 64


@dataclass(frozen=True)
class _Flash4CuteKernelConfig:
    forward_tile: tuple[int, int]
    backward_tile: tuple[int, int]
    num_threads: int
    forward_num_stages: int
    backward_num_stages_q: int
    backward_num_stages_do: int
    use_sm80_mma: bool
    allow_split_kv: bool
    allow_paged_kv: bool
    allow_block_sparse: bool
    allow_mask_mod_backward: bool


@dataclass(frozen=True)
class RotaryConfig:
    """Lightweight rotary embedding configuration."""

    theta: float = 10000.0
    scaling_factor: float | None = None


class AttentionMask(eqx.Module):
    """Grug attention mask spec.

    This is deliberately simpler than `levanter.layers.attention.AttentionMask`:
    - Stores raw JAX arrays (no NamedArray fields).
    - Supports causal masking, sliding windows, and segment IDs.
    """

    is_causal: bool = eqx.field(default=False, static=True)
    segment_ids: tuple[jax.Array, jax.Array] | None = None
    sliding_window: int | None = eqx.field(default=None, static=True)
    max_segments_per_seq: int | None = eqx.field(default=None, static=True)

    @classmethod
    def causal(
        cls,
        *,
        sliding_window: int | None = None,
        max_segments_per_seq: int | None = None,
    ) -> "AttentionMask":
        return cls(
            is_causal=True,
            sliding_window=sliding_window,
            max_segments_per_seq=max_segments_per_seq,
        )

    def with_segment_ids(
        self,
        q_segment_ids: Int[Array, "..."],
        kv_segment_ids: Int[Array, "..."] | None = None,
        *,
        max_segments_per_seq: int | None = None,
    ) -> "AttentionMask":
        kv_ids = q_segment_ids if kv_segment_ids is None else kv_segment_ids
        return AttentionMask(
            is_causal=self.is_causal,
            segment_ids=(q_segment_ids, kv_ids),
            sliding_window=self.sliding_window,
            max_segments_per_seq=self.max_segments_per_seq if max_segments_per_seq is None else max_segments_per_seq,
        )

    def with_sliding_window(self, sliding_window: int | None) -> "AttentionMask":
        return AttentionMask(
            is_causal=self.is_causal,
            segment_ids=self.segment_ids,
            sliding_window=sliding_window,
            max_segments_per_seq=self.max_segments_per_seq,
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


def align_kv_heads(x: Float[Array, "B K Hkv D"], *, num_q_heads: int) -> Float[Array, "B K Hq D"]:
    """Expand grouped-query KV heads to match query-head layout."""
    num_kv_heads = x.shape[2]
    if num_q_heads == num_kv_heads:
        return x
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    repeat = num_q_heads // num_kv_heads
    # Use reshape + broadcast instead of jnp.repeat to avoid sharding issues.
    expanded = jnp.expand_dims(x, axis=3)
    tiled = jnp.broadcast_to(expanded, (*x.shape[:3], repeat, x.shape[3]))
    return tiled.reshape(*x.shape[:2], num_q_heads, x.shape[3])


def reference_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    logits_dtype: jnp.dtype | None,
) -> Float[Array, "B Q Hq D"]:
    head_dim = q.shape[-1]
    num_q_heads = q.shape[2]
    k = align_kv_heads(k, num_q_heads=num_q_heads)
    v = align_kv_heads(v, num_q_heads=num_q_heads)

    scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q * scale, k)

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
    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    ctx = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return ctx.astype(v.dtype)


def _dense_attention_mask_to_bnts(
    mask: Bool[Array, "B Q K"] | Float[Array, "B Q K"],
    *,
    batch_size: int,
    q_len: int,
    k_len: int,
) -> jax.Array:
    if mask.ndim == 2:
        mask = mask[None, :, :]
    if mask.ndim != 3:
        raise ValueError(f"explicit mask must have shape [batch, q, k], got shape={mask.shape}")
    if mask.shape[0] not in (1, batch_size):
        raise ValueError(f"explicit mask batch dim must be 1 or {batch_size}, got {mask.shape[0]}")
    if mask.shape[1] != q_len or mask.shape[2] != k_len:
        raise ValueError(
            f"explicit mask must match attention shapes: got mask={mask.shape}, expected [batch,{q_len},{k_len}]"
        )
    return mask[:, None, :, :]


def _mask_to_jax_dot_product_attention_args(
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    batch_size: int,
    q_len: int,
    k_len: int,
) -> tuple[jax.Array | None, bool, tuple[int, int] | None]:
    """Map Grug/Splash mask semantics onto `jax.nn.dot_product_attention` args."""
    if mask is None:
        return None, False, None

    if isinstance(mask, AttentionMask):
        if mask.sliding_window is not None and mask.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")

        if mask.segment_ids is not None:
            raise NotImplementedError("JAX SDPA backends would require a dense segment_ids mask; use gpu_te instead.")

        dense_mask = None
        is_causal = mask.is_causal
        local_window_size = None

        if mask.sliding_window is not None:
            if mask.is_causal:
                local_window_size = (mask.sliding_window - 1, 0)
            else:
                q_idx = jnp.arange(q_len)[:, None]
                k_idx = jnp.arange(k_len)[None, :]
                sliding_mask = k_idx >= q_idx - (mask.sliding_window - 1)
                sliding_mask = sliding_mask[None, None, :, :]
                dense_mask = sliding_mask if dense_mask is None else jnp.logical_and(dense_mask, sliding_mask)

        return dense_mask, is_causal, local_window_size

    dense_mask = _dense_attention_mask_to_bnts(mask, batch_size=batch_size, q_len=q_len, k_len=k_len)
    if dense_mask.dtype != jnp.bool_:
        raise NotImplementedError("Additive dense masks are not supported by the native GPU attention prototype.")
    return dense_mask, False, None


def _jax_dot_product_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    implementation: Literal["xla", "cudnn"],
) -> Float[Array, "B Q Hq D"]:
    if isinstance(mask, AttentionMask) and mask.segment_ids is not None:
        raise NotImplementedError(
            "JAX SDPA backends do not support segment_ids without a dense mask; use gpu_te instead."
        )

    jax_mask, is_causal, local_window_size = _mask_to_jax_dot_product_attention_args(
        mask,
        batch_size=q.shape[0],
        q_len=q.shape[1],
        k_len=k.shape[1],
    )
    return jax.nn.dot_product_attention(
        q,
        k,
        v,
        mask=jax_mask,
        is_causal=is_causal,
        local_window_size=local_window_size,
        implementation=implementation,
    ).astype(v.dtype)


def gpu_cudnn_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
) -> Float[Array, "B Q Hq D"]:
    """Run Grug attention through JAX's native cuDNN SDPA path."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_cudnn_attention requires the JAX GPU backend.")
    return _jax_dot_product_attention(q, k, v, mask, implementation="cudnn")


def gpu_xla_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
) -> Float[Array, "B Q Hq D"]:
    """Run Grug attention through JAX's GPU SDPA lowering for JAX-supported masks."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_xla_attention requires the JAX GPU backend.")
    return _jax_dot_product_attention(q, k, v, mask, implementation="xla")


def _packed_segment_seqlens_offsets(
    segment_ids: jax.Array,
    *,
    batch_size: int,
    seq_len: int,
    max_segments_per_seq: int,
) -> tuple[Int[Array, "B M"], Int[Array, "B Mp1"]]:
    """Convert dynamic loader-style packed segment IDs to TE THD sequence lengths and offsets."""
    segment_ids = _batched_segment_ids(segment_ids, batch_size=batch_size, seq_len=seq_len)

    if max_segments_per_seq <= 0:
        raise ValueError(f"max_segments_per_seq must be positive, got {max_segments_per_seq}")

    return _seqlens_offsets_from_starts(
        _segment_starts(segment_ids),
        segment_ids >= 0,
        max_segments=max_segments_per_seq,
    )


def _batched_segment_ids(segment_ids: jax.Array, *, batch_size: int, seq_len: int) -> jax.Array:
    if segment_ids.ndim == 1:
        if segment_ids.shape[0] != seq_len:
            raise ValueError(f"1D segment_ids must match sequence length {seq_len}, got {segment_ids.shape}")
        segment_ids = jnp.broadcast_to(segment_ids[None, :], (batch_size, seq_len))
    elif segment_ids.ndim == 2:
        if segment_ids.shape[0] not in (1, batch_size) or segment_ids.shape[1] != seq_len:
            raise ValueError(f"2D segment_ids must have shape [1|{batch_size}, {seq_len}], got {segment_ids.shape}")
        if segment_ids.shape[0] == 1 and batch_size != 1:
            segment_ids = jnp.broadcast_to(segment_ids, (batch_size, seq_len))
    else:
        raise ValueError(f"segment_ids must be 1D or 2D, got ndim={segment_ids.ndim}")
    return segment_ids


def _segment_starts(segment_ids: jax.Array) -> jax.Array:
    valid = segment_ids >= 0
    previous = jnp.concatenate([segment_ids[:, :1], segment_ids[:, :-1]], axis=1)
    first = jnp.zeros_like(valid).at[:, 0].set(True)
    return valid & (first | (segment_ids != previous))


def _seqlens_offsets_from_starts(
    starts: jax.Array,
    valid: jax.Array,
    *,
    max_segments: int,
    include_terminal_offset: bool = False,
    cumulative_offsets: bool = False,
) -> tuple[Int[Array, "B M"], Int[Array, "B Mp1"]]:
    batch_size = starts.shape[0]
    num_segments = jnp.sum(starts.astype(jnp.int32), axis=1)
    starts = eqx.error_if(
        starts,
        jnp.any(num_segments > max_segments),
        "packed segment count exceeds max_segments_per_seq.",
    )
    ordinals = jnp.cumsum(starts.astype(jnp.int32), axis=1) - 1
    clipped_ordinals = jnp.clip(ordinals, 0, max_segments - 1)
    batch_idx = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    lengths = jnp.zeros((batch_size, max_segments), dtype=jnp.int32)
    lengths = lengths.at[batch_idx, clipped_ordinals].add(valid.astype(jnp.int32))

    segment_idx = jnp.arange(max_segments, dtype=jnp.int32)[None, :]
    seqlens = jnp.where(segment_idx < num_segments[:, None], lengths, -1)

    if cumulative_offsets:
        offsets = jnp.concatenate([jnp.zeros((batch_size, 1), dtype=jnp.int32), jnp.cumsum(lengths, axis=1)], axis=1)
        offset_idx = jnp.arange(max_segments + 1, dtype=jnp.int32)[None, :]
        offsets = jnp.where(offset_idx <= num_segments[:, None], offsets, -1)
        return seqlens, offsets

    offsets = jax.vmap(functools.partial(jnp.argwhere, size=max_segments + 1, fill_value=-1))(starts)
    offsets = offsets.squeeze(axis=-1).astype(jnp.int32)
    if include_terminal_offset:
        positions = jnp.arange(starts.shape[1], dtype=jnp.int32)[None, :]
        terminal_offsets = jnp.max(jnp.where(valid, positions + 1, 0), axis=1)
        terminal_slots = jnp.clip(num_segments, 0, max_segments)
        offsets = offsets.at[jnp.arange(batch_size, dtype=jnp.int32), terminal_slots].set(terminal_offsets)
    return seqlens, offsets


def _flattened_packed_segment_seqlens_offsets(
    segment_ids: jax.Array,
    *,
    batch_size: int,
    seq_len: int,
    max_segments_per_seq: int,
) -> tuple[Int[Array, "B M"], Int[Array, "B Mp1"]]:
    segment_ids = _batched_segment_ids(segment_ids, batch_size=batch_size, seq_len=seq_len)
    max_segments = batch_size * max_segments_per_seq
    if max_segments <= 0:
        raise ValueError(f"max_segments must be positive, got {max_segments}")

    starts = _segment_starts(segment_ids).reshape(1, batch_size * seq_len)
    valid = (segment_ids >= 0).reshape(1, batch_size * seq_len)
    return _seqlens_offsets_from_starts(
        starts,
        valid,
        max_segments=max_segments,
        include_terminal_offset=True,
        cumulative_offsets=batch_size == 1,
    )


def _packed_segment_ids_positions(
    segment_ids: jax.Array,
    *,
    batch_size: int,
    seq_len: int,
) -> tuple[Int[Array, "B S"], Int[Array, "B S"]]:
    segment_ids = _batched_segment_ids(segment_ids, batch_size=batch_size, seq_len=seq_len)
    valid = segment_ids >= 0
    starts = _segment_starts(segment_ids)
    local_segment_ids = jnp.where(valid, jnp.cumsum(starts.astype(jnp.int32), axis=1), 0)

    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    current_start = _packed_segment_start_positions(
        segment_ids,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    segment_positions = jnp.where(valid, positions - current_start, 0)
    return local_segment_ids, segment_positions


def _packed_segment_start_positions(
    segment_ids: jax.Array,
    *,
    batch_size: int,
    seq_len: int,
) -> Int[Array, "B S"]:
    segment_ids = _batched_segment_ids(segment_ids, batch_size=batch_size, seq_len=seq_len)
    valid = segment_ids >= 0
    starts = _segment_starts(segment_ids)
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    start_positions = jnp.where(starts, positions, 0)
    current_start = jax.lax.associative_scan(jnp.maximum, start_positions, axis=1)
    return jnp.where(valid, current_start, seq_len)


def _packed_segment_causal_lower_bounds(
    segment_ids: jax.Array,
    *,
    batch_size: int,
    seq_len: int,
    sliding_window: int | None,
) -> tuple[Int[Array, "B S"], Bool[Array, "B S"]]:
    """Return per-token inclusive key lower bounds for dynamic packed causal attention."""
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")

    segment_ids = _batched_segment_ids(segment_ids, batch_size=batch_size, seq_len=seq_len)
    valid = segment_ids >= 0
    lower_bounds = _packed_segment_start_positions(
        segment_ids,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    if sliding_window is not None:
        positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        window_lower_bounds = positions - (sliding_window - 1)
        lower_bounds = jnp.maximum(lower_bounds, window_lower_bounds)
    return jnp.where(valid, lower_bounds, seq_len), valid


def _packed_self_attention_segment_ids(
    q: jax.Array,
    k: jax.Array,
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    backend_name: str,
) -> tuple[Int[Array, "B S"], int]:
    if isinstance(mask, jax.Array):
        raise NotImplementedError(f"{backend_name} does not support dense masks.")
    if not isinstance(mask, AttentionMask) or mask.segment_ids is None:
        raise NotImplementedError(f"{backend_name} currently requires packed segment_ids.")
    if not mask.is_causal:
        raise NotImplementedError(f"{backend_name} currently supports only causal packed self-attention.")
    if q.shape[0] != k.shape[0]:
        raise NotImplementedError(f"{backend_name} requires matching q/kv batch sizes.")
    if q.shape[1] != k.shape[1]:
        raise NotImplementedError(f"{backend_name} requires self-attention q_len == k_len.")
    if mask.sliding_window is not None and mask.sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")

    q_segment_ids, kv_segment_ids = mask.segment_ids
    max_segments_per_seq = q.shape[1] if mask.max_segments_per_seq is None else mask.max_segments_per_seq
    if max_segments_per_seq <= 0:
        raise ValueError(f"max_segments_per_seq must be positive, got {max_segments_per_seq}")
    same_segment_ids = q_segment_ids is kv_segment_ids
    q_segment_ids = _batched_segment_ids(q_segment_ids, batch_size=q.shape[0], seq_len=q.shape[1])
    if not same_segment_ids:
        kv_segment_ids = _batched_segment_ids(kv_segment_ids, batch_size=k.shape[0], seq_len=k.shape[1])
        q_segment_ids = eqx.error_if(
            q_segment_ids,
            jnp.any(q_segment_ids != kv_segment_ids),
            f"{backend_name} requires matching q/kv segment_ids for packed self-attention.",
        )
    return q_segment_ids, max_segments_per_seq


def _flash4_cute_kernel_config(
    head_dim: int,
    head_dim_v: int,
    *,
    arch: int,
) -> _Flash4CuteKernelConfig:
    arch_family = arch // 10
    if arch_family == 12:
        return _Flash4CuteKernelConfig(
            forward_tile=(128, 128 if head_dim <= 64 else 64),
            backward_tile=(64, 64),
            num_threads=128,
            forward_num_stages=1,
            backward_num_stages_q=2 if head_dim <= 64 else 1,
            backward_num_stages_do=2 if head_dim <= 64 else 1,
            use_sm80_mma=True,
            allow_split_kv=False,
            allow_paged_kv=False,
            allow_block_sparse=False,
            allow_mask_mod_backward=False,
        )
    if arch_family == 8:
        return _Flash4CuteKernelConfig(
            forward_tile=(128, 64),
            backward_tile=(128, 64),
            num_threads=128,
            forward_num_stages=1,
            backward_num_stages_q=1,
            backward_num_stages_do=1,
            use_sm80_mma=True,
            allow_split_kv=True,
            allow_paged_kv=True,
            allow_block_sparse=False,
            allow_mask_mod_backward=True,
        )
    if arch_family == 9:
        return _Flash4CuteKernelConfig(
            forward_tile=(128, 128 if head_dim <= 64 else 64),
            backward_tile=(128, 64),
            num_threads=128,
            forward_num_stages=1,
            backward_num_stages_q=1,
            backward_num_stages_do=1,
            use_sm80_mma=True,
            allow_split_kv=True,
            allow_paged_kv=True,
            allow_block_sparse=False,
            allow_mask_mod_backward=False,
        )
    raise NotImplementedError(f"FA4/CuTe attention does not support SM{arch}.")


def _gpu_compute_arch() -> int:
    for device in jax.local_devices(backend="gpu"):
        compute_capability = getattr(device, "compute_capability", None)
        if callable(compute_capability):
            compute_capability = compute_capability()
        if isinstance(compute_capability, tuple) and len(compute_capability) >= 2:
            return int(compute_capability[0]) * 10 + int(compute_capability[1])
        if isinstance(compute_capability, str):
            major, _, minor = compute_capability.partition(".")
            if major.isdigit() and minor.isdigit():
                return int(major) * 10 + int(minor)
    raise RuntimeError("Could not determine CUDA compute capability for FA4/CuTe attention.")


def gpu_fa4_cute_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
) -> Float[Array, "B Q Hq D"]:
    """Run dynamic packed segment attention through a FlashAttention-4/CuTe JAX FFI backend."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_fa4_cute_attention requires the JAX GPU backend.")

    q_segment_ids, _ = _packed_self_attention_segment_ids(q, k, mask, backend_name="gpu_fa4_cute_attention")
    assert isinstance(mask, AttentionMask)
    lower_bounds, valid = _packed_segment_causal_lower_bounds(
        q_segment_ids,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        sliding_window=mask.sliding_window,
    )
    kernel_config = _flash4_cute_kernel_config(q.shape[-1], v.shape[-1], arch=_gpu_compute_arch())

    return fa4_cute_attention_forward(
        q,
        k,
        v,
        lower_bounds,
        valid,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        kernel_config=kernel_config,
    )


def gpu_te_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
) -> Float[Array, "B Q Hq D"]:
    """Run dynamic packed segment attention through Transformer Engine/cudnn THD attention."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_te_attention requires the JAX GPU backend.")
    q_segment_ids, max_segments_per_seq = _packed_self_attention_segment_ids(
        q,
        k,
        mask,
        backend_name="gpu_te_attention",
    )
    assert isinstance(mask, AttentionMask)
    from transformer_engine.jax.attention import (  # type: ignore[import]
        AttnBiasType,
        AttnMaskType,
        AttnSoftmaxType,
        QKVLayout,
        SequenceDescriptor,
        fused_attn,
    )

    te_max_segments = q.shape[0] * max_segments_per_seq
    seqlens, offsets = _flattened_packed_segment_seqlens_offsets(
        q_segment_ids,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        max_segments_per_seq=max_segments_per_seq,
    )
    sequence_descriptor = SequenceDescriptor.from_seqlens_and_offsets(
        (seqlens, seqlens),
        (offsets, offsets),
    )
    window_size = None
    if mask.sliding_window is not None and mask.sliding_window < q.shape[1]:
        window_size = (mask.sliding_window - 1, 0)
    if window_size is None:
        te_segment_ids, te_segment_positions = _packed_segment_ids_positions(
            q_segment_ids,
            batch_size=q.shape[0],
            seq_len=q.shape[1],
        )
        sequence_descriptor = SequenceDescriptor.from_segment_ids_and_pos(
            (te_segment_ids, te_segment_ids),
            (te_segment_positions, te_segment_positions),
        )
        out = fused_attn(
            (q, k, v),
            None,
            sequence_descriptor,
            None,
            AttnBiasType.NO_BIAS,
            AttnMaskType.PADDING_CAUSAL_MASK,
            QKVLayout.THD_THD_THD,
            AttnSoftmaxType.VANILLA_SOFTMAX,
            1.0 / math.sqrt(q.shape[-1]),
            0.0,
            True,
            max_segments_per_seq=max_segments_per_seq,
            window_size=None,
        )
        return out.astype(v.dtype)

    q_flat = q.reshape(1, q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
    k_flat = k.reshape(1, k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
    v_flat = v.reshape(1, v.shape[0] * v.shape[1], v.shape[2], v.shape[3])
    out = fused_attn(
        (q_flat, k_flat, v_flat),
        None,
        sequence_descriptor,
        None,
        AttnBiasType.NO_BIAS,
        AttnMaskType.PADDING_CAUSAL_MASK,
        QKVLayout.THD_THD_THD,
        AttnSoftmaxType.VANILLA_SOFTMAX,
        1.0 / math.sqrt(q.shape[-1]),
        0.0,
        True,
        max_segments_per_seq=te_max_segments,
        window_size=window_size,
    )
    return out.reshape(q.shape[0], q.shape[1], q.shape[2], v.shape[3]).astype(v.dtype)


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

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(q_pspec, k_pspec, v_pspec, segment_ids_axes, None),
        out_specs=q_pspec,
        **_SHARD_MAP_CHECK_KWARGS,
    )
    def wrap(q_bhsd, k_bhsd, v_bhsd, seg_ids, kernel):
        return jax.vmap(kernel, in_axes=(0, 0, 0, segment_batch_axis))(q_bhsd, k_bhsd, v_bhsd, seg_ids)

    out = wrap(q_, k_, v_, segment_ids, splash_kernel)
    return jnp.transpose(out, (0, 2, 1, 3)).astype(v.dtype)


def attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    implementation: GrugAttentionImplementation | None = None,
) -> Float[Array, "B Q Hq D"]:
    if implementation == "reference":
        return reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    if implementation == "gpu_xla":
        return gpu_xla_attention(q, k, v, mask)
    if implementation == "gpu_cudnn":
        return gpu_cudnn_attention(q, k, v, mask)
    if implementation == "gpu_te":
        return gpu_te_attention(q, k, v, mask)
    if implementation == "gpu_fa4_cute":
        return gpu_fa4_cute_attention(q, k, v, mask)
    if implementation == "tpu_splash":
        if isinstance(mask, jax.Array):
            raise NotImplementedError("Dense masks are not supported for splash attention.")
        return _tpu_splash_attention(q, k, v, mask)

    if jax.default_backend() == "gpu" and isinstance(mask, AttentionMask) and mask.segment_ids is not None:
        return gpu_te_attention(q, k, v, mask)

    if jax.default_backend() == "tpu":
        if isinstance(mask, jax.Array):
            return reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
        return _tpu_splash_attention(q, k, v, mask)
    return reference_attention(q, k, v, mask, logits_dtype=jnp.float32)


__all__ = [
    "AttentionMask",
    "DEFAULT_MAX_PACKED_SEGMENTS",
    "GrugAttentionImplementation",
    "RotaryConfig",
    "align_kv_heads",
    "apply_rotary_embedding",
    "attention",
    "gpu_cudnn_attention",
    "gpu_fa4_cute_attention",
    "gpu_te_attention",
    "gpu_xla_attention",
    "reference_attention",
]
