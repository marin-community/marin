# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import functools
import inspect
import math
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
from haliax.jax_utils import named_call
from haliax.partitioning import _get_mesh
from jax import numpy as jnp
from jax import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.sharding import NamedSharding
from jaxtyping import Array, Bool, Float, Int

from levanter.kernels.pallas.splash_attention import (
    DEFAULT_SPLASH_BLOCK_SIZE,
    lower_splash_attention_mask,
    lower_splash_segment_ids,
    splash_attention_mask_spec_from_fields,
    splash_attention_block_sizes,
    splash_partition_spec_shard_factor,
)
from levanter.layers.attention_mask import (
    segment_run_metadata_from_segment_id_array as _segment_run_metadata_from_segment_id_array,
)

_SHARD_MAP_CHECK_KWARG = "check_vma" if "check_vma" in inspect.signature(shard_map).parameters else "check_rep"
_SHARD_MAP_CHECK_KWARGS = {_SHARD_MAP_CHECK_KWARG: False}
GrugAttentionImplementation = Literal[
    "reference",
    "tpu_splash",
    "gpu_fa4_cute",
    "gpu_fa4_thd",
]


@dataclass(frozen=True)
class RotaryConfig:
    """Lightweight rotary embedding configuration."""

    theta: float = 10000.0
    scaling_factor: float | None = None


class ThdSegmentMetadata(eqx.Module):
    """Fixed-shape segment metadata for FA4 THD attention.

    `segment_lengths` stores the contiguous run lengths implied by packed
    `segment_ids`, padded to a fixed max segment count. The runs include the
    trailing padding run when segment id -1 is present so THD outputs still
    reshape back to dense BSHD.
    """

    segment_lengths: Int[Array, "... M"]
    num_segments: Int[Array, "..."]


def thd_segment_metadata_from_segment_ids(
    segment_ids: Int[Array, "... S"],
    *,
    max_segments: int,
) -> ThdSegmentMetadata:
    metadata = _segment_run_metadata_from_segment_id_array(segment_ids, max_segments=max_segments)
    return ThdSegmentMetadata(
        segment_lengths=metadata.segment_lengths,
        num_segments=metadata.num_segments,
    )


class AttentionMask(eqx.Module):
    """Grug attention mask spec.

    This is deliberately simpler than `levanter.layers.attention.AttentionMask`:
    - Stores raw JAX arrays (no NamedArray fields).
    - Supports causal masking, sliding windows, and segment IDs.
    """

    is_causal: bool = eqx.field(default=False, static=True)
    segment_ids: tuple[jax.Array, jax.Array] | None = None
    thd_segment_metadata: ThdSegmentMetadata | None = None
    sliding_window: int | None = eqx.field(default=None, static=True)

    @classmethod
    def causal(
        cls,
        *,
        sliding_window: int | None = None,
    ) -> "AttentionMask":
        return cls(
            is_causal=True,
            sliding_window=sliding_window,
        )

    def with_segment_ids(
        self,
        q_segment_ids: Int[Array, "..."],
        kv_segment_ids: Int[Array, "..."] | None = None,
        *,
        max_segments: int | None = None,
    ) -> "AttentionMask":
        kv_ids = q_segment_ids if kv_segment_ids is None else kv_segment_ids
        thd_segment_metadata = None
        if max_segments is not None:
            if kv_segment_ids is not None:
                q_segment_ids = eqx.error_if(
                    q_segment_ids,
                    jnp.any(q_segment_ids != kv_ids),
                    "THD segment metadata requires matching q/kv segment_ids.",
                )
            thd_segment_metadata = thd_segment_metadata_from_segment_ids(
                q_segment_ids,
                max_segments=max_segments,
            )
        return AttentionMask(
            is_causal=self.is_causal,
            segment_ids=(q_segment_ids, kv_ids),
            thd_segment_metadata=thd_segment_metadata,
            sliding_window=self.sliding_window,
        )

    def with_sliding_window(self, sliding_window: int | None) -> "AttentionMask":
        return AttentionMask(
            is_causal=self.is_causal,
            segment_ids=self.segment_ids,
            thd_segment_metadata=self.thd_segment_metadata,
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
        dtype = x.dtype
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(dtype)

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


def _tpu_splash_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | jax.Array | None,
) -> Float[Array, "B Q Hq D"]:
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

    head_shards = splash_partition_spec_shard_factor(q_pspec[1], mesh)
    q_seq_shards = splash_partition_spec_shard_factor(q_pspec[2], mesh)
    kv_seq_shards = splash_partition_spec_shard_factor(k_pspec[2], mesh)

    block_sizes = splash_attention_block_sizes(
        q_seq_len=Sq,
        kv_seq_len=Sk,
        q_seq_shards=q_seq_shards,
        kv_seq_shards=kv_seq_shards,
        max_block_size=DEFAULT_SPLASH_BLOCK_SIZE,
    )

    if mask is None:
        mask_spec = None
        segment_id_lowering = lower_splash_segment_ids()
    elif isinstance(mask, AttentionMask):
        if mask.sliding_window is not None:
            if mask.sliding_window <= 0:
                raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")

        mask_spec = splash_attention_mask_spec_from_fields(
            is_causal=mask.is_causal,
            sliding_window=mask.sliding_window,
        )

        if mask.segment_ids is not None:
            q_segment_ids, kv_segment_ids = mask.segment_ids
            q_seg_sharding = _named_sharding_of(q_segment_ids, label="segment_ids.q")
            kv_seg_sharding = _named_sharding_of(kv_segment_ids, label="segment_ids.kv")
            segment_id_lowering = lower_splash_segment_ids(
                q_segment_ids=q_segment_ids,
                kv_segment_ids=kv_segment_ids,
                q_segment_ids_axes=q_seg_sharding.spec,
                kv_segment_ids_axes=kv_seg_sharding.spec,
                q_segment_batch_axis=0 if q_segment_ids.ndim == 2 else None,
                kv_segment_batch_axis=0 if kv_segment_ids.ndim == 2 else None,
            )
        else:
            segment_id_lowering = lower_splash_segment_ids()
    else:
        raise NotImplementedError("Dense masks are not supported for splash attention.")

    mask_lowering = lower_splash_attention_mask(
        mask=mask_spec,
        q_seq_len=Sq,
        kv_seq_len=Sk,
        num_heads=Hq,
        q_seq_shards=q_seq_shards,
    )

    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=mask_lowering.kernel_mask,
        block_sizes=block_sizes,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
    )

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(q_pspec, k_pspec, v_pspec, segment_id_lowering.segment_ids_axes, None),
        out_specs=q_pspec,
        **_SHARD_MAP_CHECK_KWARGS,
    )
    def wrap(q_bhsd, k_bhsd, v_bhsd, seg_ids, kernel):
        return jax.vmap(kernel, in_axes=(0, 0, 0, segment_id_lowering.segment_batch_axis))(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            seg_ids,
        )

    # pyrefly: ignore[bad-specialization, bad-argument-count]  # jax.shard_map decorator erases wrap's real signature
    out = wrap(q_, k_, v_, segment_id_lowering.segment_ids, splash_kernel)
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
    if implementation == "gpu_fa4_cute":
        from levanter.grug.attention._fa4_cute import gpu_fa4_cute_attention  # noqa: PLC0415

        return gpu_fa4_cute_attention(q, k, v, mask)
    if implementation == "gpu_fa4_thd":
        from levanter.grug.attention._fa4_thd import gpu_fa4_thd_attention  # noqa: PLC0415

        return gpu_fa4_thd_attention(q, k, v, mask)
    if implementation == "tpu_splash":
        if isinstance(mask, jax.Array):
            raise NotImplementedError("Dense masks are not supported for splash attention.")
        return _tpu_splash_attention(q, k, v, mask)
    if implementation is not None:
        raise ValueError(f"Unknown Grug attention implementation: {implementation}")

    if jax.default_backend() == "tpu":
        if isinstance(mask, jax.Array):
            return reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
        return _tpu_splash_attention(q, k, v, mask)
    return reference_attention(q, k, v, mask, logits_dtype=jnp.float32)


__all__ = [
    "AttentionMask",
    "GrugAttentionImplementation",
    "RotaryConfig",
    "align_kv_heads",
    "apply_rotary_embedding",
    "attention",
    "reference_attention",
]
