# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Tokamax-flex-style adapter pieces for Grug attention masks."""

import math
from collections.abc import Callable
from typing import Literal

import jax
import numpy as np
from jaxtyping import Array, Bool, Float

from levanter.grug.attention import AttentionMask, reference_attention

FlexMaskMod = Callable[[tuple[int, ...]], Bool[jax.Array, "..."]]
FlexImplementation = Literal["xla", "pallas_triton"]
FlashImplementation = Literal["xla", "triton", "mosaic"]


def _is_empty_attention_mask(mask: AttentionMask) -> bool:
    return not mask.is_causal and mask.sliding_window is None and mask.segment_ids is None


def _combine_mask(
    current: Bool[jax.Array, "..."] | None,
    constraint: Bool[jax.Array, "..."],
) -> Bool[jax.Array, "..."]:
    if current is None:
        return constraint
    return jax.numpy.logical_and(current, constraint)


def grug_attention_mask_for_scores(mask: AttentionMask, scores_shape: tuple[int, ...]) -> Bool[jax.Array, "..."]:
    """Return an index-derived Grug mask broadcastable to Tokamax scores `[*B, H, Q, K]`."""
    if len(scores_shape) < 3:
        raise ValueError(f"scores_shape must include head, q, and k dimensions, got {scores_shape}")

    q_len = scores_shape[-2]
    k_len = scores_shape[-1]
    q_idx = jax.numpy.arange(q_len, dtype=jax.numpy.int32)[None, None, :, None]
    k_idx = jax.numpy.arange(k_len, dtype=jax.numpy.int32)[None, None, None, :]
    allowed = None

    if mask.is_causal:
        allowed = _combine_mask(allowed, k_idx <= q_idx)

    if mask.sliding_window is not None:
        if mask.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")
        allowed = _combine_mask(allowed, k_idx >= q_idx - (mask.sliding_window - 1))

    if mask.segment_ids is not None:
        q_seg, k_seg = mask.segment_ids
        if q_seg.ndim != k_seg.ndim:
            raise ValueError(f"segment_ids ndim mismatch: q={q_seg.ndim}, k={k_seg.ndim}")
        if q_seg.ndim == 1:
            if q_seg.shape[0] != q_len or k_seg.shape[0] != k_len:
                raise ValueError(
                    "1D segment_ids must match attention sequence lengths: "
                    f"got q={q_seg.shape}, k={k_seg.shape}, expected ({q_len},), ({k_len},)"
                )
            segment_allowed = q_seg[None, None, :, None] == k_seg[None, None, None, :]
        elif q_seg.ndim == 2:
            batch_shape = scores_shape[:-3]
            if len(batch_shape) != 1:
                raise ValueError(f"2D segment_ids require one batch axis in scores_shape, got {scores_shape}")
            if q_seg.shape[0] != k_seg.shape[0]:
                raise ValueError(f"segment_ids batch mismatch: q={q_seg.shape[0]}, k={k_seg.shape[0]}")
            if q_seg.shape[0] not in (1, batch_shape[0]):
                raise ValueError(f"segment_ids batch dim must be 1 or {batch_shape[0]}, got {q_seg.shape[0]}")
            if q_seg.shape[1] != q_len or k_seg.shape[1] != k_len:
                raise ValueError(
                    "2D segment_ids must match attention sequence lengths: "
                    f"got q={q_seg.shape}, k={k_seg.shape}, expected (*,{q_len}), (*,{k_len})"
                )
            segment_allowed = q_seg[:, None, :, None] == k_seg[:, None, None, :]
        else:
            raise ValueError(f"segment_ids must be 1D or 2D, got ndim={q_seg.ndim}")
        allowed = _combine_mask(allowed, segment_allowed)

    if allowed is None:
        raise ValueError("Do not build a flex mask_mod for an empty AttentionMask")
    return allowed


def grug_flex_mask_mod(mask: AttentionMask | None) -> FlexMaskMod | None:
    """Build a Tokamax flex `mask_mod` closure for Grug structured masks."""
    if mask is None or _is_empty_attention_mask(mask):
        return None

    def mask_mod(scores_shape: tuple[int, ...]) -> Bool[jax.Array, "..."]:
        return grug_attention_mask_for_scores(mask, scores_shape)

    return mask_mod


def _packed_segment_starts(segment_ids: jax.Array) -> jax.Array:
    positions = jax.numpy.arange(segment_ids.shape[-1], dtype=jax.numpy.int32)
    previous = jax.numpy.roll(segment_ids, 1, axis=-1)
    is_start = positions == 0
    is_start = jax.numpy.logical_or(is_start, segment_ids != previous)
    starts = jax.numpy.where(is_start, positions, 0)
    return jax.lax.cummax(starts, axis=segment_ids.ndim - 1)


def _packed_segment_ends(segment_ids: jax.Array) -> jax.Array:
    seq_len = segment_ids.shape[-1]
    positions = jax.numpy.arange(seq_len, dtype=jax.numpy.int32)
    next_segment = jax.numpy.roll(segment_ids, -1, axis=-1)
    is_end = positions == seq_len - 1
    is_end = jax.numpy.logical_or(is_end, segment_ids != next_segment)
    end_markers = jax.numpy.where(is_end, positions + 1, seq_len)
    reversed_ends = jax.numpy.flip(end_markers, axis=-1)
    return jax.numpy.flip(jax.lax.cummin(reversed_ends, axis=segment_ids.ndim - 1), axis=-1)


def _tokamax_range_axis(x: jax.Array | None) -> jax.Array | None:
    if x is None:
        return None
    if x.ndim == 1:
        return x[None, :]
    if x.ndim == 2:
        return x[:, None, :]
    raise ValueError(f"Tokamax range masks require 1D or 2D arrays, got shape={x.shape}")


def _static_array_value(x: jax.Array) -> np.ndarray:
    try:
        return np.asarray(jax.device_get(x))
    except Exception as exc:
        raise NotImplementedError("Tokamax flash segment fast path requires static segment_ids.") from exc


def _validate_same_static_segment_ids(q_seg: jax.Array, k_seg: jax.Array, *, assume_packed_segment_ids: bool) -> None:
    if q_seg is k_seg:
        return
    try:
        q_value = _static_array_value(q_seg)
        k_value = _static_array_value(k_seg)
    except NotImplementedError:
        if assume_packed_segment_ids:
            raise NotImplementedError(
                "Tokamax flash segment fast path requires q/kv segment_ids to be the same object when dynamic."
            ) from None
        raise
    if not np.array_equal(q_value, k_value):
        raise NotImplementedError(
            "Tokamax flash segment fast path requires identical q/kv segment_ids; use flex attention instead."
        )


def _validate_contiguous_segment_runs(segment_ids: jax.Array, *, assume_packed_segment_ids: bool) -> None:
    try:
        values = _static_array_value(segment_ids)
    except NotImplementedError:
        if assume_packed_segment_ids:
            return
        raise

    rows = values.reshape(-1, values.shape[-1])
    for row in rows:
        seen = set()
        previous = None
        for raw_value in row:
            value = int(raw_value)
            if value == previous:
                continue
            if value in seen:
                raise NotImplementedError(
                    "Tokamax flash segment fast path requires contiguous segment-id runs; use flex attention instead."
                )
            seen.add(value)
            previous = value


def grug_tokamax_attention_mask(
    mask: AttentionMask | None,
    *,
    q_len: int,
    k_len: int,
    batch_size: int,
    assume_packed_segment_ids: bool = False,
):
    """Build a Tokamax range mask for Grug masks representable without a dense score mask.

    Segment IDs are supported for packed self-attention, where query and key
    tokens are identical and segment runs are contiguous. Static segment IDs are
    validated. Dynamic packed training batches can opt into the range fast path
    with `assume_packed_segment_ids=True`.
    """
    from tokamax._src.ops.attention import base as tokamax_attention_base

    if mask is None or _is_empty_attention_mask(mask):
        return tokamax_attention_base.Mask()

    if mask.sliding_window is not None and mask.sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")

    k_start = None
    k_end = None
    if mask.segment_ids is not None:
        q_seg, k_seg = mask.segment_ids
        if q_len != k_len:
            raise NotImplementedError("Tokamax flash segment fast path requires self-attention q_len == k_len.")
        if q_seg.shape != k_seg.shape:
            raise NotImplementedError("Tokamax flash segment fast path requires matching q/kv segment_id shapes.")
        if q_seg.ndim == 1:
            if q_seg.shape[0] != q_len:
                raise ValueError(f"1D segment_ids must match sequence length {q_len}, got {q_seg.shape}")
        elif q_seg.ndim == 2:
            if q_seg.shape[0] not in (1, batch_size) or q_seg.shape[1] != q_len:
                raise ValueError(f"2D segment_ids must have shape [1|{batch_size}, {q_len}], got {q_seg.shape}")
        else:
            raise ValueError(f"segment_ids must be 1D or 2D, got ndim={q_seg.ndim}")
        _validate_same_static_segment_ids(q_seg, k_seg, assume_packed_segment_ids=assume_packed_segment_ids)
        _validate_contiguous_segment_runs(q_seg, assume_packed_segment_ids=assume_packed_segment_ids)

        k_start = _packed_segment_starts(q_seg)
        if not mask.is_causal:
            k_end = _packed_segment_ends(q_seg)

    if mask.sliding_window is not None:
        window_start = jax.numpy.arange(q_len, dtype=jax.numpy.int32) - (mask.sliding_window - 1)
        window_start = jax.numpy.maximum(window_start, 0)
        k_start = window_start if k_start is None else jax.numpy.maximum(k_start, window_start)

    return tokamax_attention_base.Mask(
        k_start=_tokamax_range_axis(k_start),
        k_end=_tokamax_range_axis(k_end),
        is_causal=mask.is_causal,
    )


def _tokamax_flex_op(implementation: FlexImplementation):
    _ensure_tokamax_flags_parsed()
    if implementation == "xla":
        from tokamax._src.ops.flex_attention.base import FlexAttention

        return FlexAttention()
    if implementation == "pallas_triton":
        from tokamax._src.ops.flex_attention.pallas_triton import PallasTritonFlexAttention

        return PallasTritonFlexAttention()
    raise ValueError(f"Unsupported flex implementation: {implementation}")


def _ensure_tokamax_flags_parsed() -> None:
    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["grug_flex_attention"], known_only=True)


def tokamax_flex_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | None,
    *,
    implementation: FlexImplementation,
) -> Float[Array, "B Q Hq D"]:
    """Run Grug attention through Tokamax FlexAttention."""
    if not isinstance(mask, AttentionMask | None):
        raise NotImplementedError(
            "Tokamax flex attention currently supports only structured Grug AttentionMask masks."
        )

    op = _tokamax_flex_op(implementation)
    q_scaled = q * (1.0 / math.sqrt(q.shape[-1]))
    score_mod = (lambda scores: scores) if implementation == "xla" else None
    out = op(
        q_scaled,
        k,
        v,
        score_mod=score_mod,
        mask_mod=grug_flex_mask_mod(mask),
        normalize_output=True,
    )
    return out.astype(v.dtype)


def tokamax_flash_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | None,
    *,
    implementation: FlashImplementation,
    assume_packed_segment_ids: bool = False,
) -> Float[Array, "B Q Hq D"]:
    """Run Grug attention through Tokamax FlashAttention when the mask is range-representable."""
    if not isinstance(mask, AttentionMask | None):
        raise NotImplementedError(
            "Tokamax flash attention currently supports only structured Grug AttentionMask masks."
        )

    _ensure_tokamax_flags_parsed()
    from jax.extend import backend
    from tokamax._src.ops.attention.api import IMPLEMENTATIONS

    tokamax_mask = grug_tokamax_attention_mask(
        mask,
        q_len=q.shape[1],
        k_len=k.shape[1],
        batch_size=q.shape[0],
        assume_packed_segment_ids=assume_packed_segment_ids,
    )
    implementation_key = implementation
    if implementation_key == "mosaic":
        implementation_key = "mosaic_gpu" if "NVIDIA" in backend.get_default_device().device_kind else "mosaic_tpu"
    op = IMPLEMENTATIONS[implementation_key]
    out = op(q, k, v, mask=tokamax_mask)
    return out.astype(v.dtype)


def tokamax_flex_attention_with_reference_vjp(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | None,
    *,
    implementation: FlexImplementation,
) -> Float[Array, "B Q Hq D"]:
    """Run Tokamax flex attention with a reference-derived VJP for q/k/v."""

    @jax.custom_vjp
    def _attention(q_arg, k_arg, v_arg):
        return tokamax_flex_attention(q_arg, k_arg, v_arg, mask, implementation=implementation)

    def _attention_fwd(q_arg, k_arg, v_arg):
        out = tokamax_flex_attention(q_arg, k_arg, v_arg, mask, implementation=implementation)
        return out, (q_arg, k_arg, v_arg)

    def _attention_bwd(residuals, cotangent):
        q_arg, k_arg, v_arg = residuals

        def _reference(q_ref, k_ref, v_ref):
            return reference_attention(q_ref, k_ref, v_ref, mask, logits_dtype=jax.numpy.float32)

        _, pullback = jax.vjp(_reference, q_arg, k_arg, v_arg)
        return pullback(cotangent)

    _attention.defvjp(_attention_fwd, _attention_bwd)
    return _attention(q, k, v)


def gpu_flex_pallas_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | None,
    *,
    use_reference_vjp: bool = True,
    prefer_flash: bool = True,
    assume_packed_segment_ids: bool = True,
) -> Float[Array, "B Q Hq D"]:
    """Run Grug attention through Tokamax's experimental Pallas-Triton flex backend."""
    if jax.default_backend() != "gpu":
        raise NotImplementedError("gpu_flex_pallas requires the JAX GPU backend.")
    try:
        if prefer_flash:
            try:
                return tokamax_flash_attention(
                    q,
                    k,
                    v,
                    mask,
                    implementation="triton",
                    assume_packed_segment_ids=assume_packed_segment_ids,
                )
            except NotImplementedError:
                pass
        if use_reference_vjp:
            return tokamax_flex_attention_with_reference_vjp(q, k, v, mask, implementation="pallas_triton")
        return tokamax_flex_attention(q, k, v, mask, implementation="pallas_triton")
    except ImportError as exc:
        raise NotImplementedError(
            "gpu_flex_pallas requires Tokamax flex Pallas-Triton; install the marin-levanter kernels extra."
        ) from exc
    except AttributeError as exc:
        raise NotImplementedError(
            "gpu_flex_pallas could not load Tokamax PallasTritonFlexAttention from the installed Tokamax package."
        ) from exc
