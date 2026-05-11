# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Tokamax-flex-style adapter pieces for Grug attention masks."""

import math
from collections.abc import Callable
from typing import Literal

import jax
from jaxtyping import Array, Bool, Float

from levanter.grug.attention import AttentionMask


FlexMaskMod = Callable[[tuple[int, ...]], Bool[jax.Array, "..."]]
FlexImplementation = Literal["xla", "pallas_triton"]


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


def gpu_flex_pallas_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | None,
) -> Float[Array, "B Q Hq D"]:
    """Run Grug attention through Tokamax's experimental Pallas-Triton flex backend."""
    if jax.default_backend() != "gpu":
        raise NotImplementedError("gpu_flex_pallas requires the JAX GPU backend.")
    try:
        return tokamax_flex_attention(q, k, v, mask, implementation="pallas_triton")
    except ImportError as exc:
        raise NotImplementedError(
            "gpu_flex_pallas requires Tokamax flex Pallas-Triton; install the marin-levanter kernels extra."
        ) from exc
    except AttributeError as exc:
        raise NotImplementedError(
            "gpu_flex_pallas could not load Tokamax PallasTritonFlexAttention from the installed Tokamax package."
        ) from exc
