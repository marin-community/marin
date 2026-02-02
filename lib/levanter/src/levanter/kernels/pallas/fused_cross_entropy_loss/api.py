# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence
from typing import Literal, Optional, TypeAlias, cast
import warnings

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .config import BlockSizes
from .tuned_block_sizes import infer_block_sizes
from .reference import linear_softmax_cross_entropy_loss_reference
from .xla import linear_softmax_cross_entropy_loss_xla


Implementation: TypeAlias = Literal["pallas_tpu", "xla", "reference"]
Reduction: TypeAlias = Literal["sum", "mean"] | None


ArrayImpl = Callable[..., tuple[jax.Array, jax.Array]]


IMPLEMENTATIONS: dict[str, ArrayImpl] = {
    "reference": linear_softmax_cross_entropy_loss_reference,
    "xla": linear_softmax_cross_entropy_loss_xla,
}
_DEFAULT_IMPLEMENTATION: tuple[Implementation, ...] = ("xla",)

try:
    from .pallas_tpu import PallasUnsupportedError, linear_softmax_cross_entropy_loss_pallas

    IMPLEMENTATIONS["pallas_tpu"] = linear_softmax_cross_entropy_loss_pallas
    _DEFAULT_IMPLEMENTATION = ("pallas_tpu",) + _DEFAULT_IMPLEMENTATION
except ImportError:
    PallasUnsupportedError = NotImplementedError  # type: ignore[assignment]


def _validate_inputs(x: jax.Array, labels: jax.Array, w: jax.Array) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [B, H], got shape {x.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"labels must be rank-1 [B], got shape {labels.shape}.")
    if w.ndim != 2:
        raise ValueError(f"w must be rank-2 [H, V], got shape {w.shape}.")
    if x.shape[0] != labels.shape[0]:
        raise ValueError(f"Batch mismatch: x has B={x.shape[0]}, labels has B={labels.shape[0]}.")
    if x.shape[1] != w.shape[0]:
        raise ValueError(f"Hidden mismatch: x has H={x.shape[1]}, w has H={w.shape[0]}.")
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError(f"labels must be integer dtype, got {labels.dtype}.")


def _resolve_block_sizes(
    block_size: Optional[int],
    block_sizes: Optional[BlockSizes],
    *,
    x: jax.Array,
    w: jax.Array,
    dtype: Optional[jnp.dtype],
) -> BlockSizes:
    if block_sizes is None:
        if block_size is None:
            return infer_block_sizes(x.shape[0], x.shape[1], w.shape[1], dtype=dtype)
        return BlockSizes(v_block_size=block_size)
    if block_size is not None and block_size != block_sizes.v_block_size:
        raise ValueError(
            "block_size and block_sizes.v_block_size disagree: "
            f"block_size={block_size}, block_sizes.v_block_size={block_sizes.v_block_size}."
        )
    return block_sizes


def _apply_reduction(loss: jax.Array, reduction: Reduction, weight: Optional[jax.Array]) -> jax.Array:
    if weight is not None:
        weight = weight.astype(loss.dtype)
        loss = loss * weight

    if reduction is None:
        return loss
    if reduction == "sum":
        return jnp.sum(loss)
    if reduction == "mean":
        if weight is None:
            return jnp.mean(loss)
        denom = jnp.sum(weight)
        return jnp.where(denom != 0, jnp.sum(loss) / denom, jnp.zeros_like(denom))
    raise ValueError(f"Unsupported reduction: {reduction}")


def fused_cross_entropy_loss_and_logsumexp_penalty(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    reduction: Reduction = "mean",
    weight: Optional[Float[Array, "B"]] = None,
    logsumexp_weight: Optional[float] = 0.0,
    block_size: Optional[int] = None,
    block_sizes: Optional[BlockSizes] = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: Implementation | Sequence[Implementation | ArrayImpl] | None = None,
) -> jax.Array:
    """Fused cross-entropy + logsumexp penalty on raw arrays.

    Args:
        x: [B, H] input activations.
        labels: [B] integer labels.
        w: [H, V] projection weights.
        reduction: "sum", "mean", or None to return per-example loss.
        weight: Optional per-example weights/mask, broadcastable to [B].
        logsumexp_weight: Weight for the logsumexp (z-loss) penalty.
        block_size: Optional convenience for setting block_sizes.v_block_size.
        block_sizes: Block size configuration for the kernel.
        dtype: Optional dtype for logits/softmax computations.
        logit_soft_cap: Optional tanh soft cap for logits.
        precision: Optional matmul precision override for XLA/reference paths.
        implementation: Backend selector or override implementation list.

    Returns:
        Reduced loss (scalar) or per-example loss [B] if reduction is None.
    """
    _validate_inputs(x, labels, w)
    explicit_block_sizes = block_size is not None or block_sizes is not None
    resolved_block_sizes = (
        _resolve_block_sizes(block_size, block_sizes, x=x, w=w, dtype=dtype) if explicit_block_sizes else None
    )

    if implementation is None:
        impls: Sequence[Implementation | ArrayImpl] = _DEFAULT_IMPLEMENTATION
        explicit = False
    elif isinstance(implementation, Sequence) and not isinstance(implementation, (str, bytes)):
        impls = cast(Sequence[Implementation | ArrayImpl], implementation)
        explicit = len(impls) == 1
    else:
        impls = (cast(Implementation, implementation),)
        explicit = True

    errors: list[Exception] = []
    for impl in impls:
        if explicit_block_sizes:
            block_sizes_for_impl = resolved_block_sizes
        elif impl in ("xla", "reference"):
            block_sizes_for_impl = None
        else:
            block_sizes_for_impl = infer_block_sizes(x.shape[0], x.shape[1], w.shape[1], dtype=dtype)
        if callable(impl):
            try:
                loss, lse = impl(
                    x,
                    labels,
                    w,
                    block_sizes=block_sizes_for_impl,
                    dtype=dtype,
                    logit_soft_cap=logit_soft_cap,
                    precision=precision,
                )
            except PallasUnsupportedError as e:
                if explicit:
                    raise
                warnings.warn(
                    f"Pallas fused cross-entropy unavailable, falling back to XLA: {e}",
                    RuntimeWarning,
                )
                errors.append(e)
                continue
            except NotImplementedError as e:
                if explicit:
                    raise
                warnings.warn(
                    f"Pallas fused cross-entropy unavailable, falling back to XLA: {e}",
                    RuntimeWarning,
                )
                errors.append(e)
                continue
        else:
            fn = IMPLEMENTATIONS.get(impl)
            if fn is None:
                raise ValueError(f"Unsupported implementation: {impl}")
            try:
                loss, lse = fn(
                    x,
                    labels,
                    w,
                    block_sizes=block_sizes_for_impl,
                    dtype=dtype,
                    logit_soft_cap=logit_soft_cap,
                    precision=precision,
                )
            except PallasUnsupportedError as e:
                if explicit:
                    raise
                warnings.warn(
                    f"Pallas fused cross-entropy unavailable, falling back to XLA: {e}",
                    RuntimeWarning,
                )
                errors.append(e)
                continue
            except NotImplementedError as e:
                if explicit:
                    raise
                warnings.warn(
                    f"Pallas fused cross-entropy unavailable, falling back to XLA: {e}",
                    RuntimeWarning,
                )
                errors.append(e)
                continue

        if logsumexp_weight is not None and logsumexp_weight != 0.0:
            loss = loss + logsumexp_weight * (lse**2)
        return _apply_reduction(loss, reduction, weight)

    raise ExceptionGroup("all implementations failed", errors)


__all__ = [
    "BlockSizes",
    "Implementation",
    "IMPLEMENTATIONS",
    "Reduction",
    "fused_cross_entropy_loss_and_logsumexp_penalty",
]
