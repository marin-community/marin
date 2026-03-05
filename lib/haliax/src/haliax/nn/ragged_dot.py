# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox import gmm

from ..partitioning import ResourceAxis

Implementation: TypeAlias = Literal["auto", "megablox", "xla"]
_AUTO_FALLBACK_EXCEPTIONS = (NotImplementedError, RuntimeError)
_HAS_WARNED_AUTO_FALLBACK = False


def _ragged_dot_megablox_impl(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    tile_size = (512, 1024, 1024)  # (m, k, n)
    m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[2]
    return gmm(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=lhs.dtype,
        tiling=(min(m, tile_size[0]), min(k, tile_size[1]), min(n, tile_size[2])),
        interpret=jax.default_backend() == "cpu",
    )


def _ragged_dot_xla_impl(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    return jax.lax.ragged_dot_general(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
            dot_dimension_numbers=(((1,), (1,)), ((), ())),
            lhs_ragged_dimensions=(0,),
            rhs_group_dimensions=(0,),
        ),
    )


def _preferred_implementations(implementation: Implementation) -> tuple[Implementation, ...]:
    if implementation != "auto":
        return (implementation,)

    if jax.default_backend() == "tpu":
        return ("megablox", "xla")

    return ("xla",)


def _run_impl(name: Implementation, lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    if name == "megablox":
        return _ragged_dot_megablox_impl(lhs, rhs, group_sizes)
    if name == "xla":
        return _ragged_dot_xla_impl(lhs, rhs, group_sizes)
    raise ValueError(f"Unknown ragged_dot implementation: {name}")


def ragged_dot(
    lhs_: jax.Array,
    rhs_: jax.Array,
    group_sizes_: jax.Array,
    ar: bool = False,
    implementation: Implementation = "auto",
) -> jax.Array:
    """Grouped matrix multiply with backend-dispatched ragged dot implementations.

    Args:
        lhs_: [tokens, in] input matrix.
        rhs_: [experts, in, out] expert weights.
        group_sizes_: [experts] number of tokens per expert.
        ar: Whether to perform an all-reduce over the model axis on the output.
        implementation: Backend selection policy. `"auto"` uses XLA on CPU/GPU and
            Megablox on TPU with XLA fallback.

    Returns:
        A [tokens, out] array.
    """
    hs_shape = lhs_.shape
    if hs_shape[0] % 512:
        pad_length = 512 - hs_shape[0] % 512
        lhs_ = jax.lax.pad(lhs_, jnp.zeros((), dtype=lhs_.dtype), [(0, pad_length, 0), (0, 0, 0)])

    out = None

    for impl in _preferred_implementations(implementation):
        try:
            out = _run_impl(impl, lhs_, rhs_, group_sizes_)
            break
        except _AUTO_FALLBACK_EXCEPTIONS as exc:
            if implementation == "auto" and impl == "megablox":
                global _HAS_WARNED_AUTO_FALLBACK
                if not _HAS_WARNED_AUTO_FALLBACK:
                    warnings.warn(
                        f"ragged_dot auto fallback: megablox failed ({type(exc).__name__}), trying XLA.",
                        RuntimeWarning,
                    )
                    _HAS_WARNED_AUTO_FALLBACK = True
                continue
            raise

    if out is None:
        raise RuntimeError("No ragged_dot implementation was selected")

    if ar:
        out = jax.lax.psum(out, ResourceAxis.MODEL)

    if hs_shape[0] % 512:
        out = out[: hs_shape[0]]

    return out


__all__ = ["Implementation", "ragged_dot"]
