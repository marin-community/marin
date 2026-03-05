# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import lru_cache
from typing import Literal, TypeAlias, cast

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox import gmm

from ..partitioning import ResourceAxis

logger = logging.getLogger(__name__)

Implementation: TypeAlias = Literal["auto", "megablox", "ragged_dot_general"]


@lru_cache(maxsize=1)
def _warn_gpu_fallback_once() -> None:
    logger.warning(
        "haliax.nn.gmm_sharded: falling back from megablox to ragged_dot_general on GPU "
        "after Triton dynamic-grid lowering failure"
    )


def _is_triton_dynamic_grid_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "dynamic grid bounds not supported" in msg and "triton" in msg


def _gmm_megablox_impl(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
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


def _gmm_ragged_impl(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
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

    backend = jax.default_backend()
    if backend == "gpu":
        return ("megablox", "ragged_dot_general")

    return ("megablox",)


def _run_gmm_impl(name: Implementation, lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    if name == "megablox":
        return _gmm_megablox_impl(lhs, rhs, group_sizes)
    if name == "ragged_dot_general":
        return _gmm_ragged_impl(lhs, rhs, group_sizes)
    raise ValueError(f"Unknown GMM implementation: {name}")


def gmm_sharded(
    lhs_: jnp.ndarray,
    rhs_: jnp.ndarray,
    group_sizes_: jnp.ndarray,
    ar: bool = False,
    implementation: Implementation = "auto",
) -> jnp.ndarray:
    """Grouped matrix multiply with explicit backend-dispatch behavior.

    Args:
        lhs_: [tokens, in] input matrix.
        rhs_: [experts, in, out] expert weights.
        group_sizes_: [experts] number of tokens per expert.
        ar: Whether to perform an all-reduce over the model axis on the output.
        implementation: Backend selection policy. `"auto"` tries preferred kernels,
            using structured fallback for known unsupported GPU Triton lowerings.

    Returns:
        A [tokens, out] array.
    """
    hs_shape = lhs_.shape
    if hs_shape[0] % 512:
        pad_length = 512 - hs_shape[0] % 512
        lhs_ = jax.lax.pad(lhs_, 0.0, [(0, pad_length, 0), (0, 0, 0)])

    implementations = _preferred_implementations(implementation)

    out = None
    last_exc: Exception | None = None

    for impl in implementations:
        try:
            out = _run_gmm_impl(impl, lhs_, rhs_, group_sizes_)
            break
        except Exception as exc:
            last_exc = exc
            if impl == "megablox" and implementation == "auto" and _is_triton_dynamic_grid_error(exc):
                _warn_gpu_fallback_once()
                continue
            raise

    if out is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No GMM implementation was selected")

    if ar:
        out = jax.lax.psum(out, ResourceAxis.MODEL)

    if hs_shape[0] % 512:
        out = out[: hs_shape[0]]

    return cast(jnp.ndarray, out)


__all__ = ["Implementation", "gmm_sharded"]
