# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Router top-k for Grug MoE, with a Pallas-Triton kernel for AMD (ROCm) GPUs.

On ROCm, XLA lowers ``jax.lax.top_k`` over a ``[tokens, experts]`` router matrix to rocprim radix sorts: about
28 launches and 0.57 ms per call for top-5 of 256 at 32,768 tokens per GPU. The kernel here reads each row once
and takes ``k`` rounds of max and first-index-of-max in registers, which matches ``jax.lax.top_k`` exactly: the
same indices in the same order, with ties going to the lower index. Values are gathered from the input outside the
kernel, so gradients are those of ``jax.lax.top_k``. TPU and other GPUs keep ``jax.lax.top_k``.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int

from levanter.utils.jax_utils import is_rocm_backend

_MAX_BLOCK_ROWS = 64
_NUM_WARPS = 4
_INT32_MIN = -(2**31)


def _total_order_key(x: jax.Array) -> jax.Array:
    """Map float32 bits to int32 keys whose signed order is IEEE total order (-NaN < -inf < -0 < +0 < inf < NaN)."""
    bits = jax.lax.bitcast_convert_type(x, jnp.int32)
    return bits ^ ((bits >> 31) & jnp.int32(0x7FFFFFFF))


def _top_k_indices_kernel(x_ref, idx_ref, *, k: int):
    key = _total_order_key(x_ref[...])
    rows, cols = key.shape
    col = jax.lax.broadcasted_iota(jnp.int32, (rows, cols), 1)
    out_col = jax.lax.broadcasted_iota(jnp.int32, idx_ref.shape, 1)
    available = col >= 0
    out = jnp.zeros(idx_ref.shape, jnp.int32)
    for j in range(k):
        best = jnp.max(jnp.where(available, key, _INT32_MIN), axis=1)
        is_best = available & (key == best[:, None])
        idx = jnp.min(jnp.where(is_best, col, cols), axis=1)
        out = jnp.where(out_col == j, idx[:, None], out)
        available = available & (col != idx[:, None])
    idx_ref[...] = out


def _block_rows(tokens: int) -> int:
    return math.gcd(tokens, _MAX_BLOCK_ROWS)


def triton_top_k_indices(x: Float[Array, "T E"], k: int, *, interpret: bool = False) -> Int[Array, "T k"]:
    """Indices of ``jax.lax.top_k(x, k)`` for a local float32 ``[T, E]`` array, via one Pallas-Triton pass."""
    tokens, experts = x.shape
    if x.dtype != jnp.float32:
        raise ValueError(f"triton_top_k_indices needs float32 input, got {x.dtype}")
    if experts & (experts - 1):
        raise ValueError(f"triton_top_k_indices needs a power-of-two expert count, got {experts}")
    if not 0 < k <= experts:
        raise ValueError(f"k={k} must be in [1, {experts}]")
    block_rows = _block_rows(tokens)
    k_padded = 1 << (k - 1).bit_length()
    indices = pl.pallas_call(
        functools.partial(_top_k_indices_kernel, k=k),
        grid=(tokens // block_rows,),
        in_specs=[pl.BlockSpec((block_rows, experts), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, k_padded), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((tokens, k_padded), jnp.int32),
        compiler_params=plgpu.CompilerParams(num_warps=_NUM_WARPS, num_stages=1),
        cost_estimate=pl.CostEstimate(
            flops=4 * k * tokens * experts,
            transcendentals=0,
            bytes_accessed=tokens * experts * 4 + tokens * k_padded * 4,
        ),
        interpret=interpret,
        name="grug_routing_top_k",
    )(jax.lax.stop_gradient(x))
    return indices[:, :k]


def routing_top_k(
    x: Float[Array, "T E"],
    k: int,
    *,
    mesh: jax.sharding.AbstractMesh,
    batch_axes: tuple[str, ...],
) -> tuple[Float[Array, "T k"], Int[Array, "T k"]]:
    """``jax.lax.top_k(x, k)`` over the expert axis of a token-sharded router matrix.

    On ROCm this runs the Pallas-Triton kernel per token shard; elsewhere it is ``jax.lax.top_k``.
    """
    if not is_rocm_backend():
        return jax.lax.top_k(x, k)
    spec = P(batch_axes, None)

    @functools.partial(shard_map, mesh=mesh, in_specs=spec, out_specs=(spec, spec), check_vma=False)
    def _local(x_local):
        indices = triton_top_k_indices(x_local, k)
        return jnp.take_along_axis(x_local, indices, axis=-1), indices

    return _local(reshard(x, spec))
