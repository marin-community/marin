# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Router top-k for Grug MoE, with a Pallas-Triton kernel for AMD Instinct GPUs.

On ROCm, XLA lowers ``jax.lax.top_k`` over a ``[tokens, experts]`` router matrix to a chain of rocPRIM radix-sort
launches. The kernel here reads each row once and takes ``k`` rounds of max and first-index-of-max in registers, which
matches ``jax.lax.top_k`` exactly: the same indices in the same order, with ties going to the lower index. Values are
gathered from the input outside the kernel, so gradients are those of ``jax.lax.top_k``.

The kernel runs on AMD Instinct for float32 rows whose expert count is a power of two (a Pallas-Triton block is a
power of two wide). Other devices and shapes keep ``jax.lax.top_k``. On CPU the kernel runs in the Pallas
interpreter.
"""

import functools
import math

import jax
import jax.numpy as jnp
from haliax.jax_utils import is_amd_instinct
from jax import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int

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


def _kernel_supports(x: jax.Array) -> bool:
    experts = x.shape[-1]
    return x.dtype == jnp.float32 and experts & (experts - 1) == 0


def triton_top_k_indices(x: Float[Array, "T E"], k: int) -> Int[Array, "T k"]:
    """Indices of ``jax.lax.top_k(x, k)`` for a local float32 ``[T, E]`` array, E a power of two, in one Pallas pass."""
    tokens, experts = x.shape
    if not _kernel_supports(x):
        raise ValueError(
            f"triton_top_k_indices needs float32 input and a power-of-two expert count, got {x.dtype}[{x.shape}]"
        )
    if not 0 < k <= experts:
        raise ValueError(f"k={k} must be in [1, {experts}]")
    block_rows = math.gcd(tokens, _MAX_BLOCK_ROWS)
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
        interpret=jax.default_backend() == "cpu",
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

    On AMD Instinct, for float32 input with a power-of-two expert count, this runs the Pallas-Triton kernel on each
    token shard; otherwise it is ``jax.lax.top_k``.
    """
    if not (is_amd_instinct() and _kernel_supports(x)):
        return jax.lax.top_k(x, k)
    spec = P(batch_axes, None)

    @functools.partial(shard_map, mesh=mesh, in_specs=spec, out_specs=(spec, spec), check_vma=False)
    def _local(x_local):
        indices = triton_top_k_indices(x_local, k)
        return jnp.take_along_axis(x_local, indices, axis=-1), indices

    return _local(reshard(x, spec))
