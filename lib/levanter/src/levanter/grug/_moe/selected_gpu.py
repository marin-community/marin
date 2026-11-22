# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Matrix-vector products for sparse single-token expert decode on CUDA."""

from collections.abc import Callable
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

_INPUT_TILE = 256
_OUTPUT_TILE = 64


def supports_selected_matvec(hidden: int, intermediate: int) -> bool:
    """Return whether both expert projections fit the CUDA tile geometry."""
    return all(width % _INPUT_TILE == 0 and width % _OUTPUT_TILE == 0 for width in (hidden, intermediate))


@lru_cache(maxsize=None)
def _selected_matvec_kernel(routes: int, width: int, out_width: int, dtype):
    splits = width // _INPUT_TILE
    result = jax.ShapeDtypeStruct((splits, routes, out_width), jnp.float32)

    def tile_product(x, weight):
        return jnp.sum(x.astype(jnp.float32)[:, None] * weight.astype(jnp.float32), axis=0)

    body = pl.estimate_cost(
        tile_product,
        jax.ShapeDtypeStruct((_INPUT_TILE,), dtype),
        jax.ShapeDtypeStruct((_INPUT_TILE, _OUTPUT_TILE), dtype),
    )
    programs = routes * splits * (out_width // _OUTPUT_TILE)
    cost = with_io_bytes_accessed(
        pl.CostEstimate(body.flops * programs, body.transcendentals * programs, 0),
        kernel_inputs_specs=(
            jax.ShapeDtypeStruct((routes, width), dtype),
            jax.ShapeDtypeStruct((routes,), jnp.int32),
            # Each active route reads one matrix from the local expert bank.
            jax.ShapeDtypeStruct((routes, width, out_width), dtype),
        ),
        kernel_outputs_specs=(result,),
    )

    def matvec(x_ref, expert_ref, weight_ref, output_ref):
        route, split, block = (pl.program_id(axis) for axis in range(3))
        k = split * _INPUT_TILE + jnp.arange(_INPUT_TILE)
        n = block * _OUTPUT_TILE + jnp.arange(_OUTPUT_TILE)
        expert = expert_ref[route]

        def multiply():
            return tile_product(x_ref[route, k], weight_ref[expert, k[:, None], n[None, :]])

        partial = jax.lax.cond(
            (expert >= 0) & (expert < weight_ref.shape[0]), multiply, lambda: jnp.zeros((_OUTPUT_TILE,), jnp.float32)
        )
        output_ref[split, route, n] = partial

    return pl.pallas_call(
        matvec,
        out_shape=result,
        grid=(routes, splits, out_width // _OUTPUT_TILE),
        compiler_params=plgpu.CompilerParams(num_warps=4),
        cost_estimate=cost,
        name="decode_selected_expert_matvec",
    )


def _selected_matvec(x: jax.Array, experts: jax.Array, weight: jax.Array) -> jax.Array:
    partial = _selected_matvec_kernel(*x.shape, weight.shape[-1], x.dtype)(x, experts, weight)
    return jnp.sum(partial, axis=0).astype(x.dtype)


def selected_expert_mlp(
    x: jax.Array,
    experts: jax.Array,
    weights: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    *,
    activation: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Run local experts for one token inside the expert shard map, with -1 for absent routes."""
    rows = jnp.broadcast_to(x, (experts.size, x.size))
    gate = _selected_matvec(rows, experts, w_gate)
    up = _selected_matvec(rows, experts, w_up)
    down = _selected_matvec(activation(gate) * up, experts, w_down)
    weighted = jnp.where(experts[:, None] >= 0, down * weights[:, None], 0)
    return jax.lax.fori_loop(0, experts.size, lambda i, output: output + weighted[i], jnp.zeros_like(x), unroll=True)
