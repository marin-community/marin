# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Schedule complete grouped-GEMM output row tiles across TPU cores."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.megablox.gmm import make_group_metadata


@partial(jax.jit, static_argnames=("partitions", "tile_rows", "transpose_rhs", "interpret"))
def row_partitioned_gmm(x, weights, groups, *, partitions=2, tile_rows=512, transpose_rhs=False, interpret=False):
    """Split row tiles between cores and preserve expert order within each tile."""
    m, k = x.shape
    n = weights.shape[1] if transpose_rhs else weights.shape[2]
    tm = min(tile_rows, m)
    row_tiles = m // tm
    if m % tm or row_tiles % partitions:
        raise ValueError("Expected complete row tiles evenly divided across partitions")
    metadata, active = make_group_metadata(
        group_sizes=groups,
        m=m,
        tm=tm,
        start_group=jnp.asarray(0, jnp.int32),
        num_nonzero_groups=groups.shape[0],
        visit_empty_groups=False,
    )
    tiles = metadata[2]
    valid = jnp.arange(tiles.size) < active
    # Split populated tiles; a half-full reserve must not leave one core idle.
    active_row_tiles = (jnp.sum(groups) + tm - 1) // tm
    boundaries = (jnp.arange(partitions + 1) * active_row_tiles + partitions - 1) // partitions
    starts = jnp.sum(valid[None, :] & (tiles[None, :] < boundaries[:, None]), axis=1, dtype=jnp.int32)
    counts = starts[1:] - starts[:-1]

    def slot_index(part, slot, starts, counts):
        return starts[part] + jnp.minimum(slot, counts[part] - 1)

    def kernel(meta, starts, counts, left, right, output, assembled):
        part, slot = pl.program_id(0), pl.program_id(1)
        index = slot_index(part, slot, starts, counts)
        offsets, ids, tiles = meta
        tile = tiles[index]
        expert = ids[index]
        previous = tiles[jnp.maximum(index - 1, starts[part])]
        following = tiles[jnp.minimum(index + 1, starts[part + 1] - 1)]

        @pl.when(slot < counts[part])
        def compute():
            @pl.when((slot == 0) | (tile != previous))
            def initialize():
                assembled[...] = jnp.zeros((tm, n), x.dtype)

            rhs = right[...].T if transpose_rhs else right[...]
            product = lax.dot_general(
                left[...],
                rhs,
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(x.dtype)
            row = lax.broadcasted_iota(jnp.int32, (tm, n), 0) + tile * tm
            mask = (row >= offsets[expert]) & (row < offsets[expert + 1])
            assembled[...] = jnp.where(mask, product, assembled[...])

            @pl.when((slot == counts[part] - 1) | (tile != following))
            def finish():
                output[...] = assembled[...]

    def row_index(part, slot, meta, starts, counts):
        return meta[2][slot_index(part, slot, starts, counts)], 0

    def weight_index(part, slot, meta, starts, counts):
        return meta[1][slot_index(part, slot, starts, counts)], 0, 0

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(partitions, jnp.max(counts)),
            in_specs=[
                pl.BlockSpec((tm, k), row_index),
                pl.BlockSpec((None, *weights.shape[1:]), weight_index),
            ],
            out_specs=pl.BlockSpec((tm, n), row_index),
            scratch_shapes=[pltpu.VMEM((tm, n), x.dtype)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        name="belay_row_partitioned_gmm",
        interpret=interpret,
    )(metadata, starts, counts, x, weights)
