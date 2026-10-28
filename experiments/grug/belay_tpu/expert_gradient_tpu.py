# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 The JAX Authors.
"""Partition expert weight-gradient GEMMs across TPU grid programs.

The masked tile accumulation follows JAX Megablox (Apache-2.0). Each partition
owns distinct experts; its row-tile loop remains serial for accumulation.
This operator is diagnostic-only and does not change the selected training path.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.megablox.gmm import make_group_metadata


@partial(jax.jit, static_argnames=("partitions", "tile_rows", "output_dtype", "interpret", "lhs_relu2"))
def expert_weight_gradient(
    lhs, rhs, groups, *, partitions=2, tile_rows=512, output_dtype=jnp.float32, interpret=False, lhs_relu2=False
):
    """Compute per-expert lhs.T @ rhs, optionally rebuilding ReLU² in each tile."""
    m, k = lhs.shape
    n = rhs.shape[1]
    experts = groups.shape[0]
    tm = min(tile_rows, m)
    if m % tm or experts % partitions or lhs.dtype != rhs.dtype:
        raise ValueError("Expected whole row tiles, equal expert partitions and matching input dtypes")
    count = experts // partitions

    def metadata_for(start):
        return make_group_metadata(
            group_sizes=groups,
            m=m,
            tm=tm,
            start_group=start,
            num_nonzero_groups=count,
            visit_empty_groups=True,
        )

    metadata, active_tiles = jax.vmap(metadata_for)(jnp.arange(partitions, dtype=jnp.int32) * count)

    def safe_slot(part, slot, active):
        return jnp.minimum(slot, active[part] - 1)

    def kernel(meta, active, left, right, out, accumulator):
        part, slot = pl.program_id(0), pl.program_id(1)
        offsets, ids, tiles = meta
        valid_slot = safe_slot(part, slot, active)
        group = ids[part, valid_slot]
        previous = ids[part, jnp.maximum(valid_slot - 1, 0)]
        following = ids[part, jnp.minimum(valid_slot + 1, active[part] - 1)]

        @pl.when(slot < active[part])
        def accumulate():
            @pl.when((slot == 0) | (group != previous))
            def initialize():
                accumulator[...] = jnp.zeros((k, n), jnp.float32)

            def masked_rows(value, width):
                row = lax.broadcasted_iota(jnp.int32, (tm, width), 0) + tiles[part, valid_slot] * tm
                mask = (row >= offsets[part, group]) & (row < offsets[part, group + 1])
                return jnp.where(mask, value, 0)

            left_value = jnp.square(jnp.maximum(left[...], 0)) if lhs_relu2 else left[...]
            x = masked_rows(left_value, k)
            dy = masked_rows(right[...], n)
            accumulator[...] += lax.dot_general(
                x,
                dy,
                dimension_numbers=(((0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )

            @pl.when((slot == active[part] - 1) | (group != following))
            def finish():
                out[...] = accumulator[...].astype(output_dtype)

    def row_index(part, slot, meta, active):
        return meta[2][part, safe_slot(part, slot, active)], 0

    def out_index(part, slot, meta, active):
        return meta[1][part, safe_slot(part, slot, active)], 0, 0

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((experts, k, n), output_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec((tm, k), row_index), pl.BlockSpec((tm, n), row_index)],
            out_specs=pl.BlockSpec((None, k, n), out_index),
            grid=(partitions, jnp.max(active_tiles)),
            scratch_shapes=[pltpu.VMEM((k, n), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        name="belay_expert_partitioned_weight_gradient",
        interpret=interpret,
    )(metadata, active_tiles, lhs, rhs)
