# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fuse indexed token reads with expert projection in an isolated TPU candidate."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def indexed_layout(selected, experts, *, slabs=2, tile_rows=128):
    """Build slab-local expert tiles, retaining every input assignment."""
    tokens, topk = selected.shape
    if tokens % slabs:
        raise ValueError("Token slabs must divide the input")
    tokens_per_slab = tokens // slabs
    assignments = tokens_per_slab * topk
    rows = ((assignments + experts * (tile_rows - 1) + tile_rows - 1) // tile_rows) * tile_rows
    indices = np.full((slabs, rows), -1, np.int32)
    sizes = np.zeros((slabs, experts), np.int32)
    restore = np.empty((slabs, assignments), np.int32)
    for slab in range(slabs):
        flat = selected[slab * tokens_per_slab : (slab + 1) * tokens_per_slab].reshape(-1)
        counts = np.bincount(flat, minlength=experts)
        physical = ((counts + tile_rows - 1) // tile_rows) * tile_rows
        spare = (rows - physical.sum()) // tile_rows
        physical += (spare // experts + (np.arange(experts) < spare % experts)) * tile_rows
        order = np.argsort(flat, stable=True)
        source = 0
        destination = 0
        for expert in range(experts):
            count = counts[expert]
            slots = order[source : source + count]
            indices[slab, destination : destination + count] = slots // topk
            restore[slab, slots] = slab * rows + destination + np.arange(count)
            source += count
            destination += physical[expert]
        sizes[slab] = physical
    tile_experts = np.stack([np.repeat(np.arange(experts), counts // tile_rows) for counts in sizes]).astype(np.int32)
    return indices, sizes, tile_experts, restore.reshape(-1)


@partial(jax.jit, static_argnames=("tile_rows", "interpret"))
def indexed_up(x, weights, indices, tile_experts, *, tile_rows=128, interpret=False):
    """Keep a token slab in VMEM and gather rows directly into its expert GEMM."""
    tokens, width = x.shape
    experts, weight_width, hidden = weights.shape
    del experts
    slabs, rows = indices.shape
    slab_tokens = tokens // slabs
    if tokens % slabs or rows % tile_rows or width != weight_width:
        raise ValueError("Expected complete token slabs, row tiles and matching feature dimensions")

    def kernel(metadata, inputs, slots, up, output):
        del metadata
        slot = slots[...]
        gather_indices = jnp.broadcast_to(jnp.maximum(slot, 0)[:, None], (tile_rows, width))
        values = jnp.take_along_axis(inputs[...], gather_indices, axis=0)
        valid = jnp.broadcast_to((slot >= 0)[:, None], (tile_rows, width))
        values = jnp.where(valid, values, 0)
        result = lax.dot_general(
            values,
            up[...],
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        output[...] = result.astype(x.dtype)

    def input_index(slab, tile, metadata):
        del tile, metadata
        return slab, 0

    def slot_index(slab, tile, metadata):
        del metadata
        return slab, tile

    def weight_index(slab, tile, metadata):
        return metadata[slab, tile], 0, 0

    def output_index(slab, tile, metadata):
        del metadata
        return slab, tile, 0

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((slabs, rows, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(slabs, rows // tile_rows),
            in_specs=[
                pl.BlockSpec((slab_tokens, width), input_index),
                pl.BlockSpec((None, tile_rows), slot_index),
                pl.BlockSpec((None, width, hidden), weight_index),
            ],
            out_specs=pl.BlockSpec((None, tile_rows, hidden), output_index),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret,
        name="belay_indexed_expert_up",
    )(tile_experts, x, indices, weights)
