# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reduce routed outputs while keeping a feature slice resident in TPU VMEM."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@partial(jax.jit, static_argnames=("tile_tokens", "tile_features", "interpret"))
def combine_feature_major(rows, positions, weights, *, tile_tokens=128, tile_features=8, interpret=False):
    """Combine feature-major expert rows without an expanded reordered output.

    Args:
        rows: Expert outputs shaped [features, assignments].
        positions: Expert-row indices shaped [topk, tokens].
        weights: FP32 combination weights shaped [topk, tokens].
        tile_tokens: Tokens reduced by each kernel invocation.
        tile_features: Features retained across all token tiles.
        interpret: Run the Pallas interpreter for non-training oracles.
    """
    features, assignments = rows.shape
    topk, tokens = positions.shape
    if features % tile_features or tokens % tile_tokens or positions.shape != weights.shape:
        raise ValueError("Expected complete feature/token tiles and matching routing arrays")

    def kernel(source, indices, coefficients, output):
        values = source[...]
        result = jnp.zeros((tile_features, tile_tokens), jnp.float32)
        for route in range(topk):
            index = jnp.broadcast_to(indices[route : route + 1, :], (tile_features, tile_tokens))
            selected = jnp.take_along_axis(values, index, axis=1).astype(jnp.float32)
            weight = jnp.broadcast_to(coefficients[route : route + 1, :], (tile_features, tile_tokens))
            result = result + selected * weight
        output[...] = result.astype(rows.dtype)

    def row_index(feature, token):
        del token
        return feature, 0

    def routing_index(feature, token):
        del feature
        return 0, token

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((features, tokens), rows.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(features // tile_features, tokens // tile_tokens),
            in_specs=[
                pl.BlockSpec((tile_features, assignments), row_index),
                pl.BlockSpec((topk, tile_tokens), routing_index),
                pl.BlockSpec((topk, tile_tokens), routing_index),
            ],
            out_specs=pl.BlockSpec((tile_features, tile_tokens), lambda feature, token: (feature, token)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret,
        name="belay_feature_major_combine",
    )(rows, positions, weights)


def combine_with_transpose(rows, order, inverse, weights, *, interpret=False):
    """Include conversion from the existing row-major producer and consumer."""
    del order
    positions = inverse.reshape(weights.shape).T
    return combine_feature_major(rows.T, positions, weights.T, interpret=interpret).T
