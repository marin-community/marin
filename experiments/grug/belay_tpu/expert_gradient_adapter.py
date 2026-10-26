# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use partitioned weight gradients with the existing grouped-dot forward."""

import math
from functools import partial

import jax
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm

from experiments.grug.belay_tpu.expert_gradient_tpu import expert_weight_gradient
from experiments.grug.belay_tpu.local_experts import _tiling
from experiments.grug.belay_tpu.row_partitioned_gmm import row_partitioned_gmm


def whole_width_tiling(m, k, n):
    return math.gcd(m, 512), k, n


def _grouped_dot(x, weights, groups, interpret, tiling, row_partitions, *, transpose=False):
    if row_partitions:
        return row_partitioned_gmm(
            x,
            weights,
            groups,
            partitions=row_partitions,
            tile_rows=math.gcd(x.shape[0] // row_partitions, 512),
            transpose_rhs=transpose,
            interpret=interpret,
        )
    return gmm(
        x,
        weights,
        groups,
        preferred_element_type=x.dtype,
        tiling=tiling,
        transpose_rhs=transpose,
        interpret=interpret,
    )


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def partitioned_grouped_dot(x, weights, groups, interpret=False, tiling=_tiling, row_partitions=0):
    return _grouped_dot(x, weights, groups, interpret, tiling, row_partitions)


def _forward(x, weights, groups, interpret, tiling, row_partitions):
    output = partitioned_grouped_dot(x, weights, groups, interpret, tiling, row_partitions)
    return output, (x, weights, groups)


def _backward(interpret, tiling, row_partitions, residuals, gradient):
    x, weights, groups = residuals
    dx = _grouped_dot(gradient, weights, groups, interpret, tiling, row_partitions, transpose=True)
    dw = expert_weight_gradient(
        x, gradient, groups, partitions=2, tile_rows=512, output_dtype=weights.dtype, interpret=interpret
    )
    return dx, dw, None


partitioned_grouped_dot.defvjp(_forward, _backward)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def weight_first_grouped_dot(x, weights, groups, interpret=False, tiling=_tiling, row_partitions=0):
    """Make expert weight gradients ready before the input-gradient product."""
    return _grouped_dot(x, weights, groups, interpret, tiling, row_partitions)


def _weight_first_forward(x, weights, groups, interpret, tiling, row_partitions):
    output = _grouped_dot(x, weights, groups, interpret, tiling, row_partitions)
    return output, (x, weights, groups)


def _weight_first_backward(interpret, tiling, row_partitions, residuals, gradient):
    x, weights, groups = residuals
    dw = expert_weight_gradient(
        x, gradient, groups, partitions=2, tile_rows=512, output_dtype=weights.dtype, interpret=interpret
    )
    # Consume all barrier outputs so dX cannot precede dW. The reduction of dW
    # remains independent of dX and can overlap subsequent backpropagation work.
    gradient, weights, dw = jax.lax.optimization_barrier((gradient, weights, dw))
    dx = _grouped_dot(gradient, weights, groups, interpret, tiling, row_partitions, transpose=True)
    return dx, dw, None


weight_first_grouped_dot.defvjp(_weight_first_forward, _weight_first_backward)
