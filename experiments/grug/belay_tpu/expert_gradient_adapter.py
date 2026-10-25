# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use partitioned weight gradients with the existing grouped-dot forward."""

import math
from functools import partial

import jax
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm

from experiments.grug.belay_tpu.expert_gradient_tpu import expert_weight_gradient
from experiments.grug.belay_tpu.local_experts import _tiling


def whole_width_tiling(m, k, n):
    return math.gcd(m, 512), k, n


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def partitioned_grouped_dot(x, weights, groups, interpret=False, tiling=_tiling):
    return gmm(x, weights, groups, preferred_element_type=x.dtype, tiling=tiling, interpret=interpret)


def _forward(x, weights, groups, interpret, tiling):
    output = partitioned_grouped_dot(x, weights, groups, interpret, tiling)
    return output, (x, weights, groups)


def _backward(interpret, tiling, residuals, gradient):
    x, weights, groups = residuals
    dx = gmm(
        gradient,
        weights,
        groups,
        preferred_element_type=x.dtype,
        tiling=tiling,
        transpose_rhs=True,
        interpret=interpret,
    )
    dw = expert_weight_gradient(
        x, gradient, groups, partitions=2, tile_rows=512, output_dtype=weights.dtype, interpret=interpret
    )
    return dx, dw, None


partitioned_grouped_dot.defvjp(_forward, _backward)
