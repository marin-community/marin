# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use partitioned weight gradients with the existing grouped-dot forward."""

from functools import partial

import jax
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm

from experiments.grug.belay_tpu.expert_gradient_tpu import expert_weight_gradient
from experiments.grug.belay_tpu.local_experts import _tiling


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def partitioned_grouped_dot(x, weights, groups, interpret=False):
    return gmm(x, weights, groups, preferred_element_type=x.dtype, tiling=_tiling, interpret=interpret)


def _forward(x, weights, groups, interpret):
    output = partitioned_grouped_dot(x, weights, groups, interpret)
    return output, (x, weights, groups)


def _backward(interpret, residuals, gradient):
    x, weights, groups = residuals
    dx = gmm(
        gradient,
        weights,
        groups,
        preferred_element_type=x.dtype,
        tiling=_tiling,
        transpose_rhs=True,
        interpret=interpret,
    )
    dw = expert_weight_gradient(
        x, gradient, groups, partitions=2, tile_rows=512, output_dtype=weights.dtype, interpret=interpret
    )
    return dx, dw, None


partitioned_grouped_dot.defvjp(_forward, _backward)
