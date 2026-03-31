# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Scan-safe backward metric sinks using custom_vjp.

Provides a mechanism to observe intermediate gradients during the backward pass
and return them as ordinary data, without io_callback. Works with jit, scan,
and remat.

Usage::

    sink = empty_sink("grad_sumsq", "grad_count")

    def loss(param, sink, x):
        y = param * x
        y, sink = observe_grad_sumsq(y, sink)
        return jnp.sum(y ** 2)

    grad_fn = jax.value_and_grad(loss, argnums=(0, 1))
    loss_val, (param_grad, backward_metrics) = grad_fn(param, sink, x)
    rms = grad_rms_from_sink(backward_metrics)
"""

from typing import Callable

import jax
import jax.numpy as jnp

BackwardMetricSink = dict[str, jax.Array]


def empty_sink(*names: str) -> BackwardMetricSink:
    """Create a zero-initialized backward metric sink with the given keys."""
    return {name: jnp.zeros(()) for name in names}


def make_backward_observer(
    compute_stats: Callable,
) -> Callable:
    """Create a backward observer using custom_vjp.

    Returns an ``observe(x, sink) -> (x, sink)`` function that is identity in
    the forward pass. During the backward pass, ``compute_stats`` is called
    with the cotangent of ``x`` and its return value is added element-wise to
    the cotangent of ``sink``.

    Args:
        compute_stats: receives the cotangent of x, returns a dict with the
            same keys as the sink.
    """

    @jax.custom_vjp
    def observe(x, sink):
        return x, sink

    def observe_fwd(x, sink):
        return (x, sink), ()

    def observe_bwd(_, g):
        g_x, g_sink = g
        new_stats = compute_stats(g_x)
        new_g_sink = jax.tree.map(jnp.add, g_sink, new_stats)
        return g_x, new_g_sink

    observe.defvjp(observe_fwd, observe_bwd)
    return observe


def _grad_sumsq_stats(g: jax.Array) -> BackwardMetricSink:
    """Compute gradient sum-of-squares and element count from a pytree."""
    leaves = jax.tree.leaves(g)
    sumsq = jnp.float32(0.0)
    count = 0
    for leaf in leaves:
        sumsq = sumsq + jnp.sum(leaf.astype(jnp.float32) ** 2)
        count += leaf.size
    return {"grad_sumsq": sumsq, "grad_count": jnp.array(count, dtype=jnp.float32)}


observe_grad_sumsq = make_backward_observer(_grad_sumsq_stats)
"""Pre-built observer: gradient sum-of-squares and element count.

Use with ``empty_sink("grad_sumsq", "grad_count")``. After differentiation,
the sink gradient contains accumulated sumsq and count across all observation
points. Compute RMS via ``grad_rms_from_sink``.
"""


def grad_rms_from_sink(sink_grad: BackwardMetricSink) -> jax.Array:
    """Compute gradient RMS from an accumulated grad_sumsq sink."""
    return jnp.sqrt(sink_grad["grad_sumsq"] / sink_grad["grad_count"])
