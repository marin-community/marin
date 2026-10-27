# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact route counts and capacity-prefix layouts without assignment lookups."""

import jax
import jax.numpy as jnp

from experiments.grug.belay_tpu.qb_hist_tpu import factorized_histogram


def factorized_route_counts(flat, experts):
    """Count valid expert IDs using an exact int8-by-int8 matrix product."""
    return factorized_histogram(flat[:, None], experts, group_size=1, radix=16).reshape(experts)


def _interval_route_payloads(flat, counts, accepted_counts, *payloads):
    experts = counts.shape[0]
    positions = jnp.arange(flat.size, dtype=jnp.int32)
    sorted_experts, original_order, *sorted_payloads = jax.lax.sort(
        (flat, positions, *payloads), num_keys=1, is_stable=True
    )
    starts = jnp.cumsum(counts) - counts
    stops = starts + accepted_counts
    # Each expert occupies a disjoint sorted interval. Empty intervals add and
    # subtract at the same boundary. This scatters 2E events rather than looking
    # up an expert cutoff for every one of the T*K assignments.
    events = jnp.zeros((flat.size + 1,), jnp.int32)
    events = events.at[starts].add(1).at[stops].add(-1)
    accepted_sorted = jnp.cumsum(events[:-1]) > 0
    keys = jnp.where(accepted_sorted, sorted_experts, experts)
    # The second key preserves original token/slot order among rejected routes.
    _, order, *ordered_payloads = jax.lax.sort((keys, original_order, *sorted_payloads), num_keys=2, is_stable=True)
    inverse = jnp.argsort(order)
    keep = inverse < accepted_counts.sum()
    return order, inverse, keep, *ordered_payloads


def interval_route_layout(flat, counts, accepted_counts):
    """Retain each expert's accepted prefix and the source's rejected-row order."""
    return _interval_route_payloads(flat, counts, accepted_counts)


def weighted_interval_route_layout(flat, counts, accepted_counts, weights):
    """Carry a nondifferentiable weight cache through the existing route sorts."""
    return _interval_route_payloads(flat, counts, accepted_counts, jax.lax.stop_gradient(weights.reshape(-1)))
