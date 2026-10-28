# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU routing adapters from the selected Base16 implementation (aef63f29d).

These preserve the selected experts and capacity prefixes. Router lookup uses
one vector reduction per selected slot instead of a general gather/scatter.
"""

from functools import partial

import jax
import jax.numpy as jnp


def _lookup_bits(logits, selected):
    bits = jax.lax.bitcast_convert_type(logits, jnp.uint32)
    experts = jnp.arange(logits.shape[-1])
    output = jnp.stack(
        [
            jnp.bitwise_or.reduce(jnp.where(experts == selected[:, i, None], bits, 0), axis=-1)
            for i in range(selected.shape[-1])
        ],
        axis=-1,
    )
    return jax.lax.bitcast_convert_type(output, jnp.float32)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _selected_logits(logits, selected, _experts):
    return _lookup_bits(logits, selected)


def _selected_forward(logits, selected, _experts):
    return _lookup_bits(logits, selected), selected


def _selected_backward(experts, selected, cotangent):
    # The reference scatter-add starts from positive zero.
    cotangent = jnp.where(cotangent == 0, jnp.float32(0), cotangent)
    bits = jax.lax.bitcast_convert_type(cotangent, jnp.uint32)
    expert_ids = jnp.arange(experts)
    result = jnp.zeros((selected.shape[0], experts), jnp.uint32)
    for i in range(selected.shape[-1]):
        result |= jnp.where(expert_ids == selected[:, i, None], bits[:, i, None], 0)
    return jax.lax.bitcast_convert_type(result, jnp.float32), None


_selected_logits.defvjp(_selected_forward, _selected_backward)


def unrolled_router_logits(logits, selected):
    """Select FP32 logits with unique, in-range expert IDs in each row."""
    return _selected_logits(logits, selected, logits.shape[-1])


def accepted_assignments_cutoff(flat_selected, sorted_indices, group_sizes, accepted_group_sizes):
    """Mark the same accepted prefixes using one cutoff lookup per assignment."""
    experts = group_sizes.shape[0]
    positions = jnp.argsort(sorted_indices)
    cutoff = jnp.cumsum(group_sizes) - group_sizes + accepted_group_sizes
    expert = jnp.minimum(flat_selected, experts - 1)
    return (flat_selected < experts) & (positions < cutoff[expert])


def restore_assignment_scalars_sort(values, sorted_indices):
    """Restore assignment order by sorting indices and scalar values together."""
    return jax.lax.sort_key_val(sorted_indices, values)[1]


def permute_assignment_weights_sort(weights, sorted_indices):
    """Move weights into expert order by sorting their inverse permutation."""
    return jax.lax.sort_key_val(jnp.argsort(sorted_indices), weights)[1]
