# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite expert-matrix credit using candidate 11's native capacity-limited dispatch."""

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.ep_common import _ranks_within_groups
from levanter.grug._moe.ep_fixed_pooled_wave_all_to_all import (
    _compact_received,
    _compute_pooled,
    _dispatch_pooled,
    _expand_compacted,
    _PooledDispatch,
    _receiver_kept_at_sender,
)

from experiments.grug.noise_belay.forward_credit import (
    Relu2ParameterCredit,
    relu2_parameter_credit,
    sqrt_softplus_router_credit,
)

BATCH_AXES = ("replica_dcn", "data", "expert")


class LocalRoute(NamedTuple):
    dispatch: _PooledDispatch
    kept: jax.Array
    shards: int
    pool_capacity: int
    receiver_capacity: int


def _local_route(
    x: jax.Array,
    selected: jax.Array,
    *,
    local_experts: int,
    shards: int,
    capacity_factor: float,
    transport_capacity_factor: float,
) -> LocalRoute:
    tokens, topk = selected.shape
    assignments = tokens * topk
    pool_capacity = max(math.ceil(transport_capacity_factor * assignments / shards), 1)
    receiver_capacity = max(math.ceil(capacity_factor * assignments / local_experts), 1)
    flat_experts = selected.reshape(-1)
    destinations = flat_experts // local_experts
    pool_ranks = _ranks_within_groups(destinations, num_groups=shards, valid=jnp.ones(assignments, jnp.bool_))
    sender_keep = pool_ranks < pool_capacity
    dispatch, _ = _dispatch_pooled(
        x_local=x,
        local_expert_indices=flat_experts % local_experts,
        destination_shards=destinations,
        pool_ranks=pool_ranks,
        sender_keep=sender_keep,
        local_experts=local_experts,
        expert_shards=shards,
        pool_capacity=pool_capacity,
        receiver_capacity=receiver_capacity,
        receiver_limit=jnp.asarray(receiver_capacity, jnp.int32),
        assignments_per_shard=assignments,
        topk=topk,
        fp8_dispatch=False,
    )
    keep = _receiver_kept_at_sender(dispatch, sender_keep, expert_shards=shards, pool_capacity=pool_capacity).reshape(
        tokens, topk
    )
    return LocalRoute(dispatch, keep, shards, pool_capacity, receiver_capacity)


def pooled_relu2_parameter_credit(
    inputs: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    up: jax.Array,
    down: jax.Array,
    output_credit: jax.Array,
    *,
    capacity_factor: float,
    transport_capacity_factor: float,
    relative_step: float,
) -> Relu2ParameterCredit:
    """Reconstruct matrices with fixed routing and native drop renormalization.

    Inputs and credits have shape [tokens, channels], routing arrays [tokens, K],
    and matrices [experts, input, hidden] / [experts, hidden, output]. The credit
    must come from an independent measured hub. Only fixed linear dispatch,
    combine and readout operations transport it before a single finite ReLU²
    probe; no corrected credit is propagated through another nonlinearity.

    This covers candidate 11's one-wave, non-FP8 dispatch with all tokens valid.
    Loss-masked tokens still participate in routing, as in its native forward.
    Credits are unnormalized sums. Router and latent-normalization parameters
    require separate independent probes.
    """
    num_experts = up.shape[0]

    def local(x, selected, weights, up, down, credit):
        tokens, topk = selected.shape
        local_experts = up.shape[0]
        assignments = tokens * topk
        dispatch, keep, shards, pool_capacity, receiver_capacity = _local_route(
            x,
            selected,
            local_experts=local_experts,
            shards=num_experts // local_experts,
            capacity_factor=capacity_factor,
            transport_capacity_factor=transport_capacity_factor,
        )
        # Native MoE rounds combine weights to compute precision but calculates
        # its post-drop renormalization from the original float32 weights.
        kept_sum = jnp.sum(jnp.where(keep, weights, 0), axis=-1)
        renormalize = jnp.any(~keep, axis=-1) & (kept_sum > 0)
        scale = (
            jnp.where(
                renormalize,
                jnp.sum(weights, axis=-1) / jnp.where(renormalize, kept_sum, 1),
                1,
            )
            .astype(x.dtype)
            .astype(jnp.float32)
        )
        credit = credit.astype(jnp.float32) * scale[:, None]
        valid = dispatch.assignment_sources < assignments
        source = jnp.minimum(dispatch.assignment_sources, assignments - 1)
        source_weight = weights.astype(x.dtype).reshape(-1)[source].astype(jnp.float32)
        sent_credit = jnp.where(valid[:, None], credit[source // topk] * source_weight[:, None], 0)
        received_credit = jax.lax.all_to_all(
            sent_credit.reshape(shards, pool_capacity, credit.shape[-1]),
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        ).reshape(-1, credit.shape[-1])
        expert_credit = _compact_received(
            received_credit,
            dispatch.receiver_linear_indices,
            dispatch.receiver_keep,
            local_experts,
            receiver_capacity,
        )
        preactivation = jnp.einsum("erh,ehi->eri", dispatch.compacted_x, up)
        result = jax.vmap(partial(relu2_parameter_credit, relative_step=relative_step))(
            dispatch.compacted_x, preactivation, down, expert_credit
        )
        return jax.tree.map(lambda value: jax.lax.psum(value, ("replica_dcn", "data")), result)

    specs = (
        P(BATCH_AXES, None),
        P(BATCH_AXES, None),
        P(BATCH_AXES, None),
        P("expert", None, None),
        P("expert", None, None),
        P(BATCH_AXES, None),
    )
    mapped = jax.shard_map(
        local,
        mesh=jax.sharding.get_abstract_mesh(),
        in_specs=specs,
        out_specs=Relu2ParameterCredit(P("expert", None, None), P("expert", None, None)),
        check_vma=False,
    )
    return mapped(
        *(
            jax.sharding.reshard(value, spec)
            for value, spec in zip(
                (inputs, selected_experts, combine_weights, up, down, output_credit),
                specs,
                strict=True,
            )
        )
    )


def pooled_router_parameter_credit(
    inputs: jax.Array,
    selected_experts: jax.Array,
    selected_logits: jax.Array,
    up: jax.Array,
    down: jax.Array,
    output_credit: jax.Array,
    router_inputs: jax.Array,
    *,
    capacity_factor: float,
    transport_capacity_factor: float,
    sigma: float,
    renorm_sum: float,
) -> jax.Array:
    """Measure fixed-selection router matrix credit through frozen expert outputs.

    This probes float32 combine weights; the native compute-precision weight
    cast is treated as an identity, as for Noise's mixed-precision local probes.
    Capacity and expert selection are frozen at the clean forward. Every logit
    probe is contracted with the original independently measured MLP hub credit.
    """
    num_experts = up.shape[0]

    def local(x, selected, logits, up, down, credit, router_x):
        route = _local_route(
            x,
            selected,
            local_experts=up.shape[0],
            shards=num_experts // up.shape[0],
            capacity_factor=capacity_factor,
            transport_capacity_factor=transport_capacity_factor,
        )
        dispatch = route.dispatch
        output = _compute_pooled(
            dispatch,
            moe_w13_local=up,
            moe_w2_local=down,
            activation_fn=lambda h: jnp.square(jax.nn.relu(h)),
        )
        received = _expand_compacted(
            output.compacted_output,
            dispatch.receiver_linear_indices,
            dispatch.receiver_keep,
        )
        returned = jax.lax.all_to_all(
            received.reshape(route.shards, route.pool_capacity, down.shape[-1]),
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        ).reshape(-1, down.shape[-1])
        per_assignment = returned[jnp.minimum(dispatch.sender_linear_indices, returned.shape[0] - 1)]
        per_assignment = per_assignment.reshape(*selected.shape, down.shape[-1])
        scores = jnp.sum(per_assignment.astype(jnp.float32) * credit[:, None, :], axis=-1)
        logit_credit = sqrt_softplus_router_credit(
            logits,
            scores,
            route.kept,
            sigma=sigma,
            renorm_sum=renorm_sum,
        )
        dense_credit = (
            jnp.zeros((x.shape[0], num_experts), jnp.float32)
            .at[jnp.arange(x.shape[0])[:, None], selected]
            .add(logit_credit)
        )
        return jax.lax.psum(router_x.astype(jnp.float32).T @ dense_credit, BATCH_AXES)

    specs = (
        P(BATCH_AXES, None),
        P(BATCH_AXES, None),
        P(BATCH_AXES, None),
        P("expert", None, None),
        P("expert", None, None),
        P(BATCH_AXES, None),
        P(BATCH_AXES, None),
    )
    mapped = jax.shard_map(
        local,
        mesh=jax.sharding.get_abstract_mesh(),
        in_specs=specs,
        out_specs=P(None, None),
        check_vma=False,
    )
    return mapped(
        *(
            jax.sharding.reshard(value, spec)
            for value, spec in zip(
                (
                    inputs,
                    selected_experts,
                    selected_logits,
                    up,
                    down,
                    output_credit,
                    router_inputs,
                ),
                specs,
                strict=True,
            )
        )
    )
