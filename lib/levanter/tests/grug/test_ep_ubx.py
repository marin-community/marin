# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.ep_ubx import (
    _accepted_unit_gates,
    _dense_accepted_gates,
    _scatter_slot_values_to_global_topk,
    _slot_values_from_inverse_map,
    _sorted_topk_values,
    _unsort_topk_values,
    _validate_ubx_contract,
)
from levanter.grug._moe.ep_ubx_maps import UbXRoutingMaps, build_ubx_routing_maps


def _routing_fixture() -> tuple[jax.Array, tuple[UbXRoutingMaps, UbXRoutingMaps]]:
    selected = jnp.asarray(
        (
            (0, 2),
            (0, 2),
            (0, 1),
            (1, 2),
        ),
        dtype=jnp.int32,
    )
    accepted_indices = jnp.asarray(
        (
            (0, 2, 4, 5),
            (1, 3, 7, 0),
        ),
        dtype=jnp.int32,
    )
    accepted_valid = jnp.asarray(
        (
            (True, True, True, True),
            (True, True, True, False),
        ),
        dtype=jnp.bool_,
    )
    maps = tuple(
        build_ubx_routing_maps(
            selected,
            accepted_indices,
            accepted_valid,
            rank=rank,
            local_experts=2,
        )
        for rank in range(2)
    )
    return selected, maps  # type: ignore[return-value]


def _reference_dispatch(
    x_by_rank: jax.Array,
    maps: tuple[UbXRoutingMaps, UbXRoutingMaps],
) -> jax.Array:
    ranks, tokens_per_rank, _ = x_by_rank.shape
    capacity = maps[0].inverse_map.shape[0]
    output = jnp.zeros((ranks, capacity, x_by_rank.shape[-1]), dtype=x_by_rank.dtype)
    for source_rank in range(ranks):
        for token in range(tokens_per_rank):
            for topk in range(maps[source_rank].dispatch_topk_expert.shape[1]):
                expert = int(maps[source_rank].dispatch_topk_expert[token, topk])
                if expert < 0:
                    continue
                destination_rank = expert // 2
                slot = int(maps[source_rank].dispatch_topk_slot[token, topk])
                output = output.at[destination_rank, slot].set(x_by_rank[source_rank, token])
    return output


def _reference_combine(
    expert_outputs_by_rank: jax.Array,
    dense_gates_by_rank: jax.Array,
    maps: tuple[UbXRoutingMaps, UbXRoutingMaps],
) -> jax.Array:
    ranks, _, hidden = expert_outputs_by_rank.shape
    tokens_per_rank = maps[0].topk_idx.shape[0]
    output = jnp.zeros((ranks, tokens_per_rank, hidden), dtype=expert_outputs_by_rank.dtype)
    for destination_rank in range(ranks):
        for slot in range(maps[destination_rank].inverse_map.shape[0]):
            source_rank, source_token, sorted_topk, valid = map(int, maps[destination_rank].inverse_map[slot])
            if not valid:
                continue
            expert = int(maps[source_rank].topk_idx[source_token, sorted_topk])
            weight = dense_gates_by_rank[source_rank, source_token, expert]
            output = output.at[source_rank, source_token].add(expert_outputs_by_rank[destination_rank, slot] * weight)
    return output


def test_ubx_transport_transposes_match_pure_jax_oracle() -> None:
    selected_global, maps = _routing_fixture()
    selected_by_rank = selected_global.reshape(2, 2, 2)
    accepted_by_rank = jnp.stack(tuple(route_maps.accepted_local for route_maps in maps))
    topk_idx_by_rank = jnp.stack(tuple(route_maps.topk_idx for route_maps in maps))
    weights_by_rank = jnp.asarray(
        (
            ((0.1, 0.2), (0.3, 0.4)),
            ((0.5, 0.6), (0.7, 0.8)),
        ),
        dtype=jnp.float32,
    )
    x_by_rank = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) / 7
    expert_outputs_by_rank = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) / 11
    output_cotangent = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) / 13
    dispatch_cotangent = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) / 17

    unit_gates = jnp.stack(tuple(_accepted_unit_gates(route_maps.topk_idx, num_experts=4) for route_maps in maps))
    _, dispatch_vjp = jax.vjp(lambda value: _reference_dispatch(value, maps), x_by_rank)
    (dispatch_x_cotangent,) = dispatch_vjp(dispatch_cotangent)
    dispatch_x_cotangent_from_combine = _reference_combine(dispatch_cotangent, unit_gates, maps)
    np.testing.assert_allclose(dispatch_x_cotangent, dispatch_x_cotangent_from_combine, rtol=0, atol=0)

    def combine(expert_outputs: jax.Array, weights: jax.Array) -> jax.Array:
        dense_gates = jnp.stack(
            tuple(
                _dense_accepted_gates(
                    selected_by_rank[rank],
                    weights[rank],
                    accepted_by_rank[rank],
                    num_experts=4,
                )
                for rank in range(2)
            )
        )
        return _reference_combine(expert_outputs, dense_gates, maps)

    _, combine_vjp = jax.vjp(combine, expert_outputs_by_rank, weights_by_rank)
    expected_expert_cotangent, expected_weight_cotangent = combine_vjp(output_cotangent)

    output_cotangent_dispatch = _reference_dispatch(output_cotangent, maps)
    sorted_weights_global = jnp.concatenate(
        tuple(
            _sorted_topk_values(
                weights_by_rank[rank],
                selected_by_rank[rank],
                topk_idx_by_rank[rank],
            )
            for rank in range(2)
        ),
        axis=0,
    )
    slot_weights = jnp.stack(
        tuple(
            _slot_values_from_inverse_map(
                sorted_weights_global,
                route_maps.inverse_map,
                tokens_per_rank=2,
            )
            for route_maps in maps
        )
    )
    actual_expert_cotangent = output_cotangent_dispatch * slot_weights[:, :, None]

    slot_weight_cotangents = jnp.sum(output_cotangent_dispatch * expert_outputs_by_rank, axis=-1)
    sorted_weight_cotangents_global = sum(
        (
            _scatter_slot_values_to_global_topk(
                slot_weight_cotangents[rank],
                maps[rank].inverse_map,
                tokens_per_rank=2,
                expert_axis_size=2,
                topk=2,
            )
            for rank in range(2)
        ),
        start=jnp.zeros((4, 2), dtype=jnp.float32),
    )
    actual_weight_cotangent = jnp.stack(
        tuple(
            _unsort_topk_values(
                sorted_weight_cotangents_global[rank * 2 : (rank + 1) * 2],
                topk_idx_by_rank[rank],
                selected_by_rank[rank],
                accepted_by_rank[rank],
            )
            for rank in range(2)
        )
    )

    np.testing.assert_allclose(actual_expert_cotangent, expected_expert_cotangent, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_weight_cotangent, expected_weight_cotangent, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual_weight_cotangent[1, 1, 0], 0)


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    (
        (
            lambda values: {**values, "x_local": values["x_local"].astype(jnp.float32)},
            TypeError,
            "bfloat16 activations",
        ),
        (
            lambda values: {**values, "combine_weights_local": values["combine_weights_local"].astype(jnp.bfloat16)},
            TypeError,
            "float32 combine weights",
        ),
        (
            lambda values: {
                **values,
                "x_local": jnp.zeros((2, 48), dtype=jnp.bfloat16),
                "moe_w13_local": jnp.zeros((1, 48, 16), dtype=jnp.bfloat16),
                "moe_w2_local": jnp.zeros((1, 8, 48), dtype=jnp.bfloat16),
            },
            ValueError,
            "divisible by 32",
        ),
        (
            lambda values: {**values, "num_experts": 7},
            ValueError,
            "exactly 8 expert ranks",
        ),
    ),
)
def test_ubx_contract_rejects_unsupported_runtime_layout(mutate, error, match) -> None:
    values = {
        "x_local": jnp.zeros((2, 32), dtype=jnp.bfloat16),
        "selected_experts_local": jnp.zeros((2, 2), dtype=jnp.int32),
        "combine_weights_local": jnp.zeros((2, 2), dtype=jnp.float32),
        "moe_w13_local": jnp.zeros((1, 32, 16), dtype=jnp.bfloat16),
        "moe_w2_local": jnp.zeros((1, 8, 32), dtype=jnp.bfloat16),
        "num_experts": 8,
        "capacity_factor": 1.25,
    }
    with pytest.raises(error, match=match):
        _validate_ubx_contract(**mutate(values))
