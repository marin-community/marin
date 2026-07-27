# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.ep_ubx_maps import build_ubx_routing_maps


def test_build_ubx_routing_maps_matches_ring_prefix_cap() -> None:
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

    rank0 = jax.jit(
        lambda: build_ubx_routing_maps(
            selected,
            accepted_indices,
            accepted_valid,
            rank=0,
            local_experts=2,
        )
    )()
    rank1 = jax.jit(
        lambda: build_ubx_routing_maps(
            selected,
            accepted_indices,
            accepted_valid,
            rank=1,
            local_experts=2,
        )
    )()

    np.testing.assert_array_equal(rank0.dispatch_topk_expert, ((0, 2), (0, 2)))
    np.testing.assert_array_equal(rank0.dispatch_topk_slot, ((0, 0), (1, 1)))
    np.testing.assert_array_equal(rank1.dispatch_topk_expert, ((0, 1), (2, -1)))
    np.testing.assert_array_equal(rank1.dispatch_topk_slot, ((2, 3), (2, -1)))
    np.testing.assert_array_equal(rank0.accepted_counts, (3, 1, 3, 0))
    np.testing.assert_array_equal(rank0.drops_by_expert_rank, (1, 0))
    np.testing.assert_array_equal(rank0.group_sizes, (3, 1))
    np.testing.assert_array_equal(rank1.group_sizes, (3, 1))
    np.testing.assert_array_equal(rank0.dispatch_valid, (True, True, True, True))
    np.testing.assert_array_equal(rank1.dispatch_valid, (True, True, True, False))

    np.testing.assert_array_equal(
        rank0.inverse_map[:4],
        (
            (0, 0, 0, 1),
            (0, 1, 0, 1),
            (1, 0, 0, 1),
            (1, 0, 1, 1),
        ),
    )
    np.testing.assert_array_equal(
        rank1.inverse_map[:3],
        (
            (0, 0, 1, 1),
            (0, 1, 1, 1),
            (1, 1, 0, 1),
        ),
    )
