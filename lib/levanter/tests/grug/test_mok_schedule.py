# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.mok_schedule import EXPERT_PADDING, build_dispatch_schedule

WORLD_SIZE = 4
NUM_LOCAL_TOKENS = 6
TOPK = 2
NUM_LOCAL_EXPERTS = 2
CAPACITY = NUM_LOCAL_EXPERTS * EXPERT_PADDING


def reference_schedule(topk_all, num_local_experts, rank, schedule_capacity):
    """Transcription of the CUDA count/pad/schedule loops, kept deliberately naive."""
    world_size, num_local_tokens, topk = topk_all.shape
    stride = num_local_tokens * topk
    first_expert = rank * num_local_experts
    flat = topk_all.reshape(world_size, stride)

    counts = np.zeros((num_local_experts, world_size), np.int32)
    for peer in range(world_size):
        for i in range(stride):
            expert = flat[peer, i] - first_expert
            if 0 <= expert < num_local_experts:
                counts[expert, peer] += 1

    tokens_per_expert = np.array(
        [int(np.ceil(counts[e].sum() / EXPERT_PADDING)) * EXPERT_PADDING for e in range(num_local_experts)],
        np.int32,
    )

    peer_rank = np.full(schedule_capacity, -1, np.int32)
    peer_token_idx = np.zeros(schedule_capacity, np.int32)
    for expert in range(num_local_experts):
        base = int(tokens_per_expert[:expert].sum())
        for peer in range(world_size):
            j = 0
            for i in range(stride):
                if flat[peer, i] - first_expert != expert:
                    continue
                destination = base
                for source in range(world_size):
                    n = int(counts[expert, source])
                    destination += min(n, j)
                    destination += 1 if (source < peer and n > j) else 0
                peer_rank[destination] = peer
                peer_token_idx[destination] = i
                j += 1

    return peer_rank, peer_token_idx, int(tokens_per_expert.sum()), tokens_per_expert


def random_routing(seed, num_global_experts):
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_global_experts, size=(WORLD_SIZE, NUM_LOCAL_TOKENS, TOPK), dtype=np.int32)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("rank", range(WORLD_SIZE))
def test_matches_reference_loops(seed, rank):
    topk_all = random_routing(seed, WORLD_SIZE * NUM_LOCAL_EXPERTS)
    expected = reference_schedule(topk_all, NUM_LOCAL_EXPERTS, rank, CAPACITY)

    got = build_dispatch_schedule(
        jnp.asarray(topk_all), num_local_experts=NUM_LOCAL_EXPERTS, rank=rank, schedule_capacity=CAPACITY
    )

    np.testing.assert_array_equal(np.asarray(got[0]), expected[0])
    np.testing.assert_array_equal(np.asarray(got[1]), expected[1])
    assert int(got[2]) == expected[2]
    np.testing.assert_array_equal(np.asarray(got[3]), expected[3])


@pytest.mark.parametrize("rank", range(WORLD_SIZE))
def test_every_owned_assignment_lands_exactly_once(rank):
    """The table must be a bijection onto the assignments this rank owns."""
    topk_all = random_routing(7, WORLD_SIZE * NUM_LOCAL_EXPERTS)
    peer_rank, peer_token_idx, _, _ = build_dispatch_schedule(
        jnp.asarray(topk_all), num_local_experts=NUM_LOCAL_EXPERTS, rank=rank, schedule_capacity=CAPACITY
    )
    peer_rank = np.asarray(peer_rank)
    peer_token_idx = np.asarray(peer_token_idx)

    filled = peer_rank >= 0
    scheduled = sorted(zip(peer_rank[filled].tolist(), peer_token_idx[filled].tolist()))

    flat = topk_all.reshape(WORLD_SIZE, -1)
    first_expert = rank * NUM_LOCAL_EXPERTS
    owned = sorted(
        (peer, i)
        for peer in range(WORLD_SIZE)
        for i in range(flat.shape[1])
        if first_expert <= flat[peer, i] < first_expert + NUM_LOCAL_EXPERTS
    )

    assert scheduled == owned


@pytest.mark.parametrize("rank", range(WORLD_SIZE))
def test_rows_land_in_their_own_expert_segment(rank):
    """A filled row must sit inside the padded segment of the expert it routed to."""
    topk_all = random_routing(11, WORLD_SIZE * NUM_LOCAL_EXPERTS)
    peer_rank, peer_token_idx, _, tokens_per_expert = build_dispatch_schedule(
        jnp.asarray(topk_all), num_local_experts=NUM_LOCAL_EXPERTS, rank=rank, schedule_capacity=CAPACITY
    )
    peer_rank = np.asarray(peer_rank)
    peer_token_idx = np.asarray(peer_token_idx)
    bounds = np.cumsum(np.asarray(tokens_per_expert))

    flat = topk_all.reshape(WORLD_SIZE, -1)
    for row in np.flatnonzero(peer_rank >= 0):
        expert = flat[peer_rank[row], peer_token_idx[row]] - rank * NUM_LOCAL_EXPERTS
        assert bounds[expert] - tokens_per_expert[expert] <= row < bounds[expert]


def test_unrouted_rows_stay_marked_as_padding():
    """Every rank owns experts, so a routing that avoids rank 0 leaves its table empty."""
    topk_all = np.full((WORLD_SIZE, NUM_LOCAL_TOKENS, TOPK), NUM_LOCAL_EXPERTS, np.int32)

    peer_rank, _, num_tokens, tokens_per_expert = build_dispatch_schedule(
        jnp.asarray(topk_all), num_local_experts=NUM_LOCAL_EXPERTS, rank=0, schedule_capacity=CAPACITY
    )

    assert np.all(np.asarray(peer_rank) == -1)
    assert int(num_tokens) == 0
    np.testing.assert_array_equal(np.asarray(tokens_per_expert), np.zeros(NUM_LOCAL_EXPERTS, np.int32))
