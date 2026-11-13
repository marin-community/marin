# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare ring_filtered's routing maps (Pallas-Triton counting sorts, XLA fusions) with a sequential reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.jax_utils import is_amd_instinct
from levanter.grug._moe.ep_ring_filtered import _xla_expert_rows, _xla_send_slots
from levanter.grug._moe.pallas_routing_metadata import expert_rows, send_slots

pytestmark = pytest.mark.skipif(jax.default_backend() == "tpu", reason="Pallas-Triton kernels need a GPU or CPU")


def _send_reference(selected, valid, cap, *, local_experts, destinations, pair_capacity):
    """One pass over the tokens per destination, in token order."""
    tokens, _ = selected.shape
    send_rows = destinations * pair_capacity
    dest = selected // local_experts
    slot = np.full((tokens, destinations), send_rows, np.int32)
    source = np.full((send_rows,), tokens, np.int32)
    for d in range(destinations):
        rank = 0
        for t in range(tokens):
            if valid[t] and (dest[t] == d).any():
                if rank < cap:
                    slot[t, d] = d * pair_capacity + rank
                    source[d * pair_capacity + rank] = t
                rank += 1
    kept = int(np.sum((np.take_along_axis(slot, dest, axis=1) != send_rows) & valid[:, None]))
    return slot, source, kept


def _expert_reference(ids, cap, *, local_experts, physical_capacity):
    """Stable sort of the assignments by local expert; capacity filled in expert order."""
    assignments = ids.shape[0]
    counts = np.bincount(ids[ids >= 0], minlength=local_experts)
    starts = np.cumsum(counts) - counts
    accepted = np.minimum(counts, np.maximum(cap - starts, 0))
    picked = np.full((physical_capacity,), assignments, np.int32)
    position = np.full((assignments,), physical_capacity, np.int32)
    seen = np.zeros(local_experts, np.int64)
    for a in range(assignments):
        e = ids[a]
        if e < 0:
            continue
        if seen[e] < accepted[e]:
            row = starts[e] + seen[e]
            picked[row] = a
            position[a] = row
        seen[e] += 1
    return accepted.astype(np.int32), int(counts.sum() - accepted.sum()), picked, position


def _routing(key, tokens, *, experts, topk):
    k_sel, k_valid = jax.random.split(key)
    # Skewed expert choice so some destinations and experts overflow their capacity.
    logits = jax.random.gumbel(k_sel, (tokens, experts)) + jnp.linspace(1.5, 0.0, experts)
    selected = np.asarray(jax.lax.top_k(logits, topk)[1], np.int32)
    valid = np.asarray(jax.random.bernoulli(k_valid, 0.9, (tokens,)))
    return selected, valid


def _received_metadata(rng, *, destinations, pair_capacity, topk, local_experts, empty):
    """ring_filtered's received [destinations, C + 1, 2K] buffer with skewed local expert ids, a fraction ``empty`` of
    them -1 (no assignment)."""
    rows = destinations * pair_capacity
    ids = np.minimum(rng.geometric(0.08, (rows, topk)) - 1, local_experts - 1).astype(np.int32)
    ids[rng.random((rows, topk)) < empty] = -1
    meta = np.concatenate([rng.random((rows, topk)).astype(np.float32), ids.astype(np.float32)], axis=1)
    meta = np.concatenate(
        [meta.reshape(destinations, pair_capacity, 2 * topk), np.zeros((destinations, 1, 2 * topk), np.float32)], 1
    )
    return meta, ids.reshape(-1)


def _check_send_slots(selected, valid, cap, **shape):
    expected_slot, expected_source, expected_kept = _send_reference(selected, valid, cap, **shape)
    for build in (send_slots, _xla_send_slots):
        maps = jax.jit(lambda s, v, c, build=build: build(s, v, c, **shape))(selected, valid, jnp.int32(cap))
        np.testing.assert_array_equal(np.asarray(maps.slot), expected_slot, err_msg=build.__name__)
        np.testing.assert_array_equal(np.asarray(maps.source), expected_source, err_msg=build.__name__)
        assert int(maps.kept_assignments) == expected_kept, build.__name__


def _check_expert_rows(meta, ids, cap, *, local_experts, physical_capacity):
    group_sizes, dropped, picked, position = _expert_reference(
        ids, min(cap, physical_capacity), local_experts=local_experts, physical_capacity=physical_capacity
    )
    assert dropped > 0 or cap >= ids.size
    for build in (expert_rows, _xla_expert_rows):
        rows = jax.jit(
            lambda m, c, build=build: build(m, c, local_experts=local_experts, physical_capacity=physical_capacity)
        )(meta, jnp.int32(cap))
        np.testing.assert_array_equal(np.asarray(rows.group_sizes), group_sizes, err_msg=build.__name__)
        assert int(rows.dropped) == dropped, build.__name__
        np.testing.assert_array_equal(np.asarray(rows.picked), picked, err_msg=build.__name__)
        np.testing.assert_array_equal(np.asarray(rows.position), position, err_msg=build.__name__)


@pytest.mark.parametrize(
    ("tokens", "cap", "pair_capacity"),
    [(600, 80, 100), (200, 200, 200), (40, 3, 8)],
    ids=["three_blocks_sender_drops", "one_block_no_drops", "tiny_cap"],
)
def test_send_slots_match_sequential_reference(tokens, cap, pair_capacity):
    """Each token's send row per destination, the inverse map and the kept count, over several blocks of tokens."""
    local_experts, destinations, topk = 4, 8, 4
    selected, valid = _routing(jax.random.key(tokens), tokens, experts=local_experts * destinations, topk=topk)
    _check_send_slots(
        selected, valid, cap, local_experts=local_experts, destinations=destinations, pair_capacity=pair_capacity
    )


@pytest.mark.parametrize(
    ("pair_capacity", "cap", "physical_capacity"),
    [(40, 300, 640), (40, 2000, 1280), (10, 64, 64)],
    ids=["three_blocks_receiver_drops", "no_drops_all_rows", "one_block_tight"],
)
def test_expert_rows_match_sequential_reference(pair_capacity, cap, physical_capacity):
    """Accepted rows per local expert, drops, each expert row's assignment and each assignment's expert row, with
    empty send rows and experts past the capacity."""
    destinations, topk, local_experts = 8, 4, 16
    meta, ids = _received_metadata(
        np.random.default_rng(pair_capacity + cap),
        destinations=destinations,
        pair_capacity=pair_capacity,
        topk=topk,
        local_experts=local_experts,
        empty=0.15,
    )
    _check_expert_rows(meta, ids, cap, local_experts=local_experts, physical_capacity=physical_capacity)


def test_routing_maps_match_sequential_reference_at_june_shapes_on_gpu():
    if not is_amd_instinct():
        pytest.skip("compiles the Triton kernels ring_filtered uses only on AMD Instinct")
    # One June layer-microbatch: 32768 tokens per GPU, top-4 of 256 experts over 8 shards, C = 14988 send rows per
    # destination, 131072 expert rows.
    local_experts, destinations, topk, pair_capacity = 32, 8, 4, 14988
    selected, valid = _routing(jax.random.key(3), 32768, experts=local_experts * destinations, topk=topk)
    _check_send_slots(
        selected, valid, 13000, local_experts=local_experts, destinations=destinations, pair_capacity=pair_capacity
    )
    meta, ids = _received_metadata(
        np.random.default_rng(4),
        destinations=destinations,
        pair_capacity=pair_capacity,
        topk=topk,
        local_experts=local_experts,
        empty=0.1,
    )
    _check_expert_rows(meta, ids, 120000, local_experts=local_experts, physical_capacity=131072)
