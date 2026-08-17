# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Segment-plan metadata for `put_segments`, validated on CPU.

The Mosaic-GPU kernel itself needs hardware, but its metadata (which rows
go where) is pure index math. These tests execute the plans with NumPy
copies and check (a) the dispatch plan reproduces the correctness
simulator's receive pools exactly, and (b) dispatch followed by combine is
the identity on each device's compacted send buffer.
"""

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug._moe.fused_dispatch_brd import TRANSPORT_STAGE_ROWS, expected_arrival_signals
from levanter.grug._moe.marin_ep_transport import combine_segments, dispatch_segments

from experiments.marin_ep.planref import execute_plans
from experiments.marin_ep.simcore import simulate_forward

DEVICES, TOKENS, TOPK, EXPERTS, HIDDEN, INTERMEDIATE = 4, 16, 3, 8, 8, 12
LOCAL_EXPERTS = EXPERTS // DEVICES
GROUP_SIZE = 2
CF = 1.1


def _instance():
    rng = np.random.default_rng(seed=3)
    probs = rng.dirichlet(np.full(EXPERTS, 0.4))
    experts = rng.choice(EXPERTS, size=(DEVICES, TOKENS, TOPK), p=probs).astype(np.int32)
    x = rng.standard_normal((DEVICES, TOKENS, HIDDEN)).astype(np.float32)
    weights = rng.random((DEVICES, TOKENS, TOPK)).astype(np.float32)
    w13 = rng.standard_normal((EXPERTS, HIDDEN, 2 * INTERMEDIATE)).astype(np.float32)
    w2 = rng.standard_normal((EXPERTS, INTERMEDIATE, HIDDEN)).astype(np.float32)
    result = simulate_forward(
        jnp.asarray(x),
        experts,
        jnp.asarray(weights),
        jnp.asarray(w13).reshape(DEVICES, LOCAL_EXPERTS, HIDDEN, 2 * INTERMEDIATE),
        jnp.asarray(w2).reshape(DEVICES, LOCAL_EXPERTS, INTERMEDIATE, HIDDEN),
        num_experts=EXPERTS,
        capacity_factor=CF,
        activation_fn=jax.nn.silu,
        pool_group_size=GROUP_SIZE,
    )
    assert result.dropped_total > 0  # plans must be exercised under drops
    return x, experts, result


def _compacted_send_buffer(x, experts, routing, device):
    """Kept rows in expert-major order — the layout `put_segments` reads."""
    rows = []
    for g in range(EXPERTS):
        slots, _ = routing.routes[device][g]
        rows.append(x[device][slots // TOPK])
    return np.concatenate(rows, axis=0)


def _accepted(routing):
    return np.array([[routing.routes[d][g][0].size for g in range(EXPERTS)] for d in range(DEVICES)], dtype=np.int64)


def test_dispatch_plan_reproduces_simulator_pools():
    x, experts, result = _instance()
    routing = result.saved.routing
    accepted = jnp.asarray(_accepted(routing))
    region = jnp.asarray(result.saved.region)

    sends = [_compacted_send_buffer(x, experts, routing, d) for d in range(DEVICES)]
    plans = [dispatch_segments(accepted, region, jnp.int32(d), local_experts=LOCAL_EXPERTS) for d in range(DEVICES)]
    pool_rows = LOCAL_EXPERTS * routing.capacity
    pools = execute_plans(plans, sends, pool_rows)
    for d in range(DEVICES):
        np.testing.assert_array_equal(pools[d], np.asarray(result.saved.recv_x[d]))


def test_dispatch_then_combine_roundtrips_send_buffers():
    x, experts, result = _instance()
    routing = result.saved.routing
    accepted = jnp.asarray(_accepted(routing))
    region = jnp.asarray(result.saved.region)

    sends = [_compacted_send_buffer(x, experts, routing, d) for d in range(DEVICES)]
    dispatch_plans = [
        dispatch_segments(accepted, region, jnp.int32(d), local_experts=LOCAL_EXPERTS) for d in range(DEVICES)
    ]
    pools = execute_plans(dispatch_plans, sends, LOCAL_EXPERTS * routing.capacity)

    combine_plans = [
        combine_segments(accepted, region, jnp.int32(d), local_experts=LOCAL_EXPERTS) for d in range(DEVICES)
    ]
    kept_per_device = int(np.asarray(accepted).sum(axis=1).max())
    returned = execute_plans(combine_plans, pools, kept_per_device)
    for d in range(DEVICES):
        np.testing.assert_array_equal(returned[d][: sends[d].shape[0]], sends[d])


def test_expected_arrival_signals_matches_reference_loop():
    rng = np.random.default_rng(7)
    devices, local_experts, grid_size = 4, 3, 5
    accepted = rng.integers(0, 4 * TRANSPORT_STAGE_ROWS, size=(devices, devices * local_experts)).astype(np.int32)
    accepted[0, 1] = 0  # zero-row senders must not be counted
    for shard in range(devices):
        got = expected_arrival_signals(
            jnp.asarray(accepted), jnp.int32(shard), local_experts=local_experts, grid_size=grid_size
        )
        for j in range(local_experts):
            want = 0
            for src in range(devices):
                rows = int(accepted[src, shard * local_experts + j])
                full, tail = divmod(rows, TRANSPORT_STAGE_ROWS)
                want += min(full, grid_size) + (1 if tail else 0)
            assert int(got[j]) == want, (shard, j)
