# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Segment-plan metadata for the M6 transport kernel, validated on CPU.

The Mosaic-GPU kernel itself needs hardware, but its metadata (which rows
go where) is pure index math. These tests execute the plans with NumPy
copies and check (a) the dispatch plan reproduces the correctness
simulator's receive pools exactly, and (b) dispatch followed by combine is
the identity on each device's compacted send buffer.
"""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.marin_ep.kernels.mgpu_transport import combine_segments, dispatch_segments
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


def _execute_plan(plans, sources, out_rows):
    """NumPy interpreter for SegmentPlan: what the GPU kernel would copy."""
    outputs = [np.zeros((out_rows, sources[0].shape[1]), sources[0].dtype) for _ in range(DEVICES)]
    for device, plan in enumerate(plans):
        dest_ids = np.asarray(plan.dest_ids)
        entry_start = np.asarray(plan.entry_start)
        src_lo, dst_lo, rows = (np.asarray(a) for a in (plan.src_lo, plan.dst_lo, plan.rows))
        for k, dest in enumerate(dest_ids):
            for e in range(entry_start[k], entry_start[k + 1]):
                n = rows[e]
                outputs[dest][dst_lo[e] : dst_lo[e] + n] = sources[device][src_lo[e] : src_lo[e] + n]
    return outputs


def test_dispatch_plan_reproduces_simulator_pools():
    x, experts, result = _instance()
    routing = result.saved.routing
    accepted = jnp.asarray(_accepted(routing))
    region = jnp.asarray(result.saved.region)

    sends = [_compacted_send_buffer(x, experts, routing, d) for d in range(DEVICES)]
    plans = [dispatch_segments(accepted, region, jnp.int32(d), local_experts=LOCAL_EXPERTS) for d in range(DEVICES)]
    pool_rows = LOCAL_EXPERTS * routing.capacity
    pools = _execute_plan(plans, sends, pool_rows)
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
    pools = _execute_plan(dispatch_plans, sends, LOCAL_EXPERTS * routing.capacity)

    combine_plans = [
        combine_segments(accepted, region, jnp.int32(d), local_experts=LOCAL_EXPERTS) for d in range(DEVICES)
    ]
    kept_per_device = int(np.asarray(accepted).sum(axis=1).max())
    returned = _execute_plan(combine_plans, pools, kept_per_device)
    for d in range(DEVICES):
        np.testing.assert_array_equal(returned[d][: sends[d].shape[0]], sends[d])
