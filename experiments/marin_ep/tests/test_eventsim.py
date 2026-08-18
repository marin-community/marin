# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""L1 event-sim behavior: scheduler contract, backfill, and program shapes."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.marin_ep.oracle import expert_capacity
from experiments.marin_ep.perfmodel.eventsim import (
    LayerProgramParams,
    WorkItem,
    balanced_counts,
    estimate_layer_makespan,
    run_schedule,
)
from experiments.marin_ep.perfmodel.roofline import GB200
from experiments.marin_ep.simcore import simulate_forward

SMALL_PARAMS = LayerProgramParams(
    hidden=64,
    intermediate=128,
    local_experts=2,
    capacity=10**9,  # dropless
    wire_bytes=2,
    tile_rows=64,
)


def _random_dag_items(rng, n=60):
    items = []
    resources = [f"r{i}" for i in range(4)]
    for i in range(n):
        deps = tuple(int(d) for d in rng.choice(i, size=min(i, int(rng.integers(0, 3))), replace=False))
        n_res = int(rng.integers(0, 3))
        res = tuple(rng.choice(resources, size=n_res, replace=False))
        items.append(WorkItem(name=f"w{i}", resources=res, duration=float(rng.random()), deps=deps))
    return items


def test_schedule_respects_dependencies_and_resource_exclusivity():
    rng = np.random.default_rng(seed=0)
    items = _random_dag_items(rng)
    sched = run_schedule(items)
    for i, item in enumerate(items):
        for d in item.deps:
            assert sched.finish[d] <= sched.start[i] + 1e-12
    # No resource runs two items at once.
    by_resource: dict[str, list[tuple[float, float]]] = {}
    for i, item in enumerate(items):
        for r in item.resources:
            by_resource.setdefault(r, []).append((sched.start[i], sched.finish[i]))
    for intervals in by_resource.values():
        intervals.sort()
        for (_, prev_hi), (lo, _) in itertools.pairwise(intervals):
            assert prev_hi <= lo + 1e-12


def test_makespan_bounded_by_busiest_resource_and_serial_sum():
    rng = np.random.default_rng(seed=1)
    items = _random_dag_items(rng)
    sched = run_schedule(items)
    assert sched.makespan >= max(sched.busy.values()) - 1e-12
    assert sched.makespan <= sum(item.duration for item in items) + 1e-12


def test_backfill_uses_idle_window_behind_committed_item():
    """A convoy regression: C shares only an *idle* resource with earlier
    items, so it must start immediately, not queue behind B's other-resource
    wait (the head-of-line artifact that inflated hero dispatch 3x)."""
    items = [
        WorkItem(name="A", resources=("out0", "in1"), duration=10.0),
        WorkItem(name="B", resources=("out2", "in1"), duration=5.0),
        WorkItem(name="C", resources=("out2", "in3"), duration=5.0),
    ]
    sched = run_schedule(items)
    assert sched.start[1] == 10.0  # B waits for in1 behind A
    assert sched.start[2] == 0.0  # C backfills out2's idle window


def test_pipelined_beats_bulk_on_balanced_routing():
    counts = balanced_counts(4, 8, 1024)
    pipelined = estimate_layer_makespan(counts, SMALL_PARAMS, GB200, backward=False, pipelined=True)
    bulk = estimate_layer_makespan(counts, SMALL_PARAMS, GB200, backward=False, pipelined=False)
    assert pipelined.makespan < bulk.makespan


def test_incast_routing_is_slower_than_balanced():
    """All tokens routed to one owner's experts must be ingress-bound and
    slower than the balanced program with identical totals."""
    devices, num_experts, per_device = 4, 8, 1024
    balanced = balanced_counts(devices, num_experts, per_device)
    incast = np.zeros_like(balanced)
    incast[:, : SMALL_PARAMS.local_experts] = per_device // SMALL_PARAMS.local_experts
    t_balanced = estimate_layer_makespan(balanced, SMALL_PARAMS, GB200, backward=False, pipelined=True)
    t_incast = estimate_layer_makespan(incast, SMALL_PARAMS, GB200, backward=False, pipelined=True)
    assert t_incast.makespan > t_balanced.makespan


def test_program_builds_from_simulated_routing_counts():
    """The [S, E] counts convention must match simcore.Routing.counts."""
    rng = np.random.default_rng(seed=5)
    devices, tokens, topk, num_experts, hidden, intermediate = 4, 16, 2, 8, 8, 12
    x = rng.standard_normal((devices, tokens, hidden), dtype=np.float32)
    experts = rng.integers(0, num_experts, size=(devices, tokens, topk)).astype(np.int32)
    weights = rng.random((devices, tokens, topk), dtype=np.float32)
    w13 = rng.standard_normal((num_experts, hidden, 2 * intermediate)).astype(np.float32)
    w2 = rng.standard_normal((num_experts, intermediate, hidden)).astype(np.float32)
    result = simulate_forward(
        jnp.asarray(x),
        experts,
        jnp.asarray(weights),
        jnp.asarray(w13).reshape(devices, 2, hidden, 2 * intermediate),
        jnp.asarray(w2).reshape(devices, 2, intermediate, hidden),
        num_experts=num_experts,
        capacity_factor=1.25,
        activation_fn=jax.nn.silu,
    )
    capacity = expert_capacity(devices * tokens * topk, num_experts, 1.25)
    params = LayerProgramParams(
        hidden=hidden, intermediate=intermediate, local_experts=2, capacity=capacity, wire_bytes=4, tile_rows=16
    )
    sched = estimate_layer_makespan(result.saved.routing.counts, params, GB200, backward=False, pipelined=True)
    # Kept rows in the program equal the simulator's kept assignments: the
    # schedule's total compute busy time must be exactly rows * per-row work
    # plus per-tile overhead.
    kept_rows = int(np.asarray(result.saved.routing.keep).sum())
    rows_per_expert = np.minimum(result.saved.routing.counts.sum(axis=0), capacity)
    assert kept_rows == int(rows_per_expert.sum())
    num_tiles = int(np.ceil(rows_per_expert / params.tile_rows).sum())
    expected_busy = (
        kept_rows * 6 * hidden * intermediate / (GB200.matmul_flops * GB200.gemm_efficiency)
        + num_tiles * params.gemm_tile_overhead
    )
    total_compute_busy = sum(v for k, v in sched.busy.items() if k.startswith("compute"))
    assert total_compute_busy == pytest.approx(expected_busy, rel=1e-9)
