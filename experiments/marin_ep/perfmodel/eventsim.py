# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""L1 performance model: discrete-event simulation of one Marin EP MoE layer.

Second rung of the simulator ladder. The layer is compiled into a DAG of
work items over explicit resources — per-device egress/ingress NVLink
capacity and per-device compute — and scheduled with a deterministic
greedy list scheduler. Unlike the L0 roofline this captures tile-granular
pipelining (expert GEMM tiles start as their input rows arrive), incast
(one hot owner's ingress link serializing), and skewed routing (from a
real `[S, E]` count matrix, e.g. `simcore.Routing.counts`).

Two program shapes share the same work items:

- `pipelined=True`: the Marin EP design — GEMM tiles depend only on the
  dispatch tiles covering their rows; combine tiles depend only on their
  GEMM tile.
- `pipelined=False`: bulk-synchronous baseline — full barriers between
  dispatch, GEMM, and combine phases, approximating today's XLA
  all-to-all schedule shape.

Fidelity notes (documented deviations from reality, revisited when
hardware calibration disagrees): no NVLink switch contention beyond
per-device ingress/egress serialization; no HBM bandwidth modeling; local
(same-device) dispatch/combine cost only per-message latency; compute is
one serial resource per device (no SM-level split between transport and
MMA warps yet).
"""

import bisect
import heapq
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from experiments.marin_ep.oracle import expert_capacity, kept_rows_per_expert
from experiments.marin_ep.perfmodel.roofline import GB200, HERO, Hardware


@dataclass(frozen=True)
class WorkItem:
    name: str
    resources: tuple[str, ...]  # all are occupied for the full duration
    duration: float
    deps: tuple[int, ...] = ()


@dataclass
class Schedule:
    start: list[float]
    finish: list[float]
    makespan: float
    busy: dict[str, float] = field(default_factory=dict)

    def utilization(self, resource: str) -> float:
        if self.makespan == 0:
            return 0.0
        return self.busy.get(resource, 0.0) / self.makespan


def _earliest_gap(busy_intervals: list[tuple[float, float]], ready: float, duration: float) -> float:
    """Earliest start >= ready such that [start, start+duration) misses all busy intervals."""
    t = ready
    for lo, hi in busy_intervals:
        if hi <= t:
            continue
        if lo >= t + duration:
            break
        t = hi
    return t


def _insert_interval(busy_intervals: list[tuple[float, float]], lo: float, hi: float) -> None:
    index = bisect.bisect_left(busy_intervals, (lo, hi))
    busy_intervals.insert(index, (lo, hi))


def run_schedule(items: list[WorkItem]) -> Schedule:
    """Deterministic list scheduling with backfill.

    Items become ready when all deps finish; the queue pops by (ready time,
    item index). An item is placed at the earliest instant at or after its
    ready time when *all* its resources have a gap of its duration —
    including gaps before previously placed items. Backfill matters: a
    strict-FIFO resource model produces head-of-line convoys (measured 3x
    on the hero all-to-all pattern) that packet-interleaved NVLink fabrics
    do not exhibit; single-occupancy intervals still charge bandwidth.
    """
    n = len(items)
    indegree = [len(item.deps) for item in items]
    dependents: list[list[int]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        for d in item.deps:
            if not 0 <= d < n:
                raise ValueError(f"item {i} ({item.name}) has out-of-range dep {d}")
            if d >= i:
                raise ValueError(f"item {i} ({item.name}) depends on later item {d}; items must be in topo order")
            dependents[d].append(i)

    ready_time = [0.0] * n
    start = [0.0] * n
    finish = [0.0] * n
    resource_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    busy: dict[str, float] = defaultdict(float)

    queue: list[tuple[float, int]] = [(0.0, i) for i in range(n) if indegree[i] == 0]
    heapq.heapify(queue)
    scheduled = 0
    while queue:
        ready, i = heapq.heappop(queue)
        item = items[i]
        begin = ready
        while True:
            proposal = begin
            for r in item.resources:
                proposal = max(proposal, _earliest_gap(resource_intervals[r], proposal, item.duration))
            if proposal == begin:
                break
            begin = proposal
        start[i], finish[i] = begin, begin + item.duration
        for r in item.resources:
            _insert_interval(resource_intervals[r], begin, finish[i])
            busy[r] += item.duration
        scheduled += 1
        for j in dependents[i]:
            ready_time[j] = max(ready_time[j], finish[i])
            indegree[j] -= 1
            if indegree[j] == 0:
                heapq.heappush(queue, (ready_time[j], j))
    if scheduled != n:
        raise ValueError("dependency cycle: not all items were scheduled")
    return Schedule(start=start, finish=finish, makespan=max(finish, default=0.0), busy=dict(busy))


@dataclass(frozen=True)
class LayerProgramParams:
    """Shape + hardware knobs for building a layer program from a count matrix."""

    hidden: int
    intermediate: int
    local_experts: int
    capacity: int  # per-expert base capacity C (SPEC Section 2)
    pool_group_size: int = 1
    wire_bytes: int = 2
    tile_rows: int = 4096  # transfer/GEMM chunk granularity
    message_latency: float = 2e-6  # per put: launch + flag propagation
    gemm_tile_overhead: float = 1e-6


def _clipped_rows(counts: np.ndarray, capacity: int, group_size: int) -> np.ndarray:
    """Kept rows per (source, expert) under the group-pooled capacity rule."""
    kept = kept_rows_per_expert(counts.sum(axis=0), capacity=capacity, group_size=group_size)
    base = np.cumsum(counts, axis=0) - counts
    return np.clip(np.minimum(counts, kept - base), 0, None)


def build_layer_program(
    counts: np.ndarray,
    params: LayerProgramParams,
    hw: Hardware,
    *,
    backward: bool,
    pipelined: bool,
) -> list[WorkItem]:
    """Compile one layer direction into scheduler work items.

    `counts[d, g]` is the number of assignments device `d` routes to expert
    `g` (`simcore.Routing.counts`, or synthetic). Transport per kept row is
    `hidden * wire_bytes` in each direction for both forward and backward
    (SPEC Sections 3-4); GEMM FLOPs per row are 6*H*I forward, 12*H*I
    backward.
    """
    num_devices, num_experts = counts.shape
    if num_experts != num_devices * params.local_experts:
        raise ValueError(f"counts shape {counts.shape} inconsistent with local_experts={params.local_experts}")
    link = hw.link_bytes * hw.link_efficiency
    flops_rate = hw.matmul_flops * hw.gemm_efficiency
    flops_per_row = (12 if backward else 6) * params.hidden * params.intermediate
    bytes_per_row = params.hidden * params.wire_bytes
    rows_kept = _clipped_rows(counts, params.capacity, params.pool_group_size)
    offsets = np.cumsum(rows_kept, axis=0) - rows_kept  # arrival offset of each source's block

    items: list[WorkItem] = []

    # Count exchange: one aggregated item per source device, then a barrier
    # (dispatch offsets need the full count matrix, so every dispatch item
    # waits on all counts; a zero-duration barrier keeps the edge count linear).
    count_ids = []
    for d in range(num_devices):
        count_ids.append(len(items))
        items.append(
            WorkItem(
                name=f"counts[{d}]",
                resources=(f"link_out[{d}]",),
                duration=(num_devices - 1) * num_experts * 4 / link + params.message_latency,
            )
        )
    count_barrier = len(items)
    items.append(WorkItem(name="count_barrier", resources=(), duration=0.0, deps=tuple(count_ids)))

    def rotated_experts(sender: int):
        """Experts ordered by owner starting after `sender`.

        Every source targeting owner 0 first serializes on owner 0's ingress
        (a convoy the event sim exposed: ~2x dispatch drain time); rotating
        the send order per source spreads ingress load, like a rotated
        all-to-all. The kernel must implement the same ordering.
        """
        for step in range(num_devices):
            owner = (sender + 1 + step) % num_devices
            for local_g in range(params.local_experts):
                yield owner * params.local_experts + local_g

    # Dispatch tiles, and for each expert the (tile item id, row range) pairs.
    arrivals: dict[int, list[tuple[int, int, int]]] = defaultdict(list)  # g -> [(item, row_lo, row_hi)]
    dispatch_ids = []
    for d in range(num_devices):
        for g in rotated_experts(d):
            rows = int(rows_kept[d, g])
            if rows == 0:
                continue
            owner = g // params.local_experts
            row_lo = int(offsets[d, g])
            for tile_lo in range(0, rows, params.tile_rows):
                tile_rows = min(params.tile_rows, rows - tile_lo)
                remote = (f"link_out[{d}]", f"link_in[{owner}]") if owner != d else ()
                duration = (tile_rows * bytes_per_row / link if remote else 0.0) + params.message_latency
                item_id = len(items)
                items.append(
                    WorkItem(
                        name=f"dispatch[d{d}->g{g}]",
                        resources=remote,
                        duration=duration,
                        deps=(count_barrier,),
                    )
                )
                arrivals[g].append((item_id, row_lo + tile_lo, row_lo + tile_lo + tile_rows))
                dispatch_ids.append(item_id)

    # GEMM tiles per expert; each depends on the dispatch tiles overlapping
    # its row range (pipelined) or on a full dispatch barrier (bulk).
    dispatch_barrier = len(items)
    items.append(WorkItem(name="dispatch_barrier", resources=(), duration=0.0, deps=tuple(dispatch_ids)))
    gemm_cover: dict[int, list[tuple[int, int, int]]] = defaultdict(list)  # g -> [(item, row_lo, row_hi)]
    gemm_ids = []
    for g in range(num_experts):
        owner = g // params.local_experts
        total_rows = int(rows_kept[:, g].sum())
        for tile_lo in range(0, total_rows, params.tile_rows):
            tile_hi = min(tile_lo + params.tile_rows, total_rows)
            if pipelined:
                deps = tuple(item for item, lo, hi in arrivals[g] if lo < tile_hi and hi > tile_lo)
            else:
                deps = (dispatch_barrier,)
            item_id = len(items)
            items.append(
                WorkItem(
                    name=f"gemm[g{g}]",
                    resources=(f"compute[{owner}]",),
                    duration=(tile_hi - tile_lo) * flops_per_row / flops_rate + params.gemm_tile_overhead,
                    deps=deps,
                )
            )
            gemm_cover[g].append((item_id, tile_lo, tile_hi))
            gemm_ids.append(item_id)

    # Combine tiles: expert output rows return to their source device, in
    # rotated destination order per owner (same convoy-avoidance as dispatch).
    gemm_barrier = len(items)
    items.append(WorkItem(name="gemm_barrier", resources=(), duration=0.0, deps=tuple(gemm_ids)))
    for owner in range(num_devices):
        for step in range(num_devices):
            d = (owner + 1 + step) % num_devices
            for local_g in range(params.local_experts):
                g = owner * params.local_experts + local_g
                rows = int(rows_kept[d, g])
                if rows == 0:
                    continue
                row_lo = int(offsets[d, g])
                for tile_lo in range(0, rows, params.tile_rows):
                    tile_rows = min(params.tile_rows, rows - tile_lo)
                    lo, hi = row_lo + tile_lo, row_lo + tile_lo + tile_rows
                    if pipelined:
                        deps = tuple(item for item, glo, ghi in gemm_cover[g] if glo < hi and ghi > lo)
                    else:
                        deps = (gemm_barrier,)
                    remote = (f"link_out[{owner}]", f"link_in[{d}]") if owner != d else ()
                    duration = (tile_rows * bytes_per_row / link if remote else 0.0) + params.message_latency
                    items.append(WorkItem(name=f"combine[g{g}->d{d}]", resources=remote, duration=duration, deps=deps))
    return items


def balanced_counts(num_devices: int, num_experts: int, assignments_per_device: int) -> np.ndarray:
    """Near-balanced [S, E] count matrix; the remainder goes to the first experts."""
    per_expert, remainder = divmod(assignments_per_device, num_experts)
    row = np.full(num_experts, per_expert, dtype=np.int64)
    row[:remainder] += 1
    return np.tile(row, (num_devices, 1))


def estimate_layer_makespan(
    counts: np.ndarray,
    params: LayerProgramParams,
    hw: Hardware,
    *,
    backward: bool,
    pipelined: bool,
) -> Schedule:
    return run_schedule(build_layer_program(counts, params, hw, backward=backward, pipelined=pipelined))


def hero_l1_report(hw: Hardware | None = None) -> str:
    """L1 makespans at the hero shape, pipelined vs bulk, fwd and bwd."""
    hw = hw or GB200
    counts = balanced_counts(HERO.ep, HERO.num_experts, HERO.assignments_per_device)
    capacity = expert_capacity(HERO.assignments_per_device * HERO.ep, HERO.num_experts, HERO.capacity_factor)
    params = LayerProgramParams(
        hidden=HERO.hidden,
        intermediate=HERO.intermediate,
        local_experts=HERO.num_experts // HERO.ep,
        capacity=capacity,
        pool_group_size=HERO.num_experts // HERO.ep,  # hero G = El = 3 (SPEC S2)
        wire_bytes=HERO.wire_bytes,
        tile_rows=8192,
    )
    lines = ["Marin EP L1 event-sim at hero shape (balanced routing, per layer):"]
    for backward in (False, True):
        name = "bwd" if backward else "fwd"
        for pipelined in (True, False):
            sched = estimate_layer_makespan(counts, params, hw, backward=backward, pipelined=pipelined)
            label = "pipelined" if pipelined else "bulk"
            lines.append(f"  {name} {label}: {sched.makespan * 1e3:.2f} ms")
    return "\n".join(lines)
