# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SPIKE S2 harness: two-level placement fidelity vs a single global scheduler.

Throwaway spike code. Imports the REAL Iris scheduler
(`iris.cluster.controller.scheduling.scheduler.Scheduler` + `run_preemption_pass`)
and drives it two ways over the *same* synthetic multi-backend workload:

  (a) GLOBAL: one Scheduler over every worker in the federation, with `backend`
      as a plain worker attribute. This is "today's scheduler with a backend tag".
  (b) TWO-LEVEL: a root meta-scheduler assigns task->backend from ONLY a
      per-backend `CapacitySummary` (it never sees a worker), then a per-backend
      Scheduler assigns task->worker. Jobs pin to one backend (design invariant).

Both modes call the identical real Scheduler for task->worker placement, so the
ONLY thing under test is the routing layer + the summary it routes from.

The simulator is multi-tick: jobs arrive over time, run for a duration, and free
their workers. We diff placement timing, utilization, queue depth, starvation,
preemptions, and priority inversion between (a) and (b), and ablate each summary
field to find which are load-bearing.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field, replace

from iris.cluster.constraints import (
    AttributeValue,
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    routing_constraints,
    split_hard_soft,
)
from iris.cluster.controller.scheduling.policy import PreemptionCandidate, run_preemption_pass
from iris.cluster.controller.scheduling.scheduler import (
    JobRequirements,
    RunningTaskInfo,
    Scheduler,
    SchedulingContext,
    WorkerSnapshot,
)
from iris.cluster.types import JobName, UserBudgetDefaults, WorkerId
from iris.rpc import job_pb2

PROD = job_pb2.PRIORITY_BAND_PRODUCTION  # 1 (highest)
INTER = job_pb2.PRIORITY_BAND_INTERACTIVE  # 2
BATCH = job_pb2.PRIORITY_BAND_BATCH  # 3

GIB = 1024**3
_UNLIMITED = 1 << 30  # disable per-cycle building/assignment caps; identical in both modes
GROUP_KEY = "slice-group"  # coscheduling group attribute (globally-unique slice/host id)


# ---------------------------------------------------------------------------
# Fleet + workload model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerSpec:
    """One worker (TPU VM / GPU node / CPU box) belonging to exactly one backend."""

    worker_id: str
    backend: str
    variant: str  # 'v5e-4' | 'h100' | 'cpu'
    cpu_millicores: int
    memory_bytes: int
    gpu_count: int
    tpu_count: int
    group: str | None  # coscheduling group key (tpu-name / gpu host); None for cpu
    region: str
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendDef:
    """A backend: a name, static config attributes, and its worker pool."""

    backend_id: str
    kind: str  # 'worker_daemon' | 'k8s'
    static: dict[str, set[str]]  # config-declared attributes (set-valued)
    workers: list[WorkerSpec]


@dataclass(frozen=True)
class JobSpec:
    """A workload job: one or more tasks placed atomically (gang) or independently."""

    job_id: JobName
    user: str
    band: int
    arrival: int
    duration: int
    num_tasks: int
    coscheduled: bool
    group_by: str | None
    variant: str  # 'v5e-4' | 'h100' | 'cpu'
    cpu_millicores: int
    memory_bytes: int
    gpu_count: int
    tpu_count: int
    constraints: tuple[Constraint, ...]
    pinned_backend: str | None  # explicit --backend X, else None

    def requirements(self) -> JobRequirements:
        return JobRequirements(
            req_cpu_millicores=self.cpu_millicores,
            req_memory_bytes=self.memory_bytes,
            req_gpu_count=self.gpu_count,
            req_tpu_count=self.tpu_count,
            device_variant=None if self.variant == "cpu" else self.variant,
            constraints=list(self.constraints),
            is_coscheduled=self.coscheduled,
            coscheduling_group_by=self.group_by,
        )

    def task_ids(self) -> list[JobName]:
        return [self.job_id.task(i) for i in range(self.num_tasks)]


# ---------------------------------------------------------------------------
# Worker -> scheduler projection
# ---------------------------------------------------------------------------


def _worker_attrs(w: WorkerSpec, *, include_backend: bool) -> dict[str, AttributeValue]:
    a: dict[str, AttributeValue] = {WellKnownAttribute.REGION: AttributeValue(w.region)}
    if w.variant == "cpu":
        a[WellKnownAttribute.DEVICE_TYPE] = AttributeValue("cpu")
    else:
        dt = "tpu" if w.tpu_count else "gpu"
        a[WellKnownAttribute.DEVICE_TYPE] = AttributeValue(dt)
        a[WellKnownAttribute.DEVICE_VARIANT] = AttributeValue(w.variant)
    if w.group is not None:
        a[GROUP_KEY] = AttributeValue(w.group)
        a[WellKnownAttribute.TPU_WORKER_ID] = AttributeValue(int(w.worker_id.rsplit("-", 1)[-1]))
    if include_backend:
        a["backend"] = AttributeValue(w.backend)
    for k, v in w.attrs.items():
        a[k] = AttributeValue(v)
    return a


def _snapshot(
    w: WorkerSpec, committed: tuple[int, int, int, int], *, include_backend: bool
) -> WorkerSnapshot:
    cpu, mem, gpu, tpu = committed
    return WorkerSnapshot(
        worker_id=WorkerId(w.worker_id),
        total_cpu_millicores=w.cpu_millicores,
        total_memory_bytes=w.memory_bytes,
        total_gpu_count=w.gpu_count,
        total_tpu_count=w.tpu_count,
        committed_cpu_millicores=cpu,
        committed_memory_bytes=mem,
        committed_gpu_count=gpu,
        committed_tpu_count=tpu,
        attributes=_worker_attrs(w, include_backend=include_backend),
    )


# ---------------------------------------------------------------------------
# Capacity summary (the thing the root routes from in two-level mode)
# ---------------------------------------------------------------------------


@dataclass
class CapacitySummary:
    """Per-backend summary the root sees. NO per-worker rows.

    Superset of spec.md s3.1's strawman plus two candidate extensions
    (`max_free_cpu_bin`, gang made per-variant) so ablations can prune.
    """

    backend_id: str
    static: dict[str, set[str]]
    # allocatable: free worker-slots per device-variant key (+ 'cpu' bucket).
    allocatable_units: dict[str, int]
    # CPU bin-packing signals (only meaningful for the 'cpu' bucket).
    free_cpu_millicores: int
    max_free_cpu_bin: int
    free_memory_bytes: int
    max_free_memory_bin: int
    # largest coschedulable gang currently placeable (FREE), per device-variant.
    largest_gang: dict[str, int]
    # structural max group size per device-variant (config-derivable; ignores busy).
    # Routing needs this to avoid pinning a gang to a backend that can NEVER host it.
    max_gang: dict[str, int]
    # agent-reported attempts launched-but-unobserved (double-count guard candidate).
    pending_leases: int
    stale_ms: int
    backoff: dict[str, int]  # scale-group -> cooldown-until-ms (0 == not backed off)


def compute_summary(
    backend: BackendDef,
    committed: dict[str, tuple[int, int, int, int]],
    *,
    stale_ms: int = 0,
    pending_leases: int = 0,
    backoff: dict[str, int] | None = None,
) -> CapacitySummary:
    """Roll a backend's live worker state into a CapacitySummary (no worker rows leak)."""
    units: dict[str, int] = defaultdict(int)
    free_cpu_total = 0
    free_mem_total = 0
    max_cpu_bin = 0
    max_mem_bin = 0
    # free workers per (variant, group) for the gang signal
    free_in_group: dict[tuple[str, str], int] = defaultdict(int)
    # total workers per (variant, group) for the structural max-gang signal
    total_in_group: dict[tuple[str, str], int] = defaultdict(int)

    for w in backend.workers:
        if w.variant != "cpu" and w.group is not None:
            total_in_group[(w.variant, w.group)] += 1
    for w in backend.workers:
        cpu, mem, gpu, tpu = committed.get(w.worker_id, (0, 0, 0, 0))
        free_cpu = w.cpu_millicores - cpu
        free_mem = w.memory_bytes - mem
        # CPU/mem are fungible: a CPU job can pack onto ANY worker's slack, so the
        # cpu bucket aggregates over the whole backend (incl. accelerator boxes).
        if free_cpu > 0:
            units["cpu"] += 1
        free_cpu_total += max(0, free_cpu)
        free_mem_total += max(0, free_mem)
        max_cpu_bin = max(max_cpu_bin, free_cpu)
        max_mem_bin = max(max_mem_bin, free_mem)
        if w.variant != "cpu":
            busy = (gpu > 0) or (tpu > 0)
            if not busy:
                units[w.variant] += 1
                if w.group is not None:
                    free_in_group[(w.variant, w.group)] += 1

    largest_gang: dict[str, int] = defaultdict(int)
    for (variant, _group), n in free_in_group.items():
        largest_gang[variant] = max(largest_gang[variant], n)
    max_gang: dict[str, int] = defaultdict(int)
    for (variant, _group), n in total_in_group.items():
        max_gang[variant] = max(max_gang[variant], n)

    return CapacitySummary(
        backend_id=backend.backend_id,
        static=backend.static,
        allocatable_units=dict(units),
        free_cpu_millicores=free_cpu_total,
        max_free_cpu_bin=max_cpu_bin,
        free_memory_bytes=free_mem_total,
        max_free_memory_bin=max_mem_bin,
        largest_gang=dict(largest_gang),
        max_gang=dict(max_gang),
        pending_leases=pending_leases,
        stale_ms=stale_ms,
        backoff=backoff or {},
    )


# ---------------------------------------------------------------------------
# Root meta-scheduler (task -> backend) from summaries only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterConfig:
    """Ablation switches for which summary fields the root is allowed to use."""

    use_largest_gang: bool = True  # dynamic free-gang signal (balance / place-now)
    use_max_gang_structural: bool = True  # static structural gang capacity (never-fits guard)
    use_max_cpu_bin: bool = True
    use_in_tick_decrement: bool = True  # don't double-count within one routing tick
    use_root_ledger: bool = True  # subtract own outstanding routes (cross-tick)
    use_pending_leases: bool = False  # alternative to root_ledger (agent-reported)
    use_stale_discount: bool = True  # treat stale backends as zero capacity
    use_backoff: bool = True  # avoid full+backed-off backends
    stale_threshold_ms: int = 30_000


def _backend_matches(hard_routing: list[Constraint], static: dict[str, set[str]]) -> bool:
    """Set-valued constraint match against a backend's static attributes.

    Faithful to spec s3.1's intent (`EQ` == set membership). Only routing
    constraints are evaluated here; worker-targeting constraints (tpu-name,
    tpu-worker-id, gpu-count, ...) are deferred to the per-backend agent.
    """
    for c in hard_routing:
        vals = static.get(c.key)
        present = bool(vals)
        wanted = {str(v.value) for v in c.values}
        if c.op == ConstraintOp.EXISTS:
            if not present:
                return False
        elif c.op == ConstraintOp.NOT_EXISTS:
            if present:
                return False
        elif c.op == ConstraintOp.EQ:
            if not present or next(iter(wanted)) not in vals:
                return False
        elif c.op == ConstraintOp.IN:
            if not present or wanted.isdisjoint(vals):
                return False
        elif c.op == ConstraintOp.NE:
            if present and vals == wanted:
                return False
        else:
            # ordered ops never appear on routing keys; ignore defensively.
            continue
    return True


class RootRouter:
    """Stateful task->backend router driven only by CapacitySummary.

    Holds the root's authoritative ledger of outstanding routes (per backend),
    which is what makes `pending_leases` redundant: the root authorizes every
    launch, so it already knows its own in-flight.
    """

    def __init__(self, backends: list[BackendDef], cfg: RouterConfig):
        self.backends = {b.backend_id: b for b in backends}
        self.cfg = cfg
        # backend -> shape -> count of routed-but-not-yet-observed-running tasks
        self.outstanding: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # backend -> total queued (for least-queue fallback)
        self.queued: dict[str, int] = defaultdict(int)

    def prime(self, outstanding: dict[str, dict[str, int]], queued: dict[str, int]) -> None:
        """Reset the per-tick ledgers from the root's authoritative ground truth.

        `outstanding[backend][shape]` = tasks the root routed to this backend that
        it has NOT yet observed running (queue backlog + launch window). `queued`
        = total pending tasks pinned per backend. Recomputed every tick from the
        root DB, so the ledger is never stale relative to the root's own actions.
        """
        self.outstanding = defaultdict(lambda: defaultdict(int))
        for b, shapes in outstanding.items():
            for s, n in shapes.items():
                self.outstanding[b][s] = n
        self.queued = defaultdict(int)
        for b, n in queued.items():
            self.queued[b] = n

    def _effective_units(self, summ: CapacitySummary, shape: str, tick_used: dict[str, int]) -> int:
        base = summ.allocatable_units.get(shape, 0)
        if self.cfg.use_stale_discount and summ.stale_ms > self.cfg.stale_threshold_ms:
            base = 0
        outstanding = 0
        if self.cfg.use_root_ledger:
            outstanding += self.outstanding[summ.backend_id].get(shape, 0)
        if self.cfg.use_pending_leases:
            outstanding += summ.pending_leases
        if self.cfg.use_in_tick_decrement:
            outstanding += tick_used.get(f"{summ.backend_id}:{shape}", 0)
        return base - outstanding

    def _effective_gang(self, summ: CapacitySummary, variant: str, tick_used: dict[str, int]) -> int:
        # With the dynamic signal: free coschedulable slots now. Without it: fall
        # back to the structural max group size, so any structurally-capable backend
        # reads as "place-now" (no balancing — gangs pile onto the first one).
        base = summ.largest_gang.get(variant, 0) if self.cfg.use_largest_gang else summ.max_gang.get(variant, 0)
        if self.cfg.use_stale_discount and summ.stale_ms > self.cfg.stale_threshold_ms:
            base = 0
        used = tick_used.get(f"{summ.backend_id}:gang:{variant}", 0)
        return base - used

    def route(
        self,
        job: JobSpec,
        summaries: dict[str, CapacitySummary],
        tick_used: dict[str, int],
    ) -> str | None:
        """Choose a backend for `job`. Returns backend_id, or None if UNSCHEDULABLE.

        Mirrors route_demand's structure: filter by routing constraints over
        static attrs, then pick among eligible by live capacity (most-free),
        falling back to least-queued when nothing has capacity right now.
        """
        shape = job.variant
        hard_routing, _soft = split_hard_soft(routing_constraints(list(job.constraints)))
        eligible = [s for s in summaries.values() if _backend_matches(hard_routing, s.static)]
        if job.pinned_backend is not None:
            eligible = [s for s in eligible if s.backend_id == job.pinned_backend]
        if not eligible:
            return None  # UNSCHEDULABLE: no static match

        # Structural gang guard: a gang can never land on a backend whose biggest
        # group is smaller than the gang, no matter how idle. Filter those out so
        # the queue-until-capacity fallback can't pin a gang to a black hole.
        if job.coscheduled and self.cfg.use_max_gang_structural:
            eligible = [s for s in eligible if s.max_gang.get(shape, 0) >= job.num_tasks]
            if not eligible:
                return None  # UNSCHEDULABLE: no backend structurally hosts this gang

        # Backends with live capacity for this exact task right now.
        placeable: list[tuple[int, CapacitySummary]] = []
        for s in eligible:
            if self.cfg.use_backoff and shape != "cpu":
                # a full + backed-off backend cannot grow; skip if no live units
                if all(v > 0 for v in s.backoff.values()) and self._effective_units(s, shape, tick_used) <= 0:
                    continue
            if job.coscheduled:
                room = self._effective_gang(s, shape, tick_used)
                if room >= job.num_tasks:
                    placeable.append((room, s))
            elif shape == "cpu":
                fits_bin = (not self.cfg.use_max_cpu_bin) or s.max_free_cpu_bin >= job.cpu_millicores
                if fits_bin and s.free_cpu_millicores >= job.cpu_millicores:
                    placeable.append((s.free_cpu_millicores, s))
            else:
                room = self._effective_units(s, shape, tick_used)
                if room >= 1:
                    placeable.append((room, s))

        if placeable:
            # most-free first; deterministic tie-break by backend_id
            placeable.sort(key=lambda t: (-t[0], t[1].backend_id))
            chosen = placeable[0][1].backend_id
        else:
            # nothing free now: pin to the least-queued eligible backend (queue + autoscale).
            eligible.sort(key=lambda s: (self.queued[s.backend_id], s.backend_id))
            chosen = eligible[0].backend_id

        # In-tick decrement so the next task in this same routing pass sees the
        # capacity we just consumed. Cross-tick backlog lives in `outstanding`
        # (primed from truth); this is the orthogonal within-tick guard.
        if job.coscheduled:
            tick_used[f"{chosen}:gang:{shape}"] = tick_used.get(f"{chosen}:gang:{shape}", 0) + job.num_tasks
            tick_used[f"{chosen}:{shape}"] = tick_used.get(f"{chosen}:{shape}", 0) + job.num_tasks
        else:
            tick_used[f"{chosen}:{shape}"] = tick_used.get(f"{chosen}:{shape}", 0) + 1
        self.queued[chosen] += job.num_tasks
        return chosen


# ---------------------------------------------------------------------------
# Placement (the REAL scheduler), shared by both modes
# ---------------------------------------------------------------------------


def _band_value(job: JobSpec) -> int:
    return job.band


def schedule_and_preempt(
    workers: list[WorkerSpec],
    committed: dict[str, tuple[int, int, int, int]],
    pending_task_ids: list[JobName],
    jobs: dict[JobName, JobSpec],
    running_infos: list[RunningTaskInfo],
    scheduler: Scheduler,
    *,
    include_backend: bool,
) -> tuple[list[tuple[JobName, WorkerId]], list[tuple[JobName, JobName]]]:
    """Run the real Scheduler.find_assignments + one preemption round.

    Returns (assignments, preemptions). `pending_task_ids` is already in the
    caller's priority order. Identical code path in both modes.
    """
    if not workers:
        return [], []
    job_reqs: dict[JobName, JobRequirements] = {}
    for jid, js in jobs.items():
        req = js.requirements()
        if not include_backend:
            # `--backend X` is a routing directive consumed by the root; the agent's
            # workers do not advertise `backend`, so it must be stripped before the
            # per-backend Scheduler places (else a pinned task matches no worker).
            req = replace(req, constraints=[c for c in req.constraints if c.key != "backend"])
        job_reqs[jid] = req
    snaps = [_snapshot(w, committed.get(w.worker_id, (0, 0, 0, 0)), include_backend=include_backend) for w in workers]
    ctx = SchedulingContext(
        workers=snaps,
        building_counts={},
        max_building_tasks=_UNLIMITED,
        max_assignments_per_worker=_UNLIMITED,
        pending_tasks=pending_task_ids,
        jobs=job_reqs,
        pending_task_rows=[],
        user_spend={},
        user_budget_limits={},
        requested_bands={},
        user_budget_defaults=UserBudgetDefaults(),
        running_for_preemption=running_infos,
    )
    result = scheduler.find_assignments(ctx)
    assigned = {t for t, _ in result.assignments}

    # Preemption: unplaced higher-band tasks may evict running lower-band work.
    band_map = {t: _band_value(jobs[t.parent]) for t in pending_task_ids if t.parent in jobs}
    candidates = [
        PreemptionCandidate(job_name=t, requirements=job_reqs[t.parent], band=band_map[t])
        for t in pending_task_ids
        if t not in assigned and t.parent in job_reqs
    ]
    preemptions = run_preemption_pass(candidates, running_infos, ctx) if candidates else []
    return result.assignments, preemptions
