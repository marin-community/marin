# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-tick simulator that drives the REAL Iris scheduler in two modes.

GLOBAL: one Scheduler over the whole federation (`backend` is a worker attribute).
TWO-LEVEL: RootRouter (task->backend, summary-only) + per-backend Scheduler.

Both call the identical `schedule_and_preempt` from harness.py for task->worker,
so only the routing layer differs. See harness.py for the model + summary + router.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from iris.cluster.controller.budget import resource_value
from iris.cluster.controller.scheduling.scheduler import RunningTaskInfo, Scheduler
from iris.cluster.types import JobName, WorkerId

from harness import (
    BackendDef,
    CapacitySummary,
    JobSpec,
    RootRouter,
    RouterConfig,
    WorkerSpec,
    compute_summary,
    schedule_and_preempt,
)


@dataclass
class RunRecord:
    task_id: JobName
    job_id: JobName
    worker_id: str
    backend: str
    end_tick: int


@dataclass
class Metrics:
    label: str
    horizon: int
    total_tasks: int
    placed_tasks: int  # tasks placed at least once by the horizon
    never_placed: int  # starvation
    waits: list[int] = field(default_factory=list)  # ticks from arrival to first placement
    preemptions: int = 0
    inversions: int = 0  # higher-band task waited while a lower-band same-shape task ran
    util_by_backend: dict[str, float] = field(default_factory=dict)
    util_global: float = 0.0
    mean_pending: float = 0.0
    max_pending: int = 0

    def wait_stats(self) -> tuple[float, int, int, int]:
        if not self.waits:
            return (0.0, 0, 0, 0)
        s = sorted(self.waits)
        p50 = s[len(s) // 2]
        p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
        return (statistics.mean(s), p50, p95, s[-1])

    def summary_line(self) -> str:
        mean, p50, p95, mx = self.wait_stats()
        return (
            f"{self.label:<26} placed={self.placed_tasks}/{self.total_tasks} "
            f"starved={self.never_placed} wait(mean/p50/p95/max)={mean:.2f}/{p50}/{p95}/{mx} "
            f"util={self.util_global:.1%} preempt={self.preemptions} inversions={self.inversions} "
            f"pend(mean/max)={self.mean_pending:.1f}/{self.max_pending}"
        )


def _committed(running: dict[JobName, RunRecord], jobs: dict[JobName, JobSpec]) -> dict[str, tuple[int, int, int, int]]:
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for rec in running.values():
        js = jobs[rec.job_id]
        c = out[rec.worker_id]
        c[0] += js.cpu_millicores
        c[1] += js.memory_bytes
        c[2] += js.gpu_count
        c[3] += js.tpu_count
    return {w: (v[0], v[1], v[2], v[3]) for w, v in out.items()}


def _running_infos(running: dict[JobName, RunRecord], jobs: dict[JobName, JobSpec]) -> list[RunningTaskInfo]:
    infos: list[RunningTaskInfo] = []
    for rec in running.values():
        js = jobs[rec.job_id]
        infos.append(
            RunningTaskInfo(
                task_id=rec.task_id,
                worker_id=WorkerId(rec.worker_id),
                band_sort_key=js.band,
                resource_value=resource_value(js.cpu_millicores, js.memory_bytes, js.gpu_count + js.tpu_count),
                is_coscheduled=js.coscheduled,
                cpu_millicores=js.cpu_millicores,
                memory_bytes=js.memory_bytes,
                gpu_count=js.gpu_count,
                tpu_count=js.tpu_count,
                device_variant=None if js.variant == "cpu" else js.variant,
            )
        )
    return infos


def _priority_key(task_id: JobName, jobs: dict[JobName, JobSpec]) -> tuple:
    js = jobs[task_id.parent]
    return (js.band, js.arrival, task_id.parent.to_wire(), task_id.task_index or 0)


class Simulator:
    """Holds the fleet + workload and replays it under either scheduling mode."""

    def __init__(self, backends: list[BackendDef], jobs: list[JobSpec], horizon: int):
        self.backends = backends
        self.jobs = {j.job_id: j for j in jobs}
        self.job_list = jobs
        self.horizon = horizon
        self.all_workers: list[WorkerSpec] = [w for b in backends for w in b.workers]
        self.workers_by_backend: dict[str, list[WorkerSpec]] = {b.backend_id: b.workers for b in backends}
        self.arrivals_by_tick: dict[int, list[JobSpec]] = defaultdict(list)
        for j in jobs:
            self.arrivals_by_tick[j.arrival].append(j)

    # -- shared bookkeeping -------------------------------------------------

    def _new_state(self):
        return {
            "running": {},  # task_id -> RunRecord
            "pending": set(),  # task_ids not running
            "arrival": {},  # task_id -> tick
            "placed_tick": {},  # task_id -> first placement tick
            "preempt": defaultdict(int),  # task_id -> count
            "pin": {},  # job_id -> backend (two-level)
            "preemptions": 0,
            "util_busy": defaultdict(int),  # worker_id -> busy ticks
            "pending_samples": [],
            "inversions": 0,
        }

    def _release(self, st, now: int) -> None:
        done = [tid for tid, rec in st["running"].items() if rec.end_tick == now]
        for tid in done:
            del st["running"][tid]

    def _admit_arrivals(self, st, now: int) -> None:
        for j in self.arrivals_by_tick.get(now, []):
            for tid in j.task_ids():
                st["pending"].add(tid)
                st["arrival"][tid] = now

    def _apply_assignments(self, st, assignments, now: int) -> None:
        for tid, wid in assignments:
            if tid not in st["pending"]:
                continue
            js = self.jobs[tid.parent]
            backend = next(b for b in self.backends if any(w.worker_id == str(wid) for w in b.workers)).backend_id
            st["running"][tid] = RunRecord(tid, tid.parent, str(wid), backend, now + js.duration)
            st["pending"].discard(tid)
            if tid not in st["placed_tick"]:
                st["placed_tick"][tid] = now

    def _apply_preemptions(self, st, preemptions) -> set[JobName]:
        evicted: set[JobName] = set()
        for _preemptor, victim in preemptions:
            if victim in st["running"]:
                del st["running"][victim]
                st["pending"].add(victim)
                st["preempt"][victim] += 1
                st["preemptions"] += 1
                evicted.add(victim)
        return evicted

    def _count_busy(self, st) -> None:
        for rec in st["running"].values():
            st["util_busy"][rec.worker_id] += 1
        st["pending_samples"].append(len(st["pending"]))

    def _count_inversions(self, st) -> None:
        """Higher-band task pending while a strictly-lower-band same-shape task runs."""
        pending_shapes_band: dict[str, int] = {}
        for tid in st["pending"]:
            js = self.jobs[tid.parent]
            sh = js.variant
            pending_shapes_band[sh] = min(pending_shapes_band.get(sh, 99), js.band)
        running_shape_band: dict[str, int] = {}
        for rec in st["running"].values():
            js = self.jobs[rec.job_id]
            sh = js.variant
            running_shape_band[sh] = max(running_shape_band.get(sh, 0), js.band)
        for sh, pend_band in pending_shapes_band.items():
            run_band = running_shape_band.get(sh, 0)
            if run_band > pend_band:  # a lower-priority task of the same shape is running
                st["inversions"] += 1

    def _finalize(self, st, label: str) -> Metrics:
        all_tids = [tid for j in self.job_list for tid in j.task_ids()]
        placed = sum(1 for tid in all_tids if tid in st["placed_tick"])
        never = len(all_tids) - placed
        waits = [st["placed_tick"][tid] - st["arrival"][tid] for tid in all_tids if tid in st["placed_tick"]]
        util_by_backend: dict[str, float] = {}
        total_busy = 0
        total_cap = 0
        for b in self.backends:
            busy = sum(st["util_busy"][w.worker_id] for w in b.workers)
            cap = len(b.workers) * self.horizon
            util_by_backend[b.backend_id] = busy / cap if cap else 0.0
            total_busy += busy
            total_cap += cap
        return Metrics(
            label=label,
            horizon=self.horizon,
            total_tasks=len(all_tids),
            placed_tasks=placed,
            never_placed=never,
            waits=waits,
            preemptions=st["preemptions"],
            inversions=st["inversions"],
            util_by_backend=util_by_backend,
            util_global=total_busy / total_cap if total_cap else 0.0,
            mean_pending=statistics.mean(st["pending_samples"]) if st["pending_samples"] else 0.0,
            max_pending=max(st["pending_samples"]) if st["pending_samples"] else 0,
        )

    # -- GLOBAL mode --------------------------------------------------------

    def run_global(self, scheduler: Scheduler, label: str = "GLOBAL") -> Metrics:
        st = self._new_state()
        for now in range(self.horizon):
            self._release(st, now)
            self._admit_arrivals(st, now)
            self._count_inversions(st)

            order = sorted(st["pending"], key=lambda t: _priority_key(t, self.jobs))
            jobs_subset = {tid.parent: self.jobs[tid.parent] for tid in order}
            committed = _committed(st["running"], self.jobs)
            infos = _running_infos(st["running"], self.jobs)
            assignments, preemptions = schedule_and_preempt(
                self.all_workers, committed, order, jobs_subset, infos, scheduler, include_backend=True
            )
            self._apply_assignments(st, assignments, now)
            if preemptions:
                self._apply_preemptions(st, preemptions)
                # second pass to fill freed capacity (preemptor wins by band order)
                order2 = sorted(st["pending"], key=lambda t: _priority_key(t, self.jobs))
                jobs2 = {tid.parent: self.jobs[tid.parent] for tid in order2}
                committed2 = _committed(st["running"], self.jobs)
                infos2 = _running_infos(st["running"], self.jobs)
                assignments2, _ = schedule_and_preempt(
                    self.all_workers, committed2, order2, jobs2, infos2, scheduler, include_backend=True
                )
                self._apply_assignments(st, assignments2, now)
            self._count_busy(st)
        return self._finalize(st, label)

    # -- TWO-LEVEL mode -----------------------------------------------------

    def _summaries(
        self,
        committed: dict[str, tuple[int, int, int, int]],
        stale_backends: dict[str, int],
        pending_leases_window: dict[str, int],
        down_backends: set[str],
    ) -> dict[str, CapacitySummary]:
        out: dict[str, CapacitySummary] = {}
        for b in self.backends:
            if b.backend_id in down_backends:
                # Partitioned agent: its last summary is frozen (looks idle/free) and
                # ages. The root only learns it is unusable via stale_ms.
                out[b.backend_id] = compute_summary(b, {}, stale_ms=stale_backends.get(b.backend_id, 60_000))
                continue
            out[b.backend_id] = compute_summary(
                b,
                committed,
                stale_ms=stale_backends.get(b.backend_id, 0),
                pending_leases=pending_leases_window.get(b.backend_id, 0),
                backoff={},
            )
        return out

    def _ledger_from_truth(
        self, st, now: int, poll_lag: int
    ) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, int]]:
        """Root ledger + agent pending_leases, both derived from authoritative state.

        Returns (outstanding, queued, pending_leases) where:
          - outstanding[backend][shape] = the FULL correction the lagged summary
            misses: routed-but-not-running (queue backlog, term1) PLUS running tasks
            launched within the poll-lag window (term2, not yet in the lagged
            allocatable). This is what the root knows because it owns task->backend.
          - pending_leases[backend] = term2 only (the agent's launched-but-unobserved
            count). A scalar, and blind to the queue backlog (term1).
        """
        outstanding: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        queued: dict[str, int] = defaultdict(int)
        pending_leases: dict[str, int] = defaultdict(int)
        for tid in st["pending"]:  # term1: routed, queued, not running
            backend = st["pin"].get(tid.parent)
            if backend is None:
                continue
            outstanding[backend][self.jobs[tid.parent].variant] += 1
            queued[backend] += 1
        for tid, rec in st["running"].items():  # term2: launched within the lag window
            if poll_lag > 0 and st["placed_tick"].get(tid, -1) >= now - poll_lag:
                outstanding[rec.backend][self.jobs[rec.job_id].variant] += 1
                pending_leases[rec.backend] += 1
        return outstanding, queued, pending_leases

    def run_two_level(
        self,
        scheduler: Scheduler,
        cfg: RouterConfig,
        label: str = "TWO-LEVEL",
        stale_backends: dict[str, int] | None = None,
        down_backends: set[str] | None = None,
        poll_lag: int = 0,
    ) -> Metrics:
        st = self._new_state()
        router = RootRouter(self.backends, cfg)
        stale_backends = stale_backends or {}
        down_backends = down_backends or set()
        committed_history: list[dict[str, tuple[int, int, int, int]]] = []

        for now in range(self.horizon):
            self._release(st, now)
            self._admit_arrivals(st, now)
            self._count_inversions(st)

            # ---- ROOT: task->backend from a (possibly lagged) summary only ----
            committed_now = _committed(st["running"], self.jobs)
            committed_history.append(committed_now)
            lagged = committed_history[max(0, now - poll_lag)]
            outstanding, queued, pending_leases = self._ledger_from_truth(st, now, poll_lag)
            summaries = self._summaries(lagged, stale_backends, pending_leases, down_backends)
            # The least-queued fallback is the root's OWN backlog knowledge, so it is
            # available iff the root ledger is. pending_leases (agent scalar) is a
            # weaker substitute; with neither, the root is blind to its own backlog.
            if cfg.use_root_ledger:
                queued_eff = queued
            elif cfg.use_pending_leases:
                queued_eff = dict(pending_leases)
            else:
                queued_eff = {}
            router.prime(outstanding, queued_eff)
            tick_used: dict[str, int] = {}
            # route each UNPINNED job that has pending tasks, in priority order
            unpinned_jobs = []
            seen = set()
            for tid in sorted(st["pending"], key=lambda t: _priority_key(t, self.jobs)):
                jid = tid.parent
                if jid in seen or jid in st["pin"]:
                    continue
                seen.add(jid)
                unpinned_jobs.append(self.jobs[jid])
            for js in unpinned_jobs:
                chosen = router.route(js, summaries, tick_used)
                if chosen is not None:
                    st["pin"][js.job_id] = chosen

            # ---- per-backend AGENT: task->worker via the real Scheduler ----
            for b in self.backends:
                if b.backend_id in down_backends:
                    continue  # partitioned agent cannot place; routed work queues
                local_pending = sorted(
                    (t for t in st["pending"] if st["pin"].get(t.parent) == b.backend_id),
                    key=lambda t: _priority_key(t, self.jobs),
                )
                if not local_pending:
                    continue
                jobs_subset = {t.parent: self.jobs[t.parent] for t in local_pending}
                committed = _committed(
                    {tid: r for tid, r in st["running"].items() if r.backend == b.backend_id}, self.jobs
                )
                infos = _running_infos(
                    {tid: r for tid, r in st["running"].items() if r.backend == b.backend_id}, self.jobs
                )
                assignments, preemptions = schedule_and_preempt(
                    b.workers, committed, local_pending, jobs_subset, infos, scheduler, include_backend=False
                )
                self._apply_assignments(st, assignments, now)
                if preemptions:
                    self._apply_preemptions(st, preemptions)
                    lp2 = sorted(
                        (t for t in st["pending"] if st["pin"].get(t.parent) == b.backend_id),
                        key=lambda t: _priority_key(t, self.jobs),
                    )
                    jobs2 = {t.parent: self.jobs[t.parent] for t in lp2}
                    committed2 = _committed(
                        {tid: r for tid, r in st["running"].items() if r.backend == b.backend_id}, self.jobs
                    )
                    infos2 = _running_infos(
                        {tid: r for tid, r in st["running"].items() if r.backend == b.backend_id}, self.jobs
                    )
                    assignments2, _ = schedule_and_preempt(
                        b.workers, committed2, lp2, jobs2, infos2, scheduler, include_backend=False
                    )
                    self._apply_assignments(st, assignments2, now)
            self._count_busy(st)
        return self._finalize(st, label)
