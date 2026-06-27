# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SPIKE S2 driver: run the same workload through GLOBAL vs TWO-LEVEL and ablate
each CapacitySummary field. Prints numbers for SPIKE.md.

Run:  .venv/bin/python .agents/projects/iris_multi_backend/spikes/S2_two_level_placement/run.py
"""

from __future__ import annotations

import random

from iris.cluster.constraints import ConstraintOp, Constraint, WellKnownAttribute
from iris.cluster.controller.scheduling.scheduler import Scheduler
from iris.cluster.types import JobName

from harness import GIB, INTER, PROD, BackendDef, JobSpec, RouterConfig, WorkerSpec
from sim import Metrics, Simulator
from workload import standard_fleet, standard_workload

HR = "-" * 104


def _hdr(title: str) -> None:
    print(f"\n{HR}\n{title}\n{HR}")


def _delta(g: Metrics, t: Metrics) -> str:
    gm, gp50, gp95, gmx = g.wait_stats()
    tm, tp50, tp95, tmx = t.wait_stats()
    return (
        f"  delta(two-level - global): placed={t.placed_tasks - g.placed_tasks:+d} "
        f"starved={t.never_placed - g.never_placed:+d} "
        f"wait_mean={tm - gm:+.2f} wait_p95={tp95 - gp95:+d} util={t.util_global - g.util_global:+.1%}"
    )


def main() -> None:
    scheduler = Scheduler()

    # ===================== Scenario MAIN: full mixed workload =====================
    _hdr("SCENARIO MAIN — mixed federation, GLOBAL vs TWO-LEVEL (full summary)")
    rng = random.Random(7)
    fleet = standard_fleet()
    jobs = standard_workload(rng, horizon=60, jobs_per_tick=6.0)
    total_workers = sum(len(b.workers) for b in fleet)
    print(f"fleet: {len(fleet)} backends, {total_workers} workers; workload: {len(jobs)} jobs "
          f"({sum(j.num_tasks for j in jobs)} tasks) over 60 ticks")

    sim = Simulator(fleet, jobs, horizon=60)
    g = sim.run_global(scheduler)
    t = sim.run_two_level(scheduler, RouterConfig(), label="TWO-LEVEL(full)")
    print(g.summary_line())
    print(t.summary_line())
    print(_delta(g, t))
    print("  per-backend utilization:")
    for b in fleet:
        print(f"    {b.backend_id:<18} global={g.util_by_backend[b.backend_id]:.1%}  "
              f"two-level={t.util_by_backend[b.backend_id]:.1%}")

    # ===================== Ablation: drop each summary field =====================
    _hdr("ABLATION — TWO-LEVEL with individual summary fields disabled (same workload)")
    print(f"baseline GLOBAL:     {g.summary_line()}")
    print(f"baseline FULL:       {t.summary_line()}")
    print()
    ablations = [
        ("no_largest_gang(dyn)", RouterConfig(use_largest_gang=False)),
        ("no_max_gang(struct)", RouterConfig(use_max_gang_structural=False)),
        ("no_max_cpu_bin", RouterConfig(use_max_cpu_bin=False)),
        ("no_in_tick_decr", RouterConfig(use_in_tick_decrement=False)),
        ("ledger->pending_leases", RouterConfig(use_root_ledger=False, use_pending_leases=True)),
        ("no_ledger_no_leases", RouterConfig(use_root_ledger=False, use_pending_leases=False)),
    ]
    for name, cfg in ablations:
        m = sim.run_two_level(scheduler, cfg, label=name)
        print(m.summary_line())

    # ===================== Scenario GANG — largest_gang load-bearing =====================
    _hdr("SCENARIO GANG — fragmented vs whole slices; does the root need largest_gang?")
    gang_fleet = _gang_fleet()
    gang_jobs = _gang_jobs()
    gsim = Simulator(gang_fleet, gang_jobs, horizon=30)
    gg = gsim.run_global(scheduler)
    gt_full = gsim.run_two_level(scheduler, RouterConfig(), label="two-level full")
    gt_nodyn = gsim.run_two_level(scheduler, RouterConfig(use_largest_gang=False), label="two-level -largest_gang(dyn)")
    gt_nostruct = gsim.run_two_level(
        scheduler, RouterConfig(use_max_gang_structural=False), label="two-level -max_gang(struct)"
    )
    print(f"  fleet: a-frag (4 slices x2 = max group 2), z-whole (2 slices x4 = max group 4); "
          f"{len(gang_jobs)} gangs of 4")
    print(gg.summary_line())
    print(gt_full.summary_line())
    print(gt_nodyn.summary_line())
    print(gt_nostruct.summary_line())

    # ===================== Scenario CPU-BIN — max_free_cpu_bin load-bearing =====================
    _hdr("SCENARIO CPU-BIN — large CPU jobs; does the root need max_free_cpu_bin?")
    cpu_fleet = _cpu_bin_fleet()
    cpu_jobs = _big_cpu_jobs()
    csim = Simulator(cpu_fleet, cpu_jobs, horizon=30)
    cg = csim.run_global(scheduler)
    ct_on = csim.run_two_level(scheduler, RouterConfig(use_max_cpu_bin=True), label="two-level +max_cpu_bin")
    ct_off = csim.run_two_level(scheduler, RouterConfig(use_max_cpu_bin=False), label="two-level -max_cpu_bin")
    print(f"  {len(cpu_jobs)} CPU jobs of 48 cores; TPU boxes have MORE total free cpu (192) but tiny bin (8); "
          f"cpu pools have less total (128) but 64-core bins")
    print(cg.summary_line())
    print(ct_on.summary_line())
    print(ct_off.summary_line())

    # ===================== Scenario BURST — in-tick decrement load-bearing =====================
    _hdr("SCENARIO BURST — 48 v5e jobs in ONE tick across 2 equal backends; need in-tick decrement?")
    burst_fleet = [_two_tpu("gcp-tpu-west", "us-west4"), _two_tpu("gcp-tpu-central", "us-central1")]
    burst_jobs = _burst_jobs(48, duration=25)
    bsim = Simulator(burst_fleet, burst_jobs, horizon=30)
    bg = bsim.run_global(scheduler)
    bt_on = bsim.run_two_level(scheduler, RouterConfig(use_in_tick_decrement=True), label="two-level +in_tick")
    bt_off = bsim.run_two_level(scheduler, RouterConfig(use_in_tick_decrement=False), label="two-level -in_tick")
    print("  2 backends x 24 free units = 48; a balanced split places all 48 at tick 0")
    print(bg.summary_line())
    print(bt_on.summary_line())
    print(bt_off.summary_line())

    # ============ Scenario LAG — cross-tick: root ledger vs agent pending_leases ============
    _hdr("SCENARIO LAG — oversubscribed, summary 2 polls stale; root-ledger vs pending_leases vs neither")
    lag_fleet = [_small_tpu("gcp-tpu-west", "us-west4"), _small_tpu("gcp-tpu-central", "us-central1")]
    lag_jobs = _sustained_v5e(rate=8, ticks=24, duration=4)
    lsim = Simulator(lag_fleet, lag_jobs, horizon=30)
    lg = lsim.run_global(scheduler)
    l_ledger = lsim.run_two_level(scheduler, RouterConfig(), label="two-level root-ledger", poll_lag=2)
    l_leases = lsim.run_two_level(
        scheduler, RouterConfig(use_root_ledger=False, use_pending_leases=True), label="two-level pending_leases", poll_lag=2
    )
    l_none = lsim.run_two_level(
        scheduler, RouterConfig(use_root_ledger=False, use_pending_leases=False), label="two-level neither", poll_lag=2
    )
    l_noit = lsim.run_two_level(
        scheduler, RouterConfig(use_in_tick_decrement=False), label="two-level -in_tick", poll_lag=2
    )
    print(f"  2 backends x 8 units, dur=4; arrivals 8/tick (oversubscribed); poll_lag=2; {len(lag_jobs)} jobs")
    print(lg.summary_line())
    print(l_ledger.summary_line())
    print(l_leases.summary_line())
    print(l_none.summary_line())
    print(l_noit.summary_line())

    # ============ Scenario GANG-BALANCE — dynamic largest_gang reduces gang latency ============
    _hdr("SCENARIO GANG-BALANCE — one slice partly busy; does dynamic largest_gang cut gang wait?")
    gb_fleet = [_tpu_def("a-west", "us-west4", n_slices=1, slice_size=4),
                _tpu_def("b-spare", "us-west4", n_slices=1, slice_size=4)]
    gb_jobs = _gang_balance_jobs()
    gbsim = Simulator(gb_fleet, gb_jobs, horizon=30)
    gbg = gbsim.run_global(scheduler)
    gb_on = gbsim.run_two_level(scheduler, RouterConfig(), label="two-level +largest_gang(dyn)")
    gb_off = gbsim.run_two_level(scheduler, RouterConfig(use_largest_gang=False), label="two-level -largest_gang(dyn)")
    print("  a-west slice 2/4 busy (pinned solos); b-spare idle; a gang of 4 arrives; both structurally capable")
    print(gbg.summary_line())
    print(gb_on.summary_line())
    print(gb_off.summary_line())

    # ===================== Scenario PARTITION — stale_ms load-bearing =====================
    _hdr("SCENARIO PARTITION — one TPU backend's agent is unreachable; does the root need stale_ms?")
    part_fleet = [
        _two_tpu("gcp-tpu-west", "us-west4"),
        _two_tpu("gcp-tpu-central", "us-central1"),
    ]
    part_jobs = _v5e_stream()
    psim = Simulator(part_fleet, part_jobs, horizon=30)
    down = {"gcp-tpu-central"}
    stale = {"gcp-tpu-central": 60_000}
    pt_on = psim.run_two_level(
        scheduler, RouterConfig(use_stale_discount=True), label="two-level +stale_discount",
        stale_backends=stale, down_backends=down,
    )
    pt_off = psim.run_two_level(
        scheduler, RouterConfig(use_stale_discount=False), label="two-level -stale_discount",
        stale_backends=stale, down_backends=down,
    )
    print(f"  fleet: 2 equivalent v5e backends; central PARTITIONED (looks idle, cannot place); "
          f"{len(part_jobs)} v5e jobs; west alone has the capacity")
    print(pt_on.summary_line())
    print(pt_off.summary_line())


# ---------------------------------------------------------------------------
# Micro-scenario fleets/workloads
# ---------------------------------------------------------------------------


def _eq(key: str, value: str) -> Constraint:
    return Constraint.create(key=key, op=ConstraintOp.EQ, value=value)


_V5E_CONS = (_eq(WellKnownAttribute.DEVICE_TYPE, "tpu"), _eq(WellKnownAttribute.DEVICE_VARIANT, "v5e-4"))


def _tpu_workers(backend_id: str, region: str, n_slices: int, slice_size: int) -> list[WorkerSpec]:
    ws = []
    for s in range(n_slices):
        for i in range(slice_size):
            ws.append(WorkerSpec(f"{backend_id}-s{s}-{i}", backend_id, "v5e-4", 8000, 32 * GIB, 0, 4,
                                 f"{backend_id}-s{s}", region))
    return ws


def _tpu_def(backend_id: str, region: str, n_slices: int, slice_size: int) -> BackendDef:
    return BackendDef(
        backend_id=backend_id, kind="worker_daemon",
        static={WellKnownAttribute.DEVICE_TYPE: {"tpu"}, WellKnownAttribute.DEVICE_VARIANT: {"v5e-4"},
                WellKnownAttribute.REGION: {region}, "backend": {backend_id}, "provider": {"gcp"}},
        workers=_tpu_workers(backend_id, region, n_slices, slice_size),
    )


def _gang_fleet() -> list[BackendDef]:
    # equal total free units (8 each), different max group size
    return [_tpu_def("a-frag", "us-west4", n_slices=4, slice_size=2),
            _tpu_def("z-whole", "us-west4", n_slices=2, slice_size=4)]


def _gang_jobs() -> list[JobSpec]:
    jobs = []
    for k in range(4):
        jid = JobName.root("alice", f"gang{k}")
        jobs.append(JobSpec(jid, "alice", INTER, arrival=k, duration=12, num_tasks=4, coscheduled=True,
                            group_by="slice-group", variant="v5e-4", cpu_millicores=8000, memory_bytes=16 * GIB,
                            gpu_count=0, tpu_count=4, constraints=_V5E_CONS, pinned_backend=None))
    return jobs


def _two_tpu(backend_id: str, region: str) -> BackendDef:
    return _tpu_def(backend_id, region, n_slices=6, slice_size=4)


def _small_tpu(backend_id: str, region: str) -> BackendDef:
    return _tpu_def(backend_id, region, n_slices=2, slice_size=4)  # 8 units


def _burst_jobs(n: int, duration: int) -> list[JobSpec]:
    jobs = []
    for k in range(n):
        jid = JobName.root("alice", f"b{k}")
        jobs.append(JobSpec(jid, "alice", INTER, arrival=0, duration=duration, num_tasks=1, coscheduled=False,
                            group_by=None, variant="v5e-4", cpu_millicores=8000, memory_bytes=16 * GIB,
                            gpu_count=0, tpu_count=4, constraints=_V5E_CONS, pinned_backend=None))
    return jobs


def _sustained_v5e(rate: int, ticks: int, duration: int) -> list[JobSpec]:
    jobs = []
    n = 0
    for tick in range(ticks):
        for _ in range(rate):
            n += 1
            jid = JobName.root("alice", f"s{n}")
            jobs.append(JobSpec(jid, "alice", INTER, arrival=tick, duration=duration, num_tasks=1, coscheduled=False,
                                group_by=None, variant="v5e-4", cpu_millicores=8000, memory_bytes=16 * GIB,
                                gpu_count=0, tpu_count=4, constraints=_V5E_CONS, pinned_backend=None))
    return jobs


def _v5e_stream() -> list[JobSpec]:
    jobs = []
    n = 0
    for tick in range(15):
        for _ in range(4):
            n += 1
            jid = JobName.root("alice", f"v{n}")
            jobs.append(JobSpec(jid, "alice", INTER, arrival=tick, duration=5, num_tasks=1, coscheduled=False,
                                group_by=None, variant="v5e-4", cpu_millicores=8000, memory_bytes=16 * GIB,
                                gpu_count=0, tpu_count=4, constraints=_V5E_CONS, pinned_backend=None))
    return jobs


def _cpu_only_def(backend_id: str, region: str, n_workers: int) -> BackendDef:
    workers = [WorkerSpec(f"{backend_id}-{i}", backend_id, "cpu", 64000, 256 * GIB, 0, 0, None, region)
               for i in range(n_workers)]
    return BackendDef(
        backend_id=backend_id, kind="worker_daemon",
        static={WellKnownAttribute.DEVICE_TYPE: {"cpu"}, WellKnownAttribute.REGION: {region},
                "backend": {backend_id}, "provider": {"gcp"}},
        workers=workers,
    )


def _cpu_bin_fleet() -> list[BackendDef]:
    # TPU backends carry MORE total idle cpu (24x8=192) but a tiny per-box bin (8);
    # cpu pools carry less total (2x64=128) but a 64-core bin. A 48-core job only
    # fits a cpu pool, but "most total free cpu" alone would route it to a TPU box.
    return [
        _tpu_def("gcp-tpu-west", "us-west4", n_slices=6, slice_size=4),
        _tpu_def("gcp-tpu-central", "us-central1", n_slices=6, slice_size=4),
        _cpu_only_def("cpu-pool-west", "us-west4", n_workers=2),
        _cpu_only_def("cpu-pool-central", "us-central1", n_workers=2),
    ]


def _gang_balance_jobs() -> list[JobSpec]:
    """2 solos pinned to a-west (occupy 2/4 of its only slice), then a gang of 4."""
    jobs: list[JobSpec] = []
    pin_cons = _V5E_CONS + (Constraint.create(key="backend", op=ConstraintOp.EQ, value="a-west"),)
    for k in range(2):
        jid = JobName.root("alice", f"hold{k}")
        jobs.append(JobSpec(jid, "alice", INTER, arrival=0, duration=20, num_tasks=1, coscheduled=False,
                            group_by=None, variant="v5e-4", cpu_millicores=8000, memory_bytes=16 * GIB,
                            gpu_count=0, tpu_count=4, constraints=pin_cons, pinned_backend="a-west"))
    jid = JobName.root("alice", "gang")
    jobs.append(JobSpec(jid, "alice", INTER, arrival=1, duration=8, num_tasks=4, coscheduled=True,
                        group_by="slice-group", variant="v5e-4", cpu_millicores=8000, memory_bytes=16 * GIB,
                        gpu_count=0, tpu_count=4, constraints=_V5E_CONS, pinned_backend=None))
    return jobs


def _big_cpu_jobs() -> list[JobSpec]:
    jobs = []
    n = 0
    for tick in range(12):
        for _ in range(3):
            n += 1
            jid = JobName.root("bob", f"c{n}")
            jobs.append(JobSpec(jid, "bob", INTER, arrival=tick, duration=4, num_tasks=1, coscheduled=False,
                                group_by=None, variant="cpu", cpu_millicores=48000, memory_bytes=64 * GIB,
                                gpu_count=0, tpu_count=0, constraints=(), pinned_backend=None))
    return jobs


if __name__ == "__main__":
    main()
