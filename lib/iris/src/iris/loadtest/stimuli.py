# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stimulus generators for the autoscaler load-test harness.

Three orthogonal generators — :func:`submit_burst`, :func:`preempt_workers`,
and :func:`bad_tpu_api` — which can be used independently or composed.
All three are deterministic (seedable RNG) and observable through the
harness's ``gcp_service`` + ``db``.

Probe/read-pressure clients have moved to :mod:`iris.loadtest.probes`.

Naming convention for TPU scale-groups (matches
``iris.cluster.config._expand_tpu_pools``)::

    tpu_{pool}_{size}-{zone}       e.g. tpu_v6e-preemptible_8-europe-west4-a
"""

from __future__ import annotations

import logging
import random
import re
import sqlite3
from pathlib import Path
from typing import Literal

from iris.cluster.constraints import Constraint, ConstraintOp, constraints_from_resources, merge_constraints
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.types import JobName, get_tpu_topology
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp

from iris.loadtest.harness import LoadtestHarness

logger = logging.getLogger(__name__)

# Per-task resource requests sized to pack realistically on a 200-vCPU / 600-GiB
# synthetic TPU host (see synthetic_worker._build_metadata):
#   one TPU job consumes 8 vCPU + 32 GiB, leaving 568 GiB for CPU tasks;
#   each CPU task consumes 4 vCPU + 16 GiB, so ~35 CPU tasks pack alongside
#   one TPU task. Memory-bound by design.
TPU_JOB_CPU_MILLICORES = 8000
TPU_JOB_MEMORY_BYTES = 32 * 1024**3
CPU_JOB_CPU_MILLICORES = 4000
CPU_JOB_MEMORY_BYTES = 16 * 1024**3


def _make_entrypoint() -> job_pb2.RuntimeEntrypoint:
    ep = job_pb2.RuntimeEntrypoint()
    ep.run_command.argv[:] = ["python", "-c", "pass"]
    return ep


def _tpu_variant(tpu_kind: str, size: int) -> str:
    """Map ``(tpu_kind, size)`` to the canonical accelerator variant.

    ``tpu_kind`` here is the pool name family (``v5e``, ``v6e``, ``v5p``,
    ``v4``); the full pool name includes the scheduling-class suffix
    (``v6e-preemptible``), which the caller passes as ``tpu_kind``. We strip
    the suffix to derive the family.
    """
    family = tpu_kind.split("-")[0]
    prefix_map = {
        "v4": "v4-",
        "v5e": "v5litepod-",
        "v5p": "v5p-",
        "v6e": "v6e-",
    }
    if family not in prefix_map:
        raise ValueError(f"Unknown TPU family {family!r}; supported: {sorted(prefix_map)}")
    return f"{prefix_map[family]}{size}"


def _inject_device_constraints(request: controller_pb2.Controller.LaunchJobRequest) -> None:
    auto = constraints_from_resources(request.resources)
    if not auto:
        return
    user = [Constraint.from_proto(c) for c in request.constraints]
    merged = merge_constraints(auto, user)
    del request.constraints[:]
    for c in merged:
        request.constraints.append(c.to_proto())


def submit_burst(
    harness: LoadtestHarness,
    *,
    user: str,
    job_count: int,
    tpu_kind: str,
    size: int | list[int],
    spread_seconds: float = 0.0,
    seed: int = 0,
    zone_pin: str | None = None,
) -> list[str]:
    """Submit ``job_count`` single-task TPU jobs via the real transitions path.

    Mirrors michaelryan's extract-v2 sweep shape: many single-task jobs
    targeting ``tpu_kind``. Each job gets a unique name derived from
    ``seed`` so reruns are stable. ``spread_seconds`` is reserved for later
    cadence-sensitive scenarios; today we submit as fast as sqlite allows.

    Args:
        size: A single TPU size, or a list of sizes to round-robin across.
            A list is the fleet-wide mode — it ensures demand lands on every
            eligible group at once, maximizing simultaneous scale-up pressure.
        zone_pin: If set, attaches a zone constraint to each job (legacy
            behavior). Default None leaves the autoscaler free to route
            demand to any zone — fleet-wide relies on this.

    Returns the wire-format job ids.
    """
    del spread_seconds  # deterministic; no sleeps in the unit path.
    if job_count <= 0:
        return []

    rng = random.Random(seed)
    sizes = [size] if isinstance(size, int) else list(size)
    if not sizes:
        raise ValueError("submit_burst: sizes list is empty")

    transitions = ControllerTransitions(db=harness.db)
    job_ids: list[str] = []

    for i in range(job_count):
        chosen_size = sizes[i % len(sizes)]
        variant = _tpu_variant(tpu_kind, chosen_size)
        # Multi-host slices (e.g. v6e-32 has vm_count=8) need one task per VM.
        # For single-VM variants (e.g. v6e-8 has vm_count=1) replicas=1.
        try:
            replicas = get_tpu_topology(variant).vm_count
        except ValueError:
            replicas = 1
        base_name = f"burst-{tpu_kind}-{chosen_size}-{seed}-{i}-{rng.randrange(10**9)}"
        jid = JobName.root(user, base_name)
        req = controller_pb2.Controller.LaunchJobRequest(
            name=jid.to_wire(),
            entrypoint=_make_entrypoint(),
            resources=job_pb2.ResourceSpecProto(
                cpu_millicores=TPU_JOB_CPU_MILLICORES,
                memory_bytes=TPU_JOB_MEMORY_BYTES,
                device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant=variant)),
            ),
            environment=job_pb2.EnvironmentConfig(),
            replicas=replicas,
        )
        if zone_pin is not None:
            req.constraints.append(Constraint.create(key="zone", op=ConstraintOp.EQ, value=zone_pin).to_proto())
        _inject_device_constraints(req)
        transitions.submit_job(jid, req, Timestamp.now())
        job_ids.append(jid.to_wire())

    logger.info(
        "submit_burst: submitted %d jobs for %s sizes=%s zone_pin=%s",
        len(job_ids),
        tpu_kind,
        sizes,
        zone_pin,
    )
    return job_ids


def submit_cpu_burst(
    harness: LoadtestHarness,
    *,
    user: str,
    job_count: int,
    tasks_per_job: int,
    seed: int = 0,
) -> list[str]:
    """Submit CPU-only jobs; each job has ``tasks_per_job`` replicas.

    CPU jobs carry no ``device`` field, so ``constraints_from_resources``
    emits no constraints and the scheduler matches any worker with sufficient
    CPU+RAM — including the TPU-tagged synthetic workers. This mirrors the
    prod topology where non-TPU tasks share TPU hosts.

    A prod-mix call is 10 jobs x 500 tasks = 5000 CPU tasks. Each task takes
    4 vCPU + 16 GiB; ~35 of them pack alongside one TPU task on a 600 GiB
    host.

    Returns the wire-format job ids.
    """
    if job_count <= 0 or tasks_per_job <= 0:
        return []

    rng = random.Random(seed)
    transitions = ControllerTransitions(db=harness.db)
    job_ids: list[str] = []

    for i in range(job_count):
        base_name = f"cpu-burst-{seed}-{i}-{rng.randrange(10**9)}"
        jid = JobName.root(user, base_name)
        req = controller_pb2.Controller.LaunchJobRequest(
            name=jid.to_wire(),
            entrypoint=_make_entrypoint(),
            resources=job_pb2.ResourceSpecProto(
                cpu_millicores=CPU_JOB_CPU_MILLICORES,
                memory_bytes=CPU_JOB_MEMORY_BYTES,
                # No device field: constraints_from_resources returns [] so
                # the task can land on any worker with enough CPU/RAM.
            ),
            environment=job_pb2.EnvironmentConfig(),
            replicas=tasks_per_job,
        )
        transitions.submit_job(jid, req, Timestamp.now())
        job_ids.append(jid.to_wire())

    logger.info(
        "submit_cpu_burst: submitted %d jobs x %d tasks = %d CPU tasks",
        len(job_ids),
        tasks_per_job,
        len(job_ids) * tasks_per_job,
    )
    return job_ids


def _matching_groups(harness: LoadtestHarness, group_pattern: str) -> list[str]:
    pattern = re.compile(group_pattern)
    return sorted(name for name in harness.autoscaler._groups.keys() if pattern.search(name))


def preempt_workers(
    harness: LoadtestHarness,
    *,
    group_pattern: str,
    fraction: float,
    seed: int = 0,
) -> list[str]:
    """Kill a fraction of TPUs currently backing workers in matching groups.

    Finds workers registered under groups whose name matches ``group_pattern``
    (regex), selects ``fraction * len(matching)``  of them, and removes their
    underlying TPU from the in-memory GCP service. The autoscaler's
    ``refresh()`` will observe the loss on the next tick via ``tpu_describe``
    returning ``None`` on the slice.

    If no workers are registered yet (e.g. the harness hasn't issued any
    scale-ups), this falls back to killing any TPUs whose names contain the
    group fragment — which is what ``scale_up`` creates before bootstrap.

    Returns the preempted worker/slice ids.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    rng = random.Random(seed)
    group_names = _matching_groups(harness, group_pattern)
    if not group_names:
        logger.warning("preempt_workers: no scale groups match %r", group_pattern)
        return []

    # Find matching TPUs in the fake. Slice names embed the *mangled* group
    # name (GCP disallows underscores, so the provider translates ``_``→``-``
    # via ``_build_gce_resource_name``). We mangle the group name the same
    # way before the substring test. Also truncation occurs in the real name
    # builder, so we compare progressively shorter prefixes.
    candidates: list[tuple[str, str]] = []
    for (name, zone), _info in harness.gcp_service.list_all_tpus():
        for group in group_names:
            mangled = group.replace("_", "-")
            matched = False
            for cut in range(len(mangled), max(len(mangled) - 10, 0), -1):
                if mangled[:cut] in name:
                    matched = True
                    break
            if matched:
                candidates.append((name, zone))
                break

    if not candidates:
        logger.warning("preempt_workers: no TPUs in matching groups %s", group_names)
        return []

    candidates.sort()
    pick_count = max(1, int(len(candidates) * fraction))
    rng.shuffle(candidates)
    victims = candidates[:pick_count]

    for name, zone in victims:
        harness.gcp_service.delete_tpu(name, zone)
        logger.info("preempt_workers: removed TPU %s in %s", name, zone)
    return [name for name, _ in victims]


def bad_tpu_api(
    harness: LoadtestHarness,
    *,
    group_pattern: str,
    failure_mode: Literal["timeout", "internal_error", "quota"],
    duration_seconds: float,
    latency_seconds: float = 120.0,
) -> None:
    """Make ``tpu_create`` for matching groups block then fail.

    The rule applies for ``duration_seconds`` of wall-clock time and burns
    each scale-up thread for ``latency_seconds`` before raising. Default
    latency matches production's observed 120 s hang.

    ``group_pattern`` is a regex; the rule fires when it matches any
    substring of the TPU slice name.
    """
    harness.gcp_service.configure_failure(
        operation="tpu_create",
        name_regex=group_pattern,
        failure_mode=failure_mode,
        duration_seconds=duration_seconds,
        latency_seconds=latency_seconds,
    )
    logger.info(
        "bad_tpu_api: group_pattern=%r mode=%s duration=%.1fs latency=%.1fs",
        group_pattern,
        failure_mode,
        duration_seconds,
        latency_seconds,
    )


# ---------------------------------------------------------------------------
# Observation helpers (thin wrappers around db reads for tests)
# ---------------------------------------------------------------------------


def read_scale_group_failures(harness: LoadtestHarness, group_name: str) -> int:
    """Return ``consecutive_failures`` for *group_name* from the DB."""
    db_path = Path(harness.db.db_dir) / "controller.sqlite3"
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT consecutive_failures FROM scaling_groups WHERE name = ?",
            (group_name,),
        ).fetchone()
    finally:
        conn.close()
    return int(row[0]) if row is not None else 0


def count_tasks_for_jobs(harness: LoadtestHarness, job_ids: list[str]) -> int:
    if not job_ids:
        return 0
    placeholders = ",".join("?" for _ in job_ids)
    with harness.db.read_snapshot() as q:
        row = q.execute_sql(
            f"SELECT COUNT(*) FROM tasks WHERE job_id IN ({placeholders})",
            tuple(job_ids),
        ).fetchone()
    return int(row[0])
