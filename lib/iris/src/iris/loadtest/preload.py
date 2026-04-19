# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pre-load synthetic workers to reproduce prod-scale fleet pressure.

Pre-populates the harness with N real-RPC ``SyntheticWorker`` instances
distributed across scale groups proportional to the snapshot's ``workers``
table. Each pre-loaded worker is a real Connect/RPC server on an ephemeral
localhost port, registered via ``ControllerTransitions.register_worker`` so the
controller-side prober (and any other RPC-over-the-wire consumer) hits it
through the same socket stack that fires in production.

Separate from the autoscaler-driven ``attach_worker_pool`` path: pre-loaded
workers do not have a backing ``tpu_create`` in the fake GCP service, so they
are not torn down on ``tpu_delete``. The preemption stimulus targets them via
``SyntheticWorkerPool.stop_for_slice`` directly.
"""

from __future__ import annotations

import logging
import resource
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from iris.cluster.controller.transitions import ControllerTransitions

from iris.loadtest.synthetic_worker import (
    LifecycleDelays,
    SyntheticWorker,
    SyntheticWorkerConfig,
    SyntheticWorkerPool,
)
from iris.cluster.types import WorkerId

logger = logging.getLogger(__name__)


# Default FD ceiling we request. Accounts for:
#   600 synthetic uvicorn servers (~2 FDs each while idle; accept socket + epoll)
#   Controller prober (long-lived HTTP/1 keepalives, one per worker)
#   Probe clients, sampler, DB pool, stdio, logging, pytest plumbing
# Measured ~1.2k FDs steady-state at 600 workers; 16k gives 10x headroom.
DEFAULT_FD_TARGET = 16384


@dataclass(frozen=True)
class GroupAllocation:
    """How many synthetic workers to spawn in a given scale group."""

    scale_group: str
    zone: str
    device_variant: str
    count: int


def ensure_fd_limit(target: int = DEFAULT_FD_TARGET) -> tuple[int, int]:
    """Raise ``RLIMIT_NOFILE`` soft limit to ``min(hard, target)``.

    Returns the ``(soft, hard)`` limit after the raise. Raises ``RuntimeError``
    with an actionable message if the hard limit is too low to accommodate a
    full pre-loaded fleet; callers that only want smoke-scale can pass a
    smaller ``target``.

    macOS: Python's ``getrlimit`` reports ``RLIM_INFINITY`` even when
    ``launchctl limit maxfiles`` caps the process at 256. This raise is still
    useful because ``setrlimit`` with an explicit int bumps the per-process
    *kernel* ceiling the way ``ulimit -n`` does — we just can't introspect it
    via ``getrlimit``. If the user hasn't run ``launchctl limit maxfiles
    16384`` (or similar), uvicorn will hit ``OSError: too many open files``
    when binding workers 257+. Nothing we can do from Python other than raise
    the soft limit and hope — which is what this does.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    requested = min(hard, target) if hard != resource.RLIM_INFINITY else target
    if soft < requested:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (requested, hard))
        except (ValueError, OSError) as exc:
            raise RuntimeError(
                f"Could not raise RLIMIT_NOFILE from {soft} to {requested} "
                f"(hard={hard}): {exc}. On macOS run "
                f"`sudo launchctl limit maxfiles {target} unlimited` and re-login; "
                f"on Linux edit /etc/security/limits.conf or use `ulimit -n {target}`."
            ) from exc
    return resource.getrlimit(resource.RLIMIT_NOFILE)


def group_distribution_from_snapshot(
    snapshot_db: Path,
    *,
    scale_group_configs: dict,
    scale_group_pattern: str | None = None,
) -> list[GroupAllocation]:
    """Read per-group active-worker counts from the snapshot DB.

    Only groups that the harness actually knows about (present in
    ``scale_group_configs``) are returned — otherwise the synthetic worker's
    registration row would reference a group the autoscaler can't route to.

    Args:
        snapshot_db: Path to the snapshot sqlite (read-only).
        scale_group_configs: Groups the harness has loaded (see
            ``_load_scale_group_configs``). Workers are only allocated to
            groups whose ``scale_group`` value appears here.
        scale_group_pattern: Optional SQL-LIKE pattern restricting which
            groups in the snapshot contribute. Defaults to all TPU groups.
    """
    conn = sqlite3.connect(f"file:{snapshot_db}?mode=ro", uri=True)
    try:
        if scale_group_pattern is None:
            rows = conn.execute(
                "SELECT scale_group, device_variant, md_gce_zone, COUNT(*) "
                "FROM workers WHERE active = 1 AND scale_group != '' "
                "GROUP BY scale_group, device_variant, md_gce_zone "
                "ORDER BY COUNT(*) DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT scale_group, device_variant, md_gce_zone, COUNT(*) "
                "FROM workers WHERE active = 1 AND scale_group LIKE ? "
                "GROUP BY scale_group, device_variant, md_gce_zone "
                "ORDER BY COUNT(*) DESC",
                (scale_group_pattern,),
            ).fetchall()
    finally:
        conn.close()

    allocations: list[GroupAllocation] = []
    for scale_group, device_variant, zone, count in rows:
        if scale_group not in scale_group_configs:
            continue
        allocations.append(
            GroupAllocation(
                scale_group=scale_group,
                zone=zone or _zone_from_group_name(scale_group),
                device_variant=device_variant or "",
                count=int(count),
            )
        )
    return allocations


def _zone_from_group_name(name: str) -> str:
    """Best-effort zone extraction from ``tpu_<pool>_<size>-<zone>``.

    Used when the snapshot row has no ``md_gce_zone`` filled in (older rows).
    Zone is the substring after the last ``-<size>-`` separator, matching the
    name-mangling in ``iris.cluster.config._expand_tpu_pools``.
    """
    # e.g. tpu_v6e-preemptible_8-europe-west4-a -> europe-west4-a
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        return "unknown"
    return parts[1].split("-", 1)[1] if "-" in parts[1] else "unknown"


def scale_allocations(
    allocations: list[GroupAllocation],
    *,
    total: int,
) -> list[GroupAllocation]:
    """Scale per-group counts so they sum to ``total`` while preserving shape.

    Largest-remainder rounding: groups retain their proportion of the
    snapshot's active fleet. Groups with a snapshot count below 1 round-trip
    to zero so we don't spawn a worker in an exotic group (e.g. CPU VMs).
    """
    if total <= 0:
        return []
    original_sum = sum(a.count for a in allocations)
    if original_sum == 0:
        return []

    scaled: list[tuple[GroupAllocation, float]] = []
    for alloc in allocations:
        exact = alloc.count * total / original_sum
        scaled.append((alloc, exact))

    # Floor everyone, then distribute the remainder to the largest fractional parts.
    result: list[GroupAllocation] = []
    floors: list[int] = []
    fracs: list[float] = []
    for _alloc, exact in scaled:
        floors.append(int(exact))
        fracs.append(exact - int(exact))
    remaining = total - sum(floors)
    order = sorted(range(len(scaled)), key=lambda i: fracs[i], reverse=True)
    for j in range(remaining):
        floors[order[j % len(order)]] += 1

    for (src_alloc, _), count in zip(scaled, floors, strict=True):
        if count <= 0:
            continue
        result.append(
            GroupAllocation(
                scale_group=src_alloc.scale_group,
                zone=src_alloc.zone,
                device_variant=src_alloc.device_variant,
                count=count,
            )
        )
    return result


def preload_workers(
    *,
    pool: SyntheticWorkerPool,
    transitions: ControllerTransitions,
    db,
    allocations: list[GroupAllocation],
    delays: LifecycleDelays,
    slice_prefix: str = "preload",
) -> list[SyntheticWorker]:
    """Spawn real-RPC synthetic workers per ``allocations``.

    Each worker registers with a synthetic ``slice_id`` of the form
    ``<slice_prefix>-<scale_group>-<index>`` (so the preempt stimulus can
    target them via substring match on the mangled group name, the same way
    autoscaler-spawned slices are targeted).
    """
    workers: list[SyntheticWorker] = []
    for alloc in allocations:
        for i in range(alloc.count):
            # Slice name is used for preemption targeting; embed the mangled
            # group name so stimuli.preempt_workers' mangling logic finds it.
            mangled = alloc.scale_group.replace("_", "-")
            slice_id = f"{slice_prefix}-{mangled}-{i}"
            worker_id = WorkerId(f"preload-{mangled}-{i}")
            config = SyntheticWorkerConfig(
                worker_id=worker_id,
                slice_id=slice_id,
                scale_group=alloc.scale_group,
                zone=alloc.zone,
                device_variant=alloc.device_variant,
            )
            worker = SyntheticWorker(
                config,
                transitions=transitions,
                db=db,
                delays=delays,
            )
            workers.append(worker)
    # Start subprocesses in parallel — serial startup of N workers would be
    # N * ~500 ms (Popen + wait-for-READY). The pool fans out to a bounded
    # thread pool and blocks until every child has registered.
    pool.start_workers_parallel(workers)
    for worker in workers:
        pool._register_external(worker)
    logger.info(
        "preload_workers: spawned %d synthetic workers across %d groups",
        len(workers),
        len(allocations),
    )
    return workers
