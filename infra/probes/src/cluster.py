# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris cluster-state gauges: a worker-fleet snapshot (from ``ListWorkers``) and a
root-job-state breakdown (from a raw SQL ``GROUP BY``), each run as its own
collector on its own cadence. They give the controller's live worker and job
counts a durable history in finelog/GCS.

I/O (the iris RPC / SQL call) is separated from the pure ``aggregate_*`` rollups
so the labelling/windowing is unit-testable without a live controller.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import NamedTuple

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.rpc import controller_pb2
from sample import Sample

# Label value marking the fleet-wide aggregate series (no per-pool/region label).
FLEET = "fleet"

# ---- workers --------------------------------------------------------------

# Workers report their GCP region under this attribute; absent → UNKNOWN_REGION.
WORKER_REGION_ATTRIBUTE = "region"
UNKNOWN_REGION = "unknown"

METRIC_WORKER_HEALTHY = "worker_healthy"
METRIC_WORKER_CPU_MILLICORES = "worker_cpu_millicores"
METRIC_WORKER_MEMORY_BYTES = "worker_memory_bytes"
METRIC_WORKER_TPU_CHIPS = "worker_tpu_chips"


class WorkerInfo(NamedTuple):
    """The fields of one worker the gauges roll up."""

    healthy: bool
    cpu_count: int
    memory_bytes: int
    tpu_chips: int
    region: str


def aggregate_workers(workers: Sequence[WorkerInfo]) -> list[Sample]:
    """Roll healthy workers into fleet resource totals + per-region head counts.

    Only healthy workers count, resources are summed across them, and the
    per-region series is the healthy head count keyed by the worker's ``region``
    attribute. Unhealthy workers are dropped (Iris schedules at whole-VM
    granularity, so an unhealthy VM contributes no usable capacity).
    """
    healthy = 0
    cpu_millicores = 0
    memory_bytes = 0
    tpu_chips = 0
    by_region: dict[str, int] = defaultdict(int)
    for w in workers:
        if not w.healthy:
            continue
        healthy += 1
        cpu_millicores += w.cpu_count * 1000
        memory_bytes += w.memory_bytes
        tpu_chips += w.tpu_chips
        by_region[w.region] += 1

    fleet = {"scope": FLEET}
    samples = [
        Sample.of(METRIC_WORKER_HEALTHY, healthy, **fleet),
        Sample.of(METRIC_WORKER_CPU_MILLICORES, cpu_millicores, **fleet),
        Sample.of(METRIC_WORKER_MEMORY_BYTES, memory_bytes, **fleet),
        Sample.of(METRIC_WORKER_TPU_CHIPS, tpu_chips, **fleet),
    ]
    samples.extend(Sample.of(METRIC_WORKER_HEALTHY, count, region=region) for region, count in sorted(by_region.items()))
    return samples


def _worker_info(worker: controller_pb2.Controller.WorkerHealthStatus) -> WorkerInfo:
    metadata = worker.metadata
    attributes = metadata.attributes
    region = attributes[WORKER_REGION_ATTRIBUTE].string_value if WORKER_REGION_ATTRIBUTE in attributes else ""
    return WorkerInfo(
        healthy=worker.healthy,
        cpu_count=metadata.cpu_count,
        memory_bytes=metadata.memory_bytes,
        # device is a oneof; .tpu.count reads 0 for cpu/gpu workers.
        tpu_chips=metadata.device.tpu.count,
        region=region or UNKNOWN_REGION,
    )


def collect_workers(iris: RemoteClusterClient) -> list[Sample]:
    """ListWorkers → fleet resource totals + per-region healthy worker gauges."""
    return aggregate_workers([_worker_info(w) for w in iris.list_workers()])


# ---- jobs -----------------------------------------------------------------

# Trailing window for terminal jobs; in-flight jobs are counted at any age.
JOB_WINDOW_HOURS = 24.0
JOB_WINDOW_MS = int(JOB_WINDOW_HOURS * 3600 * 1000)

METRIC_JOB_INFLIGHT = "job_inflight"
METRIC_JOB_TERMINAL = "job_terminal_24h"

# JobState enum (lib/iris/src/iris/rpc/job.proto JOB_STATE_*). In-flight states
# are a live snapshot; terminal states are windowed by finished_at_ms.
JOB_STATE_NAMES = {
    1: "pending",
    2: "building",
    3: "running",
    4: "succeeded",
    5: "failed",
    6: "killed",
    7: "worker_failed",
    8: "unschedulable",
}
IN_FLIGHT_STATES = frozenset({1, 2, 3})

# Root jobs only (parent_job_id IS NULL) so the count reflects what users
# explicitly submitted rather than every root's child fan-out. In-flight states
# (1/2/3) count regardless of age so long-running experiments still show; terminal
# states (4-8) are filtered to those finished within the window via finished_at_ms.
JOB_BREAKDOWN_SQL = f"""
  SELECT state, COUNT(*) AS n
  FROM jobs
  WHERE parent_job_id IS NULL
    AND (
      state IN (1, 2, 3)
      OR (
        state IN (4, 5, 6, 7, 8)
        AND finished_at_ms > (strftime('%s', 'now') * 1000 - {JOB_WINDOW_MS})
      )
    )
  GROUP BY state
"""


class StateCount(NamedTuple):
    """One ``(state, count)`` row from the job breakdown query."""

    state: int
    count: int


def aggregate_jobs(rows: Sequence[StateCount]) -> list[Sample]:
    """Split the per-state counts into in-flight vs terminal-24h gauges.

    Emits one sample per state (labelled ``state=<name>``) plus a fleet total per
    bucket. Unknown enum values fall through as ``state_<n>`` rather than being
    dropped.
    """
    samples: list[Sample] = []
    inflight_total = 0
    terminal_total = 0
    for state, count in rows:
        name = JOB_STATE_NAMES.get(state, f"state_{state}")
        if state in IN_FLIGHT_STATES:
            samples.append(Sample.of(METRIC_JOB_INFLIGHT, count, state=name))
            inflight_total += count
        else:
            samples.append(Sample.of(METRIC_JOB_TERMINAL, count, state=name))
            terminal_total += count
    samples.append(Sample.of(METRIC_JOB_INFLIGHT, inflight_total, scope=FLEET))
    samples.append(Sample.of(METRIC_JOB_TERMINAL, terminal_total, scope=FLEET))
    return samples


def collect_jobs(iris: RemoteClusterClient) -> list[Sample]:
    """Raw-SQL job-state GROUP BY → in-flight / terminal-24h root-job gauges."""
    rows = iris.execute_raw_query(JOB_BREAKDOWN_SQL)
    return aggregate_jobs([StateCount(int(state), int(count)) for state, count in rows])
