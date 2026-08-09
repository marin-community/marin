# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris cluster-state gauges: a node-fleet snapshot and a
root-job-state breakdown (from a raw SQL ``GROUP BY``), each run as its own
collector on its own cadence. They give the controller's live worker and job
counts a durable history in finelog/GCS.

I/O (the iris RPC / SQL call) is separated from the pure ``aggregate_*`` rollups
so the labelling/windowing is unit-testable without a live controller.
"""

import json
from collections import defaultdict
from collections.abc import Sequence
from typing import NamedTuple, Protocol

from iris.cluster.resources.node import NodeHealth, NodeQuery, NodeSummary
from iris.cluster.resources.source import Page
from iris.rpc import query_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from sample import Sample

# Label value marking the fleet-wide aggregate series (no per-pool/region label).
FLEET = "fleet"

# ---- workers --------------------------------------------------------------

# Nodes without a provider or worker region contribute to this stable label.
UNKNOWN_REGION = "unknown"

METRIC_WORKER_HEALTHY = "worker_healthy"
METRIC_WORKER_CPU_MILLICORES = "worker_cpu_millicores"
METRIC_WORKER_MEMORY_BYTES = "worker_memory_bytes"
METRIC_WORKER_TPU_CHIPS = "worker_tpu_chips"


class NodeResourceClient(Protocol):
    def list_nodes(self, query: NodeQuery = NodeQuery()) -> Page[NodeSummary]: ...


class WorkerInfo(NamedTuple):
    """The fields of one worker the gauges roll up."""

    healthy: bool
    cpu_millicores: int
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
        cpu_millicores += w.cpu_millicores
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


def _worker_info(node: NodeSummary) -> WorkerInfo:
    capacity = node.capacity
    return WorkerInfo(
        healthy=node.health is NodeHealth.READY,
        cpu_millicores=capacity.cpu_millicores,
        memory_bytes=capacity.memory_bytes,
        tpu_chips=capacity.accelerator_count if capacity.accelerator_kind == "tpu" else 0,
        region=node.region or UNKNOWN_REGION,
    )


def collect_workers(iris: NodeResourceClient) -> list[Sample]:
    """Roll the complete Node inventory into worker-fleet gauges."""
    workers: list[WorkerInfo] = []
    query = NodeQuery(page_size=500)
    while True:
        page = iris.list_nodes(query)
        workers.extend(_worker_info(node) for node in page.items)
        if page.next_page_token is None:
            return aggregate_workers(workers)
        query = NodeQuery(page_size=500, page_token=page.next_page_token)


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


def collect_jobs(client: ControllerServiceClientSync) -> list[Sample]:
    """Raw-SQL job-state GROUP BY → in-flight / terminal-24h root-job gauges.

    Talks to the controller's ``ExecuteRawQuery`` RPC directly (the same call the
    ``iris query`` CLI makes); each response row is a JSON-encoded array of cell
    values. Admin-only on the server, but the marin cluster's null-auth mode
    grants it without a bearer token.
    """
    response = client.execute_raw_query(query_pb2.RawQueryRequest(sql=JOB_BREAKDOWN_SQL))
    rows = [json.loads(row) for row in response.rows]
    return aggregate_jobs([StateCount(int(state), int(count)) for state, count in rows])
