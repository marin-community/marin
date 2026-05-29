# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct-provider dispatch: RunTaskRequest construction and drain logic.

Functions:
  run_request_template  — per-job RunTaskRequest template (LRU-cached)
  drain_for_direct_provider — promote PENDING tasks and snapshot running set
  build_run_request     — assemble a RunTaskRequest from a PendingDispatchRow

Dataclasses:
  SchedulingEvent, ClusterCapacity, DirectProviderBatch, DirectProviderSyncResult
"""

from dataclasses import dataclass, field

from rigging.timing import Timestamp
from sqlalchemy import select

from iris.cluster.controller import reads, writes
from iris.cluster.controller.codec import constraints_from_json, proto_from_json, resource_spec_from_scalars
from iris.cluster.controller.db import Tx
from iris.cluster.controller.lru_cache import LRUCache
from iris.cluster.controller.reads import (
    PENDING_DISPATCH_COLS,
    PendingDispatchRow,
    TaskScope,
    pending_dispatch_row,
)
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.schema import job_config_table, jobs_table, tasks_table
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES, RunningTaskEntry
from iris.cluster.types import JobName
from iris.rpc import job_pb2

DIRECT_PROVIDER_PROMOTION_RATE = 128
"""Token bucket capacity for task promotion (pods per minute).

The direct provider relies on the Kubernetes scheduler (and the cloud
autoscaler) for placement and capacity management.  Pods that cannot be
scheduled immediately stay Pending — that signal drives node provisioning.
This rate limit exists only to bound API server pressure."""


# Per-job RunTaskRequest templates are cached in RunTemplateCache.
# 4096 templates ~= worst-case concurrent job count we expect in a single
# controller process. Same-name replacement reuses the original ``job_id``,
# so ``submit_job`` evicts the cached entry before inserting the new row to
# prevent serving the prior submission's payload.
RUN_REQUEST_TEMPLATE_CACHE_SIZE = 4096

# LRU cache for per-job ``RunTaskRequest`` templates, keyed by wire job id.
# Templates carry the immutable per-job fields (entrypoint, environment,
# resources, constraints); per-attempt fields (``task_id``, ``attempt_id``)
# are stamped onto a copy at fan-out time.
RunTemplateCache = LRUCache[str, job_pb2.RunTaskRequest]


def new_run_template_cache() -> RunTemplateCache:
    return LRUCache(RUN_REQUEST_TEMPLATE_CACHE_SIZE)


@dataclass(frozen=True)
class SchedulingEvent:
    """A scheduling event from the execution backend (e.g. k8s events)."""

    task_id: str
    attempt_id: int
    event_type: str
    reason: str
    message: str
    timestamp: Timestamp


@dataclass(frozen=True)
class ClusterCapacity:
    """Aggregate capacity reported by the execution backend."""

    schedulable_nodes: int
    total_cpu_millicores: int
    available_cpu_millicores: int
    total_memory_bytes: int
    available_memory_bytes: int


@dataclass(frozen=True)
class DirectProviderBatch:
    """Work batch for a KubernetesProvider sync cycle.

    No worker_id — tasks run without a registered worker daemon.
    task_attempts rows use NULL worker_id. Kill targets are derived from
    a pod-listing diff inside the provider rather than from a buffered
    queue: any pod whose ``(task_id, attempt_id)`` is not in the desired
    set (``tasks_to_run`` union ``running_tasks``) is deleted.
    """

    tasks_to_run: list[job_pb2.RunTaskRequest] = field(default_factory=list)
    running_tasks: list[RunningTaskEntry] = field(default_factory=list)


@dataclass(frozen=True)
class DirectProviderSyncResult:
    """Result from a KubernetesProvider sync cycle."""

    updates: list[TaskUpdate] = field(default_factory=list)
    scheduling_events: list[SchedulingEvent] = field(default_factory=list)
    capacity: ClusterCapacity | None = None


def run_request_template(
    cache: RunTemplateCache,
    snap: Tx,
    job_id: JobName,
) -> job_pb2.RunTaskRequest | None:
    """Return a cached per-job ``RunTaskRequest`` template.

    Per-attempt fields (``task_id``, ``attempt_id``) are stamped onto a
    copy at fan-out time. Returns ``None`` for jobs that have no
    worker-bound dispatch (e.g. reservation holders, missing rows).
    """
    wire = job_id.to_wire()
    cached = cache.get(wire)
    if cached is not None:
        return cached

    job = reads.get_job_detail(snap, job_id)
    if job is None or job.is_reservation_holder:
        return None

    resources = resource_spec_from_scalars(
        job.res_cpu_millicores,
        job.res_memory_bytes,
        job.res_disk_bytes,
        job.res_device_json,
    )
    # proto_from_json returns shared cached instances — set via constructor
    # kwarg so RunTaskRequest copies, then mutate the copy's workdir_files
    # (never the cached source) to add inline files.
    template = job_pb2.RunTaskRequest(
        num_tasks=job.num_tasks,
        entrypoint=proto_from_json(job.entrypoint_json, job_pb2.RuntimeEntrypoint),
        environment=proto_from_json(job.environment_json, job_pb2.EnvironmentConfig),
        bundle_id=job.bundle_id,
        resources=resources,
        ports=job.ports_json,
        constraints=[c.to_proto() for c in constraints_from_json(job.constraints_json)],
        task_image=job.task_image,
    )
    for filename, data in reads.get_workdir_files(snap, job_id).items():
        template.entrypoint.workdir_files[filename] = data
    return cache.put(wire, template)


def build_run_request(
    cur: Tx,
    row: PendingDispatchRow,
    attempt_id: int,
) -> job_pb2.RunTaskRequest:
    """Assemble a RunTaskRequest for a direct-provider dispatch row."""
    # proto_from_json returns shared cached instances — set via constructor
    # kwarg so RunTaskRequest copies, then mutate the copy's workdir_files
    # (never the cached source) to add inline files.
    run_req = job_pb2.RunTaskRequest(
        task_id=row.task_id.to_wire(),
        num_tasks=row.num_tasks,
        entrypoint=proto_from_json(row.entrypoint_json, job_pb2.RuntimeEntrypoint),
        environment=proto_from_json(row.environment_json, job_pb2.EnvironmentConfig),
        bundle_id=row.bundle_id,
        resources=row.resources,
        ports=row.ports_json,
        attempt_id=attempt_id,
        constraints=[c.to_proto() for c in constraints_from_json(row.constraints_json)],
        task_image=row.task_image,
    )
    # Load inline workdir files from the job_workdir_files table.
    for filename, data in reads.get_workdir_files(cur, row.job_id).items():
        run_req.entrypoint.workdir_files[filename] = data
    # Propagate timeout for K8s activeDeadlineSeconds (Kubernetes-native enforcement).
    if row.timeout_ms is not None and row.timeout_ms > 0:
        run_req.timeout.milliseconds = row.timeout_ms
    return run_req


def drain_for_direct_provider(
    cur: Tx,
    *,
    cache: RunTemplateCache,
    max_promotions: int = DIRECT_PROVIDER_PROMOTION_RATE,
) -> DirectProviderBatch:
    """Drain pending tasks and snapshot running tasks for a direct provider sync cycle.

    Builds RunTaskRequest for two row classes:
    - Up to ``max_promotions`` PENDING rows, each promoted to ASSIGNED
      with a fresh attempt_id.
    - All ASSIGNED+null_worker rows whose pod creation may not have landed
      (controller crashed between assign-commit and ``provider.sync``, or
      the prior ``_apply_pod`` errored). ``kubectl apply`` is idempotent;
      re-issuing for a row whose pod already exists is a no-op.

    Every active null-worker row (ASSIGNED/BUILDING/RUNNING) populates
    ``running_tasks`` so the poll observes the pod's current phase. For
    ASSIGNED rows the pod was applied earlier in this same sync (or
    falls through the K8s provider's ``Pod not found`` grace path), so
    the first poll after dispatch transitions the row out of ASSIGNED.

    Kill targets are not enqueued: producing transitions move
    ``tasks.state`` directly to terminal, and the K8s provider's pod
    diff against the desired set deletes the corresponding pod on the
    next sync.
    """
    now_ms = Timestamp.now().epoch_ms()
    tasks_to_run: list[job_pb2.RunTaskRequest] = []

    # Snapshot redrive set BEFORE the PENDING promotion loop so newly-
    # promoted rows (which become ASSIGNED+null_worker mid-transaction)
    # don't get dispatched twice.
    redrive_rows = [
        pending_dispatch_row(r)
        for r in cur.execute(
            select(*PENDING_DISPATCH_COLS)
            .select_from(
                tasks_table.join(jobs_table, jobs_table.c.job_id == tasks_table.c.job_id).join(
                    job_config_table, job_config_table.c.job_id == jobs_table.c.job_id
                )
            )
            .where(
                tasks_table.c.state == int(job_pb2.TASK_STATE_ASSIGNED),
                tasks_table.c.current_worker_id.is_(None),
                jobs_table.c.is_reservation_holder == False,  # noqa: E712
            ),
        ).all()
    ]

    pending_rows: list[PendingDispatchRow] = []
    if max_promotions > 0:
        pending_rows = [
            pending_dispatch_row(r)
            for r in cur.execute(
                select(*PENDING_DISPATCH_COLS)
                .select_from(
                    tasks_table.join(jobs_table, jobs_table.c.job_id == tasks_table.c.job_id).join(
                        job_config_table, job_config_table.c.job_id == jobs_table.c.job_id
                    )
                )
                .where(
                    tasks_table.c.state == int(job_pb2.TASK_STATE_PENDING),
                    jobs_table.c.is_reservation_holder == False,  # noqa: E712
                )
                .limit(max_promotions),
            ).all()
        ]
    for row in pending_rows:
        attempt_id = row.current_attempt_id + 1
        writes.promote_to_direct_provider(
            cur,
            row.task_id,
            attempt_id,
            now_ms,
        )
        tasks_to_run.append(build_run_request(cur, row, attempt_id))

    # Redrive: pods for these rows may not exist yet (crash between
    # assign-commit and apply, or apply errored last cycle). `kubectl
    # apply` is idempotent so re-issuing for a row whose pod is already
    # there is a no-op.
    for row in redrive_rows:
        tasks_to_run.append(build_run_request(cur, row, row.current_attempt_id))

    # Poll every active row (including ASSIGNED) so a pod that just got
    # applied this cycle can transition out of ASSIGNED on the same sync.
    # Pods for ASSIGNED rows either exist (apply_pod ran above) or fall
    # through the K8s provider's "Pod not found" grace path.
    running_rows = reads.list_active_tasks(
        cur,
        TaskScope(null_worker=True),
        states=ACTIVE_TASK_STATES,
        order_by_task_id=True,
    )
    running_tasks = [
        RunningTaskEntry(
            task_id=row.task_id,
            attempt_id=row.current_attempt_id,
        )
        for row in running_rows
    ]

    return DirectProviderBatch(
        tasks_to_run=tasks_to_run,
        running_tasks=running_tasks,
    )
