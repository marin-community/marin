# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable Finelog copies of the controller's job, job config and attempt rows.

The controller DB deletes terminal jobs after a few days. A write path that
changes a job or an attempt marks its id here. Just before the transaction
commits, the marked rows are read back in batches, so each row is the entity's
final state in that transaction; after commit they are written to ``iris.controller_db.jobs``
/ ``iris.controller_db.job_config`` / ``iris.controller_db.task_attempts``, the same post-commit path
``iris.task_event`` uses. A rolled-back transaction writes nothing.

Only work this controller executes is written: rows for jobs whose ``cluster`` is
``LOCAL_CLUSTER``. A job handed off to a federation peer, and the mirror rows of
a peer's jobs and attempts, are written by the peer that runs them, so every job
has one writer and one key.

Each row's ``cluster`` column is Finelog's forwarding origin. A regional
Finelog forwards to the hub only rows that leave it empty, and stamps them with
its ``forwarding.cluster``. A controller behind such a store leaves it empty;
one whose Finelog is the hub (no ``forwarding:``) sets its own cluster name, so
no row reaches the hub without a cluster (see ``row_cluster``).

A failed snapshot or write is logged and drops that transaction's rows; it never
rolls back the controller's own writes or stops its later post-commit hooks.
"""

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import batched
from typing import Protocol

from finelog.client.log_client import Table
from rigging.redaction import redact_string
from rigging.timing import Timestamp
from sqlalchemy import select, tuple_
from sqlalchemy.engine.cursor import CursorResult

from iris.cluster.constraints import get_device_variant
from iris.cluster.controller.codec import proto_from_json
from iris.cluster.controller.schema import job_config_table, jobs_table, task_attempts_table, tasks_table
from iris.cluster.redaction import names_only_argv, names_only_script, names_only_submit_argv
from iris.cluster.stats.tables import IrisJob, IrisJobConfig, IrisJobTask, IrisTaskAttempt, stats_timestamp
from iris.cluster.types import LOCAL_CLUSTER, JobName, WorkerId, get_gpu_count, get_tpu_count
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

_MEMO_KEY = "controller_rows"
# Keeps each IN list under SQLite's 999 bound parameters (an attempt key is two).
_ID_CHUNK_SIZE = 400


def row_cluster(cluster_id: str, forwarding_cluster: str | None) -> str:
    """The value the controller writes to its rows' ``cluster`` column.

    Args:
        cluster_id: this cluster's config name.
        forwarding_cluster: the ``forwarding.cluster`` of this cluster's Finelog,
            or None when that Finelog does not forward (it is the hub, or local).

    Returns ``""`` when the Finelog forwards, so its forwarder ships the rows and
    stamps them; otherwise ``cluster_id``.
    """
    if not cluster_id:
        raise ValueError("controller rows need a non-empty cluster_id (the cluster config name)")
    if forwarding_cluster is None:
        return cluster_id
    if forwarding_cluster != cluster_id:
        raise ValueError(
            f"Finelog forwarding.cluster={forwarding_cluster!r} differs from cluster_id={cluster_id!r}; "
            "the hub would key this cluster's rows under another name"
        )
    return ""


@dataclass(frozen=True)
class ControllerRowTables:
    """The Finelog tables the controller copies its rows to.

    ``cluster`` is written to every row's ``cluster`` column, as ``row_cluster``
    returns it: empty when this cluster's Finelog forwards and stamps it.
    """

    cluster: str
    job_table: Table
    job_config_table: Table
    job_task_table: Table
    task_attempt_table: Table


class RowCursor(Protocol):
    """The write-transaction surface this module needs (``db.Tx``)."""

    controller_rows: ControllerRowTables | None
    memo: dict[str, object]

    def execute(self, stmt, params=None) -> CursorResult: ...

    def register(self, hook: Callable[[], None]) -> None: ...

    def register_pre_commit(self, hook: Callable[[], None]) -> None: ...


@dataclass
class _Marked:
    """Entities changed in one transaction, and their rows once snapshotted."""

    jobs: dict[JobName, None] = field(default_factory=dict)
    submitted: dict[JobName, None] = field(default_factory=dict)
    tasks: dict[JobName, None] = field(default_factory=dict)
    attempts: dict[tuple[JobName, int], None] = field(default_factory=dict)
    # Attempts' workers as they were before ``remove_worker`` cleared them.
    removed_workers: dict[tuple[JobName, int], WorkerId] = field(default_factory=dict)
    job_rows: list[IrisJob] = field(default_factory=list)
    config_rows: list[IrisJobConfig] = field(default_factory=list)
    task_rows: list[IrisJobTask] = field(default_factory=list)
    attempt_rows: list[IrisTaskAttempt] = field(default_factory=list)


def _marked(cur: RowCursor) -> _Marked | None:
    tables = cur.controller_rows
    if tables is None:
        return None
    marked = cur.memo.get(_MEMO_KEY)
    if marked is None:
        marked = _Marked()
        cur.memo[_MEMO_KEY] = marked
        cur.register_pre_commit(lambda: _snapshot_or_drop(cur, tables, marked))
        cur.register(lambda: _write(tables, marked))
    assert isinstance(marked, _Marked)
    return marked


def record_job_changes(cur: RowCursor, job_ids: Iterable[JobName]) -> None:
    """Mark ``job_ids`` as changed in this transaction."""
    marked = _marked(cur)
    if marked is None:
        return
    marked.jobs.update(dict.fromkeys(job_ids))


def record_job_submit(cur: RowCursor, job_id: JobName) -> None:
    """Mark a just-submitted job; its ``job_config`` row is written too.

    Submit is the one place a job's request is written, so ``iris.controller_db.job_config``
    gets exactly one row per job.
    """
    marked = _marked(cur)
    if marked is None:
        return
    marked.jobs[job_id] = None
    marked.submitted[job_id] = None


def record_task_changes(cur: RowCursor, task_ids: Iterable[JobName]) -> None:
    """Mark ``task_ids`` whose state, attempt, worker, error or exit code changed."""
    marked = _marked(cur)
    if marked is None:
        return
    marked.tasks.update(dict.fromkeys(task_ids))


def record_attempt_changes(cur: RowCursor, attempts: Iterable[tuple[JobName, int]]) -> None:
    """Mark ``(task_id, attempt_id)`` pairs as changed in this transaction."""
    marked = _marked(cur)
    if marked is None:
        return
    marked.attempts.update(dict.fromkeys(attempts))


def record_worker_removal(cur: RowCursor, worker_id: WorkerId) -> None:
    """Remember which attempts ran on ``worker_id`` before ``remove_worker`` clears them.

    A failed worker's attempts finish in the same transaction that removes it, so
    without this their final rows would lose ``worker_id``.
    """
    marked = _marked(cur)
    if marked is None:
        return
    a = task_attempts_table.c
    for r in cur.execute(select(a.task_id, a.attempt_id).where(a.worker_id == worker_id)):
        marked.removed_workers[(r.task_id, r.attempt_id)] = worker_id
    t = tasks_table.c
    marked.tasks.update(dict.fromkeys(cur.execute(select(t.task_id).where(t.current_worker_id == worker_id)).scalars()))


def _write(tables: ControllerRowTables, marked: _Marked) -> None:
    """Enqueue the snapshotted rows; a table whose write fails is logged and its rows dropped."""
    for table, rows in (
        (tables.job_table, marked.job_rows),
        (tables.job_config_table, marked.config_rows),
        (tables.job_task_table, marked.task_rows),
        (tables.task_attempt_table, marked.attempt_rows),
    ):
        if not rows:
            continue
        try:
            table.write(rows)
        # This runs among the post-commit hooks: raising would reach the caller of a
        # committed transaction and skip the hooks registered after it (cache invalidations).
        except Exception:
            logger.exception("controller rows: write failed, dropping %d %s rows", len(rows), type(rows[0]).__name__)


def _accelerator_columns(device_json: str | None) -> dict:
    """``iris.controller_db.job_config``'s flattened device columns, counted as the scheduler counts them."""
    device = proto_from_json(device_json, job_pb2.DeviceConfig) if device_json else job_pb2.DeviceConfig()
    if device.HasField("gpu"):
        kind, count = "gpu", get_gpu_count(device)
    elif device.HasField("tpu"):
        kind, count = "tpu", get_tpu_count(device)
    else:
        kind, count = None, 0
    return {"accelerator_kind": kind, "accelerator_type": get_device_variant(device), "accelerators_per_task": count}


def _dt_from_ms(epoch_ms: int | None) -> datetime | None:
    return Timestamp.from_ms(epoch_ms).as_naive_utc() if epoch_ms is not None else None


def _setup_scripts_json(environment_json: str) -> str:
    """The environment's setup scripts, names only. Environment variables and secrets are never copied."""
    return json.dumps([names_only_script(script) for script in json.loads(environment_json).get("setup_scripts", [])])


def _redacted_entrypoint_json(entrypoint_json: str) -> str:
    entrypoint = json.loads(entrypoint_json)
    # The controller DB already omits inline file contents; never let them reach Finelog.
    entrypoint.pop("workdir_files", None)
    entrypoint["setup_commands"] = [names_only_script(command) for command in entrypoint.get("setup_commands", [])]
    run_command = entrypoint.get("run_command")
    if run_command is not None:
        run_command["argv"] = names_only_argv(run_command.get("argv", []))
    return json.dumps(entrypoint)


def _free_text(value: str | None) -> str | None:
    """Error and status text stay readable, with token-like strings redacted."""
    return redact_string(value) if value is not None else None


def _dt(value: Timestamp | None) -> datetime | None:
    return value.as_naive_utc() if value is not None else None


def _snapshot_or_drop(cur: RowCursor, tables: ControllerRowTables, marked: _Marked) -> None:
    """Snapshot the marked rows; on failure, log and drop them so the transaction still commits.

    These rows are a telemetry copy: a bug here must not block scheduling.
    """
    try:
        _snapshot(cur, tables, marked)
    except Exception:
        logger.exception(
            "controller rows: snapshot failed, dropping %d job, %d attempt rows (jobs %s, attempts %s)",
            len(marked.jobs),
            len(marked.attempts),
            [j.to_wire() for j in list(marked.jobs)[:5]],
            [(t.to_wire(), a) for t, a in list(marked.attempts)[:5]],
        )
        marked.job_rows.clear()
        marked.config_rows.clear()
        marked.task_rows.clear()
        marked.attempt_rows.clear()


def _snapshot(cur: RowCursor, tables: ControllerRowTables, marked: _Marked) -> None:
    ts = stats_timestamp()
    j, c, a, t = jobs_table.c, job_config_table.c, task_attempts_table.c, tasks_table.c
    for chunk in batched(marked.jobs, _ID_CHUNK_SIZE):
        for r in cur.execute(select(jobs_table).where(j.job_id.in_(chunk), j.cluster == LOCAL_CLUSTER)):
            marked.job_rows.append(
                IrisJob(
                    cluster=tables.cluster,
                    root_job_id=r.root_job_id,
                    root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                    job_id=r.job_id.to_wire(),
                    submitted_at_ms=r.submitted_at_ms.epoch_ms(),
                    parent_job_id=r.parent_job_id.to_wire() if r.parent_job_id is not None else None,
                    depth=r.depth,
                    name=r.name,
                    user_id=r.user_id,
                    submitting_user=r.submitting_user,
                    state=job_pb2.JobState.Name(r.state),
                    started_at=_dt(r.started_at_ms),
                    finished_at=_dt(r.finished_at_ms),
                    exit_code=r.exit_code,
                    error=_free_text(r.error),
                    num_tasks=r.num_tasks,
                    scheduling_deadline_at=_dt_from_ms(r.scheduling_deadline_epoch_ms),
                    backend_id=r.backend_id,
                    placement=r.cluster,
                    ts=ts,
                )
            )
    for chunk in batched(marked.submitted, _ID_CHUNK_SIZE):
        query = (
            select(job_config_table, j.root_job_id, j.root_submitted_at_ms, j.submitted_at_ms)
            .join(jobs_table, j.job_id == c.job_id)
            .where(c.job_id.in_(chunk), j.cluster == LOCAL_CLUSTER)
        )
        for r in cur.execute(query):
            marked.config_rows.append(
                IrisJobConfig(
                    cluster=tables.cluster,
                    root_job_id=r.root_job_id,
                    root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                    job_id=r.job_id.to_wire(),
                    submitted_at_ms=r.submitted_at_ms.epoch_ms(),
                    res_cpu_millicores=r.res_cpu_millicores,
                    res_memory_bytes=r.res_memory_bytes,
                    res_disk_bytes=r.res_disk_bytes,
                    res_device_json=r.res_device_json,
                    **_accelerator_columns(r.res_device_json),
                    constraints_json=r.constraints_json,
                    has_coscheduling=bool(r.has_coscheduling),
                    coscheduling_group_by=r.coscheduling_group_by,
                    priority_band=job_pb2.PriorityBand.Name(r.priority_band),
                    preemption_policy=job_pb2.JobPreemptionPolicy.Name(r.preemption_policy),
                    existing_job_policy=job_pb2.ExistingJobPolicy.Name(r.existing_job_policy),
                    max_retries_failure=r.max_retries_failure,
                    max_retries_preemption=r.max_retries_preemption,
                    max_task_failures=r.max_task_failures,
                    timeout_ms=r.timeout_ms,
                    scheduling_timeout_ms=r.scheduling_timeout_ms,
                    termination_grace_ms=r.termination_grace_ms,
                    task_image=r.task_image,
                    name=r.name,
                    setup_scripts_json=_setup_scripts_json(r.environment_json),
                    entrypoint_json=_redacted_entrypoint_json(r.entrypoint_json),
                    submit_argv_json=json.dumps(names_only_submit_argv(list(r.submit_argv_json))),
                    bundle_id=r.bundle_id,
                    ports_json=json.dumps(list(r.ports_json)),
                    fail_if_exists=bool(r.fail_if_exists),
                    container_profile=job_pb2.ContainerProfile.Name(r.container_profile),
                    egress_policy=job_pb2.EgressPolicy.Name(r.egress_policy),
                    egress_allow_json=json.dumps(list(r.egress_allow_json)),
                    ts=ts,
                )
            )
    for chunk in batched(marked.tasks, _ID_CHUNK_SIZE):
        query = (
            select(tasks_table, j.root_job_id, j.root_submitted_at_ms, j.submitted_at_ms.label("job_submitted_at_ms"))
            .join(jobs_table, j.job_id == t.job_id)
            .where(t.task_id.in_(chunk), j.cluster == LOCAL_CLUSTER)
        )
        for r in cur.execute(query):
            marked.task_rows.append(
                IrisJobTask(
                    cluster=tables.cluster,
                    root_job_id=r.root_job_id,
                    root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                    job_id=r.job_id.to_wire(),
                    submitted_at_ms=r.job_submitted_at_ms.epoch_ms(),
                    task_id=r.task_id.to_wire(),
                    task_index=r.task_index,
                    state=job_pb2.TaskState.Name(r.state),
                    error=_free_text(r.error),
                    exit_code=r.exit_code,
                    task_submitted_at=r.submitted_at_ms.as_naive_utc(),
                    started_at=_dt(r.started_at_ms),
                    finished_at=_dt(r.finished_at_ms),
                    max_retries_failure=r.max_retries_failure,
                    max_retries_preemption=r.max_retries_preemption,
                    current_attempt_id=r.current_attempt_id,
                    priority_neg_depth=r.priority_neg_depth,
                    priority_root_submitted_ms=r.priority_root_submitted_ms,
                    priority_insertion=r.priority_insertion,
                    priority_band=job_pb2.PriorityBand.Name(r.priority_band),
                    container_id=r.container_id,
                    status_message=_free_text(r.status_message),
                    current_worker_id=str(r.current_worker_id) if r.current_worker_id is not None else None,
                    current_worker_address=r.current_worker_address,
                    backend_id=r.backend_id,
                    placement=r.cluster,
                    ts=ts,
                )
            )
    for chunk in batched(marked.attempts, _ID_CHUNK_SIZE):
        query = (
            select(task_attempts_table, t.job_id, j.root_job_id, j.root_submitted_at_ms, j.submitted_at_ms)
            .join(tasks_table, t.task_id == a.task_id)
            .join(jobs_table, j.job_id == t.job_id)
            .where(tuple_(a.task_id, a.attempt_id).in_(chunk), j.cluster == LOCAL_CLUSTER)
        )
        for r in cur.execute(query):
            worker_id = r.worker_id if r.worker_id is not None else marked.removed_workers.get((r.task_id, r.attempt_id))
            marked.attempt_rows.append(
                IrisTaskAttempt(
                    cluster=tables.cluster,
                    root_job_id=r.root_job_id,
                    root_submitted_at_ms=r.root_submitted_at_ms.epoch_ms(),
                    job_id=r.job_id.to_wire(),
                    submitted_at_ms=r.submitted_at_ms.epoch_ms(),
                    task_id=r.task_id.to_wire(),
                    attempt_id=r.attempt_id,
                    attempt_uid=r.attempt_uid,
                    state=job_pb2.TaskState.Name(r.state),
                    created_at=r.created_at_ms.as_naive_utc(),
                    started_at=_dt(r.started_at_ms),
                    finished_at=_dt(r.finished_at_ms),
                    exit_code=r.exit_code,
                    error=_free_text(r.error),
                    terminal_reason=r.terminal_reason,
                    worker_id=str(worker_id) if worker_id is not None else None,
                    node_name=r.node_name,
                    pod_name=r.pod_name,
                    pod_uid=r.pod_uid,
                    backend_id=r.backend_id,
                    ts=ts,
                )
            )
