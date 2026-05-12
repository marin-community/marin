# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SA Core write helpers for ``jobs``, ``job_config``, ``job_workdir_files``,
``users`` and the ``meta`` sequence counter.

Stage 11 of the SA Core migration. Ports every write that today lives on
:class:`iris.cluster.controller.stores.JobStore` into module-level
functions taking a :class:`iris.cluster.controller.db_v2.Tx`. The legacy
:class:`JobStore` methods are unchanged; parity tests in
``tests/cluster/controller/test_writes_jobs.py`` assert the two paths
leave the DB in identical states.
"""

from collections.abc import Iterable, Mapping, Sequence

from sqlalchemy import text

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema_v2 import (
    job_config_table,
    job_workdir_files_table,
    jobs_table,
    meta_table,
    users_table,
)
from iris.cluster.controller.stores import JobConfigInsertParams, JobInsertParams
from iris.cluster.controller.writes import writes_to
from iris.cluster.types import TERMINAL_JOB_STATES, JobName
from iris.rpc import job_pb2

_INSERT_JOB_SQL = text(
    "INSERT INTO jobs("
    "job_id, user_id, parent_job_id, root_job_id, depth, state, submitted_at_ms, "
    "root_submitted_at_ms, started_at_ms, finished_at_ms, scheduling_deadline_epoch_ms, "
    "error, exit_code, num_tasks, is_reservation_holder, name, has_reservation"
    ") VALUES ("
    ":job_id, :user_id, :parent_job_id, :root_job_id, :depth, :state, :submitted_at_ms, "
    ":root_submitted_at_ms, :started_at_ms, :finished_at_ms, :scheduling_deadline_epoch_ms, "
    ":error, :exit_code, :num_tasks, :is_reservation_holder, :name, :has_reservation)"
)

_INSERT_JOB_CONFIG_SQL = text(
    "INSERT INTO job_config("
    "job_id, name, has_reservation, "
    "res_cpu_millicores, res_memory_bytes, res_disk_bytes, res_device_json, "
    "constraints_json, has_coscheduling, coscheduling_group_by, "
    "scheduling_timeout_ms, max_task_failures, "
    "entrypoint_json, environment_json, bundle_id, ports_json, "
    "max_retries_failure, max_retries_preemption, timeout_ms, "
    "preemption_policy, existing_job_policy, priority_band, "
    "task_image, submit_argv_json, reservation_json, fail_if_exists"
    ") VALUES ("
    ":job_id, :name, :has_reservation, "
    ":res_cpu_millicores, :res_memory_bytes, :res_disk_bytes, :res_device_json, "
    ":constraints_json, :has_coscheduling, :coscheduling_group_by, "
    ":scheduling_timeout_ms, :max_task_failures, "
    ":entrypoint_json, :environment_json, :bundle_id, :ports_json, "
    ":max_retries_failure, :max_retries_preemption, :timeout_ms, "
    ":preemption_policy, :existing_job_policy, :priority_band, "
    ":task_image, :submit_argv_json, :reservation_json, :fail_if_exists)"
)

_DELETE_JOB_SQL = text("DELETE FROM jobs WHERE job_id = :job_id")

_INSERT_WORKDIR_FILE_SQL = text(
    "INSERT INTO job_workdir_files(job_id, filename, data) VALUES (:job_id, :filename, :data)"
)

_INSERT_USER_SQL = text("INSERT OR IGNORE INTO users(user_id, created_at_ms) VALUES (:user_id, :now_ms)")

_MARK_RUNNING_IF_PENDING_SQL = text(
    "UPDATE jobs SET state = CASE WHEN state = :pending THEN :running ELSE state END, "
    "started_at_ms = COALESCE(started_at_ms, :now_ms) WHERE job_id = :job_id"
)

_GET_SEQUENCE_SQL = text("SELECT value FROM meta WHERE key = :key")
_INSERT_SEQUENCE_SQL = text("INSERT INTO meta(key, value) VALUES (:key, :value)")
_UPDATE_SEQUENCE_SQL = text("UPDATE meta SET value = :value WHERE key = :key")

_PRIORITY_INSERTION_KEY = "task_priority_insertion"


@writes_to(jobs_table, job_config_table, job_workdir_files_table, users_table, meta_table)
def insert_job(tx: Tx, params: JobInsertParams) -> None:
    """Insert one row into ``jobs``."""
    tx.execute(
        _INSERT_JOB_SQL,
        {
            "job_id": params.job_id.to_wire(),
            "user_id": params.user_id,
            "parent_job_id": params.parent_job_id,
            "root_job_id": params.root_job_id,
            "depth": params.depth,
            "state": params.state,
            "submitted_at_ms": params.submitted_at_ms,
            "root_submitted_at_ms": params.root_submitted_at_ms,
            "started_at_ms": params.started_at_ms,
            "finished_at_ms": params.finished_at_ms,
            "scheduling_deadline_epoch_ms": params.scheduling_deadline_epoch_ms,
            "error": params.error,
            "exit_code": params.exit_code,
            "num_tasks": params.num_tasks,
            "is_reservation_holder": 1 if params.is_reservation_holder else 0,
            "name": params.name,
            "has_reservation": 1 if params.has_reservation else 0,
        },
    )


@writes_to(job_config_table)
def insert_job_config(tx: Tx, params: JobConfigInsertParams) -> None:
    """Insert one row into ``job_config``."""
    tx.execute(
        _INSERT_JOB_CONFIG_SQL,
        {
            "job_id": params.job_id.to_wire(),
            "name": params.name,
            "has_reservation": 1 if params.has_reservation else 0,
            "res_cpu_millicores": params.res_cpu_millicores,
            "res_memory_bytes": params.res_memory_bytes,
            "res_disk_bytes": params.res_disk_bytes,
            "res_device_json": params.res_device_json,
            "constraints_json": params.constraints_json,
            "has_coscheduling": 1 if params.has_coscheduling else 0,
            "coscheduling_group_by": params.coscheduling_group_by,
            "scheduling_timeout_ms": params.scheduling_timeout_ms,
            "max_task_failures": params.max_task_failures,
            "entrypoint_json": params.entrypoint_json,
            "environment_json": params.environment_json,
            "bundle_id": params.bundle_id,
            "ports_json": params.ports_json,
            "max_retries_failure": params.max_retries_failure,
            "max_retries_preemption": params.max_retries_preemption,
            "timeout_ms": params.timeout_ms,
            "preemption_policy": params.preemption_policy,
            "existing_job_policy": params.existing_job_policy,
            "priority_band": params.priority_band,
            "task_image": params.task_image,
            "submit_argv_json": params.submit_argv_json,
            "reservation_json": params.reservation_json,
            "fail_if_exists": 1 if params.fail_if_exists else 0,
        },
    )


@writes_to(jobs_table)
def delete_job(tx: Tx, job_id: JobName) -> None:
    """Delete a job row. ``ON DELETE CASCADE`` handles tasks, attempts, endpoints."""
    tx.execute(_DELETE_JOB_SQL, {"job_id": job_id.to_wire()})


@writes_to(job_workdir_files_table)
def insert_workdir_files(tx: Tx, job_id: JobName, files: Mapping[str, bytes]) -> None:
    """Insert each ``{filename: data}`` pair as a row in ``job_workdir_files``."""
    if not files:
        return
    tx.executemany(
        _INSERT_WORKDIR_FILE_SQL,
        [{"job_id": job_id.to_wire(), "filename": name, "data": data} for name, data in files.items()],
    )


@writes_to(jobs_table)
def update_state_if_not_terminal(
    tx: Tx,
    job_id: JobName,
    new_state: int,
    error: str | None,
    finished_at_ms: int | None,
) -> None:
    """Set a new state on a single job, skipping rows already in a terminal state."""
    guard_keys = [f"g{i}" for i in range(len(TERMINAL_JOB_STATES))]
    guard_clause = ",".join(f":{k}" for k in guard_keys)
    stmt = text(
        "UPDATE jobs SET state = :new_state, error = :error, "
        "finished_at_ms = COALESCE(finished_at_ms, :finished_at_ms) "
        f"WHERE job_id = :job_id AND state NOT IN ({guard_clause})"
    )
    params: dict[str, object] = {
        "new_state": new_state,
        "error": error,
        "finished_at_ms": finished_at_ms,
        "job_id": job_id.to_wire(),
    }
    for k, v in zip(guard_keys, TERMINAL_JOB_STATES, strict=True):
        params[k] = v
    tx.execute(stmt, params)


@writes_to(jobs_table)
def bulk_update_state(
    tx: Tx,
    job_ids: Sequence[JobName],
    new_state: int,
    error: str | None,
    finished_at_ms: int | None,
    guard_states: Iterable[int],
) -> None:
    """Set state on many jobs; rows in any of ``guard_states`` are skipped."""
    if not job_ids:
        return
    wire_ids = [jid.to_wire() for jid in job_ids]
    guard = tuple(guard_states)
    job_keys = [f"j{i}" for i in range(len(wire_ids))]
    guard_keys = [f"g{i}" for i in range(len(guard))]
    stmt = text(
        "UPDATE jobs SET state = :new_state, error = :error, "
        "finished_at_ms = COALESCE(finished_at_ms, :finished_at_ms) "
        f"WHERE job_id IN ({','.join(f':{k}' for k in job_keys)}) "
        f"AND state NOT IN ({','.join(f':{k}' for k in guard_keys)})"
    )
    params: dict[str, object] = {
        "new_state": new_state,
        "error": error,
        "finished_at_ms": finished_at_ms,
    }
    for k, v in zip(job_keys, wire_ids, strict=True):
        params[k] = v
    for k, v in zip(guard_keys, guard, strict=True):
        params[k] = v
    tx.execute(stmt, params)


@writes_to(jobs_table)
def mark_running_if_pending(tx: Tx, job_id: JobName, now_ms: int) -> None:
    """Advance PENDING → RUNNING and set ``started_at_ms`` if not already populated."""
    tx.execute(
        _MARK_RUNNING_IF_PENDING_SQL,
        {
            "pending": job_pb2.JOB_STATE_PENDING,
            "running": job_pb2.JOB_STATE_RUNNING,
            "now_ms": now_ms,
            "job_id": job_id.to_wire(),
        },
    )


@writes_to(jobs_table)
def apply_recomputed_state(
    tx: Tx,
    job_id: JobName,
    new_state: int,
    now_ms: int,
    error: str | None,
) -> None:
    """Write the result of ``_recompute_job_state`` back to the row.

    Sets ``started_at_ms`` (if moving to RUNNING), ``finished_at_ms``
    (if moving to a terminal state), and ``error`` (if the terminal
    reason warrants one). The caller has already decided ``new_state``
    differs from the current state.
    """
    terminal_keys = [f"t{i}" for i in range(len(TERMINAL_JOB_STATES))]
    terminal_clause = ",".join(f":{k}" for k in terminal_keys)
    stmt = text(
        "UPDATE jobs SET state = :new_state, "
        "started_at_ms = CASE WHEN :new_state_a = :running THEN COALESCE(started_at_ms, :now_ms_a) "
        "ELSE started_at_ms END, "
        f"finished_at_ms = CASE WHEN :new_state_b IN ({terminal_clause}) "
        "THEN :now_ms_b ELSE finished_at_ms END, "
        "error = CASE WHEN :new_state_c IN (:failed, :killed, :unschedulable, :worker_failed) "
        "THEN :error ELSE error END "
        "WHERE job_id = :job_id"
    )
    params: dict[str, object] = {
        "new_state": new_state,
        "new_state_a": new_state,
        "new_state_b": new_state,
        "new_state_c": new_state,
        "running": job_pb2.JOB_STATE_RUNNING,
        "now_ms_a": now_ms,
        "now_ms_b": now_ms,
        "failed": job_pb2.JOB_STATE_FAILED,
        "killed": job_pb2.JOB_STATE_KILLED,
        "unschedulable": job_pb2.JOB_STATE_UNSCHEDULABLE,
        "worker_failed": job_pb2.JOB_STATE_WORKER_FAILED,
        "error": error,
        "job_id": job_id.to_wire(),
    }
    for k, v in zip(terminal_keys, TERMINAL_JOB_STATES, strict=True):
        params[k] = v
    tx.execute(stmt, params)


@writes_to(meta_table)
def reserve_priority_insertion_base(tx: Tx) -> int:
    """Bump the ``task_priority_insertion`` sequence and return the new value.

    Callers reserving N task slots use ``base + i`` for ``i in range(N)``.
    Mirrors :meth:`ControllerDB.next_sequence` against the ``meta`` table.
    """
    row = tx.execute(_GET_SEQUENCE_SQL, {"key": _PRIORITY_INSERTION_KEY}).fetchone()
    if row is None:
        tx.execute(_INSERT_SEQUENCE_SQL, {"key": _PRIORITY_INSERTION_KEY, "value": 1})
        return 1
    value = int(row[0]) + 1
    tx.execute(_UPDATE_SEQUENCE_SQL, {"key": _PRIORITY_INSERTION_KEY, "value": value})
    return value


@writes_to(users_table)
def ensure_user(tx: Tx, user_id: str, now_ms: int) -> None:
    """Idempotently create a ``users`` row at submission time."""
    tx.execute(_INSERT_USER_SQL, {"user_id": user_id, "now_ms": now_ms})
