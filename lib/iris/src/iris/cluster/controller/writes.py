# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write-side helpers: module-level functions decorated with ``@writes_to``.

The :func:`writes_to` decorator records the table set on the function as
``fn.writes_to`` / ``fn.cascades_into`` and appends the function to
:data:`REGISTERED_WRITE_FUNCTIONS`. :func:`validate` cross-checks that
registry against the ``db.caches`` registry to verify no Projection-owned
table is written outside its owning Projection. Controller startup calls
:func:`validate`; tests call it after constructing projections.

Areas covered:
  jobs           — jobs, job_config, meta sequence
  task_attempts  — task_attempts
  tasks          — tasks (insert, assign, state update)
  workers        — workers, worker_attributes
  budgets        — user_budgets
"""

import secrets
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from rigging.timing import Timestamp
from sqlalchemy import Table, bindparam, case, delete, func, insert, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError

from iris.cluster.controller.caches import CacheRegistry
from iris.cluster.controller.db import Tx
from iris.cluster.controller.projections.attempt_counts import AttemptCountsProjection
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.schema import (
    federated_jobs_table,
    federated_tasks_table,
    federation_changelog_table,
    federation_sync_state_table,
    job_config_table,
    job_workdir_files_table,
    jobs_table,
    meta_table,
    slices_table,
    task_attempts_table,
    tasks_table,
    user_budgets_table,
    workers_table,
)
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.federation.store import FederationDirection, HandoffState
from iris.cluster.types import LOCAL_CLUSTER, TERMINAL_JOB_STATES, AttemptUid, JobName, WorkerId
from iris.rpc import job_pb2
from iris.time_proto import timestamp_from_proto

REGISTERED_WRITE_FUNCTIONS: list[Callable] = []


class ConfigurationError(RuntimeError):
    """Raised by :func:`validate` when a ``@writes_to`` declaration violates
    the Projection-owned-table invariant.

    Signals a programming error: a write function declared
    ``@writes_to(<projection-owned table>)`` from outside the owning
    Projection class, or its ``cascades_into`` fans out into a
    Projection-owned table without an explicit invalidation hook.
    """


def validate(caches: CacheRegistry) -> None:
    """Check that no Projection-owned table is mutated outside its owning Projection.

    For projection-owned tables, all SQL mutations must flow through the
    Projection so the in-memory dict can be updated atomically. A write
    function that bypasses the Projection (or whose FK cascade silently
    mutates the table) leaves the dict stale.

    The exemption is by ``fn.__qualname__``: a method whose qualified
    name starts with ``<OwningProjection>.`` is allowed to mutate the
    table directly. Free functions that need to cascade into a
    Projection-owned table must call the Projection's invalidation method
    inline; they should then drop the Projection-owned table from their
    ``cascades_into`` declaration so the linkage is documented at the
    call site rather than buried in the decorator metadata.

    Called from controller startup after all projections are constructed.

    Args:
        caches: The DB's cache registry, populated by projection constructors.

    Raises:
        ConfigurationError: when a violation is detected.
    """
    owned: dict[Table, type] = {}
    for projection in caches:
        for table in projection.owns:
            owned[table] = type(projection)

    violations: list[str] = []
    for fn in REGISTERED_WRITE_FUNCTIONS:
        for table in (*fn.writes_to, *fn.cascades_into):
            if table not in owned:
                continue
            if fn.__qualname__.startswith(owned[table].__name__ + "."):
                continue
            violations.append(
                f"  - {fn.__qualname__} writes (or cascades) into {table.name!r} owned by {owned[table].__name__}"
            )

    if violations:
        raise ConfigurationError(
            "Projection-owned tables externally written:\n"
            + "\n".join(violations)
            + "\n\nFix: either move this write onto the Projection, or have "
            "the write function call the Projection's invalidation method "
            "(e.g. projection.invalidate_for_worker(tx, ...)) and document "
            "the linkage at the call site."
        )


def writes_to(
    *tables: Table,
    cascades_into: tuple[Table, ...] = (),
) -> Callable:
    """Mark a write function with the tables it mutates.

    Pure metadata. The startup-time owned-table check in
    ``projections/__init__.py`` reads ``fn.writes_to`` and
    ``fn.cascades_into`` to verify no Projection-owned table is written
    outside its Projection.

    ``cascades_into`` lists tables mutated via FK ``ON DELETE CASCADE``
    by writes to ``tables``; the check treats them identically to direct
    writes.
    """

    def deco(fn: Callable) -> Callable:
        fn.writes_to = tables  # type: ignore[attr-defined]
        fn.cascades_into = cascades_into  # type: ignore[attr-defined]
        REGISTERED_WRITE_FUNCTIONS.append(fn)
        return fn

    return deco


# ---------------------------------------------------------------------------
# Meta sequence (shared by jobs and priority insertion)
# ---------------------------------------------------------------------------


@writes_to(meta_table)
def meta_sequence_bump(tx: Tx, key: str) -> int:
    """Bump the named sequence in ``meta`` and return the new value.

    If the key is absent it is inserted with value 1. Callers reserving N
    task slots use ``base + i`` for ``i in range(N)``.
    """
    row = tx.execute(select(meta_table.c.value).where(meta_table.c.key == key)).fetchone()
    if row is None:
        tx.execute(insert(meta_table).values(key=key, value=1))
        return 1
    value = int(row[0]) + 1
    tx.execute(update(meta_table).where(meta_table.c.key == key).values(value=value))
    return value


# ---------------------------------------------------------------------------
# Job writes (previously writes/jobs.py)
# ---------------------------------------------------------------------------

_PRIORITY_INSERTION_KEY = "task_priority_insertion"


@writes_to(jobs_table)
def insert_job(
    tx: Tx,
    *,
    job_id: JobName,
    user_id: str,
    submitting_user: str,
    parent_job_id: JobName | None,
    root_job_id: str,
    depth: int,
    state: int,
    submitted_at_ms: int,
    root_submitted_at_ms: int,
    started_at_ms: int | None,
    finished_at_ms: int | None,
    scheduling_deadline_epoch_ms: int | None,
    error: str | None,
    exit_code: int | None,
    num_tasks: int,
    name: str,
    cluster: str = LOCAL_CLUSTER,
) -> None:
    """Insert one row into ``jobs``.

    TypeDecorators handle JobName → wire string and bool → 0/1 automatically.
    ``cluster`` defaults to ``'local'`` (owned here); a peer id marks the row as a
    federated handle handed off to that peer (``backend_id`` stays "").
    """
    tx.execute(
        insert(jobs_table).values(
            job_id=job_id,
            user_id=user_id,
            submitting_user=submitting_user,
            parent_job_id=parent_job_id,
            root_job_id=root_job_id,
            depth=depth,
            state=state,
            submitted_at_ms=submitted_at_ms,
            root_submitted_at_ms=root_submitted_at_ms,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            scheduling_deadline_epoch_ms=scheduling_deadline_epoch_ms,
            error=error,
            exit_code=exit_code,
            num_tasks=num_tasks,
            name=name,
            cluster=cluster,
        )
    )


@writes_to(jobs_table, tasks_table)
def stamp_backend(tx: Tx, pins: list[tuple[JobName, str]]) -> None:
    """Stamp ``backend_id`` on each job and all of its tasks.

    ``pins`` is a list of ``(job_id, backend_id)`` produced by the task->backend
    meta-scheduler. Recording the pin lets later ticks skip routing the job; the
    same id propagates to the job's tasks.
    """
    for job_id, backend_id in pins:
        tx.execute(update(jobs_table).where(jobs_table.c.job_id == job_id).values(backend_id=backend_id))
        tx.execute(update(tasks_table).where(tasks_table.c.job_id == job_id).values(backend_id=backend_id))


@writes_to(job_config_table)
def insert_job_config(
    tx: Tx,
    *,
    job_id: JobName,
    name: str,
    res_cpu_millicores: int,
    res_memory_bytes: int,
    res_disk_bytes: int,
    res_device_json: str | None,
    constraints_json: str,
    has_coscheduling: bool,
    coscheduling_group_by: str,
    scheduling_timeout_ms: int | None,
    max_task_failures: int,
    entrypoint_json: str,
    environment_json: str,
    bundle_id: str,
    ports_json: list,
    max_retries_failure: int,
    max_retries_preemption: int,
    timeout_ms: int | None,
    preemption_policy: int,
    existing_job_policy: int,
    priority_band: int,
    task_image: str,
    container_profile: int = 0,
    submit_argv_json: list | None = None,
    fail_if_exists: bool = False,
) -> None:
    """Insert one row into ``job_config``."""
    tx.execute(
        insert(job_config_table).values(
            job_id=job_id,
            name=name,
            res_cpu_millicores=res_cpu_millicores,
            res_memory_bytes=res_memory_bytes,
            res_disk_bytes=res_disk_bytes,
            res_device_json=res_device_json,
            constraints_json=constraints_json,
            has_coscheduling=has_coscheduling,
            coscheduling_group_by=coscheduling_group_by,
            scheduling_timeout_ms=scheduling_timeout_ms,
            max_task_failures=max_task_failures,
            entrypoint_json=entrypoint_json,
            environment_json=environment_json,
            bundle_id=bundle_id,
            ports_json=ports_json,
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
            timeout_ms=timeout_ms,
            preemption_policy=preemption_policy,
            existing_job_policy=existing_job_policy,
            priority_band=priority_band,
            task_image=task_image,
            container_profile=container_profile,
            submit_argv_json=submit_argv_json if submit_argv_json is not None else [],
            fail_if_exists=fail_if_exists,
        )
    )


@writes_to(jobs_table, cascades_into=(task_attempts_table, job_config_table, job_workdir_files_table))
def delete_job(tx: Tx, job_id: JobName, *, record_tombstone: bool = True) -> None:
    """Delete a job row and drop the per-job memos its cascade would strand.

    ``ON DELETE CASCADE`` removes the job's tasks, attempts, endpoints, config, and
    workdir files.

    ``record_tombstone=False`` is for a deletion that immediately re-creates the
    job id for the same requester (a federated resubmission replacing a finished
    run): the parent must mirror the fresh submission, not drop its live handle.
    """
    # Record the tombstone BEFORE the delete so a parent federating with this peer
    # learns the job was pruned. The event resolves and stamps its requester from
    # the RECEIVED federated_jobs row (still present here) and carries no FK to
    # jobs, so it survives the CASCADE that removes that row (a no-op unless this
    # root was received via handoff).
    if record_tombstone:
        record_federation_change(tx, job_id, tombstone=True)
    tx.execute(delete(jobs_table).where(jobs_table.c.job_id == job_id))
    # The attempt-counts and run-template memos are keyed by job id and derived from
    # the cascaded rows. Drop them at this chokepoint — every job-row deletion flows
    # through here — so a job later minted with the same id (a federation set-replace
    # that drops a handle, then a re-handoff) cannot serve the dead job's counts.
    tx.caches[AttemptCountsProjection].invalidate_for_jobs(tx, [job_id])
    tx.caches[RunTemplatesProjection].invalidate_for_job(tx, job_id)


@writes_to(slices_table)
def delete_slice(tx: Tx, slice_id: str) -> None:
    """Delete one slice row. Slices have no FK cascades, so this is a bare delete."""
    tx.execute(delete(slices_table).where(slices_table.c.slice_id == slice_id))


# ---------------------------------------------------------------------------
# Federation (peer side): received-job ownership + the change log
# ---------------------------------------------------------------------------


@dataclass
class _ChangelogGate:
    """Per-transaction memo for the federation changelog gate.

    ``record_federation_change`` runs on every job/task mutation but writes a row
    only for a root this controller received via handoff. ``has_received`` caches the
    one-shot "is this controller ever a peer?" probe so a controller that never is
    short-circuits; ``requester_by_root`` caches each received root's requester
    (``""`` = not received) so a reconcile flush resolves it once per root.
    """

    has_received: bool | None = None
    requester_by_root: dict[str, str] = field(default_factory=dict)


_CHANGELOG_GATE_KEY = "federation_changelog_gate"


def _changelog_gate(tx: Tx) -> _ChangelogGate:
    """The changelog gate's cache for ``tx``, created on first use.

    Lives in ``Tx.memo`` (a generic per-transaction slot) so it is dropped with the
    transaction and can never go stale — unlike a controller-lifetime cache.
    """
    gate = tx.memo.get(_CHANGELOG_GATE_KEY)
    if not isinstance(gate, _ChangelogGate):
        gate = _ChangelogGate()
        tx.memo[_CHANGELOG_GATE_KEY] = gate
    return gate


@writes_to(federated_jobs_table)
def insert_received_handle(
    tx: Tx,
    *,
    job_id: JobName,
    requester_id: str,
    owner_principal: str,
    handoff_nonce: str,
) -> None:
    """Record that ``job_id`` was handed to this peer by ``requester_id`` as a
    RECEIVED ``federated_jobs`` row (``peer_id`` is the requester; the SENT-only
    columns stay null).

    Must run after the ``jobs`` row exists (the FK). Idempotent under a re-sent
    handoff (same ``job_id``): the row is upserted, so a retried handoff never
    duplicates it.
    """
    tx.execute(
        sqlite_insert(federated_jobs_table)
        .values(
            job_id=job_id,
            direction=int(FederationDirection.RECEIVED),
            peer_id=requester_id,
            owner_principal=owner_principal,
            handoff_nonce=handoff_nonce,
        )
        .on_conflict_do_update(
            index_elements=["job_id"],
            set_={"peer_id": requester_id, "owner_principal": owner_principal, "handoff_nonce": handoff_nonce},
        )
    )
    # Keep the per-transaction requester resolution consistent with this insert so a
    # changelog event recorded later in the same transaction sees the row.
    gate = _changelog_gate(tx)
    gate.has_received = True
    gate.requester_by_root[job_id.to_wire()] = requester_id


def _federation_has_received(tx: Tx) -> bool:
    """Whether this controller holds any RECEIVED handoff, memoized per transaction.

    Lets the changelog gate short-circuit with a single probe on a controller that
    is never a peer (its ``federated_jobs`` has no RECEIVED rows), so such a
    controller writes no changelog rows and issues no per-mutation lookup.
    """
    gate = _changelog_gate(tx)
    if gate.has_received is None:
        gate.has_received = (
            tx.execute(
                select(federated_jobs_table.c.job_id)
                .where(federated_jobs_table.c.direction == int(FederationDirection.RECEIVED))
                .limit(1)
            ).first()
            is not None
        )
    return gate.has_received


def _received_requester(tx: Tx, root: JobName) -> str:
    """The requester of a received-handoff ``root``, or '' if not a received job.

    Resolved from the RECEIVED ``federated_jobs`` row (the source of truth) and
    memoized per transaction so a reconcile flush does one lookup per distinct root.
    """
    gate = _changelog_gate(tx)
    key = root.to_wire()
    if key in gate.requester_by_root:
        return gate.requester_by_root[key]
    if not _federation_has_received(tx):
        gate.requester_by_root[key] = ""
        return ""
    row = tx.execute(
        select(federated_jobs_table.c.peer_id).where(
            federated_jobs_table.c.job_id == root,
            federated_jobs_table.c.direction == int(FederationDirection.RECEIVED),
        )
    ).first()
    requester = row[0] if row is not None else ""
    gate.requester_by_root[key] = requester
    return requester


@writes_to(federation_changelog_table)
def record_federation_change(
    tx: Tx,
    job_id: JobName,
    *,
    task_index: int | None = None,
    tombstone: bool = False,
) -> None:
    """Append a federation changelog event for a received-handoff job's mutation.

    A no-op unless ``job_id``'s root was received via handoff, so a controller that
    is never a peer writes nothing. Each row carries the ``requester_id`` it is
    reported to (resolved from the RECEIVED ``federated_jobs`` row), so
    FederationSync attributes it without a join and a tombstone survives the job
    delete it records (the changelog has no FK to ``jobs``).
    """
    requester = _received_requester(tx, job_id.root_job)
    if not requester:
        return
    tx.execute(
        insert(federation_changelog_table).values(
            job_id=job_id,
            requester_id=requester,
            task_index=task_index,
            tombstone=1 if tombstone else 0,
            written_ms=Timestamp.now().epoch_ms(),
        )
    )


@writes_to(jobs_table)
def mark_jobs_running(tx: Tx, job_ids: Iterable[JobName], now_ms: int) -> None:
    """Promote each PENDING job in ``job_ids`` to RUNNING, stamping ``started_at_ms``.

    Non-PENDING jobs keep their state; ``started_at_ms`` is set only if still NULL (first assignment wins).
    """
    for job_id in job_ids:
        tx.execute(
            update(jobs_table)
            .where(jobs_table.c.job_id == job_id)
            .values(
                state=case(
                    (jobs_table.c.state == job_pb2.JOB_STATE_PENDING, job_pb2.JOB_STATE_RUNNING),
                    else_=jobs_table.c.state,
                ),
                started_at_ms=func.coalesce(jobs_table.c.started_at_ms, now_ms),
            )
        )


@writes_to(meta_table)
def reserve_priority_insertion_base(tx: Tx) -> int:
    """Bump the ``task_priority_insertion`` sequence and return the new value.

    Callers reserving N task slots use ``base + i`` for ``i in range(N)``.
    Delegates to :func:`meta_sequence_bump`.
    """
    return meta_sequence_bump(tx, _PRIORITY_INSERTION_KEY)


# ---------------------------------------------------------------------------
# Task-attempt writes (previously writes/task_attempts.py)
# ---------------------------------------------------------------------------


_ATTEMPT_UID_MINT_ATTEMPTS = 4


@writes_to(task_attempts_table)
def insert_attempt(
    tx: Tx,
    *,
    task_id: JobName,
    attempt_id: int,
    worker_id: WorkerId | None,
    state: int,
    created_at_ms: int,
) -> AttemptUid:
    """Insert one row into ``task_attempts``, minting its ``attempt_uid``.

    Every attempt gets a controller-minted 16 hex-char ``attempt_uid`` — the
    routing key workers echo back on observations. It is generated here, the
    single ``task_attempts`` insert chokepoint, so the ``NOT NULL UNIQUE``
    column is always populated. The minted value is returned to the caller.

    A UNIQUE-index collision on ``attempt_uid`` (astronomically unlikely with
    64 bits of entropy) re-mints and retries rather than aborting the
    transition. SQLite rolls back only the failed statement, so retrying the
    INSERT within the same ``tx`` is safe.
    """
    for _ in range(_ATTEMPT_UID_MINT_ATTEMPTS):
        attempt_uid = AttemptUid(secrets.token_hex(8))
        try:
            tx.execute(
                insert(task_attempts_table).values(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    worker_id=worker_id,
                    state=state,
                    created_at_ms=created_at_ms,
                    attempt_uid=attempt_uid,
                )
            )
            return attempt_uid
        except IntegrityError as exc:
            # Re-mint only on an attempt_uid collision; any other constraint
            # violation (e.g. the (task_id, attempt_id) PK) is a real bug and
            # must propagate.
            if "attempt_uid" not in str(exc.orig):
                raise
    raise RuntimeError(f"insert_attempt: exhausted attempt_uid retries for task {task_id} attempt {attempt_id}")


# ---------------------------------------------------------------------------
# Task writes (previously writes/tasks.py)
# ---------------------------------------------------------------------------


@writes_to(tasks_table)
def bulk_insert_tasks(tx: Tx, task_rows: list[dict]) -> None:
    """Insert multiple rows into ``tasks`` in a single executemany call.

    Each dict in ``task_rows`` must contain all columns required by
    :func:`insert_task`. Use :func:`task_row` to build the dicts.
    """
    if not task_rows:
        return
    tx.execute(insert(tasks_table), task_rows)
    for row in task_rows:
        record_federation_change(tx, row["job_id"], task_index=row["task_index"])


def task_row(
    *,
    task_id: JobName,
    job_id: JobName,
    task_index: int,
    state: int,
    submitted_at_ms: int,
    max_retries_failure: int,
    max_retries_preemption: int,
    priority_neg_depth: int,
    priority_root_submitted_ms: int,
    priority_insertion: int,
    priority_band: int,
) -> dict:
    """Build a parameter dict for :func:`bulk_insert_tasks`."""
    return {
        "task_id": task_id,
        "job_id": job_id,
        "task_index": task_index,
        "state": state,
        "error": None,
        "exit_code": None,
        "submitted_at_ms": submitted_at_ms,
        "started_at_ms": None,
        "finished_at_ms": None,
        "max_retries_failure": max_retries_failure,
        "max_retries_preemption": max_retries_preemption,
        "current_attempt_id": -1,
        "priority_neg_depth": priority_neg_depth,
        "priority_root_submitted_ms": priority_root_submitted_ms,
        "priority_insertion": priority_insertion,
        "priority_band": priority_band,
    }


@writes_to(tasks_table, task_attempts_table)
def assign_to_worker(
    tx: Tx,
    task_id: JobName,
    worker_id: WorkerId,
    worker_address: str,
    attempt_id: int,
    now_ms: int,
    priority_band: int | None = None,
) -> None:
    """Insert a fresh ``task_attempts`` row and assign the task to a worker.

    Stamps ``current_worker_id`` and ``current_worker_address`` on the task
    row. ``priority_band`` is stamped when provided so the preemption pass
    treats a running task's band as fixed; ``None`` leaves the column untouched.
    """
    insert_attempt(
        tx,
        task_id=task_id,
        attempt_id=attempt_id,
        worker_id=worker_id,
        state=job_pb2.TASK_STATE_ASSIGNED,
        created_at_ms=now_ms,
    )
    values: dict = {
        "state": job_pb2.TASK_STATE_ASSIGNED,
        "current_attempt_id": attempt_id,
        "started_at_ms": func.coalesce(tasks_table.c.started_at_ms, now_ms),
        "current_worker_id": worker_id,
        "current_worker_address": worker_address,
    }
    if priority_band is not None:
        values["priority_band"] = priority_band
    tx.execute(update(tasks_table).where(tasks_table.c.task_id == task_id).values(**values))
    record_federation_change(tx, task_id.parent, task_index=task_id.task_index)


@writes_to(tasks_table, task_attempts_table)
def promote_for_dispatch(
    tx: Tx,
    task_id: JobName,
    attempt_id: int,
    now_ms: int,
) -> None:
    """Insert a fresh ``task_attempts`` row and promote the task for direct-provider dispatch.

    No worker is assigned; ``current_worker_id`` is left NULL so the
    direct-provider path can track and dispatch the task via K8s.
    """
    insert_attempt(
        tx,
        task_id=task_id,
        attempt_id=attempt_id,
        worker_id=None,
        state=job_pb2.TASK_STATE_ASSIGNED,
        created_at_ms=now_ms,
    )
    tx.execute(
        update(tasks_table)
        .where(tasks_table.c.task_id == task_id)
        .values(
            state=job_pb2.TASK_STATE_ASSIGNED,
            current_attempt_id=attempt_id,
            started_at_ms=func.coalesce(tasks_table.c.started_at_ms, now_ms),
        )
    )


# ---------------------------------------------------------------------------
# Worker writes (previously writes/workers.py)
# ---------------------------------------------------------------------------


def _build_workers_upsert():
    """Build a cacheable ``INSERT ... ON CONFLICT(worker_id) DO UPDATE`` text statement.

    SA's ``sqlite_insert(...).on_conflict_do_update(...)`` form bypasses the
    compiled-statement cache (``_generate_cache_key`` returns None), so the
    burst-register path was re-compiling the SQL once per row. A ``text()``
    statement with typed bindparams gets the cache and preserves the
    ``WorkerIdType`` TypeDecorator on ``worker_id``.
    """
    cols = list(workers_table.c)
    col_names = [c.name for c in cols]
    sql = (
        f"INSERT INTO workers ({', '.join(col_names)}) "
        f"VALUES ({', '.join(f':{n}' for n in col_names)}) "
        f"ON CONFLICT(worker_id) DO UPDATE SET "
        f"{', '.join(f'{n}=excluded.{n}' for n in col_names if n != 'worker_id')}"
    )
    return text(sql).bindparams(*(bindparam(c.name, type_=c.type) for c in cols))


_WORKER_UPSERT = _build_workers_upsert()


@writes_to(workers_table)
def upsert_worker_row(tx: Tx, row: dict) -> None:
    """Insert or refresh a row in ``workers`` keyed by ``worker_id``."""
    tx.execute(_WORKER_UPSERT, row)


@writes_to(workers_table, cascades_into=(task_attempts_table,))
def remove_worker(
    tx: Tx,
    worker_id: WorkerId,
    health: WorkerHealthTracker,
) -> None:
    """Delete a worker row and clear back-references on attempts / tasks.

    ``cascades_into`` records the FK fanout to ``task_attempts``.
    The pre-emptive ``UPDATE`` on ``task_attempts`` / ``tasks`` sets
    ``current_worker_*`` to NULL before the delete so the row history
    is observable to readers in the same write transaction.
    """
    tx.execute(update(task_attempts_table).where(task_attempts_table.c.worker_id == worker_id).values(worker_id=None))
    tx.execute(update(tasks_table).where(tasks_table.c.current_worker_id == worker_id).values(current_worker_id=None))
    tx.execute(delete(workers_table).where(workers_table.c.worker_id == worker_id))
    tx.caches[WorkerAttrsProjection].invalidate_for_worker(tx, worker_id)
    tx.register(lambda: health.forget(worker_id))


# ---------------------------------------------------------------------------
# Budget writes (previously ControllerDB methods)
# ---------------------------------------------------------------------------


@writes_to(user_budgets_table)
def set_user_budget(tx: Tx, user_id: str, budget_limit: int, max_band: int, now: Timestamp) -> None:
    """Insert or update a user's budget configuration."""
    stmt = sqlite_insert(user_budgets_table).values(
        user_id=user_id,
        budget_limit=budget_limit,
        max_band=max_band,
        updated_at_ms=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["user_id"],
        set_={"budget_limit": budget_limit, "max_band": max_band, "updated_at_ms": now},
    )
    tx.execute(stmt)


# ---------------------------------------------------------------------------
# Federation (parent side): the handle + sync mirror; changelog retention
# ---------------------------------------------------------------------------


@writes_to(federated_jobs_table)
def insert_federated_handle(
    tx: Tx,
    *,
    job_id: JobName,
    peer_id: str,
    owner_principal: str,
    handoff_state: int,
    handoff_nonce: str,
) -> None:
    """Insert the SENT ``federated_jobs`` handle for a job handed off to ``peer_id``."""
    tx.execute(
        insert(federated_jobs_table).values(
            job_id=job_id,
            direction=int(FederationDirection.SENT),
            peer_id=peer_id,
            owner_principal=owner_principal,
            handoff_state=handoff_state,
            cancel_intent_version=0,
            handoff_nonce=handoff_nonce,
        )
    )


@writes_to(federated_jobs_table)
def set_handoff_state(tx: Tx, job_id: JobName, handoff_state: int) -> None:
    """Flip a handle's ``handoff_state``."""
    tx.execute(
        update(federated_jobs_table).where(federated_jobs_table.c.job_id == job_id).values(handoff_state=handoff_state)
    )


@writes_to(federated_jobs_table, jobs_table)
def promote_queued_handoff(tx: Tx, job_id: JobName, peer_id: str) -> bool:
    """Conditionally promote a QUEUED_HANDOFF handle to PENDING_HANDOFF for ``peer_id``.

    The control tick decides promotions on a (possibly stale) read snapshot, so the
    write is a compare-and-set: it flips the handle only while it is still
    ``QUEUED_HANDOFF``, uncancelled, and its job is nonterminal. A concurrent cancel
    or terminalization between the tick's read and this commit bumps
    ``cancel_intent_version`` / the job state, the CAS matches zero rows, and the
    caller drops the promotion (releasing its reservation). Returns whether the
    handle was actually promoted; on success it also stamps ``jobs.cluster`` so the
    job now names the peer it was assigned to."""
    job_nonterminal = (
        select(jobs_table.c.job_id)
        .where(jobs_table.c.job_id == job_id, jobs_table.c.state.notin_(list(TERMINAL_JOB_STATES)))
        .exists()
    )
    result = tx.execute(
        update(federated_jobs_table)
        .where(
            federated_jobs_table.c.job_id == job_id,
            federated_jobs_table.c.direction == int(FederationDirection.SENT),
            federated_jobs_table.c.handoff_state == int(HandoffState.QUEUED_HANDOFF),
            federated_jobs_table.c.cancel_intent_version == 0,
            job_nonterminal,
        )
        .values(handoff_state=int(HandoffState.PENDING_HANDOFF), peer_id=peer_id)
    )
    if result.rowcount == 0:
        return False
    tx.execute(update(jobs_table).where(jobs_table.c.job_id == job_id).values(cluster=peer_id))
    return True


@writes_to(federated_jobs_table)
def bump_cancel_intent(tx: Tx, job_id: JobName) -> None:
    """Increment a handle's ``cancel_intent_version`` (versioned, idempotent cancel)."""
    tx.execute(
        update(federated_jobs_table)
        .where(federated_jobs_table.c.job_id == job_id)
        .values(cancel_intent_version=federated_jobs_table.c.cancel_intent_version + 1)
    )


@writes_to(jobs_table)
def mark_federated_job_killed(tx: Tx, job_id: JobName, *, now_ms: int, error: str) -> None:
    """Terminate a federated handle's local job row (a job the peer never received).

    A federated handle owns no local tasks, so there is no subtree to cancel — the
    jobs row alone flips to KILLED. Used when a ``PENDING_HANDOFF`` handoff is
    cancelled before delivery, and when the peer refuses it outright; a delivered
    job's terminal state arrives via sync.
    """
    tx.execute(
        update(jobs_table)
        .where(jobs_table.c.job_id == job_id)
        .values(state=job_pb2.JOB_STATE_KILLED, finished_at_ms=now_ms, error=error)
    )


@writes_to(jobs_table)
def mark_federated_job_unschedulable(tx: Tx, job_id: JobName, *, now_ms: int, error: str) -> None:
    """Fail a queued federated job whose scheduling deadline elapsed before promotion.

    A queued handle owns no tasks, so the jobs row alone flips to ``UNSCHEDULABLE`` —
    the same terminal state a locally scheduled job reaches on a scheduling timeout.
    """
    tx.execute(
        update(jobs_table)
        .where(jobs_table.c.job_id == job_id)
        .values(state=job_pb2.JOB_STATE_UNSCHEDULABLE, finished_at_ms=now_ms, error=error)
    )


@writes_to(jobs_table)
def mirror_federated_job(
    tx: Tx,
    *,
    job_id: JobName,
    state: int,
    error: str | None,
    exit_code: int | None,
    started_at_ms: int | None,
    finished_at_ms: int | None,
    num_tasks: int,
) -> None:
    """Mirror a peer's job state onto the local federated ``jobs`` row.

    Never touches ``cluster``/``backend_id`` (the coordinate stays), only the
    state/timing/counts the local reads render.
    """
    tx.execute(
        update(jobs_table)
        .where(jobs_table.c.job_id == job_id)
        .values(
            state=state,
            error=error,
            exit_code=exit_code,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            num_tasks=num_tasks,
        )
    )


@writes_to(tasks_table, federated_tasks_table)
def mirror_federated_task(
    tx: Tx,
    *,
    task_id: JobName,
    job_id: JobName,
    task_index: int,
    peer_id: str,
    state: int,
    error: str | None,
    exit_code: int | None,
    submitted_at_ms: int | None,
    started_at_ms: int | None,
    finished_at_ms: int | None,
    current_attempt_id: int,
    worker_address: str,
    peer_worker_label: str,
) -> None:
    """Upsert a mirrored federated task row (``cluster`` set to a peer, no worker FK).

    Priority/retry columns are placeholders — a federated task is fold-excluded
    and never scheduled locally — so only the display fields the task views read
    are meaningful. ``submitted_at_ms`` carries the peer's real value so a
    not-yet-started task keeps its true submit time (not epoch 0). Failure and
    preemption counts are NOT mirrored as scalars: the parent derives them from
    the mirrored attempt rows (:func:`mirror_federated_attempts`), so they stay
    consistent with the peer without a second source of truth on the wire.
    """
    tx.execute(
        sqlite_insert(tasks_table)
        .values(
            task_id=task_id,
            job_id=job_id,
            task_index=task_index,
            state=state,
            error=error,
            exit_code=exit_code,
            submitted_at_ms=submitted_at_ms or 0,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            max_retries_failure=0,
            max_retries_preemption=0,
            current_attempt_id=current_attempt_id,
            current_worker_id=None,
            current_worker_address=worker_address,
            backend_id="",
            cluster=peer_id,
            priority_neg_depth=0,
            priority_root_submitted_ms=0,
            priority_insertion=0,
        )
        .on_conflict_do_update(
            index_elements=["task_id"],
            set_={
                "state": state,
                "error": error,
                "exit_code": exit_code,
                "submitted_at_ms": submitted_at_ms or 0,
                "started_at_ms": started_at_ms,
                "finished_at_ms": finished_at_ms,
                "current_attempt_id": current_attempt_id,
                "current_worker_address": worker_address,
                "cluster": peer_id,
            },
        )
    )
    tx.execute(
        sqlite_insert(federated_tasks_table)
        .values(task_id=task_id, peer_worker_label=peer_worker_label)
        .on_conflict_do_update(index_elements=["task_id"], set_={"peer_worker_label": peer_worker_label})
    )


@writes_to(task_attempts_table)
def mirror_federated_attempts(
    tx: Tx,
    *,
    task_id: JobName,
    attempts: Sequence[job_pb2.TaskAttempt],
) -> None:
    """Upsert a federated task's attempt rows from a sync delta.

    A federated task has no local ``workers`` row, so ``worker_id`` is NULL.
    Upserts on the ``(task_id, attempt_id)`` PK so a re-sent delta is idempotent.
    The peer's raw ``attempt_uid`` is already unique in the global index (job ids
    are cluster-unique and the parent never runs a handed-off job locally), so it
    is written verbatim.
    """
    for attempt in attempts:
        started = timestamp_from_proto(attempt.started_at).epoch_ms() if attempt.HasField("started_at") else None
        finished = timestamp_from_proto(attempt.finished_at).epoch_ms() if attempt.HasField("finished_at") else None
        attempt_uid = attempt.attempt_uid or f"{task_id.to_wire()}:{attempt.attempt_id}"
        tx.execute(
            sqlite_insert(task_attempts_table)
            .values(
                task_id=task_id,
                attempt_id=attempt.attempt_id,
                worker_id=None,
                state=attempt.state,
                created_at_ms=started or 0,
                started_at_ms=started,
                finished_at_ms=finished,
                exit_code=attempt.exit_code or None,
                error=attempt.error or None,
                attempt_uid=attempt_uid,
                backend_id="",
            )
            .on_conflict_do_update(
                index_elements=["task_id", "attempt_id"],
                set_={
                    "state": attempt.state,
                    "started_at_ms": started,
                    "finished_at_ms": finished,
                    "exit_code": attempt.exit_code or None,
                    "error": attempt.error or None,
                },
            )
        )


@writes_to(federation_sync_state_table)
def upsert_sync_cursor(tx: Tx, peer_id: str, cursor: str) -> None:
    """Persist the delta-sync cursor for ``peer_id``."""
    tx.execute(
        sqlite_insert(federation_sync_state_table)
        .values(peer_id=peer_id, cursor=cursor)
        .on_conflict_do_update(index_elements=["peer_id"], set_={"cursor": cursor})
    )
