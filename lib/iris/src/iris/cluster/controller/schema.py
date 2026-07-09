# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy Core schema for the controller database.

Mirrors the on-disk schema produced by ``controller/migrations/``. Auth tables
live on a separate ``auth_metadata`` because they are stored in the attached
``auth.sqlite3`` database, not the main controller DB.

``server_default`` holds the literal value to store, not a SQL fragment:
SQLAlchemy quotes it into the DDL, so a string must be passed unquoted (``""``
renders ``DEFAULT ''``, ``"{}"`` renders ``DEFAULT '{}'``). Pre-wrapping it
(``"''"`` / ``"'{}'"``) stores the quote characters themselves, which then fails
``json.loads`` for the JSON-backed columns. Integer columns pass the numeric
value the same way (``"0"``); SQLite's type affinity coerces it to an int.
"""

import json
import threading
from collections import OrderedDict
from typing import Any, ClassVar

from rigging.timing import Timestamp
from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    UniqueConstraint,
    func,
    literal_column,
    select,
    text,
)
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.types import TypeDecorator

from iris.cluster.types import LOCAL_CLUSTER, JobName, WorkerId

WORKER_ATTR_VALUE_TYPE_CHECK = "value_type IN ('str', 'int', 'float')"


class JobNameType(TypeDecorator):
    """Adapts ``JobName`` to/from a TEXT column."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return value.to_wire()

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return JobName.from_wire(value)


class WorkerIdType(TypeDecorator):
    """Adapts ``WorkerId`` to/from a TEXT column."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return WorkerId(str(value))


class TimestampMsType(TypeDecorator):
    """Adapts ``Timestamp`` to/from an INTEGER (epoch milliseconds)."""

    impl = Integer
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        return value.epoch_ms()

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return Timestamp.from_ms(int(value))


class BoolIntType(TypeDecorator):
    """Adapts ``bool`` to/from an INTEGER column storing 0 or 1."""

    impl = Integer
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return 1 if value else 0

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return bool(int(value))


class CachedProto(TypeDecorator):
    """Bytes-keyed LRU memo for protobuf blob columns.

    Round-trip: ``message.SerializeToString()`` on the way in,
    ``message_cls.FromString(bytes)`` on the way out. Two rows whose
    blobs decode to identical bytes share the same Python object via a
    process-wide cache.

    The cache is global across every ``CachedProto`` instance regardless
    of ``message_cls``: a single dict, a single lock, a single eviction
    policy. When the cache reaches ``_MAX_SIZE`` entries the oldest 25%
    of entries (``_MAX_SIZE // 4``) are dropped in one batch.
    """

    impl = LargeBinary
    cache_ok = True

    _MAX_SIZE: ClassVar[int] = 8192
    _global_cache: ClassVar[OrderedDict[bytes, Any]] = OrderedDict()
    _global_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, message_cls: type) -> None:
        super().__init__()
        self._message_cls = message_cls

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return value.SerializeToString()

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        raw = bytes(value)
        with self._global_lock:
            hit = self._global_cache.get(raw)
            if hit is not None:
                return hit
        decoded = self._message_cls.FromString(raw)
        with self._global_lock:
            # Re-check under the lock to avoid two threads inserting different
            # decoded objects for the same bytes (preserving is-identity).
            existing = self._global_cache.get(raw)
            if existing is not None:
                return existing
            if len(self._global_cache) >= self._MAX_SIZE:
                evict_count = self._MAX_SIZE // 4
                for _ in range(evict_count):
                    self._global_cache.popitem(last=False)
            self._global_cache[raw] = decoded
        return decoded


class JSONList(TypeDecorator):
    """Adapts a JSON-encoded list to/from a TEXT column.

    On write: accepts a list and stores it as a JSON string.
    On read: decodes the JSON string back to a list.

    Only for plain list columns (e.g. ``list[int]``, ``list[str]``).
    Proto-encoded JSON columns use ``CachedProto`` instead.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return json.dumps(list(value))

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return []
        return json.loads(value)


class JSONDict(TypeDecorator):
    """Adapts a JSON-encoded dict to/from a TEXT column.

    On write: accepts a dict and stores it as a JSON string.
    On read: decodes the JSON string back to a dict.

    Only for plain dict columns (e.g. ``dict[str, str]``).
    Proto-encoded JSON columns use ``CachedProto`` instead.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return {}
        return json.loads(value)


metadata = MetaData()
auth_metadata = MetaData()


schema_migrations_table = Table(
    "schema_migrations",
    metadata,
    Column("name", String, primary_key=True),
    Column("applied_at_ms", Integer, nullable=False),
)


meta_table = Table(
    "meta",
    metadata,
    Column("key", String, primary_key=True),
    Column("value", Integer, nullable=False),
)


jobs_table = Table(
    "jobs",
    metadata,
    Column("job_id", JobNameType, primary_key=True),
    # Plain owner string: roles are resolved from the config-derived RolePolicy
    # (see controller/auth.py), so there is no ``users`` table to anchor an FK to.
    Column("user_id", String, nullable=False),
    # The authenticated principal that submitted this job, distinct from the
    # friendly ``user_id`` owner: an IAP/JWT email, or ``local_admin`` for a
    # CIDR/loopback (null-auth) submission. Drives per-cluster federation
    # authorization; a child inherits its root's value and a federated handoff
    # carries it as a signed claim the receiving peer re-checks.
    Column("submitting_user", String, nullable=False, server_default=""),
    Column("parent_job_id", JobNameType, ForeignKey("jobs.job_id", ondelete="CASCADE")),
    Column("root_job_id", String, nullable=False),
    Column("depth", Integer, nullable=False),
    Column("state", Integer, nullable=False),
    Column("submitted_at_ms", TimestampMsType, nullable=False),
    Column("root_submitted_at_ms", TimestampMsType, nullable=False),
    Column("started_at_ms", TimestampMsType),
    Column("finished_at_ms", TimestampMsType),
    Column("scheduling_deadline_epoch_ms", Integer),
    Column("error", String),
    Column("exit_code", Integer),
    Column("num_tasks", Integer, nullable=False),
    Column("name", String, nullable=False, server_default=""),
    Column("backend_id", String, nullable=False, server_default=""),
    # The cluster coordinate, always set: "local" == owned by this controller,
    # "<peer>" == handed off to that peer cluster. Orthogonal to backend_id, but
    # a local backend_id implies cluster="local" by construction (a federated job
    # has backend_id="").
    Column("cluster", String, nullable=False, server_default=LOCAL_CLUSTER),
    Index("idx_jobs_parent", "parent_job_id"),
    Index("idx_jobs_state", text("state"), text("submitted_at_ms DESC")),
    Index("idx_jobs_depth_state", text("depth"), text("state"), text("submitted_at_ms DESC")),
    Index("idx_jobs_user_state", "user_id", "state"),
    Index("idx_jobs_root_depth", "root_job_id", "depth"),
    Index("idx_jobs_depth_submitted", text("depth"), text("submitted_at_ms DESC")),
    Index("idx_jobs_name", "name"),
)


job_config_table = Table(
    "job_config",
    metadata,
    Column("job_id", JobNameType, ForeignKey("jobs.job_id", ondelete="CASCADE"), primary_key=True),
    Column("name", String, nullable=False, server_default=""),
    Column("res_cpu_millicores", Integer, nullable=False, server_default="0"),
    Column("res_memory_bytes", Integer, nullable=False, server_default="0"),
    Column("res_disk_bytes", Integer, nullable=False, server_default="0"),
    Column("res_device_json", String),
    Column("constraints_json", String),
    Column("has_coscheduling", BoolIntType, nullable=False, server_default="0"),
    Column("coscheduling_group_by", String, nullable=False, server_default=""),
    Column("scheduling_timeout_ms", Integer),
    Column("max_task_failures", Integer, nullable=False, server_default="0"),
    Column("entrypoint_json", String, nullable=False, server_default="{}"),
    Column("environment_json", String, nullable=False, server_default="{}"),
    Column("bundle_id", String, nullable=False, server_default=""),
    Column("ports_json", JSONList(), nullable=False, server_default="[]"),
    Column("max_retries_failure", Integer, nullable=False, server_default="0"),
    Column("max_retries_preemption", Integer, nullable=False, server_default="100"),
    Column("timeout_ms", Integer),
    Column("preemption_policy", Integer, nullable=False, server_default="0"),
    Column("existing_job_policy", Integer, nullable=False, server_default="0"),
    Column("priority_band", Integer, nullable=False, server_default="0"),
    Column("task_image", String, nullable=False, server_default=""),
    Column("submit_argv_json", JSONList(), nullable=False, server_default="[]"),
    Column("fail_if_exists", BoolIntType, nullable=False, server_default="0"),
    Column("container_profile", Integer, nullable=False, server_default="0"),
    Index("idx_job_config_name", "name"),
)


job_workdir_files_table = Table(
    "job_workdir_files",
    metadata,
    Column("job_id", JobNameType, ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False),
    Column("filename", String, nullable=False),
    Column("data", LargeBinary, nullable=False),
    PrimaryKeyConstraint("job_id", "filename"),
)


tasks_table = Table(
    "tasks",
    metadata,
    Column("task_id", JobNameType, primary_key=True),
    Column("job_id", JobNameType, ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False),
    Column("task_index", Integer, nullable=False),
    Column("state", Integer, nullable=False),
    Column("error", String),
    Column("exit_code", Integer),
    Column("submitted_at_ms", TimestampMsType, nullable=False),
    Column("started_at_ms", TimestampMsType),
    Column("finished_at_ms", TimestampMsType),
    Column("max_retries_failure", Integer, nullable=False),
    Column("max_retries_preemption", Integer, nullable=False),
    # failure_count / preemption_count are NOT stored: they are derived from the
    # task's attempt rows (iris.cluster.controller.attempt_counts) and served from
    # an in-memory cache. current_attempt_id stays denormalized — it is the live
    # in-flight-attempt pointer, written atomically with each attempt insert.
    Column("current_attempt_id", Integer, nullable=False, server_default="-1"),
    Column("priority_neg_depth", Integer, nullable=False),
    Column("priority_root_submitted_ms", Integer, nullable=False),
    Column("priority_insertion", Integer, nullable=False),
    Column("priority_band", Integer, nullable=False, server_default="2"),
    Column("container_id", String),
    Column("current_worker_id", WorkerIdType, ForeignKey("workers.worker_id", ondelete="SET NULL")),
    Column("current_worker_address", String),
    Column("backend_id", String, nullable=False, server_default=""),
    # The cluster coordinate; see jobs.cluster. A federated task has cluster set to
    # a peer id, backend_id="", and no local worker/attempt rows. Every
    # control-plane reader sources the ``local_tasks`` selectable (cluster =
    # 'local') so these rows are structurally invisible to the scheduler fold.
    Column("cluster", String, nullable=False, server_default=LOCAL_CLUSTER),
    UniqueConstraint("job_id", "task_index", name="tasks_job_id_task_index_key"),
    Index("idx_tasks_job_state", "job_id", "state"),
    Index("idx_tasks_backend_state", "backend_id", "state"),
    Index(
        "idx_tasks_pending",
        "state",
        "priority_band",
        "priority_neg_depth",
        "priority_root_submitted_ms",
        "submitted_at_ms",
        "priority_insertion",
    ),
    # Partial mirrors of the two hot control-plane scans, restricted to local
    # rows so the ``local_tasks`` fold-exclusion costs nothing at read time.
    Index(
        "idx_tasks_pending_local",
        "state",
        "priority_band",
        "priority_neg_depth",
        "priority_root_submitted_ms",
        "submitted_at_ms",
        "priority_insertion",
        sqlite_where=text(f"cluster = '{LOCAL_CLUSTER}'"),
    ),
    Index("idx_tasks_state", "state"),
    Index("idx_tasks_state_local", "state", sqlite_where=text(f"cluster = '{LOCAL_CLUSTER}'")),
    Index("idx_tasks_state_attempt", "state", "task_id", "current_attempt_id", "job_id"),
    Index(
        "idx_tasks_current_worker",
        "current_worker_id",
        sqlite_where=text("current_worker_id IS NOT NULL"),
    ),
)


# The planner mis-estimates `tasks.state IN (<active states>)` as ~14% of rows
# (sqlite_stat1 only knows the index's average rows-per-value), so it full-scans
# instead of driving off the small active set via idx_tasks_state. Wrap such
# predicates to force the state-driven plan; likelihood()'s probability must be a
# literal, not a bound parameter.
_RARE_STATE_PROBABILITY = literal_column("0.005")


def hint_rare_state(predicate: ColumnElement[bool]) -> ColumnElement[bool]:
    """Hint SQLite that ``predicate`` matches few rows, so it drives off idx_tasks_state."""
    return func.likelihood(predicate, _RARE_STATE_PROBABILITY)


# Structural fold-exclusion for federated tasks. Every control-plane reader
# (scheduling, routing, reconcile/dispatch, budget, capacity/autoscale, cancel,
# timeout, pruner) sources this selectable instead of raw ``tasks_table`` so
# rows handed off to a peer (``cluster != 'local'``) are invisible to the fold.
# SQLite flattens the subquery and drives off the ``… WHERE cluster = 'local'``
# partial indexes, so the exclusion is free at read time. User-facing read paths
# (job/task detail, list, status) keep reading raw ``tasks_table`` so federated
# rows still render in listings.
local_tasks = select(tasks_table).where(tasks_table.c.cluster == LOCAL_CLUSTER).subquery("local_tasks")


task_attempts_table = Table(
    "task_attempts",
    metadata,
    Column("task_id", JobNameType, ForeignKey("tasks.task_id", ondelete="CASCADE"), nullable=False),
    Column("attempt_id", Integer, nullable=False),
    Column("worker_id", WorkerIdType, ForeignKey("workers.worker_id", ondelete="SET NULL")),
    Column("state", Integer, nullable=False),
    Column("created_at_ms", TimestampMsType, nullable=False),
    Column("started_at_ms", TimestampMsType),
    Column("finished_at_ms", TimestampMsType),
    Column("exit_code", Integer),
    Column("error", String),
    Column("attempt_uid", String, nullable=False),
    Column("backend_id", String, nullable=False, server_default=""),
    PrimaryKeyConstraint("task_id", "attempt_id"),
    Index("idx_task_attempts_worker_task", "worker_id", "task_id", "attempt_id"),
    Index(
        "idx_task_attempts_live_workerbound",
        "worker_id",
        sqlite_where=text("worker_id IS NOT NULL AND finished_at_ms IS NULL"),
    ),
    Index("idx_task_attempts_uid", "attempt_uid", unique=True),
    Index("idx_task_attempts_backend", "backend_id"),
    # Covers the failure/preemption derivation (COUNT by state, filtered on
    # started_at_ms), grouped per task — see iris.cluster.controller.attempt_counts.
    Index("idx_task_attempts_task_state", "task_id", "state", "started_at_ms"),
)


workers_table = Table(
    "workers",
    metadata,
    Column("worker_id", WorkerIdType, primary_key=True),
    Column("address", String, nullable=False),
    Column("md_hostname", String, nullable=False, server_default=""),
    Column("md_ip_address", String, nullable=False, server_default=""),
    Column("md_cpu_count", Integer, nullable=False, server_default="0"),
    Column("md_memory_bytes", Integer, nullable=False, server_default="0"),
    Column("md_disk_bytes", Integer, nullable=False, server_default="0"),
    Column("md_tpu_name", String, nullable=False, server_default=""),
    Column("md_tpu_worker_hostnames", String, nullable=False, server_default=""),
    Column("md_tpu_worker_id", String, nullable=False, server_default=""),
    Column("md_tpu_chips_per_host_bounds", String, nullable=False, server_default=""),
    Column("md_gpu_count", Integer, nullable=False, server_default="0"),
    Column("md_gpu_name", String, nullable=False, server_default=""),
    Column("md_gpu_memory_mb", Integer, nullable=False, server_default="0"),
    Column("md_gce_instance_name", String, nullable=False, server_default=""),
    Column("md_gce_zone", String, nullable=False, server_default=""),
    Column("md_device_json", String, nullable=False, server_default="{}"),
    Column("md_provenance_json", String, nullable=False, server_default="{}"),
    Column("total_cpu_millicores", Integer, nullable=False, server_default="0"),
    Column("total_memory_bytes", Integer, nullable=False, server_default="0"),
    Column("total_gpu_count", Integer, nullable=False, server_default="0"),
    Column("total_tpu_count", Integer, nullable=False, server_default="0"),
    Column("device_type", String, nullable=False, server_default=""),
    Column("device_variant", String, nullable=False, server_default=""),
    Column("slice_id", String, nullable=False, server_default=""),
    Column("scale_group", String, nullable=False, server_default=""),
)


worker_attributes_table = Table(
    "worker_attributes",
    metadata,
    Column("worker_id", WorkerIdType, ForeignKey("workers.worker_id", ondelete="CASCADE"), nullable=False),
    Column("key", String, nullable=False),
    Column("value_type", String, nullable=False),
    Column("str_value", String),
    Column("int_value", Integer),
    Column("float_value", Float),
    PrimaryKeyConstraint("worker_id", "key"),
    CheckConstraint(WORKER_ATTR_VALUE_TYPE_CHECK, name="worker_attributes_value_type_check"),
)


endpoints_table = Table(
    "endpoints",
    metadata,
    Column("endpoint_id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("address", String, nullable=False),
    Column("job_id", JobNameType, ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False),
    Column("task_id", JobNameType, ForeignKey("tasks.task_id", ondelete="CASCADE")),
    Column("metadata_json", JSONDict, nullable=False),
    Column("registered_at_ms", TimestampMsType, nullable=False),
    # Lease expiry. Registration grants a lease; re-registering renews it. A row
    # past its deadline is hidden from reads and swept by the pruner, independent
    # of the FK CASCADE. Nullable so it can be added to an existing DB without a
    # backfill; a NULL deadline is treated as never-expiring until the registrant
    # next re-registers with a real lease.
    Column("lease_deadline_ms", TimestampMsType, nullable=True),
    # Proxy access mode (EndpointAccess int). Nullable so it can be added to an
    # existing DB without a backfill; a NULL is read as PRIVATE (today's
    # cluster-identity-required behavior), so pre-migration rows are unchanged.
    Column("access", Integer, nullable=True),
    Index("idx_endpoints_name", "name"),
    Index("idx_endpoints_task", "task_id"),
    Index("idx_endpoints_job_id", "job_id"),
)


scaling_groups_table = Table(
    "scaling_groups",
    metadata,
    Column("name", String, primary_key=True),
    Column("consecutive_failures", Integer, nullable=False, server_default="0"),
    Column("backoff_until_ms", Integer, nullable=False, server_default="0"),
    Column("last_scale_up_ms", Integer, nullable=False, server_default="0"),
    Column("last_scale_down_ms", Integer, nullable=False, server_default="0"),
    Column("quota_exceeded_until_ms", Integer, nullable=False, server_default="0"),
    Column("quota_reason", String, nullable=False, server_default=""),
    Column("updated_at_ms", Integer, nullable=False, server_default="0"),
)


slices_table = Table(
    "slices",
    metadata,
    Column("slice_id", String, primary_key=True),
    Column("scale_group", String, nullable=False),
    Column("lifecycle", String, nullable=False),
    Column("worker_ids", JSONList(), nullable=False, server_default="[]"),
    Column("created_at_ms", Integer, nullable=False, server_default="0"),
    Column("error_message", String, nullable=False, server_default=""),
    Index("idx_slices_scale_group", "scale_group"),
)


user_budgets_table = Table(
    "user_budgets",
    metadata,
    # Standalone runtime state (set via ``iris user budget set``); ``user_id`` is a
    # plain-string PK with no FK — a budget can be set for any owner id.
    Column("user_id", String, primary_key=True),
    Column("budget_limit", Integer, nullable=False, server_default="0"),
    Column("max_band", Integer, nullable=False, server_default="2"),
    Column("updated_at_ms", TimestampMsType, nullable=False),
)


# ---------------------------------------------------------------------------
# Federation sidecars.
#
# The federated job/task rows live in ``jobs``/``tasks`` with ``cluster`` set to a
# peer id; state/timing/counts are the single source of truth there. These tables
# are thin join sidecars carrying only federation-only metadata that has no home
# in the main rows — the same shape as ``jobs`` ⋈ ``job_config`` today.
# ---------------------------------------------------------------------------


# One row per job this controller has federated, in either direction:
#   SENT     — this cluster is the parent; it handed the job to peer_id and tracks
#              the handoff lifecycle (handoff_state, cancel intent). The peer runs
#              the job under the same, cluster-invariant job_id.
#   RECEIVED — this cluster is the peer; peer_id is the requester that handed it off.
# Joining jobs ⋈ federated_jobs tells you where a job was federated to/from. The
# SENT-only columns are null on RECEIVED rows.
federated_jobs_table = Table(
    "federated_jobs",
    metadata,
    Column("job_id", JobNameType, ForeignKey("jobs.job_id", ondelete="CASCADE"), primary_key=True),
    Column("direction", Integer, nullable=False),  # FederationDirection: SENT | RECEIVED
    # The counterparty cluster: the destination when SENT, the requester when RECEIVED.
    Column("peer_id", String, nullable=False),
    Column("owner_principal", String, nullable=False, server_default=""),  # end-user identity
    Column("handoff_state", Integer),  # SENT only: PENDING_HANDOFF | HANDED_OFF | HANDOFF_REJECTED
    Column("cancel_intent_version", Integer, nullable=False, server_default="0"),
    Index("idx_federated_jobs_direction_peer", "direction", "peer_id"),
)


federation_sync_state_table = Table(
    "federation_sync_state",
    metadata,
    Column("peer_id", String, primary_key=True),
    # Opaque monotonic watermark into the peer's changelog; one row per peer.
    Column("cursor", String, nullable=False, server_default=""),
)


federated_tasks_table = Table(
    "federated_tasks",
    metadata,
    Column("task_id", JobNameType, ForeignKey("tasks.task_id", ondelete="CASCADE"), primary_key=True),
    # Opaque peer-side worker name, display only (a federated task has no local worker row).
    Column("peer_worker_label", String, nullable=False, server_default=""),
)


# ---------------------------------------------------------------------------
# Peer-side federation changelog (this controller acting AS a peer).
#
# When a parent hands a job off, the receiving controller runs it as an ordinary
# local job (recorded as a RECEIVED row in federated_jobs) and appends a change
# event per job/task mutation here. Each row carries the requester it belongs to,
# so FederationSync reports a requester only its own handoffs without a join. The
# table stays empty until this controller receives a handoff, so a controller that
# is never a peer is unchanged.
# ---------------------------------------------------------------------------


federation_changelog_table = Table(
    "federation_changelog",
    metadata,
    # Monotonic sequence: SQLite aliases an INTEGER PRIMARY KEY to rowid, so it
    # autoincrements. A parent's sync cursor is the max seq it has consumed.
    Column("seq", Integer, primary_key=True, autoincrement=True),
    # No foreign key to jobs on purpose: a tombstone event must outlive the job
    # row (delete_job CASCADEs its dependents), so the parent can still learn the
    # job was pruned on a later sync.
    Column("job_id", JobNameType, nullable=False),
    Column("requester_id", String, nullable=False),  # parent cluster this event is reported to
    Column("task_index", Integer),  # NULL = a job-level change
    Column("tombstone", Integer, nullable=False, server_default="0"),  # 1 = job pruned on this peer
    Column("written_ms", TimestampMsType, nullable=False),
    Index("idx_federation_changelog_requester", "requester_id", "seq"),
    # AUTOINCREMENT: seq is a cursor watermark, so it must never be reused after a
    # delete (a plain rowid alias can reuse the max after a delete).
    sqlite_autoincrement=True,
)


auth_controller_secrets_table = Table(
    "controller_secrets",
    auth_metadata,
    Column("key", String, primary_key=True),
    Column("value", String, nullable=False),
    Column("created_at_ms", TimestampMsType, nullable=False),
)
