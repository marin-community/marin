# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job, Task, Attempt, and Endpoint tables for resource schema v2."""

from iris.cluster.controller.persistence.schema.base import metadata
from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Table,
    Text,
    UniqueConstraint,
    text,
)

jobs_table = Table(
    "jobs",
    metadata,
    Column("job_uid", Text, primary_key=True),
    Column("authority_cluster_id", Text, nullable=False),
    Column("job_id", Text, nullable=False),
    Column("execution_cluster_id", Text, nullable=False),
    Column("backend_id", Text, nullable=False),
    Column("placement_state", Text, nullable=False),
    Column("owner_id", Text, nullable=False),
    Column("submitting_principal", Text, nullable=False),
    Column("parent_job_uid", Text, ForeignKey("jobs.job_uid", ondelete="CASCADE")),
    Column("root_job_uid", Text, ForeignKey("jobs.job_uid"), nullable=False),
    Column("depth", Integer, nullable=False),
    Column("state", Integer, nullable=False),
    Column("submitted_at_ms", Integer, nullable=False),
    Column("root_submitted_at_ms", Integer, nullable=False),
    Column("started_at_ms", Integer),
    Column("finished_at_ms", Integer),
    Column("scheduling_deadline_at_ms", Integer),
    Column("error_message", Text, nullable=False, server_default=""),
    Column("exit_code", Integer),
    Column("num_tasks", Integer, nullable=False),
    Column("name", Text, nullable=False),
    CheckConstraint("authority_cluster_id <> ''"),
    CheckConstraint("execution_cluster_id <> ''"),
    CheckConstraint(
        "placement_state IN ('pending', 'known') AND "
        "((placement_state = 'pending' AND backend_id = '') OR "
        "(placement_state = 'known' AND backend_id <> ''))"
    ),
    CheckConstraint("depth >= 0"),
    CheckConstraint("num_tasks >= 0"),
    UniqueConstraint("authority_cluster_id", "job_id"),
)

Index("jobs_parent", jobs_table.c.parent_job_uid)
Index("jobs_state_submitted", jobs_table.c.state, jobs_table.c.submitted_at_ms.desc())
Index("jobs_owner_state", jobs_table.c.owner_id, jobs_table.c.state)
Index("jobs_backend_state", jobs_table.c.backend_id, jobs_table.c.state)
Index("jobs_execution_state", jobs_table.c.execution_cluster_id, jobs_table.c.state)


job_specs_table = Table(
    "job_specs",
    metadata,
    Column("job_uid", Text, ForeignKey("jobs.job_uid", ondelete="CASCADE"), primary_key=True),
    Column("spec_version", Integer, nullable=False),
    Column("resources_json", Text, nullable=False),
    Column("entrypoint_json", Text, nullable=False),
    Column("environment_json", Text, nullable=False),
    Column("constraints_json", Text, nullable=False),
    Column("coscheduling_json", Text, nullable=False),
    Column("bundle_id", Text, nullable=False),
    Column("ports_json", Text, nullable=False),
    Column("scheduling_timeout_ms", Integer),
    Column("max_task_failures", Integer, nullable=False),
    Column("max_retries_failure", Integer, nullable=False),
    Column("max_retries_preemption", Integer, nullable=False),
    Column("replicas", Integer, nullable=False),
    Column("timeout_ms", Integer),
    Column("fail_if_exists", Integer, nullable=False),
    Column("preemption_policy", Integer, nullable=False),
    Column("existing_job_policy", Integer, nullable=False),
    Column("priority_band", Integer, nullable=False),
    Column("task_image", Text, nullable=False),
    Column("submit_argv_json", Text, nullable=False),
    Column("client_revision_date", Text, nullable=False),
    Column("container_profile", Integer, nullable=False),
    CheckConstraint("spec_version = 1"),
    CheckConstraint("json_valid(resources_json)"),
    CheckConstraint("json_valid(entrypoint_json)"),
    CheckConstraint("json_valid(environment_json)"),
    CheckConstraint("json_valid(constraints_json)"),
    CheckConstraint("json_valid(coscheduling_json)"),
    CheckConstraint("json_valid(ports_json)"),
    CheckConstraint("max_task_failures >= 0"),
    CheckConstraint("max_retries_failure >= 0"),
    CheckConstraint("max_retries_preemption >= 0"),
    CheckConstraint("replicas > 0"),
    CheckConstraint("fail_if_exists IN (0, 1)"),
    CheckConstraint("json_valid(submit_argv_json)"),
)


job_workdir_files_table = Table(
    "job_workdir_files",
    metadata,
    Column("job_uid", Text, ForeignKey("jobs.job_uid", ondelete="CASCADE"), primary_key=True),
    Column("filename", Text, primary_key=True),
    Column("data", LargeBinary, nullable=False),
)


tasks_table = Table(
    "tasks",
    metadata,
    Column("task_uid", Text, primary_key=True),
    Column("authority_cluster_id", Text, nullable=False),
    Column("task_id", Text, nullable=False),
    Column("job_uid", Text, ForeignKey("jobs.job_uid", ondelete="CASCADE"), nullable=False),
    Column("task_index", Integer, nullable=False),
    Column("execution_cluster_id", Text, nullable=False),
    Column("backend_id", Text, nullable=False),
    Column("placement_state", Text, nullable=False),
    Column("state", Integer, nullable=False),
    Column("submitted_at_ms", Integer, nullable=False),
    Column("started_at_ms", Integer),
    Column("finished_at_ms", Integer),
    Column("error_message", Text, nullable=False, server_default=""),
    Column("status_message", Text, nullable=False, server_default=""),
    Column("exit_code", Integer),
    Column("max_retries_failure", Integer, nullable=False),
    Column("max_retries_preemption", Integer, nullable=False),
    Column(
        "current_attempt_uid",
        Text,
        ForeignKey("attempts.attempt_uid", deferrable=True, initially="DEFERRED"),
    ),
    Column("current_node_uid", Text),
    Column("priority_band", Integer, nullable=False),
    Column("priority_neg_depth", Integer, nullable=False),
    Column("priority_root_submitted_ms", Integer, nullable=False),
    Column("priority_insertion", Integer, nullable=False),
    CheckConstraint("authority_cluster_id <> ''"),
    CheckConstraint("task_index >= 0"),
    CheckConstraint("execution_cluster_id <> ''"),
    CheckConstraint(
        "placement_state IN ('pending', 'known') AND "
        "((placement_state = 'pending' AND backend_id = '') OR "
        "(placement_state = 'known' AND backend_id <> ''))"
    ),
    CheckConstraint("max_retries_failure >= 0"),
    CheckConstraint("max_retries_preemption >= 0"),
    UniqueConstraint("authority_cluster_id", "task_id"),
    UniqueConstraint("job_uid", "task_index"),
)

Index("tasks_job_state", tasks_table.c.job_uid, tasks_table.c.state)
Index("tasks_backend_state", tasks_table.c.backend_id, tasks_table.c.state)
Index("tasks_execution_state", tasks_table.c.execution_cluster_id, tasks_table.c.state)
Index(
    "tasks_current_attempt",
    tasks_table.c.current_attempt_uid,
    sqlite_where=tasks_table.c.current_attempt_uid.is_not(None),
)
Index(
    "tasks_pending",
    tasks_table.c.state,
    tasks_table.c.priority_band,
    tasks_table.c.priority_neg_depth,
    tasks_table.c.priority_root_submitted_ms,
    tasks_table.c.submitted_at_ms,
    tasks_table.c.priority_insertion,
)


attempts_table = Table(
    "attempts",
    metadata,
    Column("attempt_uid", Text, primary_key=True),
    Column("task_uid", Text, ForeignKey("tasks.task_uid", ondelete="CASCADE"), nullable=False),
    Column("attempt_number", Integer, nullable=False),
    Column("execution_cluster_id", Text, nullable=False),
    Column("backend_id", Text, nullable=False),
    Column("node_uid", Text),
    Column("state", Integer, nullable=False),
    Column("created_at_ms", Integer, nullable=False),
    Column("started_at_ms", Integer),
    Column("finished_at_ms", Integer),
    Column("exit_code", Integer),
    Column("error_message", Text, nullable=False, server_default=""),
    Column("terminal_reason", Text, nullable=False, server_default=""),
    CheckConstraint("attempt_number >= 0"),
    CheckConstraint("execution_cluster_id <> ''"),
    CheckConstraint("backend_id <> ''"),
    UniqueConstraint("task_uid", "attempt_number"),
)

Index("attempts_task_state", attempts_table.c.task_uid, attempts_table.c.state, attempts_table.c.started_at_ms)
Index("attempts_backend", attempts_table.c.backend_id)
Index("attempts_node", attempts_table.c.node_uid, sqlite_where=attempts_table.c.node_uid.is_not(None))


attempt_runtime_objects_table = Table(
    "attempt_runtime_objects",
    metadata,
    Column("attempt_uid", Text, ForeignKey("attempts.attempt_uid", ondelete="CASCADE"), primary_key=True),
    Column("provider_kind", Text, nullable=False),
    Column("namespace", Text, nullable=False, server_default=""),
    Column("name", Text, nullable=False, server_default=""),
    Column("provider_uid", Text, nullable=False, server_default=""),
    Column("provider_node_id", Text, nullable=False, server_default=""),
    Column("provider_node_uid", Text, nullable=False, server_default=""),
    Column("container_id", Text, nullable=False, server_default=""),
    Column("observed_at_ms", Integer, nullable=False),
    CheckConstraint("provider_kind IN ('kubernetes', 'rpc')"),
    CheckConstraint(
        "(provider_kind = 'kubernetes' AND namespace <> '' AND name <> '' AND provider_uid <> '') OR "
        "(provider_kind = 'rpc' AND provider_node_uid <> '' AND container_id <> '')"
    ),
)

Index(
    "attempt_runtime_provider_uid",
    attempt_runtime_objects_table.c.provider_kind,
    attempt_runtime_objects_table.c.provider_uid,
    sqlite_where=text("provider_uid <> ''"),
)


endpoints_table = Table(
    "endpoints",
    metadata,
    Column("endpoint_id", Text, primary_key=True),
    Column("authority_cluster_id", Text, nullable=False),
    Column("execution_cluster_id", Text, nullable=False),
    Column("name", Text, nullable=False),
    Column("address", Text, nullable=False),
    Column("owner_job_id", Text, nullable=False),
    Column("owner_task_id", Text),
    Column("owner_job_uid", Text, ForeignKey("jobs.job_uid", ondelete="CASCADE")),
    Column("owner_task_uid", Text, ForeignKey("tasks.task_uid", ondelete="CASCADE")),
    Column("peer_id", Text),
    Column("metadata_json", Text, nullable=False),
    Column("access", Integer, nullable=False),
    Column("registered_at_ms", Integer, nullable=False),
    Column("lease_deadline_at_ms", Integer),
    CheckConstraint("authority_cluster_id <> ''"),
    CheckConstraint("execution_cluster_id <> ''"),
    CheckConstraint("json_valid(metadata_json)"),
    CheckConstraint("access IN (0, 1)"),
)

Index("endpoints_name", endpoints_table.c.name)
Index("endpoints_owner_task", endpoints_table.c.authority_cluster_id, endpoints_table.c.owner_task_id)
Index("endpoints_peer", endpoints_table.c.peer_id, sqlite_where=endpoints_table.c.peer_id.is_not(None))
