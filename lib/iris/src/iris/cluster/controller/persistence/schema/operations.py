# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable action receipts and user budgets for resource schema v2."""

from iris.cluster.controller.persistence.schema.base import metadata
from sqlalchemy import CheckConstraint, Column, Index, Integer, Table, Text, UniqueConstraint, text

action_receipts_table = Table(
    "action_receipts",
    metadata,
    Column("action_id", Text, primary_key=True),
    Column("authority_cluster_id", Text, nullable=False),
    Column("authority_action_id", Text, nullable=False),
    Column("action_kind", Text, nullable=False),
    Column("target_kind", Text, nullable=False),
    Column("target_id", Text, nullable=False),
    Column("expected_target_uid", Text, nullable=False),
    Column("expected_attempt_uid", Text, nullable=False, server_default=""),
    Column("backend_id", Text, nullable=False, server_default=""),
    Column("execution_cluster_id", Text, nullable=False),
    Column("principal_id", Text, nullable=False),
    Column("client_idempotency_key", Text, nullable=False),
    Column("payload_hash", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("result_code", Text, nullable=False),
    Column("result_message", Text, nullable=False, server_default=""),
    Column("created_at_ms", Integer, nullable=False),
    Column("updated_at_ms", Integer, nullable=False),
    Column("completed_at_ms", Integer),
    CheckConstraint("authority_cluster_id <> ''"),
    CheckConstraint("action_kind IN ('cancel_job', 'retry_task', 'terminate_attempt')"),
    CheckConstraint("target_kind IN ('job', 'task', 'attempt')"),
    CheckConstraint("expected_target_uid <> ''"),
    CheckConstraint("state IN ('accepted', 'verifying', 'succeeded', 'failed')"),
    CheckConstraint(
        "result_code IN (" "'none', 'satisfied', 'target_absent', 'provider_rejected', 'internal_error'" ")"
    ),
    UniqueConstraint("authority_cluster_id", "authority_action_id"),
    UniqueConstraint("principal_id", "action_kind", "client_idempotency_key"),
    CheckConstraint(
        "(action_kind = 'cancel_job' AND target_kind = 'job' AND expected_attempt_uid = '') OR "
        "(action_kind = 'retry_task' AND target_kind = 'task' AND expected_attempt_uid <> '') OR "
        "(action_kind = 'terminate_attempt' AND target_kind = 'attempt' AND expected_attempt_uid <> '')"
    ),
    CheckConstraint("action_kind = 'cancel_job' OR backend_id <> ''"),
    CheckConstraint(
        "(state IN ('accepted', 'verifying') AND result_code = 'none' AND completed_at_ms IS NULL) OR "
        "(state = 'succeeded' AND result_code IN ('satisfied', 'target_absent') "
        "AND completed_at_ms IS NOT NULL) OR "
        "(state = 'failed' AND result_code IN ('provider_rejected', 'internal_error') "
        "AND completed_at_ms IS NOT NULL)"
    ),
)

Index("action_receipts_state", action_receipts_table.c.state, action_receipts_table.c.updated_at_ms)
Index(
    "action_receipts_target",
    action_receipts_table.c.target_kind,
    action_receipts_table.c.target_id,
    action_receipts_table.c.updated_at_ms.desc(),
)
Index(
    "action_receipts_principal",
    action_receipts_table.c.principal_id,
    action_receipts_table.c.updated_at_ms.desc(),
)


user_budgets_table = Table(
    "user_budgets",
    metadata,
    Column("owner_id", Text, primary_key=True),
    Column("budget_limit", Integer, nullable=False, server_default=text("0")),
    Column("max_band", Integer, nullable=False),
    Column("updated_at_ms", Integer, nullable=False),
)
