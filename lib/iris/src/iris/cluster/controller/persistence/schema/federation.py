# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable federation state for resource schema v2."""

from iris.cluster.controller.persistence.schema.base import metadata
from sqlalchemy import CheckConstraint, Column, ForeignKey, Index, Integer, Table, Text, UniqueConstraint, text

federated_jobs_table = Table(
    "federated_jobs",
    metadata,
    Column("job_uid", Text, ForeignKey("jobs.job_uid", ondelete="CASCADE"), primary_key=True),
    Column("direction", Text, nullable=False),
    Column("peer_id", Text, nullable=False),
    Column("owner_principal", Text, nullable=False),
    Column("handoff_state", Text),
    Column("cancel_intent_version", Integer, nullable=False, server_default=text("0")),
    Column("handoff_nonce", Text, nullable=False),
    CheckConstraint("direction IN ('sent', 'received')"),
    CheckConstraint(
        "(direction = 'sent' AND handoff_state IS NOT NULL AND "
        "handoff_state IN ('queued', 'pending', 'handed_off', 'rejected')) OR "
        "(direction = 'received' AND handoff_state IS NULL)"
    ),
    UniqueConstraint("direction", "peer_id", "handoff_nonce"),
)

Index(
    "federated_jobs_direction_peer",
    federated_jobs_table.c.direction,
    federated_jobs_table.c.peer_id,
    federated_jobs_table.c.handoff_state,
)


federation_sync_state_table = Table(
    "federation_sync_state",
    metadata,
    Column("peer_id", Text, primary_key=True),
    Column("cursor", Text, nullable=False),
)


federated_tasks_table = Table(
    "federated_tasks",
    metadata,
    Column("task_uid", Text, ForeignKey("tasks.task_uid", ondelete="CASCADE"), primary_key=True),
    Column("peer_node_label", Text, nullable=False, server_default=""),
)


federation_changelog_table = Table(
    "federation_changelog",
    metadata,
    Column("seq", Integer, primary_key=True, autoincrement=True),
    Column("authority_cluster_id", Text, nullable=False),
    Column("job_id", Text, nullable=False),
    Column("job_uid", Text, nullable=False),
    Column("task_uid", Text),
    Column("requester_id", Text, nullable=False),
    Column("tombstone", Integer, nullable=False),
    Column("written_at_ms", Integer, nullable=False),
    CheckConstraint("tombstone IN (0, 1)"),
    sqlite_autoincrement=True,
)

Index(
    "federation_changelog_requester",
    federation_changelog_table.c.requester_id,
    federation_changelog_table.c.seq,
)
