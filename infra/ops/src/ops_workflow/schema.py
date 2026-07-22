# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy Core schema for the ops workflow database."""

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    UniqueConstraint,
    column,
    literal_column,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from ops_workflow.state import CaseState, SignalDisposition, SignalState

metadata = MetaData()

source_deliveries = Table(
    "source_deliveries",
    metadata,
    Column("id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("source", Text, nullable=False),
    Column("delivery_key", Text, nullable=False),
    Column("key_id", Text, nullable=False),
    Column("source_timestamp", DateTime(timezone=True), nullable=False),
    Column("received_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("body_sha256", Text, nullable=False),
    Column("normalized_payload", JSONB, nullable=False),
    Column("result", JSONB),
    UniqueConstraint("source", "delivery_key"),
)

signals = Table(
    "signals",
    metadata,
    Column("id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("source", Text, nullable=False),
    Column("fingerprint", Text, nullable=False),
    Column("generation", Integer, nullable=False, server_default=text("1")),
    Column("state", Text, nullable=False),
    Column("receiver", Text, nullable=False),
    Column("group_key", Text, nullable=False),
    Column("alert_name", Text, nullable=False),
    Column("severity", Text, nullable=False),
    Column("cluster", Text),
    Column("namespace", Text),
    Column("object_kind", Text),
    Column("object_name", Text),
    Column("summary", Text, nullable=False),
    Column("labels", JSONB, nullable=False),
    Column("annotations", JSONB, nullable=False),
    Column("values", JSONB, nullable=False),
    Column("generator_url", Text),
    Column("silence_url", Text),
    Column("dashboard_url", Text),
    Column("panel_url", Text),
    Column("source_version", Text, nullable=False),
    Column("first_seen_at", DateTime(timezone=True), nullable=False),
    Column("last_seen_at", DateTime(timezone=True), nullable=False),
    Column("latest_source_timestamp", DateTime(timezone=True), nullable=False),
    Column("resolved_at", DateTime(timezone=True)),
    Column("latest_delivery_id", UUID(as_uuid=False), ForeignKey("source_deliveries.id"), nullable=False),
    Column("missing_successful_polls", Integer, nullable=False, server_default=text("0")),
    CheckConstraint(column("state").in_(tuple(state.value for state in SignalState))),
    CheckConstraint("missing_successful_polls >= 0"),
    UniqueConstraint("source", "fingerprint"),
)

delivery_signals = Table(
    "delivery_signals",
    metadata,
    Column(
        "delivery_id",
        UUID(as_uuid=False),
        ForeignKey("source_deliveries.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("signal_id", UUID(as_uuid=False), ForeignKey("signals.id"), nullable=False),
    Column("disposition", Text, nullable=False),
    CheckConstraint(column("disposition").in_(tuple(disposition.value for disposition in SignalDisposition))),
    PrimaryKeyConstraint("delivery_id", "signal_id"),
)

cases = Table(
    "cases",
    metadata,
    Column("id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("trigger", Text, nullable=False),
    Column("receiver", Text, nullable=False),
    Column("group_key", Text, nullable=False),
    Column("grouping_rule", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("priority", Integer, nullable=False),
    Column("title", Text, nullable=False),
    Column("question", Text),
    Column("opened_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("next_eligible_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("investigated_at", DateTime(timezone=True)),
    Column("archived_at", DateTime(timezone=True)),
    Column("outcome", Text),
    Column("summary", Text),
    CheckConstraint("trigger IN ('automatic', 'manual')"),
    CheckConstraint(column("state").in_(tuple(state.value for state in CaseState))),
    CheckConstraint("outcome IN ('no_action', 'action_recommended', 'blocked', 'unknown')"),
    CheckConstraint("(trigger = 'automatic' AND question IS NULL) OR (trigger = 'manual' AND question IS NOT NULL)"),
)

case_signals = Table(
    "case_signals",
    metadata,
    Column("case_id", UUID(as_uuid=False), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False),
    Column("signal_id", UUID(as_uuid=False), ForeignKey("signals.id"), nullable=False),
    Column("signal_generation", Integer, nullable=False),
    Column("attached_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    PrimaryKeyConstraint("case_id", "signal_id", "signal_generation"),
)

agent_sessions = Table(
    "agent_sessions",
    metadata,
    Column("id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("case_id", UUID(as_uuid=False), ForeignKey("cases.id"), nullable=False),
    Column("deterministic_name", Text, nullable=False, unique=True),
    Column("loom_session_id", Text, unique=True),
    Column("loom_session_url", Text),
    Column("state", Text, nullable=False),
    Column("repo_revision", Text, nullable=False),
    Column("skill_revision", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("archived_at", DateTime(timezone=True)),
    CheckConstraint("state IN ('new', 'ready', 'active', 'idle', 'lost', 'archived')"),
)

agent_turns = Table(
    "agent_turns",
    metadata,
    Column("id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("session_id", UUID(as_uuid=False), ForeignKey("agent_sessions.id"), nullable=False),
    Column("kind", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("priority", Integer, nullable=False),
    Column("requested_by", Text, nullable=False),
    Column("client_request_id", Text, nullable=False),
    Column("prompt", Text, nullable=False),
    Column("available_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("lease_owner", Text),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("attempt", Integer, nullable=False, server_default=text("1")),
    Column("retry_of", UUID(as_uuid=False), ForeignKey("agent_turns.id")),
    Column("loom_turn_number", BigInteger),
    Column("result_artifact_revision", Integer),
    Column("result", JSONB),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("started_at", DateTime(timezone=True)),
    Column("deadline_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
    CheckConstraint("kind IN ('automatic', 'question', 'follow_up')"),
    CheckConstraint("state IN ('queued', 'launching', 'running', 'succeeded', 'failed', 'interrupted', 'cancelled')"),
    UniqueConstraint("requested_by", "client_request_id"),
)

case_events = Table(
    "case_events",
    metadata,
    Column("id", BigInteger, Identity(always=True), primary_key=True),
    Column("case_id", UUID(as_uuid=False), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False),
    Column("turn_id", UUID(as_uuid=False), ForeignKey("agent_turns.id", ondelete="SET NULL")),
    Column("event_type", Text, nullable=False),
    Column("actor", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("data", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
)

grafana_polls = Table(
    "grafana_polls",
    metadata,
    Column("id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("poll_slot", DateTime(timezone=True), nullable=False, unique=True),
    Column("observed_at", DateTime(timezone=True), nullable=False),
    Column("alert_count", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    CheckConstraint("alert_count >= 0"),
)

slack_escalations = Table(
    "slack_escalations",
    metadata,
    Column("id", UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("incident_key", Text, nullable=False, unique=True),
    Column("case_id", UUID(as_uuid=False), ForeignKey("cases.id"), nullable=False),
    Column("turn_id", UUID(as_uuid=False), ForeignKey("agent_turns.id"), nullable=False, unique=True),
    Column("severity", Text, nullable=False),
    Column("reason", Text, nullable=False),
    Column("message", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("attempts", Integer, nullable=False, server_default=text("0")),
    Column("available_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("last_error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("sent_at", DateTime(timezone=True)),
    CheckConstraint("severity IN ('error', 'critical')"),
    CheckConstraint("state IN ('pending', 'sending', 'sent', 'abandoned')"),
    CheckConstraint("attempts >= 0"),
)

Index(
    "cases_one_open_group",
    cases.c.receiver,
    cases.c.group_key,
    unique=True,
    postgresql_where=cases.c.state != "archived",
)
Index(
    "case_signals_one_case_per_generation",
    case_signals.c.signal_id,
    case_signals.c.signal_generation,
    unique=True,
)
Index(
    "agent_sessions_one_current_per_case",
    agent_sessions.c.case_id,
    unique=True,
    postgresql_where=agent_sessions.c.state != "archived",
)
Index(
    "agent_turns_one_queued_per_session",
    agent_turns.c.session_id,
    unique=True,
    postgresql_where=agent_turns.c.state == "queued",
)
Index(
    "agent_turns_one_active_per_session",
    agent_turns.c.session_id,
    unique=True,
    postgresql_where=agent_turns.c.state.in_(("launching", "running")),
)
Index(
    "agent_turns_one_active_global",
    literal_column("(true)"),
    unique=True,
    postgresql_where=agent_turns.c.state.in_(("launching", "running")),
    _table=agent_turns,
)
Index(
    "cases_queue_order",
    cases.c.priority.desc(),
    cases.c.next_eligible_at,
    cases.c.opened_at,
    postgresql_where=cases.c.state == "pending",
)
Index("signals_group_state", signals.c.receiver, signals.c.group_key, signals.c.state, signals.c.last_seen_at.desc())
Index("case_signals_case", case_signals.c.case_id, case_signals.c.attached_at)
Index(
    "agent_turns_queue_order",
    agent_turns.c.priority.desc(),
    agent_turns.c.available_at,
    agent_turns.c.created_at,
    postgresql_where=agent_turns.c.state == "queued",
)
Index(
    "agent_turns_expired_leases",
    agent_turns.c.lease_expires_at,
    postgresql_where=agent_turns.c.state.in_(("launching", "running")),
)
Index("case_events_timeline", case_events.c.case_id, case_events.c.created_at, case_events.c.id)
Index(
    "slack_escalations_delivery_queue",
    slack_escalations.c.available_at,
    slack_escalations.c.created_at,
    postgresql_where=slack_escalations.c.state == "pending",
)
Index("slack_escalations_case", slack_escalations.c.case_id, slack_escalations.c.created_at.desc())
