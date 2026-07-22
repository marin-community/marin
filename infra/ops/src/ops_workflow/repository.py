# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL persistence for Grafana-owned alert and agent lifecycles."""

import hashlib
import json
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, cast

from sqlalchemy import case, exists, func, null, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import RowMapping, make_url
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from ops_workflow.grafana import (
    GRAFANA_BASE_URL,
    GrafanaAlert,
    GrafanaDelivery,
    GrafanaNotification,
    grafana_group_metadata,
)
from ops_workflow.grafana_source import SOURCE_VERSION, GrafanaSnapshot, PolledGrafanaAlert
from ops_workflow.result import OpsResult
from ops_workflow.schema import (
    agent_sessions,
    agent_turns,
    case_events,
    case_signals,
    cases,
    delivery_signals,
    grafana_polls,
    signals,
    slack_escalations,
    source_deliveries,
)
from ops_workflow.slack import SlackDelivery, SlackEscalationDraft
from ops_workflow.state import CaseState, SignalDisposition, SignalState, severity_priority

SOURCE = "grafana"
GROUPING_RULE = "grafana:alertname+cluster"
AUTOMATIC_REQUESTER = "grafana"
MISSING_POLLS_TO_RESOLVE = 2
POLL_KEY_ID = "grafana-api-reader"
OPS_SERVICE_ACTOR = "ops-service"
TURN_LEASE_DURATION = timedelta(minutes=10)
TURN_TIMEOUT = timedelta(minutes=20)
SLACK_LEASE_DURATION = timedelta(minutes=2)
SLACK_MAX_ATTEMPTS = 5
SLACK_MAX_RETRY_DELAY = 300
MAX_PERSISTED_ERROR_CHARS = 4_000
MAX_EVENT_ERROR_CHARS = 1_000


class TurnPendingError(RuntimeError):
    """The case already has queued work for its current session."""


class ArchiveResult(StrEnum):
    """Outcome of an archive request at the persistence boundary."""

    ARCHIVED = "archived"
    ACTIVE_TURN = "active_turn"
    ALREADY_ARCHIVED = "already_archived"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class IngestResult:
    """Durable outcome returned for a Grafana delivery or exact retry."""

    delivery_id: str
    duplicate: bool
    case_ids: tuple[str, ...]
    signal_dispositions: Mapping[str, str]
    queued_case_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "delivery_id": self.delivery_id,
            "duplicate": self.duplicate,
            "case_ids": list(self.case_ids),
            "signal_dispositions": dict(self.signal_dispositions),
            "queued_case_ids": list(self.queued_case_ids),
        }


@dataclass(frozen=True)
class TouchedSignal:
    """Named result of applying one delivery alert to its durable signal."""

    id: str
    generation: int
    state: SignalState
    disposition: SignalDisposition
    alert: GrafanaAlert


class OpsRepository:
    """SQLAlchemy Core repository for the ops workflow."""

    def __init__(self, database_url: str, *, repo_revision: str, skill_revision: str) -> None:
        url = make_url(database_url)
        if url.get_backend_name() != "postgresql":
            raise ValueError("ops workflow requires a PostgreSQL database URL")
        self._engine: AsyncEngine = create_async_engine(
            url.set(drivername="postgresql+psycopg"),
            pool_pre_ping=True,
        )
        self._repo_revision = repo_revision
        self._skill_revision = skill_revision

    async def close(self) -> None:
        """Close pooled database connections."""

        await self._engine.dispose()

    async def reconcile_grafana_snapshot(self, snapshot: GrafanaSnapshot) -> tuple[IngestResult, ...]:
        """Atomically reconcile one complete, successful Grafana API snapshot."""

        grouped: dict[tuple[str, str], list[PolledGrafanaAlert]] = defaultdict(list)
        for item in snapshot.alerts:
            grouped[(item.receiver, item.group_key)].append(item)
        present = {item.alert.fingerprint for item in snapshot.alerts}

        async with self._engine.begin() as connection:
            poll_slot = snapshot.observed_at.replace(second=0, microsecond=0)
            poll_result = await connection.execute(
                pg_insert(grafana_polls)
                .values(poll_slot=poll_slot, observed_at=snapshot.observed_at, alert_count=len(snapshot.alerts))
                .on_conflict_do_nothing(index_elements=[grafana_polls.c.poll_slot])
                .returning(grafana_polls.c.id)
            )
            if poll_result.first() is None:
                return ()
            results: list[IngestResult] = []
            for items in grouped.values():
                delivery = _poll_delivery(items, observed_at=snapshot.observed_at, status="firing")
                results.append(await self._ingest_delivery(connection, delivery, key_id=POLL_KEY_ID))

            firing_result = await connection.execute(
                select(signals).where(signals.c.source == SOURCE, signals.c.state == "firing").with_for_update()
            )
            firing = firing_result.mappings().all()
            resolving: dict[tuple[str, str], list[PolledGrafanaAlert]] = defaultdict(list)
            for signal in firing:
                fingerprint = str(signal["fingerprint"])
                if fingerprint in present:
                    if signal["missing_successful_polls"]:
                        await connection.execute(
                            update(signals).where(signals.c.id == signal["id"]).values(missing_successful_polls=0)
                        )
                    continue
                missing = int(signal["missing_successful_polls"]) + 1
                if missing < MISSING_POLLS_TO_RESOLVE:
                    await connection.execute(
                        update(signals).where(signals.c.id == signal["id"]).values(missing_successful_polls=missing)
                    )
                    continue
                item = _resolved_polled_alert(signal, observed_at=snapshot.observed_at)
                resolving[(item.receiver, item.group_key)].append(item)

            for items in resolving.values():
                delivery = _poll_delivery(items, observed_at=snapshot.observed_at, status="resolved")
                results.append(await self._ingest_delivery(connection, delivery, key_id=POLL_KEY_ID))
            return tuple(results)

    async def overview(self) -> dict[str, object]:
        async with self._engine.connect() as connection:
            counts_result = await connection.execute(
                select(cases.c.state, func.count().label("count")).group_by(cases.c.state).order_by(cases.c.state)
            )
            count_rows = counts_result.mappings().all()
            active_result = await connection.execute(
                select(
                    cases.c.id.label("case_id"),
                    cases.c.title,
                    agent_turns.c.started_at,
                    agent_sessions.c.loom_session_url,
                )
                .select_from(
                    agent_turns.join(agent_sessions, agent_sessions.c.id == agent_turns.c.session_id).join(
                        cases, cases.c.id == agent_sessions.c.case_id
                    )
                )
                .where(agent_turns.c.state.in_(("launching", "running")))
                .limit(1)
            )
            active = active_result.mappings().first()
            last_poll_at = await connection.scalar(select(func.max(grafana_polls.c.observed_at)))
        return {
            "case_counts": {row["state"]: row["count"] for row in count_rows},
            "active_investigation": dict(active) if active is not None else None,
            "last_poll_at": last_poll_at,
        }

    async def recent_grafana_polls(self, *, limit: int = 60) -> list[dict[str, object]]:
        """Return recent successful snapshots for operator diagnostics."""

        async with self._engine.connect() as connection:
            result = await connection.execute(
                select(
                    grafana_polls.c.poll_slot,
                    grafana_polls.c.observed_at,
                    grafana_polls.c.alert_count,
                    grafana_polls.c.created_at,
                )
                .order_by(grafana_polls.c.observed_at.desc())
                .limit(limit)
            )
        return [dict(row) for row in result.mappings()]

    async def list_cases(self, *, include_archived: bool = False) -> list[dict[str, object]]:
        joined = (
            cases.outerjoin(case_signals, case_signals.c.case_id == cases.c.id)
            .outerjoin(
                signals,
                (signals.c.id == case_signals.c.signal_id) & (signals.c.generation == case_signals.c.signal_generation),
            )
            .outerjoin(
                agent_sessions,
                (agent_sessions.c.case_id == cases.c.id) & (agent_sessions.c.state != "archived"),
            )
        )
        statement = (
            select(
                cases.c.id,
                cases.c.trigger,
                cases.c.state,
                cases.c.priority,
                cases.c.title,
                cases.c.receiver,
                cases.c.group_key,
                cases.c.outcome,
                cases.c.summary,
                cases.c.opened_at,
                cases.c.updated_at,
                func.count(case_signals.c.signal_id).label("signal_count"),
                func.count(case_signals.c.signal_id).filter(signals.c.state == "firing").label("firing_count"),
                func.array_remove(func.array_agg(func.distinct(signals.c.cluster)), null()).label("clusters"),
                agent_sessions.c.loom_session_url,
            )
            .select_from(joined)
            .group_by(cases.c.id, agent_sessions.c.loom_session_url)
            .order_by(
                case(
                    (cases.c.state == "investigating", 0),
                    (cases.c.state == "pending", 1),
                    else_=2,
                ),
                cases.c.priority.desc(),
                cases.c.updated_at.desc(),
            )
            .limit(200)
        )
        if not include_archived:
            statement = statement.where(cases.c.state != "archived")
        async with self._engine.connect() as connection:
            result = await connection.execute(statement)
        return [dict(row) for row in result.mappings()]

    async def case_detail(self, case_id: str) -> dict[str, object] | None:
        async with self._engine.connect() as connection:
            case_result = await connection.execute(
                select(
                    *cases.c,
                    agent_sessions.c.id.label("agent_session_id"),
                    agent_sessions.c.loom_session_id,
                    agent_sessions.c.loom_session_url,
                    agent_sessions.c.state.label("agent_session_state"),
                )
                .select_from(
                    cases.outerjoin(
                        agent_sessions,
                        (agent_sessions.c.case_id == cases.c.id) & (agent_sessions.c.state != "archived"),
                    )
                )
                .where(cases.c.id == case_id)
            )
            case_row = case_result.mappings().first()
            case_record = dict(case_row) if case_row is not None else None
            if case_record is None:
                return None
            signals_result = await connection.execute(
                select(*signals.c, case_signals.c.signal_generation, case_signals.c.attached_at)
                .select_from(case_signals.join(signals, signals.c.id == case_signals.c.signal_id))
                .where(case_signals.c.case_id == case_id, case_signals.c.signal_generation == signals.c.generation)
                .order_by(signals.c.severity, signals.c.alert_name, signals.c.fingerprint)
            )
            turns_result = await connection.execute(
                select(agent_turns)
                .select_from(agent_turns.join(agent_sessions, agent_sessions.c.id == agent_turns.c.session_id))
                .where(agent_sessions.c.case_id == case_id)
                .order_by(agent_turns.c.created_at)
            )
            events_result = await connection.execute(
                select(case_events)
                .where(case_events.c.case_id == case_id)
                .order_by(case_events.c.created_at, case_events.c.id)
            )
            escalations_result = await connection.execute(
                select(slack_escalations)
                .where(slack_escalations.c.case_id == case_id)
                .order_by(slack_escalations.c.created_at)
            )
            signal_records = [dict(row) for row in signals_result.mappings()]
            turn_records = [dict(row) for row in turns_result.mappings()]
            event_records = [dict(row) for row in events_result.mappings()]
            escalation_records = [dict(row) for row in escalations_result.mappings()]
        return {
            "case": case_record,
            "signals": signal_records,
            "turns": turn_records,
            "events": event_records,
            "escalations": escalation_records,
        }

    async def create_question(self, *, text: str, actor: str) -> str:
        case_id = str(uuid.uuid4())
        group_key = f"manual:{case_id}"
        session_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        async with self._engine.begin() as connection:
            await connection.execute(
                cases.insert().values(
                    id=case_id,
                    trigger="manual",
                    receiver="manual",
                    group_key=group_key,
                    grouping_rule="manual",
                    state="pending",
                    priority=120,
                    title=text[:160],
                    question=text,
                    opened_at=now,
                    updated_at=now,
                )
            )
            await self._insert_session(connection, session_id=session_id, case_id=case_id)
            await connection.execute(
                agent_turns.insert().values(
                    id=turn_id,
                    session_id=session_id,
                    kind="question",
                    state="queued",
                    priority=120,
                    requested_by=actor,
                    client_request_id=f"manual:{case_id}",
                    prompt=text,
                    created_at=now,
                )
            )
            await self._event(connection, case_id, "question_queued", actor, {"turn_id": turn_id})
        return case_id

    async def enqueue_follow_up(self, *, case_id: str, text: str, actor: str) -> str:
        turn_id = str(uuid.uuid4())
        async with self._engine.begin() as connection:
            session_result = await connection.execute(
                select(agent_sessions.c.id)
                .select_from(agent_sessions.join(cases, cases.c.id == agent_sessions.c.case_id))
                .where(
                    agent_sessions.c.case_id == case_id,
                    agent_sessions.c.state != "archived",
                    cases.c.state != "archived",
                )
                .with_for_update()
            )
            session = session_result.mappings().first()
            if session is None:
                raise KeyError(case_id)
            queued_result = await connection.execute(
                select(agent_turns.c.id).where(
                    agent_turns.c.session_id == session["id"], agent_turns.c.state == "queued"
                )
            )
            queued = queued_result.mappings().first()
            if queued is not None:
                raise TurnPendingError(str(queued["id"]))
            await connection.execute(
                agent_turns.insert().values(
                    id=turn_id,
                    session_id=session["id"],
                    kind="follow_up",
                    state="queued",
                    priority=110,
                    requested_by=actor,
                    client_request_id=f"{actor}:{uuid.uuid4()}",
                    prompt=text,
                )
            )
            await connection.execute(
                update(cases).where(cases.c.id == case_id).values(state="pending", updated_at=func.now())
            )
            await self._event(connection, case_id, "follow_up_queued", actor, {"turn_id": turn_id})
        return turn_id

    async def claim_next_turn(self) -> dict[str, object] | None:
        """Claim one queued turn after proving no other turn is globally active."""

        async with self._engine.begin() as connection:
            active_result = await connection.execute(
                select(agent_turns.c.id)
                .where(agent_turns.c.state.in_(("launching", "running")))
                .limit(1)
                .with_for_update()
            )
            if active_result.first() is not None:
                return None
            turn_result = await connection.execute(
                select(
                    *agent_turns.c,
                    agent_sessions.c.case_id,
                    agent_sessions.c.loom_session_id,
                    agent_sessions.c.loom_session_url,
                    cases.c.title,
                    cases.c.trigger,
                    cases.c.question,
                )
                .select_from(
                    agent_turns.join(agent_sessions, agent_sessions.c.id == agent_turns.c.session_id).join(
                        cases, cases.c.id == agent_sessions.c.case_id
                    )
                )
                .where(agent_turns.c.state == "queued", agent_turns.c.available_at <= func.now())
                .order_by(agent_turns.c.priority.desc(), agent_turns.c.available_at, agent_turns.c.created_at)
                .limit(1)
                .with_for_update(of=agent_turns, skip_locked=True)
            )
            turn = turn_result.mappings().first()
            if turn is None:
                return None
            await connection.execute(
                update(agent_turns)
                .where(agent_turns.c.id == turn["id"])
                .values(
                    state="launching",
                    lease_owner=OPS_SERVICE_ACTOR,
                    lease_expires_at=func.now() + TURN_LEASE_DURATION,
                    started_at=func.now(),
                    deadline_at=func.now() + TURN_TIMEOUT,
                )
            )
            await connection.execute(
                update(cases).where(cases.c.id == turn["case_id"]).values(state="investigating", updated_at=func.now())
            )
            await self._event(
                connection,
                str(turn["case_id"]),
                "turn_claimed",
                OPS_SERVICE_ACTOR,
                {"turn_id": str(turn["id"])},
                turn_id=str(turn["id"]),
            )
            return dict(turn)

    async def turn_started(
        self,
        *,
        turn_id: str,
        loom_session_id: str,
        loom_session_url: str,
        loom_turn_number: int | None,
    ) -> None:
        async with self._engine.begin() as connection:
            result = await connection.execute(
                select(agent_turns.c.session_id)
                .where(agent_turns.c.id == turn_id, agent_turns.c.state == "launching")
                .with_for_update()
            )
            row = result.mappings().first()
            if row is None:
                raise KeyError(turn_id)
            await connection.execute(
                update(agent_sessions)
                .where(agent_sessions.c.id == row["session_id"])
                .values(
                    loom_session_id=loom_session_id,
                    loom_session_url=loom_session_url,
                    state="active",
                    updated_at=func.now(),
                )
            )
            await connection.execute(
                update(agent_turns)
                .where(agent_turns.c.id == turn_id)
                .values(
                    state="running",
                    loom_turn_number=loom_turn_number,
                    lease_expires_at=func.now() + TURN_LEASE_DURATION,
                )
            )

    async def running_turn(self) -> dict[str, object] | None:
        async with self._engine.connect() as connection:
            result = await connection.execute(
                select(
                    *agent_turns.c,
                    agent_sessions.c.case_id,
                    agent_sessions.c.loom_session_id,
                    agent_sessions.c.loom_session_url,
                )
                .select_from(agent_turns.join(agent_sessions, agent_sessions.c.id == agent_turns.c.session_id))
                .where(agent_turns.c.state == "running")
                .limit(1)
            )
            row = result.mappings().first()
        return dict(row) if row is not None else None

    async def finish_turn(
        self,
        *,
        turn_id: str,
        result: OpsResult,
        artifact_revision: int,
        escalation: SlackEscalationDraft | None,
    ) -> None:
        async with self._engine.begin() as connection:
            row_result = await connection.execute(
                select(agent_turns.c.session_id, agent_sessions.c.case_id)
                .select_from(agent_turns.join(agent_sessions, agent_sessions.c.id == agent_turns.c.session_id))
                .where(agent_turns.c.id == turn_id, agent_turns.c.state == "running")
                .with_for_update(of=agent_turns)
            )
            row = row_result.mappings().first()
            if row is None:
                return
            await connection.execute(
                update(agent_turns)
                .where(agent_turns.c.id == turn_id)
                .values(
                    state="succeeded",
                    result=result.as_dict(),
                    result_artifact_revision=artifact_revision,
                    completed_at=func.now(),
                    lease_owner=None,
                    lease_expires_at=None,
                )
            )
            await connection.execute(
                update(agent_sessions)
                .where(agent_sessions.c.id == row["session_id"])
                .values(state="idle", updated_at=func.now())
            )
            await connection.execute(
                update(cases)
                .where(cases.c.id == row["case_id"])
                .values(
                    state="waiting_human",
                    outcome=result.outcome.value,
                    summary=result.summary,
                    investigated_at=func.now(),
                    updated_at=func.now(),
                )
            )
            await self._event(
                connection,
                str(row["case_id"]),
                "turn_finished",
                OPS_SERVICE_ACTOR,
                {"outcome": result.outcome.value, "summary": result.summary},
                turn_id=turn_id,
            )
            if escalation is not None:
                inserted = await connection.execute(
                    pg_insert(slack_escalations)
                    .values(
                        incident_key=escalation.incident_key,
                        case_id=row["case_id"],
                        turn_id=turn_id,
                        severity=escalation.severity.value,
                        reason=escalation.reason,
                        message=escalation.message,
                        state="pending",
                    )
                    .on_conflict_do_nothing(index_elements=[slack_escalations.c.incident_key])
                    .returning(slack_escalations.c.id)
                )
                if inserted.first() is not None:
                    await self._event(
                        connection,
                        str(row["case_id"]),
                        "slack_escalation_queued",
                        OPS_SERVICE_ACTOR,
                        {"severity": escalation.severity.value},
                        turn_id=turn_id,
                    )

    async def claim_slack_escalation(self) -> SlackDelivery | None:
        """Lease the oldest due Slack escalation for one delivery attempt."""

        async with self._engine.begin() as connection:
            await connection.execute(
                update(slack_escalations)
                .where(
                    slack_escalations.c.state == "sending",
                    slack_escalations.c.lease_expires_at <= func.now(),
                )
                .values(
                    state=case(
                        (slack_escalations.c.attempts >= SLACK_MAX_ATTEMPTS, "abandoned"),
                        else_="pending",
                    ),
                    lease_expires_at=None,
                )
            )
            queued_result = await connection.execute(
                select(slack_escalations.c.id)
                .where(
                    slack_escalations.c.state == "pending",
                    slack_escalations.c.available_at <= func.now(),
                )
                .order_by(slack_escalations.c.available_at, slack_escalations.c.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            queued = queued_result.mappings().first()
            if queued is None:
                return None
            claimed_result = await connection.execute(
                update(slack_escalations)
                .where(slack_escalations.c.id == queued["id"])
                .values(
                    state="sending",
                    attempts=slack_escalations.c.attempts + 1,
                    lease_expires_at=func.now() + SLACK_LEASE_DURATION,
                )
                .returning(slack_escalations)
            )
            claimed = claimed_result.mappings().first()
            assert claimed is not None
            return SlackDelivery(
                id=str(claimed["id"]),
                message=str(claimed["message"]),
                attempts=int(claimed["attempts"]),
            )

    async def slack_escalation_sent(self, escalation_id: str) -> None:
        async with self._engine.begin() as connection:
            result = await connection.execute(
                update(slack_escalations)
                .where(slack_escalations.c.id == escalation_id, slack_escalations.c.state == "sending")
                .values(state="sent", sent_at=func.now(), lease_expires_at=None, last_error=None)
                .returning(slack_escalations.c.case_id, slack_escalations.c.turn_id)
            )
            row = result.mappings().first()
            if row is not None:
                await self._event(
                    connection,
                    str(row["case_id"]),
                    "slack_escalation_sent",
                    OPS_SERVICE_ACTOR,
                    {},
                    turn_id=str(row["turn_id"]),
                )

    async def slack_escalation_retry(self, escalation_id: str, error: str) -> None:
        async with self._engine.begin() as connection:
            result = await connection.execute(
                select(
                    slack_escalations.c.case_id,
                    slack_escalations.c.turn_id,
                    slack_escalations.c.attempts,
                )
                .where(slack_escalations.c.id == escalation_id, slack_escalations.c.state == "sending")
                .with_for_update()
            )
            row = result.mappings().first()
            if row is None:
                return
            attempts = int(row["attempts"])
            state = "abandoned" if attempts >= SLACK_MAX_ATTEMPTS else "pending"
            retry_delay = timedelta(seconds=min(2**attempts, SLACK_MAX_RETRY_DELAY))
            await connection.execute(
                update(slack_escalations)
                .where(slack_escalations.c.id == escalation_id, slack_escalations.c.state == "sending")
                .values(
                    state=state,
                    available_at=func.now() + retry_delay,
                    lease_expires_at=None,
                    last_error=error[:MAX_PERSISTED_ERROR_CHARS],
                )
            )
            if state == "abandoned":
                await self._event(
                    connection,
                    str(row["case_id"]),
                    "slack_escalation_abandoned",
                    OPS_SERVICE_ACTOR,
                    {"attempts": attempts, "error": error[:MAX_EVENT_ERROR_CHARS]},
                    turn_id=str(row["turn_id"]),
                )

    async def recent_slack_escalations(self, *, limit: int = 60) -> list[dict[str, object]]:
        async with self._engine.connect() as connection:
            result = await connection.execute(
                select(
                    slack_escalations.c.id,
                    slack_escalations.c.case_id,
                    slack_escalations.c.turn_id,
                    slack_escalations.c.severity,
                    slack_escalations.c.reason,
                    slack_escalations.c.state,
                    slack_escalations.c.attempts,
                    slack_escalations.c.available_at,
                    slack_escalations.c.last_error,
                    slack_escalations.c.created_at,
                    slack_escalations.c.sent_at,
                )
                .order_by(slack_escalations.c.created_at.desc())
                .limit(limit)
            )
        return [dict(row) for row in result.mappings()]

    async def fail_turn(self, *, turn_id: str, error: str) -> None:
        async with self._engine.begin() as connection:
            result = await connection.execute(
                select(agent_turns.c.session_id, agent_sessions.c.case_id)
                .select_from(agent_turns.join(agent_sessions, agent_sessions.c.id == agent_turns.c.session_id))
                .where(agent_turns.c.id == turn_id, agent_turns.c.state.in_(("launching", "running")))
                .with_for_update(of=agent_turns)
            )
            row = result.mappings().first()
            if row is None:
                return
            bounded_error = error[:MAX_PERSISTED_ERROR_CHARS]
            await connection.execute(
                update(agent_turns)
                .where(agent_turns.c.id == turn_id)
                .values(
                    state="failed",
                    error=bounded_error,
                    completed_at=func.now(),
                    lease_owner=None,
                    lease_expires_at=None,
                )
            )
            await connection.execute(
                update(agent_sessions)
                .where(agent_sessions.c.id == row["session_id"])
                .values(state="idle", updated_at=func.now())
            )
            await connection.execute(
                update(cases)
                .where(cases.c.id == row["case_id"])
                .values(state="failed", summary=bounded_error, updated_at=func.now())
            )

    async def archive_case(self, *, case_id: str, actor: str) -> ArchiveResult:
        async with self._engine.begin() as connection:
            case_result = await connection.execute(select(cases.c.state).where(cases.c.id == case_id).with_for_update())
            case_row = case_result.mappings().first()
            if case_row is None:
                return ArchiveResult.NOT_FOUND
            if case_row["state"] == CaseState.ARCHIVED.value:
                return ArchiveResult.ALREADY_ARCHIVED
            active_result = await connection.execute(
                select(agent_turns.c.id)
                .select_from(agent_turns.join(agent_sessions, agent_sessions.c.id == agent_turns.c.session_id))
                .where(agent_sessions.c.case_id == case_id, agent_turns.c.state.in_(("launching", "running")))
            )
            if active_result.first() is not None:
                return ArchiveResult.ACTIVE_TURN
            await connection.execute(
                update(cases)
                .where(cases.c.id == case_id)
                .values(state="archived", archived_at=func.now(), updated_at=func.now())
            )
            session_ids = select(agent_sessions.c.id).where(agent_sessions.c.case_id == case_id)
            await connection.execute(
                update(agent_turns)
                .where(agent_turns.c.session_id.in_(session_ids), agent_turns.c.state == "queued")
                .values(state="cancelled", completed_at=func.now(), error="Case archived before launch")
            )
            await connection.execute(
                update(agent_sessions)
                .where(agent_sessions.c.case_id == case_id, agent_sessions.c.state != "archived")
                .values(state="archived", archived_at=func.now(), updated_at=func.now())
            )
            await self._event(connection, case_id, "case_archived", actor, {})
            return ArchiveResult.ARCHIVED

    async def _ingest_delivery(
        self,
        connection: AsyncConnection,
        delivery: GrafanaDelivery,
        *,
        key_id: str,
    ) -> IngestResult:
        existing = await self._existing_delivery(connection, delivery.delivery_key)
        if existing is not None:
            return _stored_ingest_result(existing)

        delivery_id = str(uuid.uuid4())
        insert_result = await connection.execute(
            pg_insert(source_deliveries)
            .values(
                id=delivery_id,
                source=SOURCE,
                delivery_key=delivery.delivery_key,
                key_id=key_id,
                source_timestamp=delivery.source_timestamp,
                body_sha256=delivery.body_sha256,
                normalized_payload=delivery.normalized_payload,
            )
            .on_conflict_do_nothing(index_elements=[source_deliveries.c.source, source_deliveries.c.delivery_key])
            .returning(source_deliveries.c.id)
        )
        if insert_result.first() is None:
            existing = await self._existing_delivery(connection, delivery.delivery_key)
            assert existing is not None
            return _stored_ingest_result(existing)

        touched = [
            await self._upsert_signal(connection, delivery, delivery_id, alert) for alert in delivery.notification.alerts
        ]
        case_id, should_queue = await self._materialize_case(connection, delivery, touched)
        queued_case_ids: tuple[str, ...] = ()
        if case_id is not None and should_queue:
            queued = await self._enqueue_automatic_turn(
                connection,
                case_id=case_id,
                delivery_key=delivery.delivery_key,
            )
            if queued:
                queued_case_ids = (case_id,)

        case_ids = (case_id,) if case_id is not None else ()
        result = IngestResult(
            delivery_id=delivery_id,
            duplicate=False,
            case_ids=case_ids,
            signal_dispositions={item.alert.fingerprint: item.disposition.value for item in touched},
            queued_case_ids=queued_case_ids,
        )
        await connection.execute(
            update(source_deliveries).where(source_deliveries.c.id == delivery_id).values(result=result.as_dict())
        )
        return result

    async def _existing_delivery(self, connection: AsyncConnection, delivery_key: str) -> RowMapping | None:
        result = await connection.execute(
            select(source_deliveries.c.id, source_deliveries.c.result)
            .where(source_deliveries.c.source == SOURCE, source_deliveries.c.delivery_key == delivery_key)
            .with_for_update()
        )
        return result.mappings().first()

    async def _upsert_signal(
        self,
        connection: AsyncConnection,
        delivery: GrafanaDelivery,
        delivery_id: str,
        alert: GrafanaAlert,
    ) -> TouchedSignal:
        prior_result = await connection.execute(
            select(signals)
            .where(signals.c.source == SOURCE, signals.c.fingerprint == alert.fingerprint)
            .with_for_update()
        )
        prior = prior_result.mappings().first()
        if prior is not None and delivery.source_timestamp < prior["latest_source_timestamp"]:
            await connection.execute(
                delivery_signals.insert().values(
                    delivery_id=delivery_id,
                    signal_id=prior["id"],
                    disposition="stale",
                )
            )
            return TouchedSignal(
                id=str(prior["id"]),
                generation=int(prior["generation"]),
                state=SignalState(str(prior["state"])),
                disposition=SignalDisposition.STALE,
                alert=alert,
            )

        if prior is None:
            signal_id = str(uuid.uuid4())
            generation = 1
            disposition = SignalDisposition.CREATED if alert.status == "firing" else SignalDisposition.RESOLVED
        else:
            signal_id = str(prior["id"])
            generation = int(prior["generation"])
            if prior["state"] == "resolved" and alert.status == "firing":
                generation += 1
                disposition = SignalDisposition.REOPENED
            elif prior["state"] == "firing" and alert.status == "resolved":
                disposition = SignalDisposition.RESOLVED
            else:
                disposition = SignalDisposition.UPDATED

        labels = alert.labels
        current_fields = {
            "receiver": delivery.notification.receiver,
            "group_key": delivery.notification.group_key,
            "alert_name": alert.alert_name,
            "severity": alert.severity,
            "cluster": labels.get("cluster"),
            "namespace": labels.get("namespace"),
            "object_kind": labels.get("object_kind") or labels.get("kind"),
            "object_name": labels.get("object_name") or labels.get("name") or labels.get("pod") or labels.get("node"),
            "summary": alert.summary,
            "labels": dict(alert.labels),
            "annotations": dict(alert.annotations),
            "values": dict(alert.values),
            "generator_url": alert.generator_url or None,
            "silence_url": alert.silence_url or None,
            "dashboard_url": alert.dashboard_url or None,
            "panel_url": alert.panel_url or None,
            "source_version": delivery.notification.version,
            "first_seen_at": alert.starts_at,
            "last_seen_at": delivery.source_timestamp,
            "resolved_at": alert.ends_at if alert.status == "resolved" else None,
            "latest_source_timestamp": delivery.source_timestamp,
            "latest_delivery_id": delivery_id,
        }
        if prior is None:
            await connection.execute(
                signals.insert().values(
                    id=signal_id,
                    source=SOURCE,
                    fingerprint=alert.fingerprint,
                    generation=generation,
                    state=alert.status,
                    **current_fields,
                )
            )
        else:
            await connection.execute(
                update(signals)
                .where(signals.c.id == signal_id)
                .values(
                    generation=generation,
                    state=alert.status,
                    missing_successful_polls=0,
                    **current_fields,
                )
            )
        await connection.execute(
            delivery_signals.insert().values(
                delivery_id=delivery_id,
                signal_id=signal_id,
                disposition=disposition.value,
            )
        )
        return TouchedSignal(
            id=signal_id,
            generation=generation,
            state=SignalState(alert.status),
            disposition=disposition,
            alert=alert,
        )

    async def _materialize_case(
        self,
        connection: AsyncConnection,
        delivery: GrafanaDelivery,
        touched: Sequence[TouchedSignal],
    ) -> tuple[str | None, bool]:
        notification = delivery.notification
        case_result = await connection.execute(
            select(cases)
            .where(
                cases.c.receiver == notification.receiver,
                cases.c.group_key == notification.group_key,
                cases.c.state != "archived",
            )
            .with_for_update()
        )
        case_row = case_result.mappings().first()
        novel_firing = any(
            item.state == SignalState.FIRING
            and item.disposition in (SignalDisposition.CREATED, SignalDisposition.REOPENED)
            for item in touched
        )
        any_firing = bool(
            await connection.scalar(
                select(
                    exists().where(
                        signals.c.source == SOURCE,
                        signals.c.receiver == notification.receiver,
                        signals.c.group_key == notification.group_key,
                        signals.c.state == "firing",
                    )
                )
            )
        )
        if case_row is None and not novel_firing:
            return None, False
        created = case_row is None
        if created:
            case_id = await self._open_case(connection, notification=notification, touched=touched)
        else:
            case_id = str(case_row["id"])

        attached_novel = await self._attach_signals(connection, case_id=case_id, touched=touched)

        if not any_firing:
            await self._resolve_case_group(connection, case_id=case_id)
            return case_id, False

        should_queue = created or novel_firing or attached_novel
        case_state = None if case_row is None else CaseState(str(case_row["state"]))
        await self._touch_case(connection, case_id=case_id, requeue=should_queue and not created, state=case_state)
        return case_id, should_queue

    async def _open_case(
        self,
        connection: AsyncConnection,
        *,
        notification: GrafanaNotification,
        touched: Sequence[TouchedSignal],
    ) -> str:
        case_id = str(uuid.uuid4())
        priority = max(severity_priority(item.alert.severity) for item in touched if item.state == SignalState.FIRING)
        title = notification.title or notification.common_annotations.get("summary") or _case_title(touched)
        await connection.execute(
            cases.insert().values(
                id=case_id,
                trigger="automatic",
                receiver=notification.receiver,
                group_key=notification.group_key,
                grouping_rule=GROUPING_RULE,
                state="pending",
                priority=priority,
                title=title,
                opened_at=func.now(),
                updated_at=func.now(),
            )
        )
        await self._event(
            connection,
            case_id,
            "case_opened",
            SOURCE,
            {"receiver": notification.receiver, "group_key": notification.group_key},
        )
        return case_id

    async def _attach_signals(
        self,
        connection: AsyncConnection,
        *,
        case_id: str,
        touched: Sequence[TouchedSignal],
    ) -> bool:
        attached_novel = False
        for item in touched:
            if item.disposition == SignalDisposition.STALE:
                continue
            result = await connection.execute(
                pg_insert(case_signals)
                .values(case_id=case_id, signal_id=item.id, signal_generation=item.generation)
                .on_conflict_do_nothing()
                .returning(case_signals.c.signal_id)
            )
            attached_novel = attached_novel or result.first() is not None
        return attached_novel

    async def _resolve_case_group(
        self,
        connection: AsyncConnection,
        *,
        case_id: str,
    ) -> None:
        await connection.execute(
            update(cases)
            .where(cases.c.id == case_id)
            .values(
                state=case((cases.c.state == "pending", "investigated"), else_=cases.c.state),
                outcome=case((cases.c.state == "pending", "no_action"), else_=cases.c.outcome),
                summary=case(
                    (cases.c.state == "pending", "Grafana resolved the group before investigation"),
                    else_=cases.c.summary,
                ),
                investigated_at=case(
                    (cases.c.state == "pending", func.now()),
                    else_=cases.c.investigated_at,
                ),
                updated_at=func.now(),
            )
        )
        session_ids = select(agent_sessions.c.id).where(agent_sessions.c.case_id == case_id)
        await connection.execute(
            update(agent_turns)
            .where(
                agent_turns.c.session_id.in_(session_ids),
                agent_turns.c.state == "queued",
                agent_turns.c.kind == "automatic",
            )
            .values(
                state="cancelled",
                completed_at=func.now(),
                error="Grafana resolved the group before launch",
            )
        )
        await self._event(connection, case_id, "grafana_group_resolved", SOURCE, {})

    async def _touch_case(
        self,
        connection: AsyncConnection,
        *,
        case_id: str,
        requeue: bool,
        state: CaseState | None,
    ) -> None:
        needs_pending = requeue and state in (
            CaseState.WAITING_HUMAN,
            CaseState.INVESTIGATED,
            CaseState.FAILED,
        )
        if needs_pending:
            await connection.execute(
                update(cases).where(cases.c.id == case_id).values(state="pending", outcome=None, updated_at=func.now())
            )
            return
        await connection.execute(update(cases).where(cases.c.id == case_id).values(updated_at=func.now()))

    async def _enqueue_automatic_turn(
        self,
        connection: AsyncConnection,
        *,
        case_id: str,
        delivery_key: str,
    ) -> bool:
        session_result = await connection.execute(
            select(agent_sessions.c.id)
            .where(agent_sessions.c.case_id == case_id, agent_sessions.c.state != "archived")
            .with_for_update()
        )
        session = session_result.mappings().first()
        if session is None:
            session_id = str(uuid.uuid4())
            await self._insert_session(connection, session_id=session_id, case_id=case_id)
        else:
            session_id = str(session["id"])
        queued_result = await connection.execute(
            select(agent_turns.c.id).where(agent_turns.c.session_id == session_id, agent_turns.c.state == "queued")
        )
        if queued_result.first() is not None:
            return False
        priority = await connection.scalar(select(cases.c.priority).where(cases.c.id == case_id))
        assert priority is not None
        turn_id = str(uuid.uuid4())
        insert_result = await connection.execute(
            pg_insert(agent_turns)
            .values(
                id=turn_id,
                session_id=session_id,
                kind="automatic",
                state="queued",
                priority=priority,
                requested_by=AUTOMATIC_REQUESTER,
                client_request_id=f"grafana:{case_id}:{delivery_key}",
                prompt="Investigate the current Grafana alert group using the attached case evidence.",
            )
            .on_conflict_do_nothing(index_elements=[agent_turns.c.requested_by, agent_turns.c.client_request_id])
            .returning(agent_turns.c.id)
        )
        if insert_result.first() is None:
            return False
        await self._event(connection, case_id, "automatic_turn_queued", SOURCE, {"turn_id": turn_id})
        return True

    async def _insert_session(
        self,
        connection: AsyncConnection,
        *,
        session_id: str,
        case_id: str,
    ) -> None:
        await connection.execute(
            agent_sessions.insert().values(
                id=session_id,
                case_id=case_id,
                deterministic_name=f"ops-case-{case_id}",
                state="new",
                repo_revision=self._repo_revision,
                skill_revision=self._skill_revision,
            )
        )

    async def _event(
        self,
        connection: AsyncConnection,
        case_id: str,
        event_type: str,
        actor: str,
        data: Mapping[str, object],
        *,
        turn_id: str | None = None,
    ) -> None:
        await connection.execute(
            case_events.insert().values(
                case_id=case_id,
                turn_id=turn_id,
                event_type=event_type,
                actor=actor,
                data=dict(data),
            )
        )


def _stored_ingest_result(row: Mapping[str, Any]) -> IngestResult:
    result = row["result"]
    if not isinstance(result, dict):
        raise RuntimeError(f"delivery {row['id']} committed without a result")
    return IngestResult(
        delivery_id=str(row["id"]),
        duplicate=True,
        case_ids=tuple(result.get("case_ids", ())),
        signal_dispositions=dict(result.get("signal_dispositions", {})),
        queued_case_ids=tuple(result.get("queued_case_ids", ())),
    )


def _poll_delivery(
    items: Sequence[PolledGrafanaAlert],
    *,
    observed_at: datetime,
    status: str,
) -> GrafanaDelivery:
    if not items:
        raise ValueError("a polled Grafana delivery requires at least one alert")
    first = items[0]
    if any(item.receiver != first.receiver or item.group_key != first.group_key for item in items):
        raise ValueError("all alerts in a polled delivery must share one group")
    alerts = tuple(item.alert for item in items)
    common_labels = _common_strings([alert.labels for alert in alerts])
    common_annotations = _common_strings([alert.annotations for alert in alerts])
    normalized = {
        "source": SOURCE_VERSION,
        "observed_at": observed_at.isoformat(),
        "receiver": first.receiver,
        "status": status,
        "group_key": first.group_key,
        "group_labels": dict(first.group_labels),
        "alerts": [
            {
                "fingerprint": alert.fingerprint,
                "status": alert.status,
                "labels": dict(alert.labels),
                "annotations": dict(alert.annotations),
                "values": dict(alert.values),
                "starts_at": alert.starts_at.isoformat(),
                "ends_at": alert.ends_at.isoformat() if alert.ends_at else None,
            }
            for alert in alerts
        ],
    }
    body = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode()
    body_sha256 = hashlib.sha256(body).hexdigest()
    return GrafanaDelivery(
        notification=GrafanaNotification(
            receiver=first.receiver,
            status=status,
            org_id=0,
            version=SOURCE_VERSION,
            group_key=first.group_key,
            group_labels=first.group_labels,
            common_labels=common_labels,
            common_annotations=common_annotations,
            external_url=f"{GRAFANA_BASE_URL}/",
            title=first.title,
            message="",
            alerts=alerts,
        ),
        source_timestamp=observed_at,
        delivery_key=f"poll:{body_sha256}",
        body_sha256=body_sha256,
        normalized_payload=normalized,
    )


def _resolved_polled_alert(signal: Mapping[str, Any], *, observed_at: datetime) -> PolledGrafanaAlert:
    labels = _mapping(signal["labels"], "signal labels")
    annotations = _mapping(signal["annotations"], "signal annotations")
    values = signal["values"]
    if not isinstance(values, Mapping):
        raise RuntimeError("stored signal values must be an object")
    alert_name = str(signal["alert_name"])
    cluster = str(signal["cluster"] or "")
    group = grafana_group_metadata(alert_name, cluster)
    return PolledGrafanaAlert(
        alert=GrafanaAlert(
            fingerprint=str(signal["fingerprint"]),
            status="resolved",
            labels=labels,
            annotations=annotations,
            values=values,
            starts_at=signal["first_seen_at"],
            ends_at=observed_at,
            generator_url=str(signal["generator_url"] or ""),
            silence_url=str(signal["silence_url"] or ""),
            dashboard_url=str(signal["dashboard_url"] or ""),
            panel_url=str(signal["panel_url"] or ""),
        ),
        receiver=str(signal["receiver"]),
        group_key=str(signal["group_key"]),
        group_labels=group.labels,
        title=group.title,
    )


def _mapping(value: object, field: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise RuntimeError(f"stored {field} must contain string keys and values")
    return cast(Mapping[str, str], value)


def _common_strings(values: Sequence[Mapping[str, str]]) -> Mapping[str, str]:
    first = values[0]
    return {key: item for key, item in first.items() if all(value.get(key) == item for value in values[1:])}


def _case_title(touched: Sequence[TouchedSignal]) -> str:
    alerts = [item.alert for item in touched if item.state == SignalState.FIRING]
    if not alerts:
        return "Resolved Grafana alert group"
    first = alerts[0]
    suffix = f" (+{len(alerts) - 1})" if len(alerts) > 1 else ""
    return f"{first.alert_name}: {first.summary}{suffix}"


def json_evidence(detail: Mapping[str, object]) -> str:
    """Serialize a bounded, prompt-safe evidence packet for the agent."""

    case = detail["case"]
    signals = detail["signals"]
    assert isinstance(case, Mapping)
    assert isinstance(signals, list)
    evidence = {
        "case": {
            "id": str(case["id"]),
            "title": case["title"],
            "receiver": case["receiver"],
            "group_key": case["group_key"],
            "question": case.get("question"),
        },
        "grafana_alerts": [
            {
                "fingerprint": item["fingerprint"],
                "generation": item["signal_generation"],
                "state": item["state"],
                "alert_name": item["alert_name"],
                "severity": item["severity"],
                "cluster": item["cluster"],
                "namespace": item["namespace"],
                "object_kind": item["object_kind"],
                "object_name": item["object_name"],
                "summary": item["summary"],
                "labels": item["labels"],
                "annotations": item["annotations"],
                "values": item["values"],
                "starts_at": item["first_seen_at"],
                "resolved_at": item["resolved_at"],
                "generator_url": item["generator_url"],
            }
            for item in signals
        ],
    }
    return json.dumps(evidence, default=str, indent=2)
