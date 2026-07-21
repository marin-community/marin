# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL persistence for Grafana-owned alert and agent lifecycles."""

import hashlib
import json
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ops_workflow.grafana import GrafanaAlert, GrafanaDelivery, GrafanaNotification
from ops_workflow.grafana_source import SOURCE_VERSION, GrafanaSnapshot, PolledGrafanaAlert
from ops_workflow.state import CaseState, SignalDisposition, severity_priority

SOURCE = "grafana"
GROUPING_RULE = "grafana:alertname+cluster"
AUTOMATIC_REQUESTER = "grafana"
MISSING_POLLS_TO_RESOLVE = 2
POLL_KEY_ID = "grafana-postgres-reader"


class TurnPendingError(RuntimeError):
    """The case already has queued work for its current session."""


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


class OpsRepository:
    """Small connection-per-operation repository for the ops workflow."""

    def __init__(self, database_url: str, *, repo_revision: str, skill_revision: str) -> None:
        self._database_url = database_url
        self._repo_revision = repo_revision
        self._skill_revision = skill_revision

    async def reconcile_grafana_snapshot(self, snapshot: GrafanaSnapshot) -> tuple[IngestResult, ...]:
        """Atomically reconcile one complete, successful Grafana SQL snapshot."""

        grouped: dict[tuple[str, str], list[PolledGrafanaAlert]] = defaultdict(list)
        for item in snapshot.alerts:
            grouped[(item.receiver, item.group_key)].append(item)
        present = {item.alert.fingerprint for item in snapshot.alerts}

        async with await self._connection() as connection:
            async with connection.transaction():
                poll_slot = snapshot.observed_at.replace(second=0, microsecond=0)
                poll_cursor = await connection.execute(
                    """
                    INSERT INTO grafana_polls (poll_slot, observed_at, alert_count)
                    VALUES (%s, %s, %s) ON CONFLICT (poll_slot) DO NOTHING
                    RETURNING id
                    """,
                    (poll_slot, snapshot.observed_at, len(snapshot.alerts)),
                )
                if await poll_cursor.fetchone() is None:
                    return ()
                results: list[IngestResult] = []
                for items in grouped.values():
                    webhook = _poll_webhook(items, observed_at=snapshot.observed_at, status="firing")
                    results.append(await self._ingest_delivery(connection, webhook, key_id=POLL_KEY_ID))

                cursor = await connection.execute(
                    "SELECT * FROM signals WHERE source = %s AND state = 'firing' FOR UPDATE",
                    (SOURCE,),
                )
                firing = await cursor.fetchall()
                resolving: dict[tuple[str, str], list[PolledGrafanaAlert]] = defaultdict(list)
                for signal in firing:
                    fingerprint = str(signal["fingerprint"])
                    if fingerprint in present:
                        if signal["missing_successful_polls"]:
                            await connection.execute(
                                "UPDATE signals SET missing_successful_polls = 0 WHERE id = %s",
                                (signal["id"],),
                            )
                        continue
                    missing = int(signal["missing_successful_polls"]) + 1
                    if missing < MISSING_POLLS_TO_RESOLVE:
                        await connection.execute(
                            "UPDATE signals SET missing_successful_polls = %s WHERE id = %s",
                            (missing, signal["id"]),
                        )
                        continue
                    item = _resolved_polled_alert(signal, observed_at=snapshot.observed_at)
                    resolving[(item.receiver, item.group_key)].append(item)

                for items in resolving.values():
                    webhook = _poll_webhook(items, observed_at=snapshot.observed_at, status="resolved")
                    results.append(await self._ingest_delivery(connection, webhook, key_id=POLL_KEY_ID))
                return tuple(results)

    async def overview(self) -> dict[str, object]:
        async with await self._connection() as connection:
            counts_cursor = await connection.execute(
                "SELECT state, count(*) AS count FROM cases GROUP BY state ORDER BY state"
            )
            count_rows = await counts_cursor.fetchall()
            active_cursor = await connection.execute(
                """
                SELECT c.id AS case_id, c.title, t.started_at, s.loom_session_url
                FROM agent_turns t
                JOIN agent_sessions s ON s.id = t.session_id
                JOIN cases c ON c.id = s.case_id
                WHERE t.state IN ('launching', 'running')
                LIMIT 1
                """
            )
            active = await active_cursor.fetchone()
            freshness_cursor = await connection.execute("SELECT max(observed_at) AS last_poll_at FROM grafana_polls")
            freshness = await freshness_cursor.fetchone()
        return {
            "case_counts": {row["state"]: row["count"] for row in count_rows},
            "active_investigation": active,
            "last_poll_at": freshness["last_poll_at"] if freshness else None,
        }

    async def list_cases(self, *, include_archived: bool = False) -> list[dict[str, object]]:
        async with await self._connection() as connection:
            cursor = await connection.execute(
                """
                SELECT c.id, c.trigger, c.state, c.priority, c.title, c.receiver,
                       c.group_key, c.outcome, c.summary, c.opened_at, c.updated_at,
                       count(cs.signal_id) AS signal_count,
                       count(cs.signal_id) FILTER (WHERE s.state = 'firing') AS firing_count,
                       array_remove(array_agg(DISTINCT s.cluster), NULL) AS clusters,
                       a.loom_session_url
                FROM cases c
                LEFT JOIN case_signals cs ON cs.case_id = c.id
                LEFT JOIN signals s ON s.id = cs.signal_id AND s.generation = cs.signal_generation
                LEFT JOIN agent_sessions a ON a.case_id = c.id AND a.state <> 'archived'
                WHERE %s OR c.state <> 'archived'
                GROUP BY c.id, a.loom_session_url
                ORDER BY
                    CASE WHEN c.state = 'investigating' THEN 0 WHEN c.state = 'pending' THEN 1 ELSE 2 END,
                    c.priority DESC, c.updated_at DESC
                LIMIT 200
                """,
                (include_archived,),
            )
            rows = await cursor.fetchall()
        return list(rows)

    async def case_detail(self, case_id: str) -> dict[str, object] | None:
        async with await self._connection() as connection:
            case_cursor = await connection.execute(
                """
                SELECT c.*, a.id AS agent_session_id, a.loom_session_id,
                       a.loom_session_url, a.state AS agent_session_state
                FROM cases c
                LEFT JOIN agent_sessions a ON a.case_id = c.id AND a.state <> 'archived'
                WHERE c.id = %s
                """,
                (case_id,),
            )
            case = await case_cursor.fetchone()
            if case is None:
                return None
            signals_cursor = await connection.execute(
                """
                SELECT s.*, cs.signal_generation, cs.attached_at
                FROM case_signals cs
                JOIN signals s ON s.id = cs.signal_id
                WHERE cs.case_id = %s AND cs.signal_generation = s.generation
                ORDER BY s.severity, s.alert_name, s.fingerprint
                """,
                (case_id,),
            )
            turns_cursor = await connection.execute(
                """
                SELECT t.* FROM agent_turns t
                JOIN agent_sessions a ON a.id = t.session_id
                WHERE a.case_id = %s
                ORDER BY t.created_at
                """,
                (case_id,),
            )
            events_cursor = await connection.execute(
                "SELECT * FROM case_events WHERE case_id = %s ORDER BY created_at, id",
                (case_id,),
            )
            signals = await signals_cursor.fetchall()
            turns = await turns_cursor.fetchall()
            events = await events_cursor.fetchall()
        return {"case": case, "signals": signals, "turns": turns, "events": events}

    async def create_question(self, *, text: str, actor: str) -> str:
        case_id = str(uuid.uuid4())
        group_key = f"manual:{case_id}"
        session_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        async with await self._connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO cases (
                        id, trigger, receiver, group_key, grouping_rule, state,
                        priority, title, question, opened_at, updated_at
                    ) VALUES (%s, 'manual', 'manual', %s, 'manual', 'pending', 120, %s, %s, %s, %s)
                    """,
                    (case_id, group_key, text[:160], text, now, now),
                )
                await self._insert_session(connection, session_id=session_id, case_id=case_id)
                await connection.execute(
                    """
                    INSERT INTO agent_turns (
                        id, session_id, kind, state, priority, requested_by,
                        client_request_id, prompt, created_at
                    ) VALUES (%s, %s, 'question', 'queued', 120, %s, %s, %s, %s)
                    """,
                    (turn_id, session_id, actor, f"manual:{case_id}", text, now),
                )
                await self._event(connection, case_id, "question_queued", actor, {"turn_id": turn_id})
        return case_id

    async def enqueue_follow_up(self, *, case_id: str, text: str, actor: str) -> str:
        turn_id = str(uuid.uuid4())
        async with await self._connection() as connection:
            async with connection.transaction():
                session_cursor = await connection.execute(
                    """
                    SELECT a.id FROM agent_sessions a
                    JOIN cases c ON c.id = a.case_id
                    WHERE a.case_id = %s AND a.state <> 'archived' AND c.state <> 'archived'
                    FOR UPDATE
                    """,
                    (case_id,),
                )
                session = await session_cursor.fetchone()
                if session is None:
                    raise KeyError(case_id)
                queued_cursor = await connection.execute(
                    "SELECT id FROM agent_turns WHERE session_id = %s AND state = 'queued'",
                    (session["id"],),
                )
                queued = await queued_cursor.fetchone()
                if queued is not None:
                    raise TurnPendingError(str(queued["id"]))
                await connection.execute(
                    """
                    INSERT INTO agent_turns (
                        id, session_id, kind, state, priority, requested_by,
                        client_request_id, prompt
                    ) VALUES (%s, %s, 'follow_up', 'queued', 110, %s, %s, %s)
                    """,
                    (turn_id, session["id"], actor, f"{actor}:{uuid.uuid4()}", text),
                )
                await connection.execute(
                    "UPDATE cases SET state = 'pending', updated_at = now() WHERE id = %s",
                    (case_id,),
                )
                await self._event(connection, case_id, "follow_up_queued", actor, {"turn_id": turn_id})
        return turn_id

    async def claim_next_turn(self) -> dict[str, object] | None:
        """Claim one queued turn after proving no other turn is globally active."""

        async with await self._connection() as connection:
            async with connection.transaction():
                active_cursor = await connection.execute(
                    "SELECT id FROM agent_turns WHERE state IN ('launching', 'running') LIMIT 1 FOR UPDATE"
                )
                if await active_cursor.fetchone() is not None:
                    return None
                cursor = await connection.execute(
                    """
                    SELECT t.*, a.case_id, a.loom_session_id, a.loom_session_url,
                           c.title, c.trigger, c.question
                    FROM agent_turns t
                    JOIN agent_sessions a ON a.id = t.session_id
                    JOIN cases c ON c.id = a.case_id
                    WHERE t.state = 'queued' AND t.available_at <= now()
                    ORDER BY t.priority DESC, t.available_at, t.created_at
                    LIMIT 1
                    FOR UPDATE OF t SKIP LOCKED
                    """
                )
                turn = await cursor.fetchone()
                if turn is None:
                    return None
                await connection.execute(
                    """
                    UPDATE agent_turns
                    SET state = 'launching', lease_owner = 'ops-service',
                        lease_expires_at = now() + interval '10 minutes',
                        started_at = now(), deadline_at = now() + interval '20 minutes'
                    WHERE id = %s
                    """,
                    (turn["id"],),
                )
                await connection.execute(
                    "UPDATE cases SET state = 'investigating', updated_at = now() WHERE id = %s",
                    (turn["case_id"],),
                )
                await self._event(
                    connection,
                    str(turn["case_id"]),
                    "turn_claimed",
                    "ops-service",
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
        async with await self._connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    "SELECT session_id FROM agent_turns WHERE id = %s AND state = 'launching' FOR UPDATE",
                    (turn_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise KeyError(turn_id)
                await connection.execute(
                    """
                    UPDATE agent_sessions SET loom_session_id = %s, loom_session_url = %s,
                        state = 'active', updated_at = now() WHERE id = %s
                    """,
                    (loom_session_id, loom_session_url, row["session_id"]),
                )
                await connection.execute(
                    """
                    UPDATE agent_turns SET state = 'running', loom_turn_number = %s,
                        lease_expires_at = now() + interval '10 minutes' WHERE id = %s
                    """,
                    (loom_turn_number, turn_id),
                )

    async def running_turn(self) -> dict[str, object] | None:
        async with await self._connection() as connection:
            cursor = await connection.execute(
                """
                SELECT t.*, a.case_id, a.loom_session_id, a.loom_session_url
                FROM agent_turns t
                JOIN agent_sessions a ON a.id = t.session_id
                WHERE t.state = 'running'
                LIMIT 1
                """
            )
            row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def finish_turn(self, *, turn_id: str, summary: str) -> None:
        async with await self._connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    """
                    SELECT t.session_id, a.case_id FROM agent_turns t
                    JOIN agent_sessions a ON a.id = t.session_id
                    WHERE t.id = %s AND t.state = 'running' FOR UPDATE OF t
                    """,
                    (turn_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    return
                result = {"summary": summary, "source": "loom_chat"}
                await connection.execute(
                    """
                    UPDATE agent_turns SET state = 'succeeded', result = %s,
                        completed_at = now(), lease_owner = NULL, lease_expires_at = NULL
                    WHERE id = %s
                    """,
                    (Jsonb(result), turn_id),
                )
                await connection.execute(
                    "UPDATE agent_sessions SET state = 'idle', updated_at = now() WHERE id = %s",
                    (row["session_id"],),
                )
                await connection.execute(
                    """
                    UPDATE cases SET state = 'waiting_human', outcome = 'unknown',
                        summary = %s, investigated_at = now(), updated_at = now()
                    WHERE id = %s
                    """,
                    (summary, row["case_id"]),
                )
                await self._event(
                    connection,
                    str(row["case_id"]),
                    "turn_finished",
                    "ops-service",
                    {"summary": summary},
                    turn_id=turn_id,
                )

    async def fail_turn(self, *, turn_id: str, error: str) -> None:
        async with await self._connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    """
                    SELECT t.session_id, a.case_id FROM agent_turns t
                    JOIN agent_sessions a ON a.id = t.session_id
                    WHERE t.id = %s AND t.state IN ('launching', 'running') FOR UPDATE OF t
                    """,
                    (turn_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    return
                bounded_error = error[:4_000]
                await connection.execute(
                    """
                    UPDATE agent_turns SET state = 'failed', error = %s, completed_at = now(),
                        lease_owner = NULL, lease_expires_at = NULL WHERE id = %s
                    """,
                    (bounded_error, turn_id),
                )
                await connection.execute(
                    "UPDATE agent_sessions SET state = 'idle', updated_at = now() WHERE id = %s",
                    (row["session_id"],),
                )
                await connection.execute(
                    "UPDATE cases SET state = 'failed', summary = %s, updated_at = now() WHERE id = %s",
                    (bounded_error, row["case_id"]),
                )

    async def archive_case(self, *, case_id: str, actor: str) -> bool:
        async with await self._connection() as connection:
            async with connection.transaction():
                active_cursor = await connection.execute(
                    """
                    SELECT t.id FROM agent_turns t JOIN agent_sessions a ON a.id = t.session_id
                    WHERE a.case_id = %s AND t.state IN ('launching', 'running')
                    """,
                    (case_id,),
                )
                if await active_cursor.fetchone() is not None:
                    return False
                cursor = await connection.execute(
                    """
                    UPDATE cases SET state = 'archived', archived_at = now(), updated_at = now()
                    WHERE id = %s AND state <> 'archived' RETURNING id
                    """,
                    (case_id,),
                )
                archived = await cursor.fetchone()
                if archived is None:
                    return False
                await connection.execute(
                    """
                    UPDATE agent_sessions SET state = 'archived', archived_at = now(), updated_at = now()
                    WHERE case_id = %s AND state <> 'archived'
                    """,
                    (case_id,),
                )
                await self._event(connection, case_id, "case_archived", actor, {})
                return True

    async def _connection(self) -> psycopg.AsyncConnection[dict[str, Any]]:
        connection = await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row)
        return cast(psycopg.AsyncConnection[dict[str, Any]], connection)

    async def _ingest_delivery(
        self,
        connection: psycopg.AsyncConnection[dict[str, Any]],
        webhook: GrafanaDelivery,
        *,
        key_id: str,
    ) -> IngestResult:
        existing = await self._existing_delivery(connection, webhook.delivery_key)
        if existing is not None:
            return _stored_ingest_result(existing)

        delivery_id = str(uuid.uuid4())
        insert_cursor = await connection.execute(
            """
            INSERT INTO source_deliveries (
                id, source, delivery_key, key_id, source_timestamp,
                body_sha256, normalized_payload
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, delivery_key) DO NOTHING
            RETURNING id
            """,
            (
                delivery_id,
                SOURCE,
                webhook.delivery_key,
                key_id,
                webhook.source_timestamp,
                webhook.body_sha256,
                Jsonb(webhook.normalized_payload),
            ),
        )
        if await insert_cursor.fetchone() is None:
            existing = await self._existing_delivery(connection, webhook.delivery_key)
            assert existing is not None
            return _stored_ingest_result(existing)

        touched = [
            await self._upsert_signal(connection, webhook, delivery_id, alert) for alert in webhook.notification.alerts
        ]
        case_id, should_queue = await self._materialize_case(connection, webhook, touched)
        queued_case_ids: tuple[str, ...] = ()
        if case_id is not None and should_queue:
            queued = await self._enqueue_automatic_turn(
                connection,
                case_id=case_id,
                delivery_key=webhook.delivery_key,
            )
            if queued:
                queued_case_ids = (case_id,)

        case_ids = (case_id,) if case_id is not None else ()
        result = IngestResult(
            delivery_id=delivery_id,
            duplicate=False,
            case_ids=case_ids,
            signal_dispositions={item[4].fingerprint: item[3].value for item in touched},
            queued_case_ids=queued_case_ids,
        )
        await connection.execute(
            "UPDATE source_deliveries SET result = %s WHERE id = %s",
            (Jsonb(result.as_dict()), delivery_id),
        )
        return result

    async def _existing_delivery(
        self, connection: psycopg.AsyncConnection[dict[str, Any]], delivery_key: str
    ) -> dict[str, Any] | None:
        cursor = await connection.execute(
            "SELECT id, result FROM source_deliveries WHERE source = %s AND delivery_key = %s FOR UPDATE",
            (SOURCE, delivery_key),
        )
        row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def _upsert_signal(
        self,
        connection: psycopg.AsyncConnection[dict[str, Any]],
        webhook: GrafanaDelivery,
        delivery_id: str,
        alert: GrafanaAlert,
    ) -> tuple[str, int, str, SignalDisposition, GrafanaAlert]:
        cursor = await connection.execute(
            "SELECT * FROM signals WHERE source = %s AND fingerprint = %s FOR UPDATE",
            (SOURCE, alert.fingerprint),
        )
        prior = await cursor.fetchone()
        if prior is not None and webhook.source_timestamp < prior["latest_source_timestamp"]:
            await connection.execute(
                """
                INSERT INTO delivery_signals (delivery_id, signal_id, disposition)
                VALUES (%s, %s, 'stale')
                """,
                (delivery_id, prior["id"]),
            )
            return (
                str(prior["id"]),
                int(prior["generation"]),
                str(prior["state"]),
                SignalDisposition.STALE,
                alert,
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
        current_fields = (
            webhook.notification.receiver,
            webhook.notification.group_key,
            alert.alert_name,
            alert.severity,
            labels.get("cluster"),
            labels.get("namespace"),
            labels.get("object_kind") or labels.get("kind"),
            labels.get("object_name") or labels.get("name") or labels.get("pod") or labels.get("node"),
            alert.summary,
            Jsonb(alert.labels),
            Jsonb(alert.annotations),
            Jsonb(alert.values),
            alert.generator_url or None,
            alert.silence_url or None,
            alert.dashboard_url or None,
            alert.panel_url or None,
            webhook.notification.version,
            alert.starts_at,
            webhook.source_timestamp,
            alert.ends_at if alert.status == "resolved" else None,
            webhook.source_timestamp,
            delivery_id,
        )
        if prior is None:
            await connection.execute(
                """
                INSERT INTO signals (
                    id, source, fingerprint, generation, state, receiver, group_key,
                    alert_name, severity, cluster, namespace, object_kind, object_name,
                    summary, labels, annotations, values, generator_url, silence_url,
                    dashboard_url, panel_url, source_version, first_seen_at, last_seen_at,
                    resolved_at, latest_source_timestamp, latest_delivery_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    signal_id,
                    SOURCE,
                    alert.fingerprint,
                    generation,
                    alert.status,
                    *current_fields,
                ),
            )
        else:
            await connection.execute(
                """
                UPDATE signals SET generation = %s, state = %s, receiver = %s,
                    group_key = %s, alert_name = %s, severity = %s, cluster = %s,
                    namespace = %s, object_kind = %s, object_name = %s, summary = %s,
                    labels = %s, annotations = %s, values = %s, generator_url = %s,
                    silence_url = %s, dashboard_url = %s, panel_url = %s,
                    source_version = %s, first_seen_at = %s, last_seen_at = %s,
                    resolved_at = %s, latest_source_timestamp = %s, latest_delivery_id = %s,
                    missing_successful_polls = 0
                WHERE id = %s
                """,
                (generation, alert.status, *current_fields, signal_id),
            )
        await connection.execute(
            """
            INSERT INTO delivery_signals (delivery_id, signal_id, disposition)
            VALUES (%s, %s, %s)
            """,
            (delivery_id, signal_id, disposition.value),
        )
        return signal_id, generation, alert.status, disposition, alert

    async def _materialize_case(
        self,
        connection: psycopg.AsyncConnection[dict[str, Any]],
        webhook: GrafanaDelivery,
        touched: list[tuple[str, int, str, SignalDisposition, GrafanaAlert]],
    ) -> tuple[str | None, bool]:
        notification = webhook.notification
        cursor = await connection.execute(
            """
            SELECT * FROM cases
            WHERE receiver = %s AND group_key = %s AND state <> 'archived'
            FOR UPDATE
            """,
            (notification.receiver, notification.group_key),
        )
        case = await cursor.fetchone()
        novel_firing = any(
            state == "firing" and disposition in (SignalDisposition.CREATED, SignalDisposition.REOPENED)
            for _, _, state, disposition, _ in touched
        )
        firing_cursor = await connection.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM signals
                WHERE source = %s AND receiver = %s AND group_key = %s AND state = 'firing'
            ) AS any_firing
            """,
            (SOURCE, notification.receiver, notification.group_key),
        )
        firing_row = await firing_cursor.fetchone()
        assert firing_row is not None
        any_firing = bool(firing_row["any_firing"])
        if case is None and not novel_firing:
            return None, False
        created = case is None
        if created:
            case_id = str(uuid.uuid4())
            priority = max(severity_priority(alert.severity) for _, _, state, _, alert in touched if state == "firing")
            title = notification.title or notification.common_annotations.get("summary") or _case_title(touched)
            await connection.execute(
                """
                INSERT INTO cases (
                    id, trigger, receiver, group_key, grouping_rule, state,
                    priority, title, opened_at, updated_at
                ) VALUES (%s, 'automatic', %s, %s, %s, 'pending', %s, %s, now(), now())
                """,
                (case_id, notification.receiver, notification.group_key, GROUPING_RULE, priority, title),
            )
            await self._event(
                connection,
                case_id,
                "case_opened",
                SOURCE,
                {"receiver": notification.receiver, "group_key": notification.group_key},
            )
        else:
            case_id = str(case["id"])

        attached_novel = False
        for signal_id, generation, _, disposition, _ in touched:
            if disposition == SignalDisposition.STALE:
                continue
            insert_cursor = await connection.execute(
                """
                INSERT INTO case_signals (case_id, signal_id, signal_generation)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING signal_id
                """,
                (case_id, signal_id, generation),
            )
            attached_novel = attached_novel or await insert_cursor.fetchone() is not None

        if not any_firing:
            await connection.execute(
                """
                UPDATE cases SET state = CASE WHEN state = 'pending' THEN 'investigated' ELSE state END,
                    outcome = CASE WHEN state = 'pending' THEN 'no_action' ELSE outcome END,
                    summary = CASE
                        WHEN state = 'pending' THEN 'Grafana resolved the group before investigation'
                        ELSE summary
                    END,
                    investigated_at = CASE WHEN state = 'pending' THEN now() ELSE investigated_at END,
                    updated_at = now()
                WHERE id = %s
                """,
                (case_id,),
            )
            await connection.execute(
                """
                UPDATE agent_turns SET state = 'cancelled', completed_at = now(),
                    error = 'Grafana resolved the group before launch'
                WHERE session_id IN (SELECT id FROM agent_sessions WHERE case_id = %s)
                  AND state = 'queued' AND kind = 'automatic'
                """,
                (case_id,),
            )
            await self._event(connection, case_id, "grafana_group_resolved", SOURCE, {})
            return case_id, False

        should_queue = created or novel_firing or attached_novel
        if should_queue and not created:
            assert case is not None
            if case["state"] in (
                CaseState.WAITING_HUMAN.value,
                CaseState.INVESTIGATED.value,
                CaseState.FAILED.value,
            ):
                await connection.execute(
                    "UPDATE cases SET state = 'pending', outcome = NULL, updated_at = now() WHERE id = %s",
                    (case_id,),
                )
            else:
                await connection.execute("UPDATE cases SET updated_at = now() WHERE id = %s", (case_id,))
        else:
            await connection.execute("UPDATE cases SET updated_at = now() WHERE id = %s", (case_id,))
        return case_id, should_queue

    async def _enqueue_automatic_turn(
        self,
        connection: psycopg.AsyncConnection[dict[str, Any]],
        *,
        case_id: str,
        delivery_key: str,
    ) -> bool:
        cursor = await connection.execute(
            "SELECT id FROM agent_sessions WHERE case_id = %s AND state <> 'archived' FOR UPDATE",
            (case_id,),
        )
        session = await cursor.fetchone()
        if session is None:
            session_id = str(uuid.uuid4())
            await self._insert_session(connection, session_id=session_id, case_id=case_id)
        else:
            session_id = str(session["id"])
        queued_cursor = await connection.execute(
            "SELECT id FROM agent_turns WHERE session_id = %s AND state = 'queued'",
            (session_id,),
        )
        if await queued_cursor.fetchone() is not None:
            return False
        case_cursor = await connection.execute("SELECT priority FROM cases WHERE id = %s", (case_id,))
        case = await case_cursor.fetchone()
        assert case is not None
        turn_id = str(uuid.uuid4())
        await connection.execute(
            """
            INSERT INTO agent_turns (
                id, session_id, kind, state, priority, requested_by,
                client_request_id, prompt
            ) VALUES (%s, %s, 'automatic', 'queued', %s, %s, %s, %s)
            ON CONFLICT (requested_by, client_request_id) DO NOTHING
            """,
            (
                turn_id,
                session_id,
                case["priority"],
                AUTOMATIC_REQUESTER,
                f"grafana:{case_id}:{delivery_key}",
                "Investigate the current Grafana alert group using the attached case evidence.",
            ),
        )
        await self._event(connection, case_id, "automatic_turn_queued", SOURCE, {"turn_id": turn_id})
        return True

    async def _insert_session(
        self,
        connection: psycopg.AsyncConnection[dict[str, Any]],
        *,
        session_id: str,
        case_id: str,
    ) -> None:
        await connection.execute(
            """
            INSERT INTO agent_sessions (
                id, case_id, deterministic_name, state, repo_revision, skill_revision
            ) VALUES (%s, %s, %s, 'new', %s, %s)
            """,
            (session_id, case_id, f"ops-case-{case_id}", self._repo_revision, self._skill_revision),
        )

    async def _event(
        self,
        connection: psycopg.AsyncConnection[dict[str, Any]],
        case_id: str,
        event_type: str,
        actor: str,
        data: Mapping[str, object],
        *,
        turn_id: str | None = None,
    ) -> None:
        await connection.execute(
            "INSERT INTO case_events (case_id, turn_id, event_type, actor, data) VALUES (%s, %s, %s, %s, %s)",
            (case_id, turn_id, event_type, actor, Jsonb(data)),
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


def _poll_webhook(
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
            external_url="https://grafana.oa.dev/",
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
        group_labels={"alertname": alert_name, "cluster": cluster},
        title=f"{alert_name} · {cluster}" if cluster else alert_name,
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


def _case_title(touched: list[tuple[str, int, str, SignalDisposition, GrafanaAlert]]) -> str:
    alerts = [item[4] for item in touched if item[2] == "firing"]
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
