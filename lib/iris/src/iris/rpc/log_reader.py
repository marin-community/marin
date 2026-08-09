# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native read boundary over Finelog logs and task activity."""

from dataclasses import dataclass
from datetime import UTC, datetime

from connectrpc.errors import ConnectError
from finelog.client import LogClient
from finelog.errors import StatsError
from finelog.rpc import logging_pb2
from rigging.timing import Timestamp

from iris.cluster.stats.tables import TASK_EVENT_NAMESPACE
from iris.resources.log import (
    LogEntry,
    LogLevel,
    LogMatchScope,
    LogQuery,
    LogReadError,
    TaskEvent,
    TaskEventKey,
    TaskEventQuery,
)


@dataclass(frozen=True, slots=True)
class FinelogLogReader:
    """Adapt a Finelog client to native Iris log records."""

    client: LogClient

    def fetch_logs(
        self,
        *,
        source: str,
        match_scope: LogMatchScope,
        query: LogQuery,
    ) -> tuple[tuple[LogEntry, ...], int]:
        minimum_level = "" if query.minimum_level is LogLevel.UNKNOWN else query.minimum_level.name
        request = logging_pb2.FetchLogsRequest(
            source=source,
            match_scope=int(match_scope),
            cursor=query.cursor,
            max_lines=query.max_lines,
            substring=query.substring,
            min_level=minimum_level,
            tail=query.tail,
        )
        if query.after is not None:
            request.since_ms = query.after.epoch_ms()
        try:
            response = self.client.fetch_logs(request)
        except (ConnectError, ConnectionError, OSError, RuntimeError) as error:
            raise LogReadError(str(error)) from error
        return tuple(_log_entry(entry) for entry in response.entries), response.cursor

    def task_events(self, query: TaskEventQuery) -> tuple[TaskEvent, ...]:
        try:
            rows = self.client.query(_task_event_sql(query), max_rows=query.limit).to_pylist()
        except (ConnectError, ConnectionError, OSError, RuntimeError, StatsError) as error:
            raise LogReadError(str(error)) from error
        return tuple(_task_event(row) for row in rows)


def _log_entry(entry: logging_pb2.LogEntry) -> LogEntry:
    return LogEntry(
        timestamp=Timestamp.from_ms(entry.timestamp.epoch_ms) if entry.HasField("timestamp") else None,
        source=entry.source,
        data=entry.data,
        attempt_id=entry.attempt_id,
        level=LogLevel(entry.level),
        key=entry.key,
        sequence=entry.seq,
    )


def _task_event(row: dict[str, object]) -> TaskEvent:
    attempt_id = row["attempt_id"]
    attempt_uid = row["attempt_uid"]
    occurred_at = row["ts"]
    event_type = row["type"]
    reason = row["reason"]
    message = row["message"]
    source = row["source"]
    count = row["count"]
    if type(attempt_id) is not int or not isinstance(attempt_uid, str):
        raise ValueError("finelog task event has invalid Attempt identity")
    if not isinstance(occurred_at, datetime):
        raise ValueError("finelog task event has invalid timestamp")
    if not all(isinstance(value, str) for value in (event_type, reason, message, source)) or type(count) is not int:
        raise ValueError("finelog task event has invalid typed fields")
    normalized = occurred_at.replace(tzinfo=UTC) if occurred_at.tzinfo is None else occurred_at.astimezone(UTC)
    return TaskEvent(
        attempt_id=attempt_id,
        attempt_uid=attempt_uid,
        occurred_at=Timestamp.from_seconds(normalized.timestamp()),
        event_type=event_type,
        reason=reason,
        message=message,
        source=source,
        count=count,
    )


def _task_event_sql(query: TaskEventQuery) -> str:
    task_literal = _text_literal(query.task_id)
    uid_literals = ", ".join(_text_literal(attempt_uid) for attempt_uid in query.attempt_uids)
    predicates = [f"task_id = {task_literal}", f"attempt_uid IN ({uid_literals})"]
    if query.after is not None:
        predicates.append(f"ts > {_timestamp_literal(query.after)}")
    if query.before is not None:
        before_timestamp = _timestamp_literal(query.before.occurred_at)
        before_predicate = f"ts < {before_timestamp}"
        if query.before.key is not None:
            equal_time = _key_before(query.before.key)
            before_predicate = f"(ts < {before_timestamp} OR (ts = {before_timestamp} AND ({equal_time})))"
        predicates.append(before_predicate)
    where = " AND ".join(predicates)
    return (
        "SELECT attempt_id, attempt_uid, ts, type, reason, message, source, count "
        f'FROM "{TASK_EVENT_NAMESPACE}" WHERE {where} '
        "ORDER BY ts DESC, attempt_id DESC, attempt_uid DESC, type DESC, reason DESC, "
        f"message DESC, source DESC, count DESC LIMIT {query.limit}"
    )


def _key_before(key: TaskEventKey) -> str:
    columns = ("attempt_id", "attempt_uid", "type", "reason", "message", "source", "count")
    literals = (
        str(key.attempt_id),
        _text_literal(key.attempt_uid),
        _text_literal(key.event_type),
        _text_literal(key.reason),
        _text_literal(key.message),
        _text_literal(key.source),
        str(key.count),
    )
    terms = []
    for index, (column, literal) in enumerate(zip(columns, literals, strict=True)):
        equal_prefix = " AND ".join(
            f"{prefix_column} = {prefix_literal}"
            for prefix_column, prefix_literal in zip(columns[:index], literals[:index], strict=True)
        )
        comparison = f"{column} < {literal}"
        terms.append(f"({equal_prefix} AND {comparison})" if equal_prefix else comparison)
    return " OR ".join(terms)


def _timestamp_literal(value: Timestamp) -> str:
    return f"to_timestamp({value.epoch_ms()} / 1000.0)"


def _text_literal(value: str) -> str:
    return f"'{value.replace("'", "''")}'"
