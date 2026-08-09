# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog boundary adapters for native resource log records."""

from finelog.client import LogClient
from finelog.rpc import logging_pb2
from rigging.timing import Timestamp

from iris.resources.log import LogEntry, LogLevel, LogQuery


def fetch_log_entries(
    client: LogClient,
    *,
    source: str,
    match_scope: str,
    query: LogQuery,
) -> tuple[tuple[LogEntry, ...], int]:
    minimum_level = "" if query.minimum_level is LogLevel.UNKNOWN else query.minimum_level.name
    request = logging_pb2.FetchLogsRequest(
        source=source,
        match_scope=match_scope,
        cursor=query.cursor,
        max_lines=query.max_lines,
        substring=query.substring,
        min_level=minimum_level,
        tail=query.tail,
    )
    if query.after is not None:
        request.since_ms = query.after.epoch_ms()
    response = client.fetch_logs(request)
    return tuple(_log_entry(entry) for entry in response.entries), response.cursor


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
