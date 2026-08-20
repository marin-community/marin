# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from finelog.rpc import logging_pb2
from iris.cluster.stats.tables import TASK_EVENT_NAMESPACE, TASK_EVENT_STORAGE_POLICY, TaskEventRow
from iris.resources.log import LogLevel, LogMatchScope, LogQuery, TaskEventCursor, TaskEventQuery
from iris.rpc.log_reader import FinelogLogReader
from rigging.timing import Timestamp


def test_finelog_reader_returns_native_filtered_log_entries(log_client, log_service) -> None:
    selected = logging_pb2.LogEntry(source="stderr", data="selected", level=logging_pb2.LOG_LEVEL_ERROR)
    selected.timestamp.epoch_ms = 2_000
    excluded = logging_pb2.LogEntry(source="stdout", data="excluded", level=logging_pb2.LOG_LEVEL_INFO)
    excluded.timestamp.epoch_ms = 3_000
    log_service.push_logs(logging_pb2.PushLogsRequest(key="/owner/job/0:1", entries=[selected]))
    log_service.push_logs(logging_pb2.PushLogsRequest(key="/owner/job/0:10", entries=[excluded]))

    entries, cursor = FinelogLogReader(log_client).fetch_logs(
        source="/owner/job/0:1",
        match_scope=LogMatchScope.EXACT,
        query=LogQuery(after=Timestamp.from_ms(1_000), minimum_level=LogLevel.WARNING),
    )

    assert [(entry.data, entry.level, entry.attempt_id, entry.timestamp) for entry in entries] == [
        ("selected", LogLevel.ERROR, 1, Timestamp.from_ms(2_000))
    ]
    assert cursor == entries[0].sequence


def test_finelog_reader_pages_task_events_with_native_keyset_cursor(log_client) -> None:
    table = log_client.get_table(TASK_EVENT_NAMESPACE, TaskEventRow, storage_policy=TASK_EVENT_STORAGE_POLICY)
    task_id = "/owner/o'clock/0"
    attempt_uid = "uid'current"
    middle = datetime(2026, 1, 2, 3, 4, 5)
    earlier = datetime(2026, 1, 2, 3, 4, 4)
    table.write(
        [
            TaskEventRow(task_id, 0, attempt_uid, middle, "Warning", "z-last", "z", "controller", 1),
            TaskEventRow(task_id, 0, attempt_uid, middle, "Warning", "y-middle", "y", "controller", 1),
            TaskEventRow(task_id, 0, attempt_uid, earlier, "Normal", "x-first", "x", "controller", 1),
            TaskEventRow(task_id, 0, "other-uid", middle, "Warning", "excluded-uid", "", "controller", 1),
            TaskEventRow("/owner/other/0", 0, attempt_uid, middle, "Warning", "excluded-task", "", "controller", 1),
        ]
    )
    table.flush()
    reader = FinelogLogReader(log_client)
    query = TaskEventQuery(task_id=task_id, attempt_uids=(attempt_uid,), limit=10)

    all_events = reader.task_events(query)
    first_page = reader.task_events(TaskEventQuery(task_id=task_id, attempt_uids=(attempt_uid,), limit=2))
    cursor = TaskEventCursor(first_page[-1].occurred_at, first_page[-1].key)
    second_page = reader.task_events(
        TaskEventQuery(task_id=task_id, attempt_uids=(attempt_uid,), before=cursor, limit=2)
    )

    assert [event.reason for event in all_events] == ["z-last", "y-middle", "x-first"]
    assert first_page + second_page == all_events
    assert (
        reader.task_events(
            TaskEventQuery(
                task_id=task_id,
                attempt_uids=(attempt_uid,),
                after=Timestamp.from_seconds(earlier.timestamp()),
                limit=10,
            )
        )
        == all_events[:2]
    )
