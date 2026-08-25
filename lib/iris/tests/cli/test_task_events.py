# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from finelog.client import FlushResult
from iris.cli.task import TaskEventView, build_task_events_sql
from iris.cluster.stats.tables import (
    TASK_EVENT_NAMESPACE,
    TASK_EVENT_STORAGE_POLICY,
    TaskEventRow,
)
from iris.cluster.types import TaskAttempt
from rigging.timing import Timestamp


def test_task_event_query_returns_the_newest_window_from_the_current_incarnation(log_client):
    table = log_client.get_table(
        TASK_EVENT_NAMESPACE,
        TaskEventRow,
        storage_policy=TASK_EVENT_STORAGE_POLICY,
    )
    now = Timestamp.now()
    rows = [
        TaskEventRow(
            task_id="/alice/job/0",
            attempt_id=0,
            attempt_uid="old-incarnation",
            ts=now.add_ms(-4_000).as_naive_utc(),
            type="Warning",
            reason="OldRunFailed",
            message="old",
            source="iris/controller",
            count=1,
        ),
        *[
            TaskEventRow(
                task_id="/alice/job/0",
                attempt_id=attempt_id,
                attempt_uid=f"current-{attempt_id}",
                ts=now.add_ms(offset).as_naive_utc(),
                type="Normal",
                reason=reason,
                message=reason,
                source="iris/controller",
                count=1,
            )
            for attempt_id, offset, reason in [
                (0, -3_000, "First"),
                (1, -2_000, "Second"),
                (1, -1_000, "Third"),
            ]
        ],
    ]
    table.write(rows)
    assert table.flush(timeout=5) == FlushResult.SUCCEEDED

    result = log_client.query(
        build_task_events_sql(
            TaskAttempt.from_wire("/alice/job/0"),
            ["current-0", "current-1"],
            limit=2,
        )
    )
    events = [TaskEventView.from_row(row) for row in result.to_pylist()]

    assert [(event.attempt_id, event.reason) for event in events] == [
        (1, "Second"),
        (1, "Third"),
    ]
