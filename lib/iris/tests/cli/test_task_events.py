# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

from iris.cli.task import TaskEventView, build_task_event_display_rows, build_task_events_sql
from iris.cluster.types import TaskAttempt


def test_build_task_events_sql_queries_all_attempts_in_time_order():
    sql = build_task_events_sql(TaskAttempt.from_wire("/alice/job/0"), limit=25)

    assert "WHERE task_id = '/alice/job/0'" in sql
    assert "attempt_id =" not in sql
    assert "ORDER BY ts ASC" in sql
    assert sql.endswith("LIMIT 25")


def test_build_task_events_sql_can_select_one_attempt():
    sql = build_task_events_sql(TaskAttempt.from_wire("/alice/job/0:3"), limit=10)

    assert "task_id = '/alice/job/0'" in sql
    assert "attempt_id = 3" in sql


def test_build_task_event_display_rows_preserves_timeline_fields():
    events = [
        TaskEventView(
            attempt_id=0,
            ts=datetime(2026, 7, 25, 21, 26, 47, tzinfo=UTC),
            event_type="Warning",
            reason="WorkloadEvictedDueToPreempted",
            message="Preempted due to ClusterQueue prioritization",
            source="k8s/kueue",
            count=1,
        ),
        TaskEventView(
            attempt_id=0,
            ts=datetime(2026, 7, 25, 21, 27, 25, tzinfo=UTC),
            event_type="Normal",
            reason="TaskRetryScheduled",
            message="Backend reported WORKER_FAILED; controller returned the task to PENDING.",
            source="iris/controller",
            count=1,
        ),
    ]

    assert build_task_event_display_rows(events) == [
        [
            "2026-07-25 21:26:47",
            0,
            "Warning",
            "k8s/kueue",
            "WorkloadEvictedDueToPreempted",
            1,
            "Preempted due to ClusterQueue prioritization",
        ],
        [
            "2026-07-25 21:27:25",
            0,
            "Normal",
            "iris/controller",
            "TaskRetryScheduled",
            1,
            "Backend reported WORKER_FAILED; controller returned the task to PENDING.",
        ],
    ]
