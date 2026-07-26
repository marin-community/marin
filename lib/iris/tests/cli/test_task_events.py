# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

from iris.cli.task import build_task_events_sql, render_task_events_text
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


def test_render_task_events_text_shows_backend_and_controller_actions():
    text = render_task_events_text(
        "/alice/job/0",
        [
            {
                "attempt_id": 0,
                "ts": datetime(2026, 7, 25, 21, 26, 47, tzinfo=UTC),
                "type": "Warning",
                "reason": "WorkloadEvictedDueToPreempted",
                "message": "Preempted due to ClusterQueue prioritization",
                "source": "k8s/kueue",
                "count": 1,
            },
            {
                "attempt_id": 0,
                "ts": datetime(2026, 7, 25, 21, 27, 25, tzinfo=UTC),
                "type": "Normal",
                "reason": "TaskRetryScheduled",
                "message": "Backend reported WORKER_FAILED; controller returned the task to PENDING.",
                "source": "iris/controller",
                "count": 1,
            },
        ],
    )

    assert "Task: /alice/job/0" in text
    assert "2026-07-25 21:26:47" in text
    assert "WorkloadEvictedDueToPreempted" in text
    assert "TaskRetryScheduled" in text
    assert "k8s/kueue" in text
    assert "iris/controller" in text
