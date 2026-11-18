# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.belay import sync
from experiments.belay.submit import tasks_started

GATED = """Job: /loom/belay-X0229v2-seed0-coord/grug-train-belay-X0229v2-seed0
State: running  exit=0  failures=0  preemptions=0
Tasks: 0/1 completed  building=1

  TASK  STATE     EXIT    DURATION    PEAK MEM    CUR MEM    DIAGNOSTIC
     0  building  -       -           -           -          SchedulingGated: Scheduling is blocked due to non-empty
"""


def test_a_run_whose_task_waits_for_a_node_has_not_started() -> None:
    assert not tasks_started(GATED)
    assert tasks_started(GATED.replace("building=1", "running=1"))
    assert tasks_started(GATED.replace("0/1 completed  building=1", "1/1 completed"))


def test_describe_output_without_a_task_summary_is_an_error() -> None:
    with pytest.raises(ValueError):
        tasks_started("Job: /loom/x\nState: running\n")


def test_the_light_sync_reports_only_runs_the_page_shows_live_that_iris_shows_ended(monkeypatch) -> None:
    page_runs = [{"name": "belay-X0001-seed0", "state": "running"}, {"name": "belay-X0001-control", "state": "running"}]
    queue = {"runs": page_runs, "queued": [{"id": 7}], "sessions": [], "settings": {"dispatch_enabled": True}}
    posted = []
    monkeypatch.setattr(sync, "call", lambda path, body=None: posted.append(body) if body else queue)
    monkeypatch.setattr(
        sync,
        "iris_runs",
        lambda: {
            "belay-X0001-seed0": sync.IrisRun("/loom/belay-X0001-seed0-coord", None, "succeeded", "t"),
            "belay-X0001-control": sync.IrisRun("/loom/belay-X0001-control-coord", None, "running", "t"),
            "belay-X0002-seed0": sync.IrisRun("/loom/belay-X0002-seed0-coord", None, "failed", "t"),
        },
    )
    status = sync.light_sync(reap=False)
    assert [r["name"] for r in posted[0]["runs"]] == [
        "belay-X0001-seed0"
    ]  # not the live control, nor a run the page has closed
    assert status == {
        "reaped": [],
        "ended": ["belay-X0001-seed0"],
        "queued_event_ids": [7],
        "session_active": False,
        "dispatch_enabled": True,
    }


def test_the_light_sync_reaps_only_when_asked(monkeypatch) -> None:
    queue = {"runs": [], "queued": [{"id": 3}], "sessions": [], "settings": {"dispatch_enabled": True}}
    calls = []

    def call(path, body=None):
        calls.append(path)
        return {"lost": ["loom-1118-223158"]} if path == "agent/sessions/reap" else queue

    monkeypatch.setattr(sync, "call", call)
    assert sync.light_sync(reap=False)["reaped"] == [] and "agent/sessions/reap" not in calls
    assert sync.light_sync(reap=True)["reaped"] == ["loom-1118-223158"]
