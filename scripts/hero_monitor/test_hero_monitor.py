import json
import math

import pytest

from scripts.hero_monitor import hero_monitor


JOB = "/root/train"
TASK_0 = f"{JOB}/0"
TASK_1 = f"{JOB}/1"
NOW = 2_000_000_000.0


def execution(
    job=JOB,
    attempts=None,
    selected_task=None,
    started_at=NOW,
    cluster="cw-test",
):
    attempts = attempts or {f"{job}/0": 0}
    selected_task = selected_task or next(iter(attempts))
    return hero_monitor.ExecutionSnapshot(
        cluster=cluster,
        job_id=job,
        selected_execution_uid=f"iris:{selected_task}:attempt:{attempts[selected_task]}",
        execution_started_at=started_at,
        attempts=attempts,
    )


def task_state(age=0, pending=0, assigned=0, building=0, running=1):
    return hero_monitor.TaskStateSnapshot(
        root_job_id=JOB,
        age_seconds=age,
        pending=pending,
        assigned=assigned,
        building=building,
        running=running,
    )


def monitor_state(job_id=None, attempts=None, retry_keys=None):
    return {
        "state_version": hero_monitor.STATE_VERSION,
        "last_fired": {},
        "last_digest": 0,
        "last_job_id": job_id,
        "seen_job_ids": [job_id] if job_id else [],
        "attempts_by_job": (
            {job_id: attempts} if job_id and attempts is not None else {}
        ),
        "retry_event_keys_by_job": (
            {job_id: retry_keys} if job_id and retry_keys is not None else {}
        ),
        "notified_attempts_by_job": (
            {job_id: dict(attempts or {})}
            if job_id and retry_keys is not None
            else {}
        ),
        "last_eval_paloma": None,
        "restart_count": 0,
        "query_failures": 0,
        "run_finished": False,
    }


def fresh_series(phase=1, step=100, progress_time=NOW):
    stamp = NOW * 1000
    return {
        "phase": [(stamp, phase)],
        "step": [(stamp, step)],
        "progress_time_seconds": [(stamp, progress_time)],
    }


@pytest.fixture
def notifications(monkeypatch):
    delivered = []

    def fake_pushover(title, message, priority=0):
        delivered.append((title, message, priority))
        return True

    monkeypatch.setattr(hero_monitor, "pushover", fake_pushover)
    monkeypatch.setattr(hero_monitor.time, "time", lambda: NOW)
    return delivered


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (f"iris:{TASK_0}:attempt:3", (TASK_0, 3)),
        (f"other:{TASK_0}:attempt:3", None),
        (f"iris:{TASK_0}:attempt:not-an-int", None),
        (f"iris:{TASK_0}:attempt:-1", None),
    ],
)
def test_execution_uid_wire_format(value, expected):
    assert hero_monitor.parse_execution_uid(value) == expected


def test_execution_snapshot_selects_latest_attempt(monkeypatch):
    selected_uid = f"iris:{TASK_0}:attempt:1"
    rows = [
        {
            "origin_cluster": "cw-test",
            "job_id": JOB,
            "execution_uid": f"iris:{TASK_0}:attempt:0",
            "first_ts": 10,
            "last_ts": 100,
            "selected_ts": 120,
            "selected_seq": 4,
            "selected_execution_uid": selected_uid,
        },
        {
            "origin_cluster": "cw-test",
            "job_id": JOB,
            "execution_uid": f"iris:{TASK_1}:attempt:0",
            "first_ts": 11,
            "last_ts": 110,
            "selected_ts": 120,
            "selected_seq": 4,
            "selected_execution_uid": selected_uid,
        },
        {
            "origin_cluster": "cw-test",
            "job_id": JOB,
            "execution_uid": selected_uid,
            "first_ts": 20,
            "last_ts": 120,
            "selected_ts": 120,
            "selected_seq": 4,
            "selected_execution_uid": selected_uid,
        },
    ]
    monkeypatch.setattr(hero_monitor, "finelog_query", lambda _: rows)

    snapshot = hero_monitor.fetch_execution_snapshot()

    assert snapshot == hero_monitor.ExecutionSnapshot(
        cluster="cw-test",
        job_id=JOB,
        selected_execution_uid=selected_uid,
        execution_started_at=0.02,
        attempts={TASK_0: 1, TASK_1: 0},
    )


def test_retry_event_rows_preserve_controller_identity(monkeypatch):
    rows = [
        {
            "task_id": TASK_0,
            "attempt_id": 2,
            "attempt_uid": "deadbeef",
            "ts": NOW * 1000,
        }
    ]
    monkeypatch.setattr(hero_monitor, "finelog_query", lambda _: rows)

    assert hero_monitor.fetch_retry_events(execution()) == [
        hero_monitor.RetryEvent(
            key="uid:deadbeef",
            task_id=TASK_0,
            attempt_id=2,
            observed_at=NOW,
        )
    ]


def test_worker_heartbeat_rotation_does_not_notify(notifications):
    state = monitor_state(JOB, {TASK_0: 0, TASK_1: 0}, [])
    current = execution(
        attempts={TASK_1: 0, TASK_0: 0},
        selected_task=TASK_1,
    )

    hero_monitor.check_restart(state, current, [])
    hero_monitor.check_restart(state, current, [])

    assert notifications == []
    assert state["restart_count"] == 0


def test_fast_retry_cycle_notifies_once(notifications):
    state = monitor_state(JOB, {TASK_0: 0}, [])
    current = execution(attempts={TASK_0: 1})
    event = hero_monitor.RetryEvent(
        key="uid:fast-cycle",
        task_id=TASK_0,
        attempt_id=0,
        observed_at=NOW - 30,
    )

    hero_monitor.check_restart(state, current, [event])
    hero_monitor.check_restart(state, current, [event])

    assert [title for title, _, _ in notifications] == [
        "hero task retry scheduled"
    ]
    assert notifications[0][2] == 0
    assert state["restart_count"] == 1


def test_phase_fallback_and_delayed_event_notify_once(notifications):
    state = monitor_state(JOB, {TASK_0: 0}, [])
    current = execution(attempts={TASK_0: 1})
    event = hero_monitor.RetryEvent(
        key="uid:delayed",
        task_id=TASK_0,
        attempt_id=0,
        observed_at=NOW,
    )

    hero_monitor.check_restart(state, current, [])
    hero_monitor.check_restart(state, current, [event])

    assert len(notifications) == 1
    assert "phase attempt increment" in notifications[0][1]
    assert state["restart_count"] == 1


def test_failed_delivery_leaves_retry_event_unconsumed(monkeypatch):
    attempted = []
    outcomes = iter((False, True))

    def fake_pushover(title, message, priority=0):
        attempted.append((title, message, priority))
        return next(outcomes)

    monkeypatch.setattr(hero_monitor, "pushover", fake_pushover)
    monkeypatch.setattr(hero_monitor.time, "time", lambda: NOW)
    state = monitor_state(JOB, {TASK_0: 0}, [])
    current = execution(attempts={TASK_0: 0})
    event = hero_monitor.RetryEvent(
        key="uid:delivery-retry",
        task_id=TASK_0,
        attempt_id=0,
        observed_at=NOW,
    )

    hero_monitor.check_restart(state, current, [event])
    assert state["retry_event_keys_by_job"][JOB] == []
    hero_monitor.check_restart(state, current, [event])

    assert len(attempted) == 2
    assert state["retry_event_keys_by_job"][JOB] == [event.key]
    assert state["restart_count"] == 1


def test_job_handoff_bounce_notifies_once(notifications):
    old_job = "/root/train-old"
    new_job = "/root/train-new"
    old = execution(job=old_job, attempts={f"{old_job}/0": 0})
    new = execution(job=new_job, attempts={f"{new_job}/0": 0})
    state = monitor_state(old_job, old.attempts, [])

    hero_monitor.check_restart(state, new, [])
    hero_monitor.check_restart(state, old, [])
    hero_monitor.check_restart(state, new, [])

    assert [title for title, _, _ in notifications] == ["hero new attempt"]
    assert state["restart_count"] == 1


def test_pending_tasks_are_not_reported_down(notifications):
    state = monitor_state()

    hero_monitor.check(
        fresh_series(phase=0, step=0, progress_time=0),
        task_state(pending=4, running=0),
        state,
        execution(started_at=NOW - 60),
    )

    assert notifications == []


@pytest.mark.parametrize(
    ("series", "started_at", "expected_title"),
    [
        (
            fresh_series(phase=0, step=0, progress_time=0),
            NOW - 46 * 60,
            f"HERO INIT STALLED {hero_monitor.CONFIG['RUN_ID']}",
        ),
        (
            fresh_series(progress_time=NOW - 16 * 60),
            NOW - 60 * 60,
            f"HERO STALLED {hero_monitor.CONFIG['RUN_ID']}",
        ),
    ],
)
def test_stall_thresholds_match_training_phase(
    notifications,
    series,
    started_at,
    expected_title,
):
    hero_monitor.check(
        series,
        task_state(),
        monitor_state(),
        execution(started_at=started_at),
    )

    assert [title for title, _, _ in notifications] == [expected_title]
    assert notifications[0][2] == 2


def test_finished_phase_suppresses_telemetry_outage(notifications):
    state = monitor_state()

    hero_monitor.check(
        fresh_series(phase=2),
        task_state(),
        state,
        execution(),
    )
    hero_monitor.check({}, None, state)

    assert notifications == []
    assert state["run_finished"] is True


def test_raw_loss_spike_notifies(notifications):
    stats = hero_monitor.AlertStats(
        baseline_samples=20,
        baseline_loss=1.0,
        baseline_stddev=0.01,
        baseline_floor=0.9,
        recent_samples=5,
        recent_loss=1.2,
        recent_floor=1.15,
        recent_peak=1.25,
        recent_skips=0,
    )

    hero_monitor.check(
        fresh_series(),
        task_state(),
        monitor_state(),
        execution(),
        stats,
    )

    assert [title for title, _, _ in notifications] == [
        f"hero LOSS SPIKE {hero_monitor.CONFIG['RUN_ID']}"
    ]
    assert notifications[0][2] == 1


def test_nonfinite_loss_pages(notifications):
    stats = hero_monitor.AlertStats(
        baseline_samples=20,
        baseline_loss=1.0,
        baseline_stddev=0.01,
        baseline_floor=0.9,
        recent_samples=5,
        recent_loss=math.nan,
        recent_floor=1.0,
        recent_peak=math.nan,
        recent_skips=0,
    )

    hero_monitor.check(
        fresh_series(),
        task_state(),
        monitor_state(),
        execution(),
        stats,
    )

    assert [title for title, _, _ in notifications] == [
        f"HERO NON-FINITE LOSS {hero_monitor.CONFIG['RUN_ID']}"
    ]
    assert notifications[0][2] == 2


@pytest.mark.parametrize(
    ("series", "expected_priority"),
    [
        (fresh_series(), -1),
        ({}, 0),
    ],
)
def test_digest_priority_reflects_telemetry_availability(
    notifications,
    series,
    expected_priority,
):
    hero_monitor.digest(series, task_state(), monitor_state())

    assert len(notifications) == 1
    assert notifications[0][0] == "hero status"
    assert notifications[0][2] == expected_priority


def test_legacy_state_discards_false_worker_restart_history(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "last_fired": {
                    "attempt_iris:/root/train/73:attempt:0": 123,
                    "stall_warn": 456,
                },
                "last_execution_uid": "iris:/root/train/73:attempt:0",
                "restart_count": 4,
            }
        )
    )
    monkeypatch.setitem(hero_monitor.CONFIG, "STATE_FILE", str(state_file))

    state = hero_monitor.load_state()

    assert state["state_version"] == hero_monitor.STATE_VERSION
    assert state["last_job_id"] is None
    assert state["restart_count"] == 0
    assert state["last_fired"] == {"stall_warn": 456}
    assert "last_execution_uid" not in state
