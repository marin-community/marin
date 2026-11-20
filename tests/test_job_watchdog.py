# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the job watchdog: registry parsing, state transitions, alert
dedup, sink delivery contracts, and the probe output parsers."""

import dataclasses
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from scripts.ops import job_watchdog as jw


def make_registry(entries: str = "") -> jw.Registry:
    raw = {
        "coordinator": "coordinator-session",
        "defaults": {
            "stall_threshold_minutes": 60,
            "gated_threshold_minutes": 30,
            "cooldown_minutes": 120,
        },
        "entries": [
            {
                "id": "demo",
                "owner": "owner-session",
                "description": "demo entry",
                "probe": {"type": "storage_objects", "url": "s3://bucket/prefix"},
            },
            *(yaml.safe_load(entries)["entries"] if entries else []),
        ],
    }
    return jw.parse_registry(raw)


def test_registry_defaults_propagate_and_entries_override() -> None:
    registry = make_registry()
    entry = registry.entries[0]
    assert entry.stall_threshold == timedelta(minutes=60)
    assert entry.gated_threshold == timedelta(minutes=30)
    assert entry.cooldown == timedelta(minutes=120)
    assert entry.preempt_alert_count == 2
    assert registry.entries[0].owner == "owner-session"


def test_registry_rejects_unknown_probe_type_and_duplicate_ids() -> None:
    with pytest.raises(jw.RegistryError, match=r"probe\.type"):
        jw.parse_registry(
            {
                "coordinator": "c",
                "entries": [{"id": "a", "owner": "o", "probe": {"type": "telemetry", "url": "x"}}],
            }
        )
    with pytest.raises(jw.RegistryError, match="duplicate"):
        jw.parse_registry(
            {
                "coordinator": "c",
                "entries": [
                    {"id": "a", "owner": "o", "probe": {"type": "storage_objects", "url": "s3://x/y"}},
                    {"id": "a", "owner": "o", "probe": {"type": "storage_objects", "url": "s3://x/z"}},
                ],
            }
        )


def test_registry_log_regex_pattern_must_compile() -> None:
    with pytest.raises(jw.RegistryError, match="pattern does not compile"):
        jw.parse_registry(
            {
                "coordinator": "c",
                "entries": [
                    {
                        "id": "a",
                        "owner": "o",
                        "probe": {"type": "log_regex", "cluster": "cl", "job": "/u/j", "pattern": "("},
                    }
                ],
            }
        )


def stall_entry(tmp_path: Path) -> jw.RegistryEntry:
    registry = make_registry()
    return registry.entries[0]


def observation(state: jw.EntryState = jw.EntryState.RUNNING, signature: str = "sig1") -> jw.Observation:
    return jw.Observation(state=state, progress="1 objects", signature=signature, detail="detail line")


def run_one_tick(registry, state, runner, sink, now):
    return jw.run_tick(registry, state, runner, sink, now)


def test_stall_requires_baseline_and_threshold(tmp_path: Path) -> None:
    entry = stall_entry(tmp_path)
    now = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)

    first = jw.effective_state(entry, jw.EntryRecord(), observation(), now)
    assert first == jw.EntryState.RUNNING

    record = jw.EntryRecord(progress_changed_at=now - timedelta(minutes=61))
    assert jw.effective_state(entry, record, observation(), now) == jw.EntryState.NO_PROGRESS

    record = jw.EntryRecord(progress_changed_at=now - timedelta(minutes=59))
    assert jw.effective_state(entry, record, observation(), now) == jw.EntryState.RUNNING


def test_terminal_states_pass_through_effective_state(tmp_path: Path) -> None:
    entry = stall_entry(tmp_path)
    record = jw.EntryRecord(progress_changed_at=datetime(2026, 11, 17, tzinfo=UTC))
    for state in (jw.EntryState.FAILED, jw.EntryState.KILLED, jw.EntryState.SUCCEEDED, jw.EntryState.GATED):
        assert jw.effective_state(entry, record, observation(state), datetime.now(UTC)) == state


def collect_alerts() -> tuple[list[jw.Alert], jw.FanOutSink]:
    alerts = []

    class RecordingSink:
        def deliver(self, alert: jw.Alert) -> None:
            alerts.append(alert)

    return alerts, jw.FanOutSink((RecordingSink(),))


def probe_runner_for(observations: list[jw.Observation]):
    calls = {"n": 0}

    def runner(entry, now) -> jw.Observation:
        index = min(calls["n"], len(observations) - 1)
        calls["n"] += 1
        return observations[index]

    return runner


def test_failed_then_recovered_then_failed_refires(tmp_path: Path) -> None:
    registry = make_registry()
    state = jw.WatchdogState()
    alerts, sink = collect_alerts()
    start = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
    observations = [
        observation(),
        observation(jw.EntryState.FAILED),
        observation(jw.EntryState.FAILED),
        observation(),
        observation(jw.EntryState.FAILED),
    ]
    for i, obs in enumerate(observations):
        run_one_tick(registry, state, probe_runner_for([obs]), sink, start + timedelta(minutes=5 * i))
    kinds = [alert.kind for alert in alerts]
    # Steady FAILED does not re-alert, but failing again after a restart is a new event.
    assert kinds == ["JOB_FAILED", "JOB_FAILED"]


def test_no_progress_refires_only_after_cooldown(tmp_path: Path) -> None:
    registry = make_registry()
    state = jw.WatchdogState()
    alerts, sink = collect_alerts()
    start = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
    # Tick 1: baseline. Ticks 2-3: same signature, past the 60 min stall threshold,
    # within the 120 min cooldown. Tick 4: cooldown elapsed.
    times = [start, start + timedelta(minutes=61), start + timedelta(minutes=100), start + timedelta(minutes=201)]
    stale = observation(signature="same")
    for now in times:
        run_one_tick(registry, state, probe_runner_for([stale]), sink, now)
    assert [alert.kind for alert in alerts] == ["NO_PROGRESS", "NO_PROGRESS"]
    assert alerts[0].key == "demo:NO_PROGRESS"


def test_gated_alert_requires_gated_threshold(tmp_path: Path) -> None:
    registry = make_registry()
    state = jw.WatchdogState()
    alerts, sink = collect_alerts()
    start = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
    gated = observation(jw.EntryState.GATED, signature="gated")
    # Tick 1 starts the gate; tick 2 is 10 minutes in (below the 30 minute threshold);
    # tick 3 is 45 minutes in.
    run_one_tick(registry, state, probe_runner_for([gated]), sink, start)
    run_one_tick(registry, state, probe_runner_for([gated]), sink, start + timedelta(minutes=10))
    assert alerts == []
    run_one_tick(registry, state, probe_runner_for([gated]), sink, start + timedelta(minutes=45))
    assert [alert.kind for alert in alerts] == ["GATED"]


def test_preempted_alert_fires_on_repeated_preemptions_and_resets(tmp_path: Path) -> None:
    registry = make_registry()
    state = jw.WatchdogState()
    alerts, sink = collect_alerts()
    start = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)

    def preempted(total: int) -> jw.Observation:
        return dataclasses.replace(observation(), preemptions=total)

    # One preemption is below the default threshold of 2.
    run_one_tick(registry, state, probe_runner_for([preempted(1)]), sink, start)
    assert alerts == []
    # The second preemption (cumulative counter 2) crosses the threshold.
    run_one_tick(registry, state, probe_runner_for([preempted(2)]), sink, start + timedelta(minutes=5))
    assert [alert.kind for alert in alerts] == ["JOB_PREEMPTED"]
    assert state.entries["demo"].preempted_since_alert == 0
    # A later batch of two more preemptions re-arms the condition; the state is unchanged
    # so the alert re-fires only once the cooldown passes.
    run_one_tick(registry, state, probe_runner_for([preempted(4)]), sink, start + timedelta(minutes=10))
    assert len(alerts) == 1
    run_one_tick(registry, state, probe_runner_for([preempted(4)]), sink, start + timedelta(minutes=140))
    assert [alert.kind for alert in alerts] == ["JOB_PREEMPTED", "JOB_PREEMPTED"]


def test_probe_error_alerts_and_recover_clears(tmp_path: Path) -> None:
    registry = make_registry()
    state = jw.WatchdogState()
    alerts, sink = collect_alerts()
    start = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)

    def failing_runner(entry, now):
        raise jw.ProbeError("connection refused")

    run_one_tick(registry, state, failing_runner, sink, start)
    assert [alert.kind for alert in alerts] == ["PROBE_ERROR"]
    assert state.entries["demo"].state == jw.EntryState.UNKNOWN

    run_one_tick(registry, state, probe_runner_for([observation()]), sink, start + timedelta(minutes=5))
    assert len(alerts) == 1


def test_tick_persists_state_history_dashboard_and_alert_log(tmp_path: Path) -> None:
    registry = make_registry()
    state_dir = tmp_path / "state"
    dashboard = tmp_path / "dashboard.html"
    alerts_path = tmp_path / "alerts.jsonl"
    _, sink = collect_alerts()
    start = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)

    watchdog_state = jw.WatchdogState()
    result = run_one_tick(registry, watchdog_state, probe_runner_for([observation(jw.EntryState.FAILED)]), sink, start)
    jw.save_state(state_dir / "state.json", watchdog_state)
    jw.append_history(state_dir / "history.jsonl", result.statuses)
    jw.write_dashboard(dashboard, result.statuses, result.ticked_at)
    for alert in result.fired:
        jw.JsonlAlertSink(alerts_path).deliver(alert)

    record = json.loads(alerts_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["key"] == "demo:JOB_FAILED"
    assert record["severity"] == "critical"
    assert record["entry"] == "demo"
    assert record["state"] == "FAILED"
    assert record["time_utc"] and record["time_local"]
    assert record["evidence"]["owner"] == "owner-session"
    assert record["evidence"]["detail"] == "detail line"

    state_json = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    assert state_json["entries"]["demo"]["state"] == "FAILED"
    assert state_json["alerts"]["demo:JOB_FAILED"]["count"] == 1

    history_row = json.loads((state_dir / "history.jsonl").read_text(encoding="utf-8"))
    assert history_row["id"] == "demo" and history_row["state"] == "FAILED"

    dashboard_html = dashboard.read_text(encoding="utf-8")
    assert "job-watchdog fleet state" in dashboard_html
    assert "FAILED" in dashboard_html


def test_state_roundtrip_preserves_rolling_fields(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = jw.WatchdogState()
    changed = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)
    state.entries["demo"] = jw.EntryRecord(
        state=jw.EntryState.NO_PROGRESS,
        progress_signature="sig",
        progress="3 objects",
        detail="d",
        progress_changed_at=changed,
        gated_since=None,
        preempted_seen=4,
        preempted_since_alert=1,
    )
    state.alerts["demo:NO_PROGRESS"] = jw.AlertRecord(last_state="NO_PROGRESS", last_sent_at=changed, count=2)
    jw.save_state(path, state)
    loaded = jw.load_state(path)
    record = loaded.entries["demo"]
    assert record.state == jw.EntryState.NO_PROGRESS
    assert record.progress_signature == "sig"
    assert record.progress_changed_at == changed
    assert record.preempted_seen == 4
    assert record.preempted_since_alert == 1
    assert loaded.alerts["demo:NO_PROGRESS"].count == 2
    assert loaded.alerts["demo:NO_PROGRESS"].last_sent_at == changed


def test_desktop_notification_sink_severity_gate(monkeypatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(jw.subprocess, "run", fake_run)
    sink = jw.DesktopNotificationSink()
    info = jw.Alert("e:SUCCEEDED", "SUCCEEDED", "info", "e", "SUCCEEDED", "m", {})
    warning = jw.Alert(
        "e:NO_PROGRESS", "NO_PROGRESS", "warning", "e", "NO_PROGRESS", "[job-watchdog]\nstate: X\nboom", {}
    )
    sink.deliver(info)
    assert calls == []
    sink.deliver(warning)
    assert len(calls) == 1
    assert "display notification" in calls[0][2]
    assert "boom" in calls[0][2]


def test_parse_iris_job_describe_real_output() -> None:
    sample = """Job: /benfeuer/qc-v1 (/benfeuer/qc-v1)
State: running  exit=0  failures=0  preemptions=2
Tasks: 4/8 completed  running=3  succeeded=4  failed=1

  TASK  STATE    EXIT    DURATION                   PEAK MEM    CUR MEM    DIAGNOSTIC
     0  running  -       1 minute and 4.05 seconds  -           -
"""
    view = jw.parse_iris_job_describe(sample)
    assert view.job_id == "/benfeuer/qc-v1"
    assert view.state == "running"
    assert view.preemptions == 2
    assert view.task_counts == {"running": 3, "succeeded": 4, "failed": 1, "completed": 4}


def test_parse_iris_job_list_rows_with_reason() -> None:
    sample = """JOB ID                                           STATE    SUBMITTED                         REASON
/romain/knotmeter-1  failed   2026-11-17T10:39:01+00:00  Cumulative failed task attempts exceeded max_task_failures: ...
/romain/knotmeter-2  running  2026-11-17T10:39:02+00:00

Showing the 2 most recent jobs.
"""
    views = jw.parse_iris_job_list(sample)
    assert [view.job_id for view in views] == ["/romain/knotmeter-1", "/romain/knotmeter-2"]
    assert views[0].state == "failed"
    assert "max_task_failures" in views[0].reason
    assert views[1].reason == ""


def test_kueue_gate_view_reads_quota_conditions(monkeypatch) -> None:
    workload = {
        "metadata": {"name": "iris-qc-v1-0-1"},
        "status": {
            "conditions": [
                {
                    "type": "QuotaReserved",
                    "status": "False",
                    "reason": "QuotaExceeded",
                    "message": "no quota left in iris-cq",
                },
            ]
        },
    }
    monkeypatch.setattr(
        jw,
        "k8s_cluster_settings",
        lambda cluster: jw.K8sClusterSettings(namespace="iris", kubeconfig="kc", context="ctx"),
    )
    monkeypatch.setattr(jw, "run_readonly", lambda command, timeout_seconds=180: json.dumps({"items": [workload]}))
    gated, message = jw.kueue_gate_view("cw-us-east-08a", "/benfeuer/qc-v1")
    assert gated
    assert "QuotaExceeded" in message and "no quota left" in message


def test_iris_lifecycle_maps_states_and_gate_reasons() -> None:
    running = jw.IrisJobView("/u/j", "running", 0, {"running": 2})
    state, detail = jw.iris_lifecycle_observation(running, "marin")
    assert state == jw.EntryState.RUNNING

    gated_reason = jw.IrisJobView("/u/j", "pending", 0, {}, reason="SchedulingGated: quota reserved false")
    state, detail = jw.iris_lifecycle_observation(gated_reason, "cw-us-east-08a", kueue=[])
    assert state == jw.EntryState.GATED
    assert "SchedulingGated" in detail

    failed = jw.IrisJobView("/u/j", "worker_failed", 0, {})
    assert jw.iris_lifecycle_observation(failed, "marin")[0] == jw.EntryState.FAILED

    unschedulable = jw.IrisJobView("/u/j", "unschedulable", 0, {})
    state, detail = jw.iris_lifecycle_observation(unschedulable, "cw-us-east-08a")
    assert state == jw.EntryState.GATED
    assert "unschedulable" in detail


def test_expected_replacement_downgrades_owner_initiated_deaths(tmp_path: Path) -> None:
    entry = dataclasses.replace(
        stall_entry(tmp_path),
        expected_replacement=jw.IrisJobRef(cluster="cw-us-east-02a", job="/benfeuer/job-v2"),
    )
    record = jw.EntryRecord()
    now = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)

    # Replacement alive: the death is a planned handover, recorded as info.
    alert = jw.evaluate_alert(
        entry, "JOB_KILLED", jw.EntryState.RUNNING, jw.EntryState.KILLED, record, jw.AlertRecord(), now, "running"
    )
    assert alert is not None and alert.severity == "info"
    assert "expected replacement /benfeuer/job-v2: running" in alert.message

    # Replacement missing: the death stays critical and says so.
    alert = jw.evaluate_alert(
        entry, "JOB_KILLED", jw.EntryState.RUNNING, jw.EntryState.KILLED, record, jw.AlertRecord(), now, None
    )
    assert alert is not None and alert.severity == "critical"
    assert "not in fleet list" in alert.message

    # The downgrade only applies to death alerts.
    alert = jw.evaluate_alert(
        entry, "NO_PROGRESS", jw.EntryState.RUNNING, jw.EntryState.NO_PROGRESS, record, jw.AlertRecord(), now, "running"
    )
    assert alert is not None and alert.severity == "warning"


def test_registry_parses_expected_replacement() -> None:
    registry = jw.parse_registry(
        {
            "coordinator": "c",
            "entries": [
                {
                    "id": "a",
                    "owner": "o",
                    "probe": {"type": "log_regex", "cluster": "cl", "job": "/u/v1"},
                    "expected_replacement": {"cluster": "cl", "job": "/u/v2"},
                }
            ],
        }
    )
    assert registry.entries[0].expected_replacement == jw.IrisJobRef(cluster="cl", job="/u/v2")
    with pytest.raises(jw.RegistryError, match="expected_replacement"):
        jw.parse_registry(
            {
                "coordinator": "c",
                "entries": [
                    {
                        "id": "a",
                        "owner": "o",
                        "probe": {"type": "log_regex", "cluster": "cl", "job": "/u/v1"},
                        "expected_replacement": "/u/v2",
                    }
                ],
            }
        )


def test_humanize_duration() -> None:
    assert jw.humanize_duration(timedelta(seconds=45)) == "45s"
    assert jw.humanize_duration(timedelta(minutes=5)) == "5m"
    assert jw.humanize_duration(timedelta(hours=3, minutes=4)) == "3h04m"
    assert jw.humanize_duration(timedelta(days=2, hours=5)) == "2d5h"


def test_fleet_snapshot_parses_lists_and_reduces_finelog_rows(monkeypatch) -> None:
    iris_output = (
        "JOB ID                     STATE     SUBMITTED                      REASON\n"
        "/benfeuer/job-a            running   2026-11-17T10:00:00+00:00\n"
        "/benfeuer/job-a/child      running   2026-11-17T10:00:01+00:00\n"
        "/benfeuer/job-b            pending   2026-11-17T10:00:02+00:00      SchedulingGated\n"
        "\nShowing the 3 most recent jobs.\n"
    )
    finelog_rows = "\n".join(
        json.dumps(row)
        for row in [
            {"key": "/benfeuer/job-a/0:0", "seq": 10, "epoch_ms": 1, "data": "older a", "cluster": "02a"},
            {"key": "/benfeuer/job-a/0:0", "seq": 12, "epoch_ms": 2, "data": "newest a", "cluster": "02a"},
            {"key": "/benfeuer/job-b/0:1", "seq": 7, "epoch_ms": 3, "data": "newest b", "cluster": "rno2a"},
            "I20261117 noise line, not json",
        ]
    )

    def fake_run_readonly(command, timeout_seconds=180):
        if "finelog" in command:
            return finelog_rows
        return iris_output

    monkeyed_registry = jw.Registry(
        coordinator="c",
        entries=(
            jw.RegistryEntry(
                id="a",
                owner="o",
                description="",
                probe=jw.LogRegexProbe(cluster="cw-us-east-02a", job="/benfeuer/job-a", pattern=""),
                iris=None,
                stall_threshold=timedelta(minutes=60),
                gated_threshold=timedelta(minutes=30),
                cooldown=timedelta(minutes=120),
                preempt_alert_count=2,
                expected_terminal=None,
            ),
        ),
        defaults={},
    )
    monkeypatch.setattr(jw, "run_readonly", fake_run_readonly)
    monkeypatch.setattr(jw, "kueue_workloads", lambda cluster: [{"metadata": {"name": "job-b-workload"}}])
    snapshot = jw.collect_fleet_snapshot(monkeyed_registry, datetime.now(UTC))
    assert snapshot.iris[("cw-us-east-02a", "/benfeuer/job-a")].state == "running"
    assert snapshot.iris[("cw-us-east-02a", "/benfeuer/job-b")].reason == "SchedulingGated"
    assert snapshot.logs["/benfeuer/job-a"].data == "newest a"
    assert snapshot.logs["/benfeuer/job-b"].data == "newest b"
    assert snapshot.kueue["cw-us-east-02a"] == [{"metadata": {"name": "job-b-workload"}}]


def test_snapshot_runner_resolves_logs_and_lifecycle() -> None:
    snapshot = jw.FleetSnapshot()
    snapshot.iris[("cw-us-east-02a", "/benfeuer/job-a")] = jw.IrisJobView("/benfeuer/job-a", "running", 1, {})
    snapshot.logs["/benfeuer/job-a"] = jw.LogLine("/benfeuer/job-a", "step 42 done", 99, "cw-us-east-02a", 5)
    entry = jw.RegistryEntry(
        id="a",
        owner="o",
        description="",
        probe=jw.LogRegexProbe(cluster="cw-us-east-02a", job="/benfeuer/job-a", pattern=r"step (\d+)"),
        iris=None,
        stall_threshold=timedelta(minutes=60),
        gated_threshold=timedelta(minutes=30),
        cooldown=timedelta(minutes=120),
        preempt_alert_count=2,
        expected_terminal=None,
    )
    runner = jw.snapshot_probe_runner(snapshot)
    obs = runner(entry, datetime.now(UTC))
    assert obs.progress == "42"
    assert obs.signature == "/benfeuer/job-a|99"
    assert obs.preemptions == 1
    assert "state=running" in obs.detail

    snapshot.logs.clear()
    obs = runner(entry, datetime.now(UTC))
    assert obs.progress == "(no log lines)"

    # prefix probes follow the newest matching job's log line
    prefix_entry = dataclasses.replace(
        entry, probe=jw.LogRegexProbe(cluster="cw-us-east-02a", job="/benfeuer/job-", job_is_prefix=True, pattern="")
    )
    snapshot.logs["/benfeuer/job-b"] = jw.LogLine("/benfeuer/job-b", "other", 200, "cw-us-east-02a", 6)
    obs = runner(prefix_entry, datetime.now(UTC))
    assert obs.signature.endswith("|200")


@pytest.mark.parametrize(
    ("iris_state", "expected"),
    [("failed", jw.EntryState.FAILED), ("succeeded", jw.EntryState.SUCCEEDED), ("running", jw.EntryState.RUNNING)],
)
def test_silent_job_takes_its_state_from_iris(iris_state: str, expected: jw.EntryState) -> None:
    """A job with no recent log lines still reports the Iris lifecycle state, so a silent terminal job alerts."""
    snapshot = jw.FleetSnapshot()
    snapshot.iris[("cw-us-east-02a", "/benfeuer/quiet")] = jw.IrisJobView("/benfeuer/quiet", iris_state, 0, {})
    entry = jw.RegistryEntry(
        id="quiet",
        owner="o",
        description="",
        probe=jw.LogRegexProbe(cluster="cw-us-east-02a", job="/benfeuer/quiet", pattern=""),
        iris=None,
        stall_threshold=timedelta(minutes=60),
        gated_threshold=timedelta(minutes=30),
        cooldown=timedelta(minutes=120),
        preempt_alert_count=2,
        expected_terminal=None,
    )
    obs = jw.snapshot_probe_runner(snapshot)(entry, datetime.now(UTC))
    assert obs.state == expected
    assert obs.progress == "(no log lines)"
    assert f"state={iris_state}" in obs.detail


def test_snapshot_runner_missing_job_is_probe_error() -> None:
    snapshot = jw.FleetSnapshot()
    entry = jw.RegistryEntry(
        id="a",
        owner="o",
        description="",
        probe=jw.LogRegexProbe(cluster="cw-us-east-02a", job="/benfeuer/gone", pattern=""),
        iris=None,
        stall_threshold=timedelta(minutes=60),
        gated_threshold=timedelta(minutes=30),
        cooldown=timedelta(minutes=120),
        preempt_alert_count=2,
        expected_terminal=None,
    )
    runner = jw.snapshot_probe_runner(snapshot)
    with pytest.raises(jw.ProbeError, match="not in the fleet list"):
        runner(entry, datetime.now(UTC))


def test_filtered_log_probe_keeps_escalations_visible_after_newer_progress(monkeypatch):
    now = datetime(2026, 11, 20, 12, tzinfo=UTC)
    registry = jw.parse_registry(
        {
            "coordinator": "c",
            "entries": [
                {
                    "id": "difficulty",
                    "owner": "o",
                    "iris": {"cluster": "cl", "job": "/u/controller"},
                    "probe": {
                        "type": "log_regex",
                        "cluster": "cl",
                        "job": "/u/controller",
                        "log_contains": "DIFFICULTY_ESCALATION",
                        "log_window_minutes": 17,
                        "failure_pattern": "DIFFICULTY_ESCALATION",
                    },
                }
            ],
        }
    )

    def external_query(command, timeout_seconds=120):
        if "iris" in command:
            return "/u/controller  running  2026-11-20T10:00:00+00:00"
        # Simulate Finelog applying its SQL content predicate before LIMIT 1.
        filtered = "contains(data, 'DIFFICULTY_ESCALATION')" in command[-1]
        if filtered:
            assert f"epoch_ms > {int(now.timestamp() * 1000) - 17 * 60 * 1000}" in command[-1]
        return json.dumps(
            {
                "key": "/u/controller/0:0",
                "seq": 10 if filtered else 11,
                "epoch_ms": int(now.timestamp() * 1000) - (1000 if filtered else 0),
                "cluster": "cl",
                "data": 'DIFFICULTY_ESCALATION {"action_id":"a"}' if filtered else "ordinary progress",
            }
        )

    monkeypatch.setattr(jw, "run_readonly", external_query)
    snapshot = jw.collect_fleet_snapshot(registry, now)
    _, sink = collect_alerts()
    result = jw.run_tick(registry, jw.WatchdogState(), jw.snapshot_probe_runner(snapshot), sink, now)
    assert result.statuses[0].state == jw.EntryState.FAILED
    assert len(result.fired) == 1
    assert "DIFFICULTY_ESCALATION" in result.fired[0].evidence["detail"]
