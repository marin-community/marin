# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from marin.dispatch.schema import (
    IrisRunConfig,
    MonitoringCollection,
    RayRunConfig,
    RunPointer,
    RunState,
    RunStatus,
    RunTrack,
    TickEvent,
    TickEventKind,
)


def test_run_track_values():
    assert RunTrack.RAY == "ray"
    assert RunTrack.IRIS == "iris"


def test_run_status_values():
    assert RunStatus.UNKNOWN == "unknown"
    assert RunStatus.FAILED == "failed"


def test_ray_run_pointer():
    rp = RunPointer(
        track=RunTrack.RAY,
        ray=RayRunConfig(job_id="j1", cluster="us-central2", experiment="exp.py"),
    )
    assert rp.track == RunTrack.RAY
    assert rp.ray is not None
    assert rp.iris is None


def test_iris_run_pointer():
    rp = RunPointer(
        track=RunTrack.IRIS,
        iris=IrisRunConfig(job_id="j2", config="c.yaml", resubmit_command="iris job run ..."),
    )
    assert rp.track == RunTrack.IRIS
    assert rp.iris is not None


def test_run_pointer_validation_ray_missing():
    with pytest.raises(ValueError, match="ray config"):
        RunPointer(track=RunTrack.RAY)


def test_run_pointer_validation_iris_missing():
    with pytest.raises(ValueError, match="iris config"):
        RunPointer(track=RunTrack.IRIS)


def test_monitoring_collection_frozen():
    c = MonitoringCollection(name="test", prompt="do stuff", logbook="lb.md", branch="b", issue=1)
    with pytest.raises(AttributeError):
        c.name = "other"  # type: ignore[misc]


def test_monitoring_collection_runs_tuple():
    rp = RunPointer(
        track=RunTrack.RAY,
        ray=RayRunConfig(job_id="j1", cluster="c", experiment="e.py"),
    )
    c = MonitoringCollection(name="t", prompt="p", logbook="l.md", branch="b", issue=1, runs=(rp,))
    assert len(c.runs) == 1


def test_tick_event_construction():
    rp = RunPointer(
        track=RunTrack.RAY,
        ray=RayRunConfig(job_id="j1", cluster="c", experiment="e.py"),
    )
    event = TickEvent(
        kind=TickEventKind.MANUAL,
        collection_name="test",
        run_index=0,
        run_pointer=rp,
        prompt="check it",
        logbook="lb.md",
        branch="b",
        issue=1,
        timestamp="2026-03-22T00:00:00Z",
    )
    assert event.kind == TickEventKind.MANUAL


def test_run_state_mutable():
    s = RunState()
    assert s.last_status == RunStatus.UNKNOWN
    s.last_status = RunStatus.RUNNING
    assert s.last_status == RunStatus.RUNNING
    s.consecutive_failures = 5
    assert s.consecutive_failures == 5
