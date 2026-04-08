# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from marin.dispatch.schema import (
    MonitoringCollection,
    RayRunConfig,
    RunPointer,
    RunState,
    RunStatus,
    RunTrack,
)
from marin.dispatch.storage import (
    load_collection,
    load_state,
    save_collection,
    save_state,
)


@pytest.fixture
def sample_collection() -> MonitoringCollection:
    return MonitoringCollection(
        name="test-sweep",
        prompt="Monitor the sweep run.\nCheck W&B for convergence.",
        logbook=".agents/logbooks/sweep.md",
        branch="research/sweep",
        issue=42,
        runs=(
            RunPointer(
                track=RunTrack.RAY,
                ray=RayRunConfig(job_id="ray-123", cluster="us-central2", experiment="exp.py"),
            ),
        ),
        created_at="2026-03-22T10:00:00Z",
    )


def test_save_and_load_collection(tmp_path, sample_collection):
    save_collection(tmp_path, sample_collection)
    loaded = load_collection(tmp_path, "test-sweep")
    assert loaded.name == sample_collection.name
    assert loaded.prompt == sample_collection.prompt
    assert loaded.branch == sample_collection.branch
    assert loaded.issue == sample_collection.issue
    assert len(loaded.runs) == 1
    assert loaded.runs[0].track == RunTrack.RAY
    assert loaded.runs[0].ray.job_id == "ray-123"


def test_save_and_load_state(tmp_path):
    states = [
        RunState(last_status=RunStatus.RUNNING, last_check="2026-03-22T10:00:00Z"),
        RunState(last_status=RunStatus.FAILED, consecutive_failures=2, last_error="OOM"),
    ]
    save_state(tmp_path, "test", states)
    loaded = load_state(tmp_path, "test")
    assert len(loaded) == 2
    assert loaded[0].last_status == RunStatus.RUNNING
    assert loaded[1].consecutive_failures == 2
    assert loaded[1].last_error == "OOM"


def test_roundtrip_multiline_prompt(tmp_path):
    c = MonitoringCollection(
        name="multi",
        prompt="Line one.\nLine two.\nLine three.",
        logbook="lb.md",
        branch="b",
        issue=1,
    )
    save_collection(tmp_path, c)
    loaded = load_collection(tmp_path, "multi")
    assert loaded.prompt == c.prompt
