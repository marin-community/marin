"""Trajectory dump cadence: per-trial by default, per-turn opt-in.

The per-episode trajectory dump was a per-turn cross-region storage write that
collapsed rollout throughput (py-spy: coordinators blocked in s3fs uploads +
format_trajectory_json). The default cadence is now FINAL: one dump when
``run()`` finishes, plus the MANDATORY continuation-boundary dumps in
linear-history mode — those segments are truncated from memory right after the
dump, so skipping them would lose data. PER_TURN restores the old
every-episode snapshots via config or the HARBOR_TRAJECTORY_DUMP_CADENCE env.
"""

import json
import threading

import pytest

from harbor.agents.terminus_2 import terminus_2 as t2
from harbor.agents.terminus_2.terminus_2 import (
    TRAJECTORY_DUMP_CADENCE_ENV,
    TrajectoryDumpCadence,
    _resolve_trajectory_dump_cadence,
)
from harbor.models.trajectories import Step
from harbor.utils.artifact_writer import ArtifactWriter
from tests.unit.agents.terminus_2.bare_agent import make_bare_terminus2


def _make_agent(logs_dir, cadence: TrajectoryDumpCadence, *, linear_history=False):
    return make_bare_terminus2(
        logs_dir,
        _trajectory_dump_cadence=cadence,
        _linear_history=linear_history,
        _trajectory_steps=[Step(step_id=1, source="user", message="original step")],
    )


# ── cadence resolution (config knob + env override) ──────────────────────────


@pytest.mark.unit
def test_cadence_defaults_to_final(monkeypatch):
    monkeypatch.delenv(TRAJECTORY_DUMP_CADENCE_ENV, raising=False)
    assert _resolve_trajectory_dump_cadence(None) is TrajectoryDumpCadence.FINAL


@pytest.mark.unit
def test_cadence_env_override(monkeypatch):
    monkeypatch.setenv(TRAJECTORY_DUMP_CADENCE_ENV, "per_turn")
    assert _resolve_trajectory_dump_cadence(None) is TrajectoryDumpCadence.PER_TURN


@pytest.mark.unit
def test_cadence_explicit_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv(TRAJECTORY_DUMP_CADENCE_ENV, "per_turn")
    assert _resolve_trajectory_dump_cadence("final") is TrajectoryDumpCadence.FINAL


@pytest.mark.unit
def test_cadence_rejects_unknown_value():
    with pytest.raises(ValueError):
        _resolve_trajectory_dump_cadence("sometimes")


# ── per-turn hook is cadence-gated ────────────────────────────────────────────


@pytest.mark.unit
def test_final_cadence_skips_per_turn_dump(tmp_path, monkeypatch):
    writer = ArtifactWriter()
    monkeypatch.setattr(t2, "get_artifact_writer", lambda: writer)
    agent = _make_agent(tmp_path, TrajectoryDumpCadence.FINAL)

    agent._maybe_persist_trajectory_per_turn()
    writer.flush(timeout=30)

    assert not (tmp_path / "trajectory.json").exists()


@pytest.mark.unit
def test_per_turn_cadence_writes_every_turn_dump(tmp_path, monkeypatch):
    writer = ArtifactWriter()
    monkeypatch.setattr(t2, "get_artifact_writer", lambda: writer)
    agent = _make_agent(tmp_path, TrajectoryDumpCadence.PER_TURN)

    agent._maybe_persist_trajectory_per_turn()
    writer.flush(timeout=30)

    data = json.loads((tmp_path / "trajectory.json").read_text())
    assert data["steps"][0]["message"] == "original step"


# ── continuation-boundary dumps are mandatory and snapshot-safe ───────────────


@pytest.mark.unit
def test_continuation_boundary_dump_written_even_with_final_cadence(
    tmp_path, monkeypatch
):
    """Linear-history summarization boundaries must dump regardless of cadence:
    the segment's steps are truncated from memory right afterwards."""
    writer = ArtifactWriter()
    monkeypatch.setattr(t2, "get_artifact_writer", lambda: writer)
    agent = _make_agent(tmp_path, TrajectoryDumpCadence.FINAL, linear_history=True)
    # Second summarization: one boundary dump already completed, so this split
    # writes the cont-1 segment and points at cont-2.
    agent._summarization_count = 2
    agent._continuation_segments_persisted = 1

    agent._split_trajectory_on_summarization("handoff prompt")
    writer.flush(timeout=30)

    segment_path = tmp_path / "trajectory.cont-1.json"
    assert segment_path.exists(), "continuation-boundary dump was skipped"
    data = json.loads(segment_path.read_text())
    # Continuation ref chain stays intact: this segment points at the next one.
    assert data["continued_trajectory_ref"] == "trajectory.cont-2.json"
    assert data["agent"]["extra"]["continuation_index"] == 1
    assert data["steps"][0]["message"] == "original step"


@pytest.mark.unit
def test_boundary_dump_snapshot_survives_post_split_truncation(tmp_path, monkeypatch):
    """The boundary dump must capture the steps as of the split, even when the
    background write happens AFTER the agent truncates/rebuilds its step list
    (which the real flow does immediately after the split)."""
    gate = threading.Event()
    writer = ArtifactWriter(num_workers=1)
    monkeypatch.setattr(t2, "get_artifact_writer", lambda: writer)
    agent = _make_agent(tmp_path, TrajectoryDumpCadence.FINAL, linear_history=True)
    agent._summarization_count = 1

    def gated_payload() -> str:
        gate.wait(timeout=30)
        return "gate opened"

    # Occupy the single worker so the boundary dump serializes AFTER the
    # post-split mutation below.
    writer.submit_optional(tmp_path / "gate.txt", gated_payload, description="gate")

    agent._split_trajectory_on_summarization("handoff prompt")
    # Simulate the post-split reset: the old step objects are gone from the
    # agent's list before the background writer ever ran.
    agent._trajectory_steps = [
        Step(step_id=1, source="user", message="continuation step")
    ]
    gate.set()
    writer.flush(timeout=30)

    # First summarization -> continuation_index 0 -> trajectory.json.
    data = json.loads((tmp_path / "trajectory.json").read_text())
    assert [s["message"] for s in data["steps"]] == ["original step"]
