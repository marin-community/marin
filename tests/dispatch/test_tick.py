# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch


from marin.dispatch.agent_adapter import AgentResult
from marin.dispatch.schema import (
    MonitoringCollection,
    RayRunConfig,
    RunPointer,
    RunState,
    RunStatus,
    RunTrack,
    TickEvent,
    TickEventKind,
)
from marin.dispatch.storage import save_collection, save_state
from marin.dispatch.tick import ESCALATION_THRESHOLD, process_tick, should_dispatch

# ── should_dispatch tests ──


def test_dispatch_on_manual():
    assert should_dispatch(RunStatus.RUNNING, RunStatus.RUNNING, TickEventKind.MANUAL, 0)


def test_dispatch_on_status_change():
    assert should_dispatch(RunStatus.FAILED, RunStatus.RUNNING, TickEventKind.SCHEDULED_POLL, 0)


def test_dispatch_on_running_health_check():
    assert should_dispatch(RunStatus.RUNNING, RunStatus.RUNNING, TickEventKind.SCHEDULED_POLL, 0)


def test_no_dispatch_when_succeeded_unchanged():
    assert not should_dispatch(RunStatus.SUCCEEDED, RunStatus.SUCCEEDED, TickEventKind.SCHEDULED_POLL, 0)


def test_no_dispatch_after_escalation_threshold():
    assert not should_dispatch(RunStatus.FAILED, RunStatus.RUNNING, TickEventKind.SCHEDULED_POLL, ESCALATION_THRESHOLD)


def test_dispatch_failure_alert_on_failed():
    assert should_dispatch(RunStatus.FAILED, RunStatus.RUNNING, TickEventKind.FAILURE_ALERT, 0)


def test_no_dispatch_failure_alert_on_running():
    assert not should_dispatch(RunStatus.RUNNING, RunStatus.RUNNING, TickEventKind.FAILURE_ALERT, 0)


def test_manual_overrides_escalation_threshold():
    # Manual dispatch is blocked by escalation threshold too.
    assert not should_dispatch(RunStatus.FAILED, RunStatus.FAILED, TickEventKind.MANUAL, ESCALATION_THRESHOLD)


# ── process_tick tests ──


class MockAgent:
    def __init__(self, result: AgentResult):
        self.result = result
        self.calls: list[tuple[TickEvent, Path]] = []

    def launch(self, event: TickEvent, worktree_path: Path) -> AgentResult:
        self.calls.append((event, worktree_path))
        return self.result


def _make_collection(name: str = "test") -> MonitoringCollection:
    return MonitoringCollection(
        name=name,
        prompt="check the run",
        logbook=".agents/logbooks/test.md",
        branch="research/test",
        issue=99,
        runs=(
            RunPointer(
                track=RunTrack.RAY,
                ray=RayRunConfig(job_id="ray-1", cluster="c", experiment="e.py"),
            ),
        ),
    )


@patch("marin.dispatch.tick.query_run_status", return_value=RunStatus.FAILED)
@patch("marin.dispatch.tick.setup_worktree")
@patch("marin.dispatch.tick.cleanup_worktree")
@patch("marin.dispatch.tick.append_logbook")
@patch("marin.dispatch.tick.commit_and_push", return_value=True)
@patch("marin.dispatch.tick.post_issue_comment")
@patch("marin.dispatch.tick.post_escalation")
def test_process_tick_success(
    mock_escalation, mock_comment, mock_push, mock_append, mock_cleanup, mock_worktree, mock_status, tmp_path
):
    mock_worktree.return_value = tmp_path / "wt"
    (tmp_path / "wt").mkdir()

    save_collection(tmp_path, _make_collection())
    save_state(tmp_path, "test", [RunState(last_status=RunStatus.RUNNING)])

    agent = MockAgent(AgentResult(success=True, logbook_entry="## Entry\nAll good.", issue_comment="Progress update"))
    outcome = process_tick("test", TickEventKind.SCHEDULED_POLL, agent, tmp_path)

    assert outcome.dispatched == 1
    assert outcome.succeeded == 1
    assert outcome.failed == 0
    assert len(agent.calls) == 1
    mock_append.assert_called_once()
    mock_push.assert_called_once()
    mock_comment.assert_called_once()
    mock_escalation.assert_not_called()


@patch("marin.dispatch.tick.query_run_status", return_value=RunStatus.FAILED)
@patch("marin.dispatch.tick.setup_worktree")
@patch("marin.dispatch.tick.cleanup_worktree")
@patch("marin.dispatch.tick.post_escalation")
def test_process_tick_escalation_on_repeated_failure(
    mock_escalation, mock_cleanup, mock_worktree, mock_status, tmp_path
):
    mock_worktree.return_value = tmp_path / "wt"
    (tmp_path / "wt").mkdir()

    save_collection(tmp_path, _make_collection())
    save_state(tmp_path, "test", [RunState(last_status=RunStatus.RUNNING, consecutive_failures=2)])

    agent = MockAgent(AgentResult(success=False, logbook_entry="", error="OOM again"))
    outcome = process_tick("test", TickEventKind.SCHEDULED_POLL, agent, tmp_path)

    assert outcome.failed == 1
    assert outcome.escalated == 1
    mock_escalation.assert_called_once()


@patch("marin.dispatch.tick.query_run_status", return_value=RunStatus.RUNNING)
def test_process_tick_paused_collection(mock_status, tmp_path):
    from dataclasses import replace

    c = replace(_make_collection(), paused=True)
    save_collection(tmp_path, c)

    agent = MockAgent(AgentResult(success=True, logbook_entry="x"))
    outcome = process_tick("test", TickEventKind.SCHEDULED_POLL, agent, tmp_path)
    assert outcome.dispatched == 0
    assert len(agent.calls) == 0
