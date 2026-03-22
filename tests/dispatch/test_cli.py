# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from click.testing import CliRunner

from marin.dispatch.cli import cli
from marin.dispatch.schema import RunState, RunStatus
from marin.dispatch.storage import load_collection, load_state, save_state


@patch("marin.dispatch.cli._resolve_repo_root")
def test_register_and_list(mock_root, tmp_path):
    mock_root.return_value = tmp_path
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "register",
            "--name",
            "sweep-1",
            "--prompt",
            "Monitor the sweep",
            "--logbook",
            ".agents/logbooks/sweep.md",
            "--branch",
            "research/sweep",
            "--issue",
            "42",
        ],
    )
    assert result.exit_code == 0
    assert "Registered" in result.output

    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "sweep-1" in result.output


@patch("marin.dispatch.cli._resolve_repo_root")
def test_show(mock_root, tmp_path):
    mock_root.return_value = tmp_path
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "register",
            "--name",
            "s1",
            "--prompt",
            "p",
            "--logbook",
            "l.md",
            "--branch",
            "b",
            "--issue",
            "1",
        ],
    )
    result = runner.invoke(cli, ["show", "s1"])
    assert result.exit_code == 0
    assert "s1" in result.output
    assert "#1" in result.output


@patch("marin.dispatch.cli._resolve_repo_root")
def test_add_and_remove_run(mock_root, tmp_path):
    mock_root.return_value = tmp_path
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "register",
            "--name",
            "s2",
            "--prompt",
            "p",
            "--logbook",
            "l.md",
            "--branch",
            "b",
            "--issue",
            "1",
        ],
    )

    result = runner.invoke(
        cli,
        [
            "add-run",
            "s2",
            "--track",
            "ray",
            "--job-id",
            "ray-123",
            "--cluster",
            "us-central2",
            "--experiment",
            "exp.py",
        ],
    )
    assert result.exit_code == 0
    assert "Added" in result.output

    c = load_collection(tmp_path, "s2")
    assert len(c.runs) == 1
    assert c.runs[0].ray.job_id == "ray-123"

    result = runner.invoke(cli, ["remove-run", "s2", "--index", "0"])
    assert result.exit_code == 0
    c = load_collection(tmp_path, "s2")
    assert len(c.runs) == 0


@patch("marin.dispatch.cli._resolve_repo_root")
def test_update_pause(mock_root, tmp_path):
    mock_root.return_value = tmp_path
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "register",
            "--name",
            "s3",
            "--prompt",
            "p",
            "--logbook",
            "l.md",
            "--branch",
            "b",
            "--issue",
            "1",
        ],
    )
    result = runner.invoke(cli, ["update", "s3", "--paused", "true"])
    assert result.exit_code == 0, result.output

    c = load_collection(tmp_path, "s3")
    assert c.paused is True


@patch("marin.dispatch.cli._resolve_repo_root")
def test_remove_run_updates_state(mock_root, tmp_path):
    """Removing a run also removes its corresponding state entry."""
    mock_root.return_value = tmp_path
    runner = CliRunner()
    runner.invoke(
        cli,
        ["register", "--name", "s5", "--prompt", "p", "--logbook", "l.md", "--branch", "b", "--issue", "1"],
    )
    for jid in ["ray-1", "ray-2", "ray-3"]:
        runner.invoke(cli, ["add-run", "s5", "--track", "ray", "--job-id", jid, "--cluster", "c", "--experiment", "e"])

    # Simulate state for all 3 runs.
    save_state(
        tmp_path,
        "s5",
        [
            RunState(last_status=RunStatus.RUNNING),
            RunState(last_status=RunStatus.FAILED, consecutive_failures=2),
            RunState(last_status=RunStatus.SUCCEEDED),
        ],
    )

    # Remove the middle run (index 1).
    result = runner.invoke(cli, ["remove-run", "s5", "--index", "1"])
    assert result.exit_code == 0

    c = load_collection(tmp_path, "s5")
    assert len(c.runs) == 2
    assert c.runs[0].ray.job_id == "ray-1"
    assert c.runs[1].ray.job_id == "ray-3"

    states = load_state(tmp_path, "s5")
    assert len(states) == 2
    assert states[0].last_status == RunStatus.RUNNING
    assert states[1].last_status == RunStatus.SUCCEEDED


@patch("marin.dispatch.cli._resolve_repo_root")
def test_delete(mock_root, tmp_path):
    mock_root.return_value = tmp_path
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "register",
            "--name",
            "s4",
            "--prompt",
            "p",
            "--logbook",
            "l.md",
            "--branch",
            "b",
            "--issue",
            "1",
        ],
    )
    result = runner.invoke(cli, ["delete", "s4"])
    assert result.exit_code == 0
    assert "Deleted" in result.output

    result = runner.invoke(cli, ["list"])
    assert "s4" not in result.output
