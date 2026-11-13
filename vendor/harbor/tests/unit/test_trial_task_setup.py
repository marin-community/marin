"""The hidden task_setup step: runs as root before the agent, leaves nothing in the guest, fails as a task error."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from harbor.environments.base import ExecResult
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.trial.errors import TaskSetupError
from harbor.trial.single_step import SingleStepTrial
from harbor.trial.trial import TASK_SETUP_GUEST_PREFIX, TASK_SETUP_RECORD


class RecordingEnvironment:
    def __init__(self, setup_result: ExecResult | BaseException):
        self.setup_result = setup_result
        self.uploads: list[tuple[Path, str]] = []
        self.execs: list[dict] = []

    async def upload_dir(self, source_dir, target_dir):
        self.uploads.append((Path(source_dir), target_dir))

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.execs.append({"command": command, "cwd": cwd, "timeout_sec": timeout_sec, "user": user})
        if command.startswith("rm -rf "):
            return ExecResult(return_code=0, stdout="", stderr="")
        if isinstance(self.setup_result, BaseException):
            raise self.setup_result
        return self.setup_result


def _trial(tmp_path: Path, environment: RecordingEnvironment, *, setup: bool = True) -> SingleStepTrial:
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Do it.")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("exit 0\n")
    (task_dir / "task.toml").write_text('[environment]\ndocker_image = "test/image"\nbuild_timeout_sec = 90\n')
    if setup:
        (task_dir / "task_setup" / "files" / "app").mkdir(parents=True)
        (task_dir / "task_setup" / "files" / "app" / "input.csv").write_text("a,b\n")
        (task_dir / "task_setup" / "setup.sh").write_text("echo setting up\n")
    trial = object.__new__(SingleStepTrial)
    trial.task = Task(task_dir)
    trial.paths = TrialPaths(trial_dir=tmp_path / "trial")
    trial.paths.mkdir()
    trial.agent_environment = environment
    trial._environment_build_timeout_sec = 90.0
    trial.logger = MagicMock()
    return trial


@pytest.mark.asyncio
async def test_setup_runs_as_root_records_output_on_host_and_removes_its_inputs(tmp_path):
    environment = RecordingEnvironment(ExecResult(return_code=0, stdout="setting up\n", stderr=""))
    trial = _trial(tmp_path, environment)

    await trial._run_task_setup()

    (source, guest), = environment.uploads
    assert source == trial.task.paths.task_setup_dir
    assert guest.startswith(TASK_SETUP_GUEST_PREFIX)
    setup, cleanup = environment.execs
    assert setup["user"] == "root" and setup["cwd"] == "/" and setup["timeout_sec"] == 90
    assert f"cp -R {guest}/files/. /" in setup["command"] and f"bash {guest}/setup.sh" in setup["command"]
    assert cleanup["command"] == f"rm -rf {guest}" and cleanup["user"] == "root"
    record = json.loads((trial.paths.trial_dir / TASK_SETUP_RECORD).read_text())
    assert record == {"exit_code": 0, "stdout": "setting up\n", "stderr": ""}


@pytest.mark.asyncio
async def test_nonzero_setup_is_a_task_setup_error_and_still_cleans_up(tmp_path):
    environment = RecordingEnvironment(ExecResult(return_code=3, stdout="", stderr="refused: apt-get install foo"))
    trial = _trial(tmp_path, environment)

    with pytest.raises(TaskSetupError, match="exited 3: refused: apt-get install foo"):
        await trial._run_task_setup()

    assert environment.execs[-1]["command"].startswith("rm -rf ")
    assert json.loads((trial.paths.trial_dir / TASK_SETUP_RECORD).read_text())["exit_code"] == 3


@pytest.mark.asyncio
async def test_setup_timeout_is_a_task_setup_error_and_still_cleans_up(tmp_path):
    environment = RecordingEnvironment(TimeoutError("Sandbox command timed out"))
    trial = _trial(tmp_path, environment)

    with pytest.raises(TaskSetupError, match="timed out after 90 seconds"):
        await trial._run_task_setup()

    assert environment.execs[-1]["command"].startswith("rm -rf ")


@pytest.mark.asyncio
async def test_task_without_setup_dir_touches_nothing(tmp_path):
    environment = RecordingEnvironment(ExecResult(return_code=0, stdout="", stderr=""))
    trial = _trial(tmp_path, environment, setup=False)

    await trial._run_task_setup()

    assert environment.uploads == [] and environment.execs == []
    assert not (trial.paths.trial_dir / TASK_SETUP_RECORD).exists()


def test_setup_runs_after_start_and_before_the_agent_network_lockdown(tmp_path):
    trial = _trial(tmp_path, RecordingEnvironment(ExecResult(return_code=0, stdout="", stderr="")))
    calls: list[str] = []

    async def start():
        calls.append("start")

    async def setup():
        calls.append("task_setup")

    trial._emit = lambda _event: _noop()
    trial._result = SimpleNamespace(environment_setup=None)
    trial._now = lambda: None
    trial._start_agent_environment = start
    trial._run_task_setup = setup

    async def lockdown():
        calls.append("network_lockdown")

    trial.agent_environment.apply_agent_network_policy = lockdown

    asyncio.run(trial._setup_agent_environment())
    assert calls == ["start", "task_setup", "network_lockdown"]


async def _noop():
    return None
