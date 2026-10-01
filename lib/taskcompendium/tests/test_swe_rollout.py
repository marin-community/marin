# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE task conversion and grading with real Git in temporary repositories."""

import asyncio
import json
import os
import shutil
import subprocess

import pytest
from shellbox.machine import ExitReason, Result

from taskcompendium.environment import EnvironmentKind, EnvironmentSpec, LocalImage
from taskcompendium.grading import Outcome
from taskcompendium.importers.swe import SWEInstance, swe_task
from taskcompendium.models import Source
from taskcompendium.parquet import read_tasks, write_tasks

from .test_rollout import ReplayModel, engine, run_task


class LocalGitMachine:
    """A machine-boundary fixture for authored commands, without container isolation."""

    def __init__(self, root, spec):
        self.root = root
        self.spec = spec
        self.closed = False

    def path(self, value):
        return self.root / value.lstrip("/")

    async def run(self, command):
        if self.closed:
            raise RuntimeError("Machine is closed")
        argv = [str(self.path(value)) if value.startswith("/") else value for value in command.argv]
        result = await asyncio.to_thread(
            subprocess.run,
            argv,
            cwd=self.path(command.cwd or self.spec.workdir),
            env={**os.environ, **self.spec.env, **command.env},
            input=command.stdin,
            capture_output=True,
            timeout=command.timeout,
        )
        return Result(result.returncode, result.stdout, result.stderr, False, False, ExitReason.EXITED)

    async def upload(self, source, target):
        path = self.path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, path)

    async def download(self, source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path(source), target)

    async def close(self):
        self.closed = True


@pytest.fixture
def git_image(tmp_path):
    repository = tmp_path / "image"
    repository.mkdir()
    (repository / "value.txt").write_text("broken\n")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.name", "Test Fixture"),
        ("git", "config", "user.email", "fixture@example.test"),
        ("git", "add", "."),
        ("git", "commit", "-qm", "Initial fixture"),
    ):
        subprocess.run(command, cwd=repository, check=True, capture_output=True)
    return repository


@pytest.mark.parametrize("answer,reward", [("fixed", 1.0), ("wrong", 0.0)])
async def test_swe_parquet_task_applies_and_grades_the_patch_in_a_fresh_repository(tmp_path, git_image, answer, reward):
    task = swe_task(
        SWEInstance(
            instance_id="fixture-1",
            problem_statement="Repair value.txt.",
            eval_script='test "$(cat value.txt)" = fixed',
        ),
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        environment=EnvironmentSpec(
            kind=EnvironmentKind.DOCKER, image=LocalImage(reference="fixture"), workdir="/testbed"
        ),
        verifier_timeout=5,
    )
    path = str(tmp_path / "tasks.parquet")
    write_tasks(path, iter([task]))
    machines = []

    class Factory:
        async def create(self, spec):
            root = tmp_path / f"machine-{len(machines)}"
            shutil.copytree(git_image, root / "testbed")
            machine = LocalGitMachine(root, spec)
            machines.append(machine)
            return machine

    model = ReplayModel(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "edit",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": f"echo {answer} > value.txt"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Completed."},
        ]
    )
    result = await run_task(engine(model, {EnvironmentKind.DOCKER: Factory()}), next(read_tasks(path)))
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, reward)
    assert (git_image / "value.txt").read_text() == "broken\n"
    assert len(machines) == 2
    assert (machines[1].root / "testbed/value.txt").read_text() == f"{answer}\n"
    assert all(machine.closed for machine in machines)
    assert result.loss_mask == (1, 0, 0, 1)
    assert all("eval_script" not in json.dumps(request.messages) for request in model.requests)
