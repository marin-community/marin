# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-stage Harbor import, private rewards, and unsupported task rejection."""

import json
from dataclasses import replace

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, ShellSimBuiltins
from taskcompendium.grading_result import Outcome
from taskcompendium.importers.harbor import harbor_task
from taskcompendium.models import EnvironmentRequirements, Source, TaskSpec

from rolloutengine.lowering import lower_task
from rolloutengine.task_session import WORKSPACE_INSTRUCTION

from .test_rollout import RecordingShellSimFactory, ReplayModel, engine, lowered, machine_runtime, shell_call

FIXTURE_IMAGE = "fixture@sha256:" + "0" * 64


@pytest.mark.parametrize("verifier_options", ["", 'environment_mode = "shared"'])
def test_shared_harbor_grading_is_rejected_during_import(tmp_path, verifier_options):
    directory = harbor_package(tmp_path, verifier_options=verifier_options)
    with pytest.raises(NotImplementedError):
        harbor_task(directory, source=Source(dataset="fixture", revision="1", row="task", importer_revision="1"))


@pytest.mark.parametrize("session_name", ["shellbox", "custom"])
async def test_shell_grading_without_an_image_is_rejected_before_execution(tmp_path, session_name):
    directory = harbor_package(tmp_path, verifier_options='environment_mode = "separate"')
    task = harbor_task(directory, source=Source(dataset="fixture", revision="1", row="task", importer_revision="1"))
    task = task.model_copy(
        update={"verifier": task.verifier.model_copy(update={"environment_requirements": EnvironmentRequirements()})}
    )
    record = lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime(), task_session=session_name)
    factory = RecordingShellSimFactory()
    model = ReplayModel([])
    sessions = {"custom": lambda task, machine: None}
    with pytest.raises(ValueError):
        lower_task(task, record.runtime, record.session, factories={"local": factory}, sessions=sessions)
    with pytest.raises(ValueError):
        await engine(model, {"local": factory}, sessions=sessions).run(record)
    assert factory.machines == []
    assert model.requests == []


def harbor_package(tmp_path, *, task_options="", environment_options="", verifier_options=""):
    directory = tmp_path / "task"
    (directory / "tests").mkdir(parents=True)
    (directory / "setup_files").mkdir()
    (directory / "instruction.md").write_text("Copy /setup_files/input to /logs/artifacts/answer.")
    (directory / "task.toml").write_text(
        f'{task_options}\n[environment]\ndocker_image = "{FIXTURE_IMAGE}"\n'
        f'workdir = "/workspace"\n{environment_options}\n'
        f"[verifier]\ntimeout_sec = 5\n{verifier_options}\n"
    )
    (directory / "setup_files/input").write_text("answer")
    (directory / "tests/test.sh").write_text(
        'if [ "$(cat /logs/artifacts/answer)" = answer ]; then\n'
        "    echo '{\"reward\":0.75}' > /logs/verifier/reward.json\n"
        "else\n    echo '{\"reward\":0}' > /logs/verifier/reward.json\nfi\n"
        "echo 1 > /logs/verifier/reward.txt\nexit 7\n"
    )
    return directory


@pytest.mark.parametrize("answer,reward", [("answer", 0.75), ("wrong", 0.0)])
async def test_prebuilt_harbor_task_keeps_tests_private_and_grades_first_reward_file(tmp_path, answer, reward):
    directory = harbor_package(tmp_path, verifier_options='environment_mode = "separate"')
    (directory / "setup_files/input").write_text(answer)
    task = harbor_task(directory, source=Source(dataset="fixture", revision="1", row="task", importer_revision="1"))
    task = TaskSpec.model_validate_json(task.model_dump_json())
    machines = []

    class FixtureImageFactory:
        async def create(self, spec):
            machine = await ShellSimMachineFactory().create(replace(spec, source=ShellSimBuiltins()))
            machines.append(machine)
            return machine

    model = ReplayModel(
        [
            shell_call("test ! -f /tests/test.sh && cat /setup_files/input > /logs/artifacts/answer"),
            {"role": "assistant", "content": "Done."},
        ]
    )
    record = await engine(model, {"local": FixtureImageFactory()}).run(
        lowered(
            task,
            machine=machine_runtime(),
            verifier_machine=machine_runtime(),
        )
    )
    assert (record.grade.status, record.grade.reward, record.grade.passed) == (Outcome.GRADED, reward, reward > 0)
    assert len(task.context.events) == 1
    assert model.requests[0].messages == (
        {"role": "user", "content": (directory / "instruction.md").read_text()},
        {"role": "user", "content": WORKSPACE_INSTRUCTION},
    )
    assert record.grade.diagnostics["exit_code"] == 7
    assert record.loss_mask == (1, 0, 0, 1)
    assert json.loads(model.requests[1].messages[-1]["content"])["exit_code"] == 0
    for machine in machines:
        with pytest.raises(RuntimeError):
            await machine.run(Command(("true",)))


@pytest.mark.parametrize("unsupported", ["stages", "build", "healthcheck", "shared_verifier_env"])
def test_unsupported_harbor_package_is_rejected(tmp_path, unsupported):
    directory = harbor_package(
        tmp_path,
        task_options='steps = [{name = "first"}]' if unsupported == "stages" else "",
        environment_options='healthcheck = {command = "true"}' if unsupported == "healthcheck" else "",
        verifier_options=(
            'env = {PRIVATE_TOKEN = "secret"}'
            if unsupported == "shared_verifier_env"
            else 'environment_mode = "separate"'
        ),
    )
    if unsupported == "build":
        (directory / "task.toml").write_text("[environment]\n[verifier]\ntimeout_sec = 5\n")
        (directory / "environment").mkdir()
        (directory / "environment/Dockerfile").write_text("FROM busybox")
    with pytest.raises(NotImplementedError):
        harbor_task(directory, source=Source(dataset="fixture", revision="1", row="task", importer_revision="1"))
