# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-stage Harbor import, private rewards, and unsupported task rejection."""

import json

import pytest
from shellbox.machine import Command
from taskcompendium.grading_result import Outcome
from taskcompendium.importers.harbor import harbor_task
from taskcompendium.models import Source, TaskSpec

from rolloutengine.task_session import WORKSPACE_INSTRUCTION

from .test_rollout import (
    FIXTURE_IMAGE,
    RecordingShellSimFactory,
    ReplayModel,
    engine,
    lowered,
    machine_runtime,
    shell_call,
)


@pytest.mark.parametrize("mode", [None, "shared"])
@pytest.mark.parametrize("verifier_environment", [False, True])
def test_shared_harbor_grading_is_rejected_during_import(tmp_path, mode, verifier_environment):
    verifier_options = "" if mode is None else f'environment_mode = "{mode}"'
    if verifier_environment:
        verifier_options += f'\n[verifier.environment]\ndocker_image = "{FIXTURE_IMAGE}"\n'
    directory = harbor_package(tmp_path, verifier_options=verifier_options)
    exception = ValueError if verifier_environment and mode == "shared" else NotImplementedError
    with pytest.raises(exception):
        harbor_task(directory, source=Source(dataset="fixture", revision="1", row="task", importer_revision="1"))


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
@pytest.mark.parametrize("verifier_environment", [False, True])
async def test_prebuilt_harbor_task_keeps_tests_private_and_grades_first_reward_file(
    tmp_path, answer, reward, verifier_environment
):
    options = 'environment_mode = "separate"'
    verifier_workdir = "/private-verifier"
    if verifier_environment:
        options += (
            '\n[verifier.environment]\ndocker_image = "verifier@sha256:'
            + "1" * 64
            + f'"\nworkdir = "{verifier_workdir}"\n'
        )
    directory = harbor_package(tmp_path, verifier_options=options)
    if verifier_environment:
        grader = directory / "tests/test.sh"
        grader.write_text(f'test "$(pwd)" = {verifier_workdir} || exit 99\n' + grader.read_text())
    (directory / "setup_files/input").write_text(answer)
    task = harbor_task(directory, source=Source(dataset="fixture", revision="1", row="task", importer_revision="1"))
    task = TaskSpec.model_validate_json(task.model_dump_json())
    factory = RecordingShellSimFactory()
    model = ReplayModel(
        [
            shell_call("test ! -f /tests/test.sh && cat /setup_files/input > /logs/artifacts/answer"),
            {"role": "assistant", "content": "Done."},
        ]
    )
    record = await engine(model, {"local": factory}).run(
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
    assert len(factory.machines) == 2
    for machine in factory.machines:
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
