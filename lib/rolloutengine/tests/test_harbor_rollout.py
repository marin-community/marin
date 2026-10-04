# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor task packages through Parquet, machine creation, and private grading."""

import asyncio
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import ExitReason, Result, ShellSimBuiltins
from taskcompendium.environment import DockerBuild, EnvironmentKind, ShellVerifierSpec
from taskcompendium.grading import Outcome
from taskcompendium.importers.harbor import harbor_task
from taskcompendium.models import Source, TaskSpec

from rolloutengine.contracts import RolloutInterrupted, RolloutOperation

from .test_rollout import ReplayModel, engine, run_task

GRADER = b"""#!/bin/sh
if [ "$(cat /logs/artifacts/answer)" = "$EXPECTED" ]; then
    echo '{"reward": 0.75}' > /logs/verifier/reward.json
else
    echo '{"reward": 0}' > /logs/verifier/reward.json
fi
echo 1 > /logs/verifier/reward.txt
exit 7
"""


class TarMachine:
    """Use host tar at the machine boundary because ShellSim has no exclusion option."""

    def __init__(self, machine):
        self.machine = machine

    async def run(self, command):
        if command.argv[0] != "tar":
            return await self.machine.run(command)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / "context"
            context.mkdir()
            argv = list(command.argv)
            context_index = argv.index("-C") + 1
            await self.machine.download(argv[context_index], context)
            remote_archive = argv[2]
            archive = root / "artifact.tar"
            argv[2] = str(archive)
            argv[context_index] = str(context)
            result = await asyncio.to_thread(subprocess.run, argv, capture_output=True, timeout=command.timeout)
            if result.returncode == 0:
                await self.machine.upload(archive, remote_archive)
            return Result(result.returncode, result.stdout, result.stderr, False, False, ExitReason.EXITED)

    async def upload(self, source, target):
        await self.machine.upload(source, target)

    async def download(self, source, target):
        await self.machine.download(source, target)

    async def close(self):
        await self.machine.close()


@pytest.mark.parametrize(
    "separate,selected_artifacts,expected,reward,passed",
    [
        (False, False, "answer", 0.75, True),
        (False, False, "wrong", 0.0, False),
        (True, False, "answer", 0.75, True),
        (True, True, "answer", 0.75, True),
    ],
)
async def test_harbor_package_grades_private_files_after_json_reload(
    tmp_path, monkeypatch, separate, selected_artifacts, expected, reward, passed
):
    directory = tmp_path / "source"
    for name in ("environment", "tests", "setup_files"):
        (directory / name).mkdir(parents=True)
    (directory / "instruction.md").write_text("Copy /setup_files/input to /logs/artifacts/answer.")
    (directory / "task.toml").write_text(
        (
            'artifacts = [{source = "/submission", destination = "export", exclude = ["*.tmp"]}, "/missing"]\n'
            if selected_artifacts
            else ""
        )
        + '[environment]\nallow_internet = false\nworkdir = "/workspace"\n'
        "[environment.healthcheck]\ninterval_sec = 0\nretries = 2\n"
        'command = "if [ -f /workspace/started ]; then touch /workspace/ready; '
        'else touch /workspace/started; false; fi"\n'
        '[verifier]\ntimeout_sec = 5\n[verifier.env]\nEXPECTED = "${HARBOR_TEST_EXPECTED}"\n'
        + (
            '[verifier.environment]\ndocker_image = "fixture-grader"\nallow_internet = false\n'
            'workdir = "/workspace"\n'
            if separate
            else ""
        )
    )
    (directory / "environment/Dockerfile").write_text("FROM busybox\nWORKDIR /workspace\n")
    (directory / "environment/public.bin").write_bytes(b"\x00\xff")
    (directory / "environment/public.bin").chmod(0o755)
    (directory / "setup_files/input").write_text("answer\n")
    grader = (
        GRADER.replace(b"/logs/artifacts/answer", b"/submission/answer").replace(
            b"; then", b" && [ ! -f /submission/ignored.tmp ]; then"
        )
        if selected_artifacts
        else GRADER
    )
    (directory / "tests/test.sh").write_bytes(grader)
    source = Source(dataset="harbor-fixture", revision="1", row="task", importer_revision="1")
    task = harbor_task(directory, source=source)
    task = TaskSpec.model_validate_json(task.model_dump_json())
    shutil.rmtree(directory)
    monkeypatch.setenv("HARBOR_TEST_EXPECTED", expected)
    machines = []
    build_directories = []

    class ImageFactory:
        """Replace container creation with a fixed ShellSim image at the machine boundary."""

        async def create(self, spec):
            if isinstance(spec.source, DockerfileSource):
                build_directories.append(spec.source.context)
                assert spec.source.dockerfile.read_text() == "FROM busybox\nWORKDIR /workspace\n"
                assert (spec.source.context / "public.bin").read_bytes() == b"\x00\xff"
                assert (spec.source.context / "public.bin").stat().st_mode & 0o777 == 0o755
                assert not (spec.source.context / "tests").exists()
            else:
                assert isinstance(spec.source, RegistryImage)
                assert spec.source.reference == "fixture-grader"
            machine = await ShellSimMachineFactory().create(replace(spec, source=ShellSimBuiltins()))
            if isinstance(spec.source, RegistryImage):
                script = tmp_path / "image-test.sh"
                script.write_bytes(grader)
                await machine.upload(script, "/tests/test.sh")
            machines.append(machine)
            return TarMachine(machine)

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
                            "arguments": json.dumps(
                                {
                                    "command": (
                                        "test -f /workspace/ready && test ! -f /tests/test.sh && "
                                        "cat /setup_files/input > /logs/artifacts/answer"
                                        + (
                                            " && mkdir -p /submission && cp /logs/artifacts/answer /submission/answer"
                                            " && echo private > /submission/ignored.tmp"
                                            if selected_artifacts
                                            else ""
                                        )
                                    )
                                }
                            ),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Completed."},
        ]
    )
    result = await run_task(engine(model, {EnvironmentKind.DOCKER: ImageFactory()}), task)
    assert (result.grade.status, result.grade.reward, result.grade.passed) == (Outcome.GRADED, reward, passed)
    assert result.grade.diagnostics["exit_code"] == 7
    assert result.response_token_ids == (20, 90, 91, 21)
    assert result.loss_mask == (1, 0, 0, 1)
    assert len(machines) == (2 if separate else 1)
    assert all(not directory.exists() for directory in build_directories)
    for machine in machines:
        with pytest.raises(RuntimeError, match="closed"):
            await machine.upload(tmp_path / "unused", "/unused")
    assert "HARBOR_TEST_EXPECTED" not in json.dumps([request.messages for request in model.requests])


def test_separate_verifier_can_reuse_the_agent_build_context(tmp_path):
    directory = tmp_path / "source"
    for name in ("environment", "tests", "setup_files"):
        (directory / name).mkdir(parents=True)
    (directory / "instruction.md").write_text("Complete the task.")
    (directory / "task.toml").write_text(
        '[environment]\nworkdir = "/workspace"\n' '[verifier]\nenvironment_mode = "separate"\ntimeout_sec = 5\n'
    )
    (directory / "environment/Dockerfile").write_text("FROM busybox\n")
    (directory / "tests/test.sh").write_text("echo 1\n")

    task = harbor_task(
        directory,
        source=Source(dataset="harbor-fixture", revision="1", row="task", importer_revision="1"),
    )
    specification = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)

    assert specification.environment is not None
    assert isinstance(specification.environment.image, DockerBuild)
    assert {file.path for file in specification.environment.image.files} == {"/Dockerfile"}
    assert {file.path for file in specification.environment.files} == {"/tests/test.sh"}


@pytest.mark.parametrize(
    "strategy,first_answer,last_grader,reward,status,stage_count",
    [
        ("mean", "first", "valid", 0.5, Outcome.GRADED, 2),
        ("final", "first", "valid", 0.75, Outcome.GRADED, 2),
        ("mean", "wrong", "valid", 0.0, Outcome.GRADED, 1),
        ("mean", "first", "broken", 0.25, Outcome.GRADED, 2),
        ("final", "first", "broken", None, Outcome.INFRA_ERROR, 2),
        ("mean", "first", "setup_failed", 0.25, Outcome.GRADED, 2),
        ("final", "first", "setup_failed", None, Outcome.UNAVAILABLE, 2),
    ],
)
async def test_harbor_stages_preserve_state_gates_and_exact_training_tokens(
    tmp_path, strategy, first_answer, last_grader, reward, status, stage_count
):
    directory = tmp_path / "source"
    for name in ("tests", "steps/first/tests", "steps/second/tests", "steps/second/workdir"):
        (directory / name).mkdir(parents=True)
    (directory / "task.toml").write_text(
        f'multi_step_reward_strategy = "{strategy}"\n'
        '[environment]\ndocker_image = "fixture"\nworkdir = "/workspace"\nallow_internet = false\n'
        "[agent]\ntimeout_sec = 5\n"
        '[[steps]]\nname = "first"\nmin_reward = {safety = 1}\n'
        '[[steps]]\nname = "second"\n'
        '[steps.healthcheck]\ncommand = "test -f /workspace/ready"\ninterval_sec = 0\nretries = 1\n'
    )
    (directory / "steps/first/instruction.md").write_text("Write first to /workspace/state.")
    (directory / "steps/second/instruction.md").write_text("Write second to /workspace/state.")
    (directory / "tests/helper.sh").write_text("private helper\n")
    (directory / "steps/first/tests/test.sh").write_text(
        'if [ "$(cat /workspace/state)" = first ]; then '
        'echo \'{"reward": 0.25, "safety": 1}\'; else echo \'{"reward": 0, "safety": 0}\'; fi '
        "> /logs/verifier/reward.json\n"
    )
    (directory / "steps/second/workdir/setup.sh").write_text(
        'test "$(cat /workspace/state)" = first && touch /workspace/ready\n'
        if last_grader != "setup_failed"
        else "exit 1\n"
    )
    (directory / "steps/second/tests/test.sh").write_text(
        'if [ "$(cat /workspace/state)" = second ]; then '
        "echo '{\"reward\": 0.75}'; else echo '{\"reward\": 0}'; fi > /logs/verifier/reward.json\n"
        if last_grader == "valid"
        else "echo broken > /logs/verifier/reward.json\n"
    )
    task = harbor_task(directory, source=Source(dataset="stages", revision="1", row="0", importer_revision="1"))
    task = TaskSpec.model_validate_json(task.model_dump_json())
    shutil.rmtree(directory)
    machines = []

    class Factory:
        async def create(self, spec):
            machine = await ShellSimMachineFactory().create(replace(spec, source=ShellSimBuiltins()))
            machines.append(machine)
            return machine

    actions = [
        f"test ! -f /tests/test.sh && echo {first_answer} > /workspace/state",
        "test ! -f /tests/test.sh && test ! -f /tests/helper.sh && test -f /workspace/ready "
        "&& echo second > /workspace/state",
    ]
    replies = []
    for index, action in enumerate(actions):
        replies.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"write-{index}",
                            "type": "function",
                            "function": {"name": "shell", "arguments": json.dumps({"command": action})},
                        }
                    ],
                },
                {"role": "assistant", "content": "Completed."},
            ]
        )
    model = ReplayModel(replies)
    runner = engine(model, {EnvironmentKind.DOCKER: Factory()})
    if last_grader == "setup_failed":
        with pytest.raises(RolloutInterrupted) as caught:
            await run_task(runner, task)
        result = caught.value.rollout
        assert caught.value.operation == RolloutOperation.PREPARE
        assert isinstance(caught.value.__cause__, RuntimeError)
    else:
        result = await run_task(runner, task)
    assert (result.grade.status, result.grade.reward) == (status, reward)
    assert len(result.grade.diagnostics["stages"]) == stage_count
    assert len(machines) == 1
    assert len(model.requests) == (2 if last_grader == "setup_failed" else stage_count * 2)
    assert "Write second" not in json.dumps(model.requests[0].messages)
    assert "private helper" not in json.dumps([request.messages for request in model.requests])
    if stage_count == 2 and last_grader != "setup_failed":
        assert model.requests[2].prefix_token_ids == (10, 11, 20, 90, 91, 21)
        assert model.requests[2].messages[-1]["content"] == "Write second to /workspace/state."
        assert result.response_token_ids == (20, 90, 91, 21, 90, 91, 22, 90, 91, 23)
        assert result.loss_mask == (
            (1, 0, 0, 1, 0, 0, 1, 0, 0, 1) if last_grader == "valid" else (1, 0, 0, 1, 0, 0, 0, 0, 0, 0)
        )
    else:
        assert result.response_token_ids == (20, 90, 91, 21)
        assert result.loss_mask == (1, 0, 0, 1)
    if status == Outcome.GRADED:
        assert sum(step.transition.reward for step in result.steps) == reward
        if strategy == "mean":
            assert result.grade.rewards == {
                "reward": reward,
                "safety": 0.0 if first_answer == "wrong" else 0.5 if last_grader == "valid" else 1.0,
            }
