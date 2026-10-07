# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable task evidence, reset behavior and curation replay."""

import asyncio
import io
import json
import os
import sys
import tarfile
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from shellbox.machine import Backend, Command, DockerImage, ExitReason, MachineSpec, Result
from verifyit.spec import ScriptSpec, render_spec, spec_from_table

from taskcompendium.datasets import nemo_actions
from taskcompendium.datasets.direct_contracts import source_contract_package
from taskcompendium.grader import grader_package, native_command_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    ProviderRequirement,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.native_grader import NativeCommandSpec
from taskcompendium.pipeline.models import CheckStatus, RawRow
from taskcompendium.pipeline.verification import verify_task
from taskcompendium.runtime.episode import ScriptedActor, run_episode
from taskcompendium.runtime.grading import grade_submission
from taskcompendium.runtime.models import RuntimeEvidence, Termination
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import BASH, CONTROL_PATH, INTERFACE, OUTPUT_PATH, ShellFactory
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.submission import PlainText


@pytest.fixture
def shell_task():
    source = Source(dataset="mock/shell-files", revision="1", row="0", importer_revision="1")
    package = source_contract_package("test fixture", "1", {}, ("Test overrides the verifier where needed",))
    return TaskSpec(
        id="shell-0",
        source=source,
        context=ConversationInput(
            events=(TextMessage(role="user", content="Read people.csv and save the selected names."),)
        ),
        environment_requirements=EnvironmentRequirements(
            compatible_backends=(Backend.DOCKER,),
            capabilities=("shell", "filesystem"),
            tool_providers={"shell": ProviderRequirement(action_interface=INTERFACE, initial_state={})},
        ),
        interaction_tools=(BASH,),
        resources=ResourceGroups(
            worker=(inline_resource("workspace/people.csv", b"name,team\nperson-0-0,team-0\nperson-0-1,other\n"),),
            oracle=(inline_resource(CONTROL_PATH.lstrip("/"), b"#!/bin/sh\ntrue\n"),),
        ),
        output_paths=(OUTPUT_PATH,),
        answer_type=AnswerType.FILE,
        verifier=package.verifier,
    )


@dataclass
class FileMachine:
    """External machine boundary with uploaded files and bounded capture reads."""

    files: dict[str, bytes] = field(default_factory=dict)
    closed: bool = False

    async def run(self, command: Command):
        if command.argv[0] == "mkdir":
            return Result(0, b"", b"", False, False, ExitReason.EXITED)
        if command.argv[0] == "rm":
            if command.argv[1] not in self.files:
                return Result(1, b"", b"missing file", False, False, ExitReason.EXITED)
            del self.files[command.argv[1]]
            return Result(0, b"", b"", False, False, ExitReason.EXITED)
        path = command.argv[4]
        data = self.files.get(path)
        if data is None:
            return Result(44, b"", b"", False, False, ExitReason.EXITED)
        limit = command.output_limit_bytes
        return Result(0, data[:limit], b"", len(data) > limit, False, ExitReason.EXITED)

    async def upload(self, source: Path, target: str):
        self.files[target] = source.read_bytes()

    async def download(self, source: str, target: Path):
        target.write_bytes(self.files[source])

    async def close(self):
        self.closed = True


@dataclass
class FileMachines:
    backend = Backend.DOCKER
    machines: list[FileMachine] = field(default_factory=list)

    async def create(self, spec):
        machine = FileMachine()
        self.machines.append(machine)
        return machine


@dataclass(kw_only=True)
class LocalGradingMachine(FileMachine):
    """Translate container paths while running the real trusted grader subprocess."""

    root: Path
    workdir: str
    remove_exit_code: int = 0

    async def upload(self, source: Path, target: str):
        await super().upload(source, target)
        path = self.root / target.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())

    async def download(self, source: str, target: Path):
        target.write_bytes((self.root / source.lstrip("/")).read_bytes())

    async def run(self, command):
        if command.argv[0] == "rm":
            target = command.argv[-1]
            if self.remove_exit_code and target == "/tmp/taskcompendium-submission.tar":
                return Result(self.remove_exit_code, b"", b"cleanup failed", False, False, ExitReason.EXITED)
            (self.root / target.lstrip("/")).unlink(missing_ok="-f" in command.argv)
            self.files.pop(target, None)
            return Result(0, b"", b"", False, False, ExitReason.EXITED)
        if command.argv[0] == "mkdir":
            (self.root / command.argv[-1].lstrip("/")).mkdir(parents=True, exist_ok=True)
            return Result(0, b"", b"", False, False, ExitReason.EXITED)
        if command.argv[0] == "tar":
            with tarfile.open(fileobj=io.BytesIO(self.files[command.argv[2]])) as archive:
                for member in archive.getmembers():
                    stream = archive.extractfile(member)
                    assert stream is not None
                    path = self.root / member.name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(stream.read())
            return Result(0, b"", b"", False, False, ExitReason.EXITED)
        if command.argv[0] in {"bash", "test"}:
            argv = [str(self.root / value.lstrip("/")) if value.startswith("/") else value for value in command.argv]
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self.root / (command.cwd or self.workdir).lstrip("/"),
                env={**os.environ, **command.env},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return Result(process.returncode, stdout, stderr, False, False, ExitReason.EXITED)
        if command.argv[0] != "python3":
            return await super().run(command)
        argv = [str(self.root / value.lstrip("/")) if value.startswith("/") else value for value in command.argv[1:]]
        if "/tests/verifier.toml" in command.argv:
            spec_path = self.root / "tests/verifier.toml"
            spec = spec_from_table(tomllib.loads(spec_path.read_text()))
            assert isinstance(spec, ScriptSpec)
            spec_path.write_text(render_spec(replace(spec, workspace=str(self.root / "app"))))
            argv.extend(("--logs-dir", str(self.root / "logs/verifier")))
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            *argv,
            cwd=self.root / (command.cwd or self.workdir).lstrip("/"),
            env={**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path), **command.env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        verdict = self.root / "logs/verifier/verdict.json"
        if verdict.exists():
            self.files["/logs/verifier/verdict.json"] = verdict.read_bytes()
        return Result(process.returncode, stdout, stderr, False, False, ExitReason.EXITED)


@dataclass
class LocalGradingMachines:
    root: Path
    remove_exit_code: int = 0
    machines: list[LocalGradingMachine] = field(default_factory=list)
    backend = Backend.DOCKER

    async def create(self, spec):
        machine = LocalGradingMachine(root=self.root, workdir=spec.workdir, remove_exit_code=self.remove_exit_code)
        self.machines.append(machine)
        return machine


@pytest.fixture
def trusted_bootstrap_task(shell_task):
    script = (
        b"import json, os, subprocess, sys\n"
        b"from pathlib import Path\n"
        b"workspace = Path(os.environ['VERIFYIT_WORKSPACE'])\n"
        b"expected = json.loads((Path(os.environ['VERIFYIT_TESTS_DIR']) / 'config.json').read_text())['expected']\n"
        b"result = subprocess.run([sys.executable, str(workspace / 'numbers.py')], capture_output=True)\n"
        b"reward = float(result.returncode == 0 and result.stdout.strip() == expected.encode())\n"
        b"(Path(os.environ['VERIFYIT_LOGS_DIR']) / 'verdict.json').write_text(\n"
        b"    json.dumps({'status': 'scored', 'reward': reward, 'detail': {}}))\n"
    )
    package = grader_package(
        ScriptSpec(path="grader.py", verdict_file="verdict.json", timeout=60),
        (inline_resource("grader.py", script), inline_resource("config.json", b'{"expected": "42"}')),
    )
    task = shell_task
    image = "fixture@sha256:" + "a" * 64
    task = task.model_copy(
        update={
            "output_paths": ("/app/numbers.py",),
            "verifier": package.verifier.model_copy(
                update={
                    "environment_requirements": task.verifier.environment_requirements.model_copy(
                        update={"docker_image": image, "compatible_backends": (Backend.DOCKER,)}
                    )
                }
            ),
            "resources": task.resources.model_copy(update={"verifier": package.resources}),
        }
    )
    return task


@pytest.mark.parametrize("candidate,expected", [(b"print(42)\n", 1.0), (b"raise RuntimeError('candidate')\n", 0.0)])
async def test_trusted_grader_bootstrap_excludes_submitted_stdlib_names(
    tmp_path, trusted_bootstrap_task, candidate, expected
):
    task = trusted_bootstrap_task
    image = task.verifier.environment_requirements.docker_image
    assert image is not None
    result = await grade_submission(
        task,
        {"/app/numbers.py": candidate},
        LocalGradingMachines(tmp_path),
        machine_spec=MachineSpec(DockerImage(image)),
    )
    assert (result.status, result.reward) == (Outcome.GRADED, expected), result.error


@pytest.mark.parametrize("remove_exit_code", [0, 1])
async def test_private_fixture_archive_removed_before_grading_or_grader_never_runs(
    tmp_path, trusted_bootstrap_task, remove_exit_code
):
    image = trusted_bootstrap_task.verifier.environment_requirements.docker_image
    assert image is not None
    machines = LocalGradingMachines(tmp_path, remove_exit_code=remove_exit_code)
    result = await grade_submission(
        trusted_bootstrap_task,
        {"/app/numbers.py": b"print(42)\n"},
        machines,
        machine_spec=MachineSpec(DockerImage(image)),
    )
    machine = machines.machines[0]
    archive = tmp_path / "tmp/taskcompendium-submission.tar"
    assert (tmp_path / "tests/config.json").is_file()
    assert machine.closed
    if remove_exit_code:
        assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None)
        assert archive.is_file()
        assert not (tmp_path / "logs/verifier/verdict.json").exists()
    else:
        assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
        assert not archive.exists()
        assert "/tmp/taskcompendium-submission.tar" not in machine.files


class NativeUnavailableMachines:
    backend = Backend.DOCKER

    async def create(self, spec):
        raise OSError("machine unavailable")


NATIVE_SHELL_SCRIPT = (
    b"#!/bin/bash\n"
    b"if [ -f numbers.py ]; then\n"
    b'  if [ "$(python3 numbers.py 2>/dev/null)" = 42 ]; then reward=1; else reward=0; fi\n'
    b'elif [ -f state.json ] && [ "$(cat state.json)" = \'{"ready":true}\' ]; then reward=1\n'
    b"else reward=0\n"
    b"fi\n"
    b"printf '%s\\n' \"$reward\" > ../logs/verifier/reward.txt\n"
)


@pytest.fixture
def native_grading_task(trusted_bootstrap_task):
    package = native_command_package(
        NativeCommandSpec(
            argv=("bash", "/tests/test.sh"),
            cwd="/app",
            result_format="reward_file",
            result_path="/logs/verifier/reward.txt",
            timeout=30,
        ),
        (inline_resource("test.sh", NATIVE_SHELL_SCRIPT),),
    )
    task = trusted_bootstrap_task.model_copy(
        update={
            "verifier": package.verifier.model_copy(
                update={"environment_requirements": trusted_bootstrap_task.verifier.environment_requirements}
            ),
            "resources": trusted_bootstrap_task.resources.model_copy(update={"verifier": package.resources}),
        }
    )
    return TaskSpec.model_validate_json(task.model_dump_json())


def native_task_with_script(task: TaskSpec, script: bytes, *, result_format: str, result_path: str) -> TaskSpec:
    native = NativeCommandSpec.model_validate_json(task.verifier.parameters_json).model_copy(
        update={"result_format": result_format, "result_path": result_path}
    )
    return task.model_copy(
        update={
            "verifier": task.verifier.model_copy(update={"parameters_json": native.model_dump_json()}),
            "resources": task.resources.model_copy(update={"verifier": (inline_resource("test.sh", script),)}),
        }
    )


@pytest.mark.parametrize("candidate,reward", [(b"print(42)\n", 1.0), (b"print(0)\n", 0.0)])
def test_native_source_grader_is_shared_by_verification_and_runtime(tmp_path, native_grading_task, candidate, reward):
    factory = LocalGradingMachines(tmp_path)
    image = native_grading_task.verifier.environment_requirements.docker_image
    assert image is not None
    spec = MachineSpec(DockerImage(image))
    files = {"/app/numbers.py": candidate}
    verification = asyncio.run(grade_submission(native_grading_task, files, factory, machine_spec=spec))
    conversation = ConversationTrace(
        events=(*native_grading_task.context.events, TextMessage(role="assistant", content="done"))
    )
    runtime = grade_task(
        native_grading_task,
        PlainText(id="native-test"),
        conversation,
        RuntimeEvidence(files, "{}"),
        machine_factory=factory,
        machine_spec=spec,
    )
    assert (verification.status, verification.reward) == (Outcome.GRADED, reward)
    assert (runtime.status, runtime.reward) == (Outcome.GRADED, reward)
    assert (tmp_path / "tests/test.sh").read_bytes() == NATIVE_SHELL_SCRIPT
    assert all(machine.closed for machine in factory.machines)


def test_native_state_grader_reads_captured_state(tmp_path, native_grading_task):
    task = native_grading_task.model_copy(update={"answer_type": AnswerType.STATE, "output_paths": ()})
    factory = LocalGradingMachines(tmp_path)
    image = task.verifier.environment_requirements.docker_image
    assert image is not None
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="done")))
    result = grade_task(
        task,
        PlainText(id="native-state"),
        conversation,
        RuntimeEvidence({}, '{"ready":true}'),
        machine_factory=factory,
        machine_spec=MachineSpec(DockerImage(image)),
    )
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


async def test_native_source_grader_preserves_non_unit_reward(tmp_path, native_grading_task):
    task = native_task_with_script(
        native_grading_task,
        b"printf '2.5\\n' > ../logs/verifier/reward.txt\n",
        result_format="reward_file",
        result_path="/logs/verifier/reward.txt",
    )
    factory = LocalGradingMachines(tmp_path)
    image = native_grading_task.verifier.environment_requirements.docker_image
    assert image is not None
    result = await grade_submission(
        task,
        {"/app/numbers.py": b"print(42)\n"},
        factory,
        machine_spec=MachineSpec(DockerImage(image)),
    )
    assert (result.status, result.reward) == (Outcome.GRADED, 2.5)


async def test_native_source_grader_uses_written_reward_after_nonzero_exit(tmp_path, native_grading_task):
    task = native_task_with_script(
        native_grading_task,
        b"printf '0\\n' > ../logs/verifier/reward.txt\n"
        b"if [ ! -f numbers.py ]; then exit 1; fi\n"
        b"printf '1\\n' > ../logs/verifier/reward.txt\n",
        result_format="reward_file",
        result_path="/logs/verifier/reward.txt",
    )
    image = task.verifier.environment_requirements.docker_image
    assert image is not None
    result = await grade_submission(
        task,
        {},
        LocalGradingMachines(tmp_path),
        machine_spec=MachineSpec(DockerImage(image)),
    )
    assert (result.status, result.reward) == (Outcome.GRADED, 0.0)


@pytest.mark.parametrize(
    "reward_bytes,expected",
    [(b'{"reward":0.75}', 0.75), (b'{"reward":"bad"}', None), (b'{"reward":0.75,"detail":{}}', None)],
)
async def test_native_source_grader_reads_original_json_reward(tmp_path, native_grading_task, reward_bytes, expected):
    task = native_task_with_script(
        native_grading_task,
        b"printf '%s\\n' '" + reward_bytes + b"' > ../logs/verifier/reward.json\n",
        result_format="reward_json",
        result_path="/logs/verifier/reward.json",
    )
    factory = LocalGradingMachines(tmp_path)
    image = task.verifier.environment_requirements.docker_image
    assert image is not None
    result = await grade_submission(
        task,
        {"/app/numbers.py": b"print(42)\n"},
        factory,
        machine_spec=MachineSpec(DockerImage(image)),
    )
    assert (result.status, result.reward) == (
        (Outcome.GRADED, expected) if expected is not None else (Outcome.INVALID_TASK, None)
    )


async def test_native_source_grader_preserves_score_detail(tmp_path, native_grading_task):
    task = native_task_with_script(
        native_grading_task,
        b"""printf '%s\\n' '{"reward":0.75,"detail":{"cases":[{"passed":true}]}}' > ../logs/verifier/score.json
""",
        result_format="score_json",
        result_path="/logs/verifier/score.json",
    )
    image = task.verifier.environment_requirements.docker_image
    assert image is not None
    result = await grade_submission(
        task,
        {"/app/numbers.py": b"print(42)\n"},
        LocalGradingMachines(tmp_path),
        machine_spec=MachineSpec(DockerImage(image)),
    )
    assert (result.status, result.reward, result.detail) == (
        Outcome.GRADED,
        0.75,
        {"cases": [{"passed": True}]},
    )


@pytest.mark.parametrize(
    "script",
    [b"printf broken > ../logs/verifier/reward.txt\n", b"exit 2\n"],
)
async def test_native_source_grader_broken_contract_is_invalid_task(tmp_path, native_grading_task, script):
    task = native_task_with_script(
        native_grading_task,
        script,
        result_format="reward_file",
        result_path="/logs/verifier/reward.txt",
    )
    factory = LocalGradingMachines(tmp_path)
    image = native_grading_task.verifier.environment_requirements.docker_image
    assert image is not None
    result = await grade_submission(
        task,
        {"/app/numbers.py": b"print(42)\n"},
        factory,
        machine_spec=MachineSpec(DockerImage(image)),
    )
    assert (result.status, result.reward) == (Outcome.INVALID_TASK, None)


async def test_native_source_grader_backend_failure_is_infrastructure_error(native_grading_task):
    image = native_grading_task.verifier.environment_requirements.docker_image
    assert image is not None
    result = await grade_submission(
        native_grading_task,
        {"/app/numbers.py": b"print(42)\n"},
        NativeUnavailableMachines(),
        machine_spec=MachineSpec(DockerImage(image)),
    )
    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None)


async def test_shell_uploads_public_files_without_oracle_and_captures_submission(shell_task):
    task = shell_task
    machines = FileMachines()
    factory = ShellFactory(machines, MachineSpec(DockerImage("test")), {"backend": "file-machine"}, 1, 1024)
    env = await factory.create(task)
    assert "/workspace/people.csv" in machines.machines[0].files
    assert CONTROL_PATH not in machines.machines[0].files
    assert (await env.evidence()).files == {}
    machines.machines[0].files[OUTPUT_PATH] = b"person-0-0\n"
    assert (await env.evidence()).files[OUTPUT_PATH] == b"person-0-0\n"
    await env.close()
    fresh = await factory.create(task)
    assert (await fresh.evidence()).files == {}
    await fresh.close()
    assert all(machine.closed for machine in machines.machines)


@dataclass(frozen=True)
class UnavailableFactory:
    @property
    def identity(self):
        return {"backend": "unavailable"}

    async def create(self, task):
        raise RuntimeError("Machine service unavailable")


async def test_environment_failure_is_recorded_without_reward(native_grading_task):
    rollout = await run_episode(
        native_grading_task, ScriptedActor(()), UnavailableFactory(), max_steps=2, control="failure"
    )
    assert rollout.termination == Termination.INFRA_ERROR
    assert rollout.artifacts == ()
    assert rollout.detail == "Machine service unavailable"


def test_nemo_pipeline_normalizes_untyped_source_messages_and_checks_actions():
    path = Path(__file__).parent / "fixtures/nemo/predicted-action.json"
    data = json.loads(path.read_text())
    for item in data["responses_create_params"]["input"]:
        if item.get("type") == "message" and item["role"] in {"system", "user"}:
            del item["type"]
    source = Source(
        dataset="fixture/nemo",
        revision="1",
        row="0",
        importer_revision="1",
    )
    task = nemo_actions.normalize(RawRow("action-0", source, data))
    assert isinstance(task, TaskSpec)
    assert all(check.status == CheckStatus.PASS for check in verify_task(task))
    assert task.context.events[-1].content == data["responses_create_params"]["input"][-1]["content"]
