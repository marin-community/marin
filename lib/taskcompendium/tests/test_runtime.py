# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable task evidence, reset behavior and curation replay."""

import asyncio
import json
import os
import sys
import tarfile
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from shellbox.machine import Backend, Command, DockerImage, ExitReason, MachineSpec, Result, UnsupportedMachineSpec
from verifyit.spec import ScriptSpec, render_spec, spec_from_table

from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.importers.nemo_predicted_action import canonical_sha256, import_row
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    DockerBuildContext,
    EnvironmentRequirements,
    FileReward,
    GradingAttempt,
    NoGrader,
    PlainText,
    ProviderRequirement,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import CheckStatus
from taskcompendium.pipeline.verification import verify_task
from taskcompendium.runtime.episode import ScriptedActor, run_episode
from taskcompendium.runtime.grading import SPEC_PATH, STAGING_ARCHIVE, grade_empty_in_sandbox, grade_in_sandbox
from taskcompendium.runtime.models import RuntimeEvidence, Termination, grading_attempt
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import BASH, CONTROL_PATH, INTERFACE, OUTPUT_PATH, ShellFactory
from taskcompendium.runtime.task_grading import grade_task

GRADER_IMAGE = "fixture@sha256:" + "a" * 64
GRADER_ENVIRONMENT = EnvironmentRequirements(docker_image=GRADER_IMAGE, compatible_backends=(Backend.DOCKER,))
GRADER_MACHINE = MachineSpec(DockerImage(GRADER_IMAGE))
EXITED = Result(0, b"", b"", False, False, ExitReason.EXITED)


@pytest.fixture
def shell_task():
    source = Source(dataset="mock/shell-files", revision="1", row="0", importer_revision="1")
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
        answer_format=PlainText(),
        grader=NoGrader(reason="Test overrides the grader where needed"),
    )


def finished(task: TaskSpec) -> ConversationTrace:
    """The conversation of an agent that worked in its machine and then stopped."""
    return ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="done")))


@dataclass
class FileMachine:
    """External machine boundary with uploaded files and bounded capture reads."""

    files: dict[str, bytes] = field(default_factory=dict)
    closed: bool = False

    async def run(self, command: Command):
        if command.argv[0] == "mkdir":
            return EXITED
        if command.argv[:2] == ("rm", "-f"):
            for path in command.argv[2:]:
                self.files.pop(path, None)
            return EXITED
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


@pytest.mark.parametrize("entrypoint", ["factory", "episode"])
def test_unresolved_actor_build_never_creates_a_machine(shell_task, entrypoint):
    requirements = shell_task.environment_requirements.model_copy(
        update={"docker_build": DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM mutable:latest"),))}
    )
    task = shell_task.model_copy(update={"environment_requirements": requirements})
    machines = FileMachines()
    factory = ShellFactory(machines, GRADER_MACHINE, {}, 10, 1024)
    with pytest.raises(UnsupportedMachineSpec):
        if entrypoint == "factory":
            asyncio.run(factory.create(task))
        else:
            asyncio.run(run_episode(task, ScriptedActor(()), factory, max_steps=1, control="fixture"))
    assert machines.machines == []


@pytest.mark.parametrize("entrypoint", ["grade", "empty", "no_machine_spec"])
def test_unresolved_grader_build_never_creates_a_machine(shell_task, entrypoint):
    environment = EnvironmentRequirements(
        docker_build=DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM mutable:latest"),))
    )
    task = shell_task.model_copy(
        update={"grader": ScriptGrader(argv=("true",), answer_path=None, environment=environment)}
    )
    machines = FileMachines()
    with pytest.raises(UnsupportedMachineSpec):
        if entrypoint == "grade":
            asyncio.run(grade_in_sandbox(task, GradingAttempt(finished(task)), machines, GRADER_MACHINE))
        elif entrypoint == "empty":
            asyncio.run(grade_empty_in_sandbox(task, machines, GRADER_MACHINE))
        else:
            grade_task(task, GradingAttempt(finished(task)))
    assert machines.machines == []


@dataclass(kw_only=True)
class LocalGradingMachine:
    """Run grading commands as host subprocesses under ``root``, translating absolute container paths."""

    root: Path
    workdir: str
    remove_exit_code: int = 0
    closed: bool = False

    def local(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def local_arguments(self, arguments) -> list[str]:
        return [str(self.local(value)) if value.startswith("/") else value for value in arguments]

    async def upload(self, source: Path, target: str):
        path = self.local(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())

    async def download(self, source: str, target: Path):
        target.write_bytes(self.local(source).read_bytes())

    async def close(self):
        self.closed = True

    async def run(self, command: Command):
        program = command.argv[0]
        if command.argv[:2] == ("rm", "-f"):
            if self.remove_exit_code and STAGING_ARCHIVE in command.argv:
                return Result(self.remove_exit_code, b"", b"cleanup failed", False, False, ExitReason.EXITED)
            for path in command.argv[2:]:
                self.local(path).unlink(missing_ok=True)
            return EXITED
        if command.argv[:2] == ("mkdir", "-p"):
            for path in command.argv[2:]:
                self.local(path).mkdir(parents=True, exist_ok=True)
            return EXITED
        if command.argv[:2] == ("tar", "-xf"):
            with tarfile.open(self.local(command.argv[2])) as archive:
                archive.extractall(self.root, filter="data")
            return EXITED
        argv = self.local_arguments(command.argv)
        env = {**os.environ, **command.env}
        if program == "python3":
            argv[0] = sys.executable
            env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
            if SPEC_PATH in command.argv:
                spec_path = self.local(SPEC_PATH)
                spec = spec_from_table(tomllib.loads(spec_path.read_text()))
                assert isinstance(spec, ScriptSpec)
                spec_path.write_text(render_spec(replace(spec, workspace=str(self.local("/app")))))
                argv.extend(("--logs-dir", str(self.local("/logs/verifier"))))
        else:
            assert program in {"bash", "sh"}, command.argv
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.local(command.cwd or self.workdir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
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
    package = verifyit_package(
        ScriptSpec(path="grader.py", verdict_file="verdict.json", timeout=60),
        (inline_resource("grader.py", script), inline_resource("config.json", b'{"expected": "42"}')),
        environment=GRADER_ENVIRONMENT,
    )
    task = shell_task.model_copy(
        update={
            "output_paths": ("/app/numbers.py",),
            "grader": package.grader,
            "resources": shell_task.resources.model_copy(update={"verifier": package.resources}),
        }
    )
    return TaskSpec.model_validate_json(task.model_dump_json())


@pytest.mark.parametrize("candidate,expected", [(b"print(42)\n", 1.0), (b"raise RuntimeError('candidate')\n", 0.0)])
@pytest.mark.asyncio
async def test_trusted_grader_bootstrap_excludes_submitted_stdlib_names(
    tmp_path, trusted_bootstrap_task, candidate, expected
):
    task = trusted_bootstrap_task
    attempt = GradingAttempt(finished(task), {"/app/numbers.py": candidate})
    result = await grade_in_sandbox(task, attempt, LocalGradingMachines(tmp_path), GRADER_MACHINE)
    assert (result.status, result.reward) == (Outcome.GRADED, expected), result.error


@pytest.mark.parametrize("remove_exit_code", [0, 1])
@pytest.mark.asyncio
async def test_private_fixture_archive_removed_before_grading_or_grader_never_runs(
    tmp_path, trusted_bootstrap_task, remove_exit_code
):
    machines = LocalGradingMachines(tmp_path, remove_exit_code=remove_exit_code)
    attempt = GradingAttempt(finished(trusted_bootstrap_task), {"/app/numbers.py": b"print(42)\n"})
    result = await grade_in_sandbox(trusted_bootstrap_task, attempt, machines, GRADER_MACHINE)
    archive = tmp_path / STAGING_ARCHIVE.lstrip("/")
    assert (tmp_path / "tests/config.json").is_file()
    assert machines.machines[0].closed
    if remove_exit_code:
        assert (result.status, result.reward, result.failure) == (Outcome.INFRA_ERROR, None, GradingFailure.EXECUTION)
        assert archive.is_file()
        assert not (tmp_path / "logs/verifier/verdict.json").exists()
    else:
        assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
        assert not archive.exists()


SCRIPT_GRADER = (
    b"#!/bin/bash\n"
    b"if [ -f numbers.py ]; then\n"
    b'  if [ "$(python3 numbers.py 2>/dev/null)" = 42 ]; then reward=1; else reward=0; fi\n'
    b'elif [ -f state.json ] && python3 -c \'import json; assert json.load(open("state.json")) == {"ready": True}\'\n'
    b"then reward=1\n"
    b"else reward=0\n"
    b"fi\n"
    b"printf '%s\\n' \"$reward\" > ../logs/verifier/reward.txt\n"
)


@pytest.fixture
def script_grading_task(trusted_bootstrap_task):
    grader = ScriptGrader(
        argv=("bash", "/tests/test.sh"),
        environment=GRADER_ENVIRONMENT,
        answer_path=None,
        reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),)),
        timeout=30,
    )
    task = trusted_bootstrap_task.model_copy(
        update={
            "grader": grader,
            "resources": trusted_bootstrap_task.resources.model_copy(
                update={"verifier": (inline_resource("test.sh", SCRIPT_GRADER),)}
            ),
        }
    )
    return TaskSpec.model_validate_json(task.model_dump_json())


@pytest.mark.parametrize("candidate,reward", [(b"print(42)\n", 1.0), (b"print(0)\n", 0.0)])
def test_script_grader_scores_captured_files_in_sandbox_and_task_grading(
    tmp_path, script_grading_task, candidate, reward
):
    factory = LocalGradingMachines(tmp_path)
    attempt = grading_attempt(finished(script_grading_task), RuntimeEvidence({"/app/numbers.py": candidate}, "{}"))
    sandbox = asyncio.run(grade_in_sandbox(script_grading_task, attempt, factory, GRADER_MACHINE))
    runtime = grade_task(script_grading_task, attempt, machine_factory=factory, machine_spec=GRADER_MACHINE)
    assert (sandbox.status, sandbox.reward) == (Outcome.GRADED, reward)
    assert (runtime.status, runtime.reward) == (Outcome.GRADED, reward)
    assert (tmp_path / "tests/test.sh").read_bytes() == SCRIPT_GRADER
    assert all(machine.closed for machine in factory.machines)


@pytest.mark.parametrize("state,reward", [('{"ready":true}', 1.0), ('{"ready":false}', 0.0)])
def test_script_grader_reads_captured_state(tmp_path, script_grading_task, state, reward):
    task = script_grading_task.model_copy(update={"answer_type": AnswerType.STATE, "output_paths": ()})
    result = grade_task(
        task,
        grading_attempt(finished(task), RuntimeEvidence({}, state)),
        machine_factory=LocalGradingMachines(tmp_path),
        machine_spec=GRADER_MACHINE,
    )
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


class UnavailableMachines:
    backend = Backend.DOCKER

    async def create(self, spec):
        raise OSError("machine unavailable")


def test_grading_machine_failure_is_infrastructure_error(script_grading_task):
    attempt = GradingAttempt(finished(script_grading_task), {"/app/numbers.py": b"print(42)\n"})
    result = grade_task(script_grading_task, attempt, machine_factory=UnavailableMachines(), machine_spec=GRADER_MACHINE)
    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None)


def test_no_grader_task_is_unavailable_without_a_grading_machine(shell_task):
    grader = shell_task.grader
    assert isinstance(grader, NoGrader)
    machines = FileMachines()
    attempt = GradingAttempt(finished(shell_task), {OUTPUT_PATH: b"person-0-0\n"})
    result = grade_task(shell_task, attempt, machine_factory=machines, machine_spec=GRADER_MACHINE)
    assert (result.status, result.reward, result.error) == (Outcome.UNAVAILABLE, None, grader.reason)
    assert not machines.machines


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_environment_failure_is_recorded_without_reward(script_grading_task):
    rollout = await run_episode(
        script_grading_task, ScriptedActor(()), UnavailableFactory(), max_steps=2, control="failure"
    )
    assert rollout.termination == Termination.INFRA_ERROR
    assert rollout.artifacts == ()
    assert rollout.detail == "Machine service unavailable"


def test_nemo_import_accepts_untyped_source_messages_and_checks_actions():
    path = Path(__file__).parent / "fixtures/nemo/predicted-action.json"
    data = json.loads(path.read_text())
    for item in data["responses_create_params"]["input"]:
        if item.get("type") == "message" and item["role"] in {"system", "user"}:
            del item["type"]
    task = import_row(data, expected_sha256=canonical_sha256(data))
    assert all(check.status == CheckStatus.PASS for check in verify_task(task))
    assert task.context.events[-1].content == data["responses_create_params"]["input"][-1]["content"]
