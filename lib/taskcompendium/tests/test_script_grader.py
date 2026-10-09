# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sandbox graders score attempts in fresh ShellSim machines that stand in for a pinned grader image."""

import asyncio
import errno
import json
import os
import re
import shutil
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import (
    Backend,
    Command,
    ExitReason,
    Machine,
    MachineSpec,
    RegistryImage,
    Result,
    ShellSimBuiltins,
)
from verifyit.grade import main as verifyit_main
from verifyit.spec import NumericSpec

import taskcompendium.runtime.grading as grading
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    ExitCodeReward,
    FileReward,
    Grader,
    GradingAttempt,
    JsonAnswer,
    MissingArtifactPolicy,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    StdoutReward,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
    VerifierCommand,
)
from taskcompendium.runtime.grading import grade_in_sandbox
from taskcompendium.runtime.resources import inline_resource

FIXTURE_IMAGE = "fixture@sha256:" + "0" * 64
GRADER_ENVIRONMENT = EnvironmentRequirements(docker_image=FIXTURE_IMAGE, compatible_backends=(Backend.DOCKER,))
GRADER_MACHINE = MachineSpec(RegistryImage(FIXTURE_IMAGE))
VERIFYIT_ENTRYPOINT = ("python3", "-c", "from verifyit.grade import main; raise SystemExit(main())")
REWARD_TXT = "/logs/verifier/reward.txt"
REWARD_JSON = "/logs/verifier/reward.json"


@dataclass
class VerifyitImageMachine:
    """A ShellSim machine whose image provides the verifyit CLI.

    ShellSim's Python lacks the standard library verifyit needs, so the CLI runs on the host against
    copies of the machine's ``/tests`` and workspace, and its verdict is written back.
    """

    machine: Machine

    async def run(self, command: Command) -> Result:
        if command.argv[:3] != VERIFYIT_ENTRYPOINT:
            return await self.machine.run(command)
        spec_path, workspace_flag, workspace = command.argv[3:]
        assert workspace_flag == "--workspace"
        spec = PurePosixPath(spec_path)
        with TemporaryDirectory() as directory:
            local = Path(directory)
            tests, local_workspace, logs = local / "tests", local / "workspace", local / "logs"
            tests.mkdir()
            local_workspace.mkdir()
            await self.machine.download(str(spec.parent), tests)
            await self.machine.download(workspace, local_workspace)
            code = verifyit_main([str(tests / spec.name), "--workspace", str(local_workspace), "--logs-dir", str(logs)])
            await self.machine.upload(logs / "verdict.json", "/logs/verifier/verdict.json")
        return Result(code, b"", b"", False, False, ExitReason.EXITED)

    async def upload(self, source: Path, target: str) -> None:
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        await self.machine.close()


class FixtureImageFactory:
    """Start the pinned fixture image as ShellSim built-ins."""

    backend = Backend.DOCKER

    async def create(self, spec: MachineSpec) -> VerifyitImageMachine:
        return VerifyitImageMachine(await ShellSimMachineFactory().create(replace(spec, source=ShellSimBuiltins())))


def arithmetic_task(grader: Grader) -> TaskSpec:
    return TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )


def answered(task: TaskSpec, answer: str) -> GradingAttempt:
    return GradingAttempt(
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer)))
    )


@pytest.fixture
def local_artifact_factory(tmp_path):
    class Machine:
        def __init__(self, root, spec, archive_members, archive_failure):
            self.root = root
            self.spec = spec
            self.archive_members = archive_members
            self.archive_failure = archive_failure
            self.closed = False
            (root / "workspace").mkdir(parents=True)
            (root / "tmp").mkdir()

        def path(self, value):
            assert value == "/" or value.split("/")[1] in {"workspace", "tests", "logs", "tmp"}
            return self.root / value.lstrip("/")

        async def run(self, command):
            if command.argv[0] == "tar" and self.archive_failure:
                if self.archive_failure == "tar_timeout":
                    return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
                return Result(
                    2,
                    b"",
                    b"Archive failed",
                    False,
                    False,
                    ExitReason.EXITED,
                )

            def rewrite(value):
                for path in re.findall(r"(?<![\w/%*])/[^\s'\";)}]*", value):
                    self.path(path)
                return re.sub(r"/(workspace|tests|logs|tmp)(?=/|$)", lambda match: str(self.path(match[0])), value)

            process = await asyncio.create_subprocess_exec(
                *(rewrite(value) for value in command.argv),
                cwd=self.path(command.cwd or "/workspace"),
                env={**os.environ, **self.spec.env, **command.env},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(command.stdin), timeout=command.timeout)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            limit = command.output_limit_bytes
            return Result(
                process.returncode,
                stdout[:limit],
                stderr[:limit],
                len(stdout) > limit,
                len(stderr) > limit,
                ExitReason.EXITED,
            )

        async def upload(self, source, target):
            destination = self.path(target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination, follow_symlinks=False)

        async def download(self, source, target):
            origin = self.path(source)
            if source.startswith("/tmp/taskcompendium-artifact-"):
                if self.archive_members:
                    with tarfile.open(origin, "w") as archive:
                        for member in self.archive_members:
                            archive.addfile(member)
            if origin.is_dir():
                shutil.copytree(origin, target, symlinks=True, dirs_exist_ok=True)
            else:
                shutil.copy2(origin, target, follow_symlinks=False)

        async def close(self):
            self.closed = True

    class Factory:
        def __init__(self, prepare_artifacts, *, archive_members=(), archive_failure=None):
            self.machines = []
            self.prepare_artifacts = prepare_artifacts
            self.archive_members = archive_members
            self.archive_failure = archive_failure

        async def create(self, spec):
            machine = Machine(tmp_path / str(len(self.machines)), spec, self.archive_members, self.archive_failure)
            self.machines.append(machine)
            if spec.env.get("ARTIFACT_TASK_MACHINE") == "1":
                self.prepare_artifacts(machine)
            return machine

    return Factory


@pytest.mark.parametrize(
    "artifact_case",
    [
        "directory",
        "file",
        "relative_private",
        "absolute_private",
        "in_tree",
        "hardlink",
        "directory_root",
        "file_root",
        "parent_link",
        "missing",
        "kind_mismatch",
        "excluded_link",
        "oversized_expanded",
        "too_many_members",
        "long_member",
        "conflicting_member",
        "directory_file_conflict",
        "file_directory_conflict",
        "host_enospc",
        "host_edquot",
        "host_eio",
        "host_emfile",
        "tar_timeout",
        "tar_failed",
    ],
)
async def test_artifact_transfer_grades_valid_files_and_rejects_invalid_submissions(
    local_artifact_factory, artifact_case, monkeypatch
):
    def prepare_artifacts(machine):
        if artifact_case == "missing":
            return
        artifacts = machine.path("/logs/artifacts")
        artifacts.mkdir(parents=True)
        answer = artifacts / "answer"
        if artifact_case in {"directory_root", "parent_link"}:
            artifacts.rename(artifacts.with_name("real"))
            artifacts.symlink_to("real", target_is_directory=True)
        if artifact_case in {"relative_private", "absolute_private"}:
            answer.symlink_to("../../tests/expected" if artifact_case == "relative_private" else "/tests/expected")
        elif artifact_case in {"in_tree", "hardlink", "file_root"}:
            submitted = artifacts / "submitted"
            submitted.write_bytes(b"secret")
            if artifact_case == "hardlink":
                os.link(submitted, answer)
            else:
                answer.symlink_to("submitted")
        else:
            answer.write_bytes(b"secret")
        if artifact_case == "excluded_link":
            (artifacts / "cache").mkdir()
            (artifacts / "cache/private").symlink_to("../../../tests/expected")
        if artifact_case == "too_many_members":
            for index in range(5):
                (artifacts / str(index)).touch()

    archive_members = []
    if artifact_case in {"conflicting_member", "file_directory_conflict", "directory_file_conflict"}:
        first = tarfile.TarInfo("file")
        if artifact_case == "directory_file_conflict":
            first.type = tarfile.DIRTYPE
        archive_members.append(first)
        second = tarfile.TarInfo("file/child" if artifact_case == "conflicting_member" else "file")
        if artifact_case == "file_directory_conflict":
            second.type = tarfile.DIRTYPE
        archive_members.append(second)
    elif artifact_case == "long_member":
        archive_members.append(tarfile.TarInfo("x" * 256))
    root = local_artifact_factory(
        prepare_artifacts,
        archive_members=archive_members,
        archive_failure=artifact_case if artifact_case in {"tar_failed", "tar_timeout"} else None,
    )
    if artifact_case == "oversized_expanded":
        monkeypatch.setattr(grading, "MAX_ARTIFACT_EXPANDED_BYTES", 1)
    if artifact_case == "too_many_members":
        monkeypatch.setattr(grading, "MAX_ARTIFACT_MEMBERS", 3)
    is_file = artifact_case in {"file", "file_root", "parent_link"}
    source = "/logs/artifacts/answer" if is_file else "/logs/artifacts"
    artifact = VerifierArtifact(
        source=source,
        target=source,
        kind=ArtifactKind.FILE if is_file or artifact_case == "kind_mismatch" else ArtifactKind.DIRECTORY,
        exclude=("cache",) if artifact_case == "excluded_link" else (),
    )
    grader = ScriptGrader(
        argv=("sh", "-c", "cmp /logs/artifacts/answer /tests/expected"),
        cwd="/workspace",
        environment=GRADER_ENVIRONMENT,
        answer_path=None,
        reward=ExitCodeReward(),
        artifacts=(artifact,),
    )
    task = arithmetic_task(grader).model_copy(
        update={"resources": ResourceGroups(verifier=(inline_resource("expected", b"secret"),))}
    )
    agent = await root.create(MachineSpec(RegistryImage(FIXTURE_IMAGE), env={"ARTIFACT_TASK_MACHINE": "1"}))
    host_errors = {
        "host_enospc": errno.ENOSPC,
        "host_edquot": errno.EDQUOT,
        "host_eio": errno.EIO,
        "host_emfile": errno.EMFILE,
    }
    if artifact_case in host_errors:

        def failed_extract(*args, **kwargs):
            raise OSError(host_errors[artifact_case], "Host artifact extraction failed")

        monkeypatch.setattr(tarfile.TarFile, "extract", failed_extract)
    try:
        if artifact_case in host_errors:
            with pytest.raises(OSError) as caught:
                await grade_in_sandbox(
                    task, answered(task, "12"), FixtureImageFactory(), GRADER_MACHINE, task_machine=agent
                )
            assert caught.value.errno == host_errors[artifact_case]
            return
        result = await grade_in_sandbox(
            task, answered(task, "12"), FixtureImageFactory(), GRADER_MACHINE, task_machine=agent
        )
        if artifact_case in {"tar_failed", "tar_timeout"}:
            assert (result.status, result.reward, result.failure) == (
                Outcome.INFRA_ERROR,
                None,
                GradingFailure.TIMEOUT if artifact_case == "tar_timeout" else GradingFailure.EXECUTION,
            )
        else:
            valid = artifact_case in {"directory", "file", "excluded_link"}
            assert (result.status, result.reward) == (
                (Outcome.GRADED, 1.0) if valid else (Outcome.SUBMISSION_FAILURE, 0.0)
            ), result.error
        assert not list(agent.path("/tmp").glob("taskcompendium-artifact-*"))
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_script_grader_collects_and_copies_task_machine_outputs_before_grading():
    agent = await ShellSimMachineFactory().create(MachineSpec(ShellSimBuiltins(), workdir="/app"))
    await agent.run(
        Command(("sh", "-c", "mkdir -p /app/out/nested && echo a > /app/out/a.txt && echo b > /app/out/nested/b.txt"))
    )
    report = (
        "import json, os, pathlib\n"
        "out = pathlib.Path('/grader/out')\n"
        "detail = {\n"
        "    'collected': open('/grader/collected.txt').read(),\n"
        "    'out': {str(p.relative_to(out)): p.read_text() for p in sorted(out.rglob('*')) if p.is_file()},\n"
        "    'missing': os.path.exists('/grader/missing'),\n"
        "}\n"
        "json.dump({'reward': 1, 'detail': detail}, open('/logs/verifier/reward.json', 'w'))\n"
    )
    grader = ScriptGrader(
        argv=("python3", "-c", report),
        environment=GRADER_ENVIRONMENT,
        collect=(VerifierCommand(argv=("sh", "-c", "echo collected > /app/collected.txt")),),
        artifacts=(
            VerifierArtifact(source="/app/collected.txt", target="/grader/collected.txt", kind=ArtifactKind.FILE),
            VerifierArtifact(source="/app/out", target="/grader/out", kind=ArtifactKind.DIRECTORY),
            VerifierArtifact(
                source="/app/missing",
                target="/grader/missing",
                kind=ArtifactKind.AUTO,
                missing=MissingArtifactPolicy.SKIP,
            ),
        ),
        reward=FileReward(files=(RewardFile(path=REWARD_JSON, format=RewardFileFormat.JSON),)),
    )
    task = arithmetic_task(grader)

    result = await grade_in_sandbox(
        task, answered(task, "12"), FixtureImageFactory(), GRADER_MACHINE, task_machine=agent
    )

    assert (result.status, result.reward) == (Outcome.GRADED, 1.0), result.error
    assert result.detail == {
        "collected": "collected\n",
        "out": {"a.txt": "a\n", "nested/b.txt": "b\n"},
        "missing": False,
    }
    collected = await agent.run(Command(("cat", "/app/collected.txt")))
    assert collected.stdout == b"collected\n"


@pytest.mark.asyncio
async def test_script_grader_reads_extracted_answer_and_conversation_at_declared_paths():
    report = (
        "import json\n"
        "detail = {\n"
        "    'answer': open('/submission/final.txt').read(),\n"
        "    'conversation': json.load(open('/tests/transcript.json')),\n"
        "}\n"
        "json.dump({'reward': 1, 'detail': detail}, open('/logs/verifier/reward.json', 'w'))\n"
    )
    grader = ScriptGrader(
        argv=("python3", "-c", report),
        environment=GRADER_ENVIRONMENT,
        answer_path="/submission/final.txt",
        conversation_path="/tests/transcript.json",
        reward=FileReward(files=(RewardFile(path=REWARD_JSON, format=RewardFileFormat.JSON),)),
    )
    task = arithmetic_task(grader).model_copy(update={"answer_format": JsonAnswer()})
    final = json.dumps({"answer": "12"})

    result = await grade_in_sandbox(task, answered(task, final), FixtureImageFactory(), GRADER_MACHINE)

    assert result.status == Outcome.GRADED, result.error
    assert result.detail == {
        "answer": "12",
        "conversation": [
            {"role": "user", "content": "What is six plus six?"},
            {"role": "assistant", "content": final},
        ],
    }


NUMBER_FILE = (RewardFile(path=REWARD_TXT, format=RewardFileFormat.NUMBER),)
JSON_FILE = (RewardFile(path=REWARD_JSON, format=RewardFileFormat.JSON),)
DETAIL = {"cases": [{"passed": True}]}


@pytest.mark.parametrize(
    "reward,script,expected",
    [
        pytest.param(StdoutReward(), "echo 0.25", (Outcome.GRADED, 0.25, None, None, None), id="stdout"),
        pytest.param(
            StdoutReward(),
            "echo 'loading scorer'; echo; echo 0.25; echo",
            (Outcome.GRADED, 0.25, None, None, None),
            id="stdout-chatter-before-reward",
        ),
        pytest.param(
            StdoutReward(),
            "echo 0.25; echo done",
            (Outcome.INFRA_ERROR, None, None, None, GradingFailure.INVALID_REWARD),
            id="stdout-chatter-after-reward",
        ),
        pytest.param(
            StdoutReward(),
            "echo high",
            (Outcome.INFRA_ERROR, None, None, None, GradingFailure.INVALID_REWARD),
            id="stdout-text",
        ),
        pytest.param(ExitCodeReward(), "exit 0", (Outcome.GRADED, 1.0, True, None, None), id="exit-success"),
        pytest.param(ExitCodeReward(), "exit 3", (Outcome.GRADED, 0.0, False, None, None), id="exit-failure"),
        pytest.param(
            FileReward(files=NUMBER_FILE, pass_above=0.4),
            f"echo 2.5 > {REWARD_TXT}",
            (Outcome.GRADED, 2.5, True, None, None),
            id="number-file-non-unit",
        ),
        pytest.param(
            FileReward(files=NUMBER_FILE),
            f"echo 0 > {REWARD_TXT}; exit 1",
            (Outcome.GRADED, 0.0, None, None, None),
            id="number-file-after-failed-exit",
        ),
        pytest.param(
            FileReward(files=JSON_FILE),
            f"printf '%s' '{json.dumps({'reward': 0.75, 'detail': DETAIL})}' > {REWARD_JSON}",
            (Outcome.GRADED, 0.75, None, DETAIL, None),
            id="json-file-detail",
        ),
        pytest.param(
            FileReward(files=(*JSON_FILE, *NUMBER_FILE)),
            f"echo 0.5 > {REWARD_TXT}",
            (Outcome.GRADED, 0.5, None, None, None),
            id="first-existing-file",
        ),
        pytest.param(
            FileReward(files=NUMBER_FILE),
            "exit 2",
            (Outcome.INFRA_ERROR, None, None, None, GradingFailure.MISSING_REWARD),
            id="missing-file",
        ),
        pytest.param(
            FileReward(files=NUMBER_FILE),
            f"printf '' > {REWARD_TXT}",
            (Outcome.INFRA_ERROR, None, None, None, GradingFailure.EMPTY_REWARD),
            id="empty-file",
        ),
        pytest.param(
            FileReward(files=NUMBER_FILE),
            f"printf broken > {REWARD_TXT}",
            (Outcome.INFRA_ERROR, None, None, None, GradingFailure.INVALID_REWARD),
            id="invalid-number",
        ),
        pytest.param(
            FileReward(files=JSON_FILE),
            f"""printf '%s' '{{"reward": "bad"}}' > {REWARD_JSON}""",
            (Outcome.INFRA_ERROR, None, None, None, GradingFailure.INVALID_REWARD),
            id="invalid-json-reward",
        ),
        pytest.param(
            FileReward(files=JSON_FILE),
            f"""printf '%s' '{{"reward": "0.75"}}' > {REWARD_JSON}""",
            (Outcome.INFRA_ERROR, None, None, None, GradingFailure.INVALID_REWARD),
            id="numeric-string-json-reward",
        ),
    ],
)
@pytest.mark.asyncio
async def test_script_grader_reward_kinds(reward, script, expected):
    grader = ScriptGrader(argv=("sh", "-c", script), environment=GRADER_ENVIRONMENT, reward=reward)
    task = arithmetic_task(grader)

    result = await grade_in_sandbox(task, answered(task, "12"), FixtureImageFactory(), GRADER_MACHINE)

    assert (result.status, result.reward, result.passed, result.detail, result.failure) == expected, result.error


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
@pytest.mark.asyncio
async def test_verifyit_grader_with_environment_runs_verifyit_cli_in_grading_machine(answer, reward):
    package = verifyit_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0), environment=GRADER_ENVIRONMENT)
    task = arithmetic_task(package.grader)

    result = await grade_in_sandbox(task, answered(task, answer), FixtureImageFactory(), GRADER_MACHINE)

    assert (result.status, result.reward) == (Outcome.GRADED, reward), result.error
