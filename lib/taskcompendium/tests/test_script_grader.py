# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sandbox graders score attempts in fresh ShellSim machines that stand in for a pinned grader image."""

import json
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
async def test_script_grader_reward_kinds(reward, script, expected):
    grader = ScriptGrader(argv=("sh", "-c", script), environment=GRADER_ENVIRONMENT, reward=reward)
    task = arithmetic_task(grader)

    result = await grade_in_sandbox(task, answered(task, "12"), FixtureImageFactory(), GRADER_MACHINE)

    assert (result.status, result.reward, result.passed, result.detail, result.failure) == expected, result.error


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
async def test_verifyit_grader_with_environment_runs_verifyit_cli_in_grading_machine(answer, reward):
    package = verifyit_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0), environment=GRADER_ENVIRONMENT)
    task = arithmetic_task(package.grader)

    result = await grade_in_sandbox(task, answered(task, answer), FixtureImageFactory(), GRADER_MACHINE)

    assert (result.status, result.reward) == (Outcome.GRADED, reward), result.error
