# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade an attempt in a fresh machine with the verifyit command or a grader script.

Grader inputs cross into the grading machine as one archive: verifier resources under
``/tests``, shared and worker resources, captured agent files, the extracted answer, the captured
state, the conversation, and copied artifacts. The agent's own machine never grades.
"""

import io
import json
import math
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from harbor_config.env import resolve_env_vars
from shellbox.machine import Command, ExitReason, Machine, MachineFactory, MachineSpec, Result
from verifyit.spec import Spec, render_spec

from taskcompendium.grading import answer_submission, parse_grade_result
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    ActionSubmission,
    AnswerType,
    ArtifactKind,
    ConversationTrace,
    EnvironmentRequirements,
    ExitCodeReward,
    FileReward,
    GradingAttempt,
    JsonSubmission,
    MissingArtifactPolicy,
    RewardFileFormat,
    ScriptGrader,
    Submission,
    SubmissionFailure,
    TaskResource,
    TaskSpec,
    TextMessage,
    TextSubmission,
    VerifierArtifact,
    VerifierCommand,
    VerifyitGrader,
    grader_workspace,
    under_grader_root,
    verifyit_answer_file,
    verifyit_spec,
)
from taskcompendium.runtime.output_capture import selected_directory_files, validate_output_directories
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.shell import MISSING_CAPTURE_EXIT_CODE, require_environment_source
from taskcompendium.submission import conversation_messages, require_submission_compatibility

GRADING_TIMEOUT = 600.0
SPEC_PATH = "/tests/verifier.toml"
VERDICT_PATH = "/logs/verifier/verdict.json"
STATE_PATH = "/app/state.json"
STAGING_ARCHIVE = "/tmp/taskcompendium-grading.tar"
DIAGNOSTIC_OUTPUT_BYTES = 16_384
ROOT = "0"


class _StepFailed(Exception):
    """A grading step failed; ``result`` reports it."""

    def __init__(self, result: GradeResult):
        super().__init__(result.error)
        self.result = result


@dataclass(frozen=True)
class _StagedFile:
    path: str
    data: bytes
    mode: int = 0o644
    mtime_ns: int | None = None


def _verifyit_answer_bytes(submission: Submission) -> bytes:
    """A text or JSON submission as the verifyit spec's answer file."""
    match submission:
        case TextSubmission(value=value):
            return value.encode()
        case JsonSubmission(value=value):
            return json.dumps(value, allow_nan=False).encode()
    raise TypeError(f"A {type(submission).__name__} cannot be written as an answer file")


def _script_answer_bytes(submission: Submission) -> bytes:
    """The answer file a script grader reads: a verifyit answer file, or a native action's final message as JSON."""
    if isinstance(submission, ActionSubmission):
        return submission.message.model_dump_json().encode()
    return _verifyit_answer_bytes(submission)


def _captured_files(task: TaskSpec, attempt: GradingAttempt, paths: tuple[str, ...]) -> list[_StagedFile]:
    selected = {path: attempt.files[path] for path in paths if path in attempt.files}
    for selection in task.output_directories:
        for path, data in selected_directory_files(selection, attempt.files).items():
            selected.setdefault(path, data)
    for path in selected:
        candidate = PurePosixPath(path)
        if not candidate.is_absolute() or ".." in candidate.parts or under_grader_root(path):
            raise ValueError(f"Captured file cannot be staged for grading: {path}")
    return [_StagedFile(path, data) for path, data in selected.items()]


def _resource_files(resources: tuple[TaskResource, ...], root: str) -> list[_StagedFile]:
    return [
        _StagedFile(
            f"{root}/{resource.path}",
            resource_bytes(resource),
            0o644 if resource.mode is None else int(resource.mode, 8),
            resource.mtime_ns,
        )
        for resource in resources
    ]


def _archive_member(path: str, *, size: int = 0, mode: int = 0o644, mtime: int = 0) -> tarfile.TarInfo:
    member = tarfile.TarInfo(path.removeprefix("/"))
    member.size = size
    member.mode = mode
    member.mtime = mtime
    return member


def _write_archive(archive_path: Path, files: list[_StagedFile], artifacts: list[tuple[Path, str]]) -> None:
    """Write files and downloaded artifacts; integer timestamps avoid PAX headers where possible."""
    with tarfile.open(archive_path, "w") as archive:
        for file in files:
            member = _archive_member(file.path, size=len(file.data), mode=file.mode)
            if file.mtime_ns is not None:
                seconds, nanos = divmod(file.mtime_ns, 1_000_000_000)
                member.mtime = seconds
                member.pax_headers = {"mtime": f"{seconds}.{nanos:09d}"}
            archive.addfile(member, io.BytesIO(file.data))
        for local, target in artifacts:
            for path in [local, *sorted(local.rglob("*"))] if local.is_dir() else [local]:
                name = target if path == local else f"{target}/{path.relative_to(local).as_posix()}"
                info = archive.gettarinfo(str(path), name.removeprefix("/"))
                info.mtime, info.uid, info.gid, info.uname, info.gname = int(info.mtime), 0, 0, "", ""
                if info.isfile():
                    with path.open("rb") as data:
                        archive.addfile(info, data)
                else:
                    archive.addfile(info)


async def _run_checked(machine: Machine, command: Command, failure: str) -> None:
    result = await machine.run(command)
    if result.reason == ExitReason.TIMED_OUT:
        raise _StepFailed(_infra_error(f"{failure}: timed out", GradingFailure.TIMEOUT, _diagnostics(result)))
    if result.exit_code != 0:
        raise _StepFailed(
            _infra_error(f"{failure}: exit={result.exit_code}", GradingFailure.EXECUTION, _diagnostics(result))
        )


async def _download_artifact(machine: Machine, artifact: VerifierArtifact, target: Path, timeout: float) -> bool:
    """Download an artifact. Return false only when its missing-file policy permits omission."""
    kind = artifact.kind
    if kind == ArtifactKind.AUTO or artifact.missing == MissingArtifactPolicy.SKIP:
        result = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    'if [ -d "$1" ]; then printf directory; elif [ -f "$1" ]; then printf file; '
                    f"else exit {MISSING_CAPTURE_EXIT_CODE}; fi",
                    "artifact-kind",
                    artifact.source,
                ),
                timeout=timeout,
                user=ROOT,
            )
        )
        if result.exit_code == MISSING_CAPTURE_EXIT_CODE and artifact.missing == MissingArtifactPolicy.SKIP:
            return False
        if result.exit_code != 0:
            raise _StepFailed(
                _infra_error(
                    f"Cannot inspect grading artifact {artifact.source}: exit={result.exit_code}",
                    GradingFailure.EXECUTION,
                    _diagnostics(result),
                )
            )
        kind = ArtifactKind(result.stdout.decode())
    if kind == ArtifactKind.DIRECTORY:
        target.mkdir()
    if not artifact.exclude or kind != ArtifactKind.DIRECTORY:
        await machine.download(artifact.source, target)
        return True
    remote_archive = f"/tmp/taskcompendium-artifact-{uuid4().hex}.tar"
    await _run_checked(
        machine,
        Command(
            (
                "tar",
                "-cf",
                remote_archive,
                *(f"--exclude={pattern}" for pattern in artifact.exclude),
                "-C",
                artifact.source,
                ".",
            ),
            timeout=timeout,
            user=ROOT,
        ),
        f"Cannot archive grading artifact {artifact.source}",
    )
    archive_path = target.with_suffix(".tar")
    await machine.download(remote_archive, archive_path)
    await _run_checked(
        machine, Command(("rm", "-f", remote_archive), timeout=timeout, user=ROOT), "Cannot remove artifact archive"
    )
    with tarfile.open(archive_path) as archive:
        archive.extractall(target, filter="data")
    return True


def _diagnostics(result: Result) -> dict[str, Any]:
    return {
        "stdout": result.stdout.decode(errors="replace"),
        "stderr": result.stderr.decode(errors="replace"),
        "exit_code": result.exit_code,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
    }


def _infra_error(message: str, failure: GradingFailure, diagnostics: dict[str, Any]) -> GradeResult:
    return GradeResult(Outcome.INFRA_ERROR, None, message, diagnostics=diagnostics, failure=failure)


def _reward_value(data: bytes, file_format: RewardFileFormat, key: str) -> tuple[float, dict | None]:
    detail = None
    if file_format == RewardFileFormat.JSON:
        document = json.loads(data)
        if not isinstance(document, dict):
            raise ValueError("A JSON reward file must hold an object")
        value = document[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("A JSON reward must be a number")
        detail = document.get("detail")
        if detail is not None and not isinstance(detail, dict):
            raise ValueError("A JSON reward detail must be an object")
    else:
        value = data.decode()
    reward = float(value)
    if not math.isfinite(reward):
        raise ValueError("Nonfinite reward")
    return reward, detail


async def _file_reward(machine: Machine, reward: FileReward, timeout: float, diagnostics: dict[str, Any]) -> GradeResult:
    for file in reward.files:
        result = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    f'if [ -f "$1" ]; then cat "$1"; else exit {MISSING_CAPTURE_EXIT_CODE}; fi',
                    "reward-file",
                    file.path,
                ),
                timeout=timeout,
            )
        )
        if result.exit_code == MISSING_CAPTURE_EXIT_CODE:
            continue
        if result.exit_code != 0 or result.stdout_truncated:
            return _infra_error(f"Cannot read reward file: {file.path}", GradingFailure.EXECUTION, diagnostics)
        if not result.stdout.strip():
            return _infra_error(f"Empty reward file: {file.path}", GradingFailure.EMPTY_REWARD, diagnostics)
        try:
            value, detail = _reward_value(result.stdout, file.format, file.key)
        except (UnicodeError, ValueError, TypeError, KeyError) as error:
            return _infra_error(f"Invalid reward file {file.path}: {error}", GradingFailure.INVALID_REWARD, diagnostics)
        passed = None if reward.pass_above is None else value > reward.pass_above
        return GradeResult(Outcome.GRADED, value, detail=detail, passed=passed, diagnostics=diagnostics)
    return _infra_error("Grader did not write a reward file", GradingFailure.MISSING_REWARD, diagnostics)


async def _script_reward(machine: Machine, grader: ScriptGrader, timeout: float) -> GradeResult:
    if isinstance(grader.reward, FileReward):
        paths = tuple(file.path for file in grader.reward.files)
        directories = tuple(sorted({str(PurePosixPath(path).parent) for path in paths}))
        for argv in (("mkdir", "-p", *directories), ("rm", "-f", *paths)):
            await _run_checked(machine, Command(argv, timeout=timeout), "Cannot prepare reward files")
    result = await machine.run(
        Command(grader.argv, cwd=grader.cwd, env=grader.env, timeout=timeout, output_limit_bytes=DIAGNOSTIC_OUTPUT_BYTES)
    )
    diagnostics = _diagnostics(result)
    if result.reason == ExitReason.TIMED_OUT:
        return _infra_error("Grader command timed out", GradingFailure.TIMEOUT, diagnostics)
    if isinstance(grader.reward, ExitCodeReward):
        passed = result.exit_code == 0
        return GradeResult(Outcome.GRADED, float(passed), passed=passed, diagnostics=diagnostics)
    if isinstance(grader.reward, FileReward):
        return await _file_reward(machine, grader.reward, timeout, diagnostics)
    if result.exit_code != 0:
        return _infra_error(
            f"Grader command failed: {result.reason}, exit={result.exit_code}", GradingFailure.EXECUTION, diagnostics
        )
    lines = [line.strip() for line in result.stdout.decode(errors="replace").splitlines() if line.strip()]
    try:
        reward = float(lines[-1])
    except (IndexError, ValueError):
        return _infra_error(
            "The last line of grader stdout must be a finite numeric reward", GradingFailure.INVALID_REWARD, diagnostics
        )
    if not math.isfinite(reward):
        return _infra_error("Grader returned a nonfinite reward", GradingFailure.INVALID_REWARD, diagnostics)
    return GradeResult(Outcome.GRADED, reward, diagnostics=diagnostics)


async def _verifyit_reward(machine: Machine, spec: Spec, workspace: str, timeout: float) -> GradeResult:
    result = await machine.run(
        Command(
            (
                "python3",
                "-c",
                "from verifyit.grade import main; raise SystemExit(main())",
                SPEC_PATH,
                "--workspace",
                workspace,
            ),
            # Bootstrap trusted imports outside submitted Python modules.
            cwd="/",
            timeout=timeout,
            output_limit_bytes=DIAGNOSTIC_OUTPUT_BYTES,
        )
    )
    diagnostics = _diagnostics(result)
    if result.reason == ExitReason.TIMED_OUT:
        return _infra_error("Verifier command timed out", GradingFailure.TIMEOUT, diagnostics)
    if result.exit_code != 0:
        return _infra_error(
            f"Verifier command failed (exit_code={result.exit_code}): {diagnostics['stderr']}",
            GradingFailure.EXECUTION,
            diagnostics,
        )
    with TemporaryDirectory(prefix="taskcompendium-verdict-") as directory:
        verdict = Path(directory) / "verdict.json"
        await machine.download(VERDICT_PATH, verdict)
        return parse_grade_result(spec, verdict.read_bytes())


@dataclass(frozen=True)
class _SandboxGrading:
    """A sandbox grader with its spec, image, working directory, time limit, and inputs from the task machine."""

    grader: VerifyitGrader | ScriptGrader
    spec: Spec | None
    environment: EnvironmentRequirements
    workspace: str
    limit: float
    collect: tuple[VerifierCommand, ...]
    artifacts: tuple[VerifierArtifact, ...]


def _sandbox_grading(
    task: TaskSpec, machine_spec: MachineSpec, task_machine: Machine | None, timeout: float | None
) -> _SandboxGrading:
    """The task's sandbox grader, checked against the grading machine and the task machine it collects from."""
    grader = task.grader
    if isinstance(grader, VerifyitGrader) and grader.environment is not None:
        limit = GRADING_TIMEOUT if timeout is None else timeout
        grading = _SandboxGrading(
            grader, verifyit_spec(grader), grader.environment, grader_workspace(grader), limit, (), ()
        )
    elif isinstance(grader, ScriptGrader):
        limit = grader.timeout if timeout is None else min(timeout, grader.timeout)
        grading = _SandboxGrading(grader, None, grader.environment, grader.cwd, limit, grader.collect, grader.artifacts)
    else:
        raise TypeError(
            f"Sandbox grading requires a verifyit grader with an environment or a script grader, not {grader.kind}"
        )
    if (grading.collect or grading.artifacts) and task_machine is None:
        raise ValueError("Collecting grader inputs requires the task machine")
    require_environment_source(machine_spec, grading.environment)
    validate_output_directories(task.output_directories, grading.workspace)
    return grading


def _grading_files(
    task: TaskSpec, attempt: GradingAttempt, grading: _SandboxGrading, submissions: list[_StagedFile]
) -> list[_StagedFile]:
    """The archive entries for one attempt, in install order."""
    # Later archive entries replace earlier ones, so agent files replace the resources they edit.
    files = [
        *_resource_files(task.resources.verifier, "/tests"),
        *_resource_files(task.resources.all + task.resources.worker, ""),
        *submissions,
    ]
    if attempt.state is not None and all(file.path != STATE_PATH for file in submissions):
        files.append(_StagedFile(STATE_PATH, json.dumps(attempt.state.value, allow_nan=False).encode()))
    if grading.spec is not None:
        files.append(_StagedFile(SPEC_PATH, render_spec(grading.spec).encode()))
    if isinstance(grading.grader, ScriptGrader):
        messages = conversation_messages(attempt.conversation.events)
        files.append(_StagedFile(grading.grader.conversation_path, json.dumps(messages, allow_nan=False).encode()))
    return files


async def grade_in_sandbox(
    task: TaskSpec,
    attempt: GradingAttempt,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    *,
    task_machine: Machine | None = None,
    timeout: float | None = None,
) -> GradeResult:
    """Grade with a verifyit grader that has an environment, or with a script grader.

    The grading machine comes from ``factory`` with ``machine_spec``, its working directory set to
    the grader's workspace and the environment's variables added. ``task_machine`` is the agent's
    machine; graders that collect inputs or copy artifacts require it. Machine failures propagate.
    """
    grading = _sandbox_grading(task, machine_spec, task_machine, timeout)
    answer_file = None if grading.spec is None else verifyit_answer_file(grading.spec)
    submissions = _captured_files(task, attempt, (*task.output_paths, *([answer_file] if answer_file else [])))
    try:
        if grading.spec is not None and task.answer_type in CONVERSATION_ANSWERS:
            require_submission_compatibility(task)
            assert answer_file is not None
            submissions.append(_StagedFile(answer_file, _verifyit_answer_bytes(answer_submission(task, attempt))))
        elif isinstance(grading.grader, ScriptGrader) and grading.grader.answer_path is not None:
            answer = _script_answer_bytes(answer_submission(task, attempt))
            submissions.append(_StagedFile(grading.grader.answer_path, answer))
    except SubmissionFailure as error:
        return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
    if (
        grading.spec is not None
        and not submissions
        and not (task.answer_type == AnswerType.STATE and attempt.state is not None)
    ):
        return GradeResult(Outcome.GRADED, 0.0, "Missing submission")
    return await _grade_staged(task, attempt, grading, submissions, factory, machine_spec, task_machine)


async def grade_empty_in_sandbox(
    task: TaskSpec, factory: MachineFactory, machine_spec: MachineSpec, *, timeout: float | None = None
) -> GradeResult:
    """Grade an agent that submitted nothing, staged in a fresh machine as ``grade_in_sandbox`` stages a rollout.

    The grader receives the task's resources, its conversation ending in an empty assistant reply,
    an empty file where it reads an answer file, and no workspace files. ``grade_in_sandbox`` rejects
    an empty reply or an empty verifyit workspace before a machine starts; this runs the grader on
    them, so its result shows whether the grader runs and what it awards an empty answer. Graders
    that collect inputs from the task machine are unsupported. Machine failures propagate.
    """
    grading = _sandbox_grading(task, machine_spec, None, timeout)
    if isinstance(grading.grader, ScriptGrader):
        answer_file = grading.grader.answer_path
    else:
        assert grading.spec is not None
        answer_file = verifyit_answer_file(grading.spec)
    attempt = GradingAttempt(ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=""))))
    submissions = [] if answer_file is None else [_StagedFile(answer_file, b"")]
    return await _grade_staged(task, attempt, grading, submissions, factory, machine_spec, None)


async def _grade_staged(
    task: TaskSpec,
    attempt: GradingAttempt,
    grading: _SandboxGrading,
    submissions: list[_StagedFile],
    factory: MachineFactory,
    machine_spec: MachineSpec,
    task_machine: Machine | None,
) -> GradeResult:
    """Stage the submissions with the task's other grader inputs in a fresh machine and grade them there."""
    files = _grading_files(task, attempt, grading, submissions)
    try:
        for command in grading.collect:
            assert task_machine is not None
            await _run_checked(
                task_machine,
                Command(
                    command.argv, cwd=command.cwd, env=resolve_env_vars(command.env), timeout=grading.limit, user=ROOT
                ),
                "Cannot collect grading inputs",
            )
        with TemporaryDirectory(prefix="taskcompendium-grading-") as directory:
            root = Path(directory)
            downloaded = []
            for index, artifact in enumerate(grading.artifacts):
                assert task_machine is not None
                local = root / f"artifact-{index}"
                if await _download_artifact(task_machine, artifact, local, grading.limit):
                    downloaded.append((local, artifact.target))
            archive_path = root / "grading.tar"
            _write_archive(archive_path, files, downloaded)
            machine = await factory.create(
                replace(
                    machine_spec,
                    workdir=grading.workspace,
                    env={**machine_spec.env, **resolve_env_vars(grading.environment.environment_variables)},
                )
            )
            try:
                return await _grade_on(
                    machine,
                    archive_path,
                    grading.workspace,
                    grading.environment,
                    grading.spec,
                    grading.grader,
                    grading.limit,
                )
            finally:
                await machine.close()
    except _StepFailed as failure:
        return failure.result


async def _grade_on(
    machine: Machine,
    archive_path: Path,
    workspace: str,
    environment: EnvironmentRequirements,
    spec: Spec | None,
    grader: VerifyitGrader | ScriptGrader,
    timeout: float,
) -> GradeResult:
    await machine.upload(archive_path, STAGING_ARCHIVE)
    await _run_checked(
        machine,
        Command(("tar", "-xf", STAGING_ARCHIVE, "-C", "/"), cwd="/", timeout=timeout, user=ROOT),
        "Could not unpack grading inputs",
    )
    # The archive holds references and tests; protecting /tests does not protect a copy in /tmp.
    await _run_checked(
        machine,
        Command(("rm", "-f", STAGING_ARCHIVE), cwd="/", timeout=timeout, user=ROOT),
        "Could not remove the grading archive",
    )
    await _run_checked(
        machine,
        Command(("mkdir", "-p", workspace), cwd="/", timeout=timeout),
        "Could not initialize the grading workspace",
    )
    for setup in environment.setup_commands:
        await _run_checked(
            machine,
            Command(("sh", "-c", setup), cwd="/", timeout=timeout, user=ROOT),
            "Grading environment setup failed",
        )
    if spec is not None:
        return await _verifyit_reward(machine, spec, workspace, timeout)
    assert isinstance(grader, ScriptGrader)
    return await _script_reward(machine, grader, timeout)
