# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private grader execution and reward collection."""

import asyncio
import errno
import json
import math
import tarfile
from collections.abc import Mapping
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from harbor_config.env import resolve_env_vars
from shellbox.machine import (
    DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    Command,
    DownloadLimitExceeded,
    ExitReason,
    Machine,
    MachineFactory,
)
from taskcompendium.chat import chat_conversation
from taskcompendium.grading import parse_grade_result
from taskcompendium.grading_contract import GradingAttempt, SubmissionFailure, TextSubmission, resolve_verifier
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import AnswerType, TaskResource
from taskcompendium.runtime.models import RuntimeEvidence
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.shell_verifier import (
    ArtifactKind,
    ExitCodeReward,
    FileReward,
    MissingArtifactPolicy,
    RewardFileFormat,
    ShellVerifierSpec,
    VerifierArtifact,
)
from taskcompendium.submission import SubmissionConvention
from verifyit.spec import DEFAULT_OUTPUT, GotestSpec, JunitSpec, PytestSpec, ScriptSpec, StdioSpec, render_spec

from rolloutengine.cleanup import _Cleanup
from rolloutengine.machines import _install_resources, _prepare_machine
from rolloutengine.spec import LoweredTaskSpec

MISSING_FILE_EXIT = 44
LINKED_ARTIFACT_EXIT = 45
MAX_ARTIFACT_ARCHIVE_BYTES = 1024**3
# Sparse members can produce more bytes than the archive contains.
MAX_ARTIFACT_EXPANDED_BYTES = 1024**3
MAX_ARTIFACT_MEMBERS = 100_000
INVALID_ARTIFACT_ERRNOS = {errno.ENAMETOOLONG, errno.ENOTDIR, errno.EISDIR, errno.EEXIST}
SPEC_PATH = "/tests/verifier.toml"
VERDICT_PATH = "/logs/verifier/verdict.json"


async def _grade_rollout(
    lowered: LoweredTaskSpec,
    convention: SubmissionConvention,
    messages: tuple[dict[str, Any], ...],
    machine: Machine | None,
    factories: Mapping[str, MachineFactory],
    cleanup: _Cleanup,
    resources: AsyncExitStack,
) -> GradeResult:
    """Grade evidence without exposing private files to model inference."""
    task = lowered.task
    timeout = lowered.session.verifier_timeout
    selection = lowered.runtime.verifier_machine
    if task.verifier.kind == "skipped":
        return GradeResult(Outcome.SKIPPED, None, json.loads(task.verifier.parameters_json)["reason"])
    if task.verifier.kind != "shell":
        evidence = RuntimeEvidence(await _capture_outputs(machine, task.output_paths, timeout), "{}")
        if selection is None:
            return await asyncio.to_thread(grade_task, task, convention, chat_conversation(list(messages)), evidence)
        grading_machine = await _prepare_machine(
            task.verifier.environment_requirements,
            selection,
            task.resources.all,
            factories,
            cleanup,
            resources,
        )
        assert grading_machine is not None
        return await _verifyit_grade(lowered, convention, messages, evidence, grading_machine)
    verifier = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)
    for command in verifier.collect:
        assert machine is not None
        result = await machine.run(
            Command(command.argv, cwd=command.cwd, env=resolve_env_vars(command.env), timeout=timeout, user="0")
        )
        if result.exit_code != 0:
            return GradeResult(
                Outcome.INFRA_ERROR, None, "Cannot collect grading inputs", failure=GradingFailure.EXECUTION
            )
    grading_machine = await _prepare_machine(
        task.verifier.environment_requirements, selection, task.resources.all, factories, cleanup, resources
    )
    assert grading_machine is not None
    with TemporaryDirectory(prefix="rollout-artifacts-") as directory:
        try:
            for index, artifact in enumerate(verifier.artifacts):
                assert machine is not None
                path = Path(directory) / str(index)
                if await _download_artifact(machine, artifact, path, timeout, cleanup, resources):
                    await grading_machine.upload(path, artifact.target)
        except (SubmissionFailure, DownloadLimitExceeded) as error:
            return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
    return await _shell_grade(
        verifier,
        messages,
        grading_machine,
        task.resources.verifier,
        timeout,
    )


async def _capture_outputs(machine: Machine | None, paths: tuple[str, ...], timeout: float | None) -> dict[str, bytes]:
    if machine is None:
        return {}
    files = {}
    for path in paths:
        result = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    f'if [ -f "$1" ]; then head -c "$2" -- "$1"; else exit {MISSING_FILE_EXIT}; fi',
                    "capture-output",
                    path,
                    str(DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES + 1),
                ),
                timeout=timeout,
                user="0",
                output_limit_bytes=DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES + 1,
            )
        )
        if result.exit_code == MISSING_FILE_EXIT:
            continue
        if result.exit_code != 0 or result.stdout_truncated or len(result.stdout) > DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES:
            raise RuntimeError(f"Cannot capture task output within the size limit: {path}")
        files[path] = result.stdout
    return files


async def _verifyit_grade(
    lowered: LoweredTaskSpec,
    convention: SubmissionConvention,
    messages: tuple[dict[str, Any], ...],
    evidence: RuntimeEvidence,
    machine: Machine,
) -> GradeResult:
    task = lowered.task
    spec = resolve_verifier(task.verifier)
    files = dict(evidence.files)
    if task.answer_type in {AnswerType.TEXT, AnswerType.NUMBER}:
        assert not isinstance(spec, StdioSpec | PytestSpec | JunitSpec | GotestSpec)
        try:
            submission = convention.extract(GradingAttempt(chat_conversation(list(messages))))
        except SubmissionFailure as error:
            return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
        if not isinstance(submission, TextSubmission):
            raise TypeError("Runtime text grading requires a text submission")
        files[DEFAULT_OUTPUT if isinstance(spec, ScriptSpec) else spec.output] = submission.value.encode()
    await _install_resources(machine, task.resources.verifier, root="/tests")
    await _install_resources(
        machine,
        (
            *(inline_resource(path.removeprefix("/"), data) for path, data in files.items()),
            inline_resource(SPEC_PATH.removeprefix("/"), render_spec(spec).encode()),
        ),
    )
    result = await machine.run(
        Command(
            ("python3", "-c", "from verifyit.grade import main; raise SystemExit(main())", SPEC_PATH),
            timeout=lowered.session.verifier_timeout,
        )
    )
    if result.reason == ExitReason.TIMED_OUT:
        return GradeResult(Outcome.INFRA_ERROR, None, "Verifier command timed out", failure=GradingFailure.TIMEOUT)
    if result.exit_code != 0:
        return GradeResult(Outcome.INFRA_ERROR, None, "Verifier command failed", failure=GradingFailure.EXECUTION)
    with TemporaryDirectory(prefix="rollout-verdict-") as directory:
        path = Path(directory) / "verdict.json"
        await machine.download(VERDICT_PATH, path)
        return parse_grade_result(spec, path.read_bytes())


async def _remove_archive(machine: Machine, path: str, timeout: float | None) -> None:
    result = await machine.run(Command(("rm", "-f", path), timeout=timeout, user="0"))
    if result.exit_code != 0:
        raise RuntimeError("Cannot remove private artifact archive")


async def _download_artifact(
    machine: Machine,
    artifact: VerifierArtifact,
    target: Path,
    timeout: float | None,
    cleanup: _Cleanup,
    resources: AsyncExitStack,
) -> bool:
    """Download an artifact. Return false only when its missing-file policy permits omission."""
    result = await machine.run(
        Command(
            argv=(
                "sh",
                "-c",
                'path=${1%/}; while [ "$path" ] && [ "$path" != / ]; do '
                f'[ ! -L "$path" ] || exit {LINKED_ARTIFACT_EXIT}; '
                'case "$path" in */*) path=${path%/*};; *) break;; esac; done; '
                'if [ -d "$1" ]; then printf directory; elif [ -f "$1" ]; then printf file; '
                f"else exit {MISSING_FILE_EXIT}; fi",
                "artifact-kind",
                artifact.source,
            ),
            timeout=timeout,
            user="0",
        )
    )
    if result.exit_code == MISSING_FILE_EXIT and artifact.missing == MissingArtifactPolicy.SKIP:
        return False
    if result.exit_code == MISSING_FILE_EXIT:
        raise SubmissionFailure(f"Grading artifact is missing: {artifact.source}")
    if result.exit_code == LINKED_ARTIFACT_EXIT:
        raise SubmissionFailure(f"Grading artifact path contains a link: {artifact.source}")
    if result.exit_code != 0 or result.stdout_truncated:
        raise RuntimeError(f"Cannot inspect grading artifact {artifact.source}: exit={result.exit_code}")
    try:
        kind = ArtifactKind(result.stdout.decode())
    except ValueError as error:
        raise SubmissionFailure(f"Invalid grading artifact kind: {artifact.source}") from error
    if artifact.kind != ArtifactKind.AUTO and artifact.kind != kind:
        raise SubmissionFailure(f"Grading artifact has the wrong kind: {artifact.source}")
    remote_archive = f"/tmp/taskcompendium-artifact-{uuid4().hex}.tar"
    resources.push_async_callback(
        cleanup.run, "artifact_archive_remove", partial(_remove_archive, machine, remote_archive, timeout)
    )
    result = await machine.run(
        Command(
            argv=(
                "tar",
                "-cf",
                remote_archive,
                *(f"--exclude={pattern}" for pattern in artifact.exclude if kind == ArtifactKind.DIRECTORY),
                "-C",
                artifact.source if kind == ArtifactKind.DIRECTORY else str(PurePosixPath(artifact.source).parent),
                "--",
                "." if kind == ArtifactKind.DIRECTORY else PurePosixPath(artifact.source).name,
            ),
            timeout=timeout,
            user="0",
        )
    )
    if result.exit_code != 0:
        raise SubmissionFailure(f"Cannot archive grading artifact {artifact.source}: exit={result.exit_code}")
    archive_path = target.with_suffix(".tar")
    await machine.download(remote_archive, archive_path, max_bytes=MAX_ARTIFACT_ARCHIVE_BYTES)
    extracted = target.with_suffix(".contents")
    extracted.mkdir()
    expanded_bytes = 0
    try:
        with tarfile.open(archive_path, "r:") as archive:
            for count, member in enumerate(archive, start=1):
                if count > MAX_ARTIFACT_MEMBERS:
                    raise SubmissionFailure(f"Grading artifact exceeds the member count limit: {artifact.source}")
                if member.issym() or member.islnk():
                    raise SubmissionFailure(f"Grading artifact contains a link: {member.name}")
                expanded_bytes += member.size
                if expanded_bytes > MAX_ARTIFACT_EXPANDED_BYTES:
                    raise SubmissionFailure(f"Grading artifact exceeds the expanded size limit: {artifact.source}")
                try:
                    archive.extract(member, extracted, filter="data")
                except OSError as error:
                    if error.errno not in INVALID_ARTIFACT_ERRNOS:
                        raise
                    raise SubmissionFailure(f"Invalid grading artifact member: {member.name}") from error
    except tarfile.TarError as error:
        raise SubmissionFailure(f"Invalid grading artifact archive: {artifact.source}") from error
    if kind == ArtifactKind.DIRECTORY:
        extracted.rename(target)
    else:
        file = extracted / PurePosixPath(artifact.source).name
        if not file.is_file():
            raise SubmissionFailure(f"Grading artifact archive has no regular file: {artifact.source}")
        file.rename(target)
    return True


async def _shell_grade(
    verifier: ShellVerifierSpec,
    messages: tuple[dict[str, Any], ...],
    machine: Machine,
    resources: tuple[TaskResource, ...],
    timeout: float | None,
) -> GradeResult:
    if isinstance(verifier.reward, FileReward):
        paths = tuple(file.path for file in verifier.reward.files)
        directories = tuple(sorted({str(PurePosixPath(path).parent) for path in paths}))
        for argv in (("mkdir", "-p", *directories), ("rm", "-f", *paths)):
            prepared = await machine.run(Command(argv=argv, timeout=timeout))
            if prepared.exit_code != 0:
                return GradeResult(
                    Outcome.INFRA_ERROR, None, "Cannot prepare private reward files", failure=GradingFailure.EXECUTION
                )
    await _install_resources(machine, resources, root="/tests")
    result = await machine.run(
        Command(
            argv=verifier.argv,
            stdin=json.dumps(messages).encode(),
            timeout=timeout,
        )
    )
    diagnostics = {
        "stdout": result.stdout.decode(errors="replace"),
        "stderr": result.stderr.decode(errors="replace"),
        "exit_code": result.exit_code,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
    }
    if result.reason == ExitReason.TIMED_OUT:
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            "Grader command timed out",
            diagnostics=diagnostics,
            failure=GradingFailure.TIMEOUT,
        )
    if isinstance(verifier.reward, ExitCodeReward):
        passed = result.exit_code == 0
        return GradeResult(Outcome.GRADED, float(passed), passed=passed, diagnostics=diagnostics)
    if isinstance(verifier.reward, FileReward):
        return await _file_grade(machine, verifier.reward, timeout, diagnostics)
    if result.exit_code != 0 or result.stdout_truncated:
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            f"Grader command failed: {result.reason}, exit={result.exit_code}",
            diagnostics=diagnostics,
            failure=GradingFailure.EXECUTION,
        )
    try:
        reward = float(result.stdout.decode().strip())
    except (UnicodeError, ValueError):
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            "Grader stdout must contain one finite numeric reward",
            diagnostics=diagnostics,
            failure=GradingFailure.INVALID_REWARD,
        )
    if not math.isfinite(reward):
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            "Grader returned a nonfinite reward",
            diagnostics=diagnostics,
            failure=GradingFailure.INVALID_REWARD,
        )
    return GradeResult(Outcome.GRADED, reward, diagnostics=diagnostics)


async def _file_grade(
    machine: Machine, specification: FileReward, timeout: float | None, diagnostics: dict[str, Any]
) -> GradeResult:
    for file in specification.files:
        result = await machine.run(
            Command(
                argv=(
                    "sh",
                    "-c",
                    f'if [ -f "$1" ]; then cat "$1"; else exit {MISSING_FILE_EXIT}; fi',
                    "reward-file",
                    file.path,
                ),
                timeout=timeout,
            )
        )
        if result.exit_code == MISSING_FILE_EXIT:
            continue
        if result.exit_code != 0 or result.stdout_truncated:
            return GradeResult(
                Outcome.INFRA_ERROR,
                None,
                f"Cannot read reward file: {file.path}",
                diagnostics=diagnostics,
                failure=GradingFailure.EXECUTION,
            )
        if not result.stdout.strip():
            return GradeResult(
                Outcome.INFRA_ERROR,
                None,
                f"Empty reward file: {file.path}",
                diagnostics=diagnostics,
                failure=GradingFailure.EMPTY_REWARD,
            )
        try:
            values = json.loads(result.stdout) if file.format == RewardFileFormat.JSON else None
            value = (
                values[file.key]
                if isinstance(values, dict)
                else values if values is not None else result.stdout.decode()
            )
            if isinstance(value, bool):
                raise ValueError("A boolean is not a numeric reward")
            reward = float(value)
            if not math.isfinite(reward):
                raise ValueError("Nonfinite reward")
        except (UnicodeError, ValueError, TypeError, KeyError) as error:
            return GradeResult(
                Outcome.INFRA_ERROR,
                None,
                f"Invalid reward file {file.path}: {error}",
                diagnostics=diagnostics,
                failure=GradingFailure.INVALID_REWARD,
            )
        return GradeResult(
            Outcome.GRADED,
            reward,
            passed=None if specification.pass_above is None else reward > specification.pass_above,
            diagnostics=diagnostics,
        )
    return GradeResult(
        Outcome.INFRA_ERROR,
        None,
        "Grader did not write a reward file",
        diagnostics=diagnostics,
        failure=GradingFailure.MISSING_REWARD,
    )
