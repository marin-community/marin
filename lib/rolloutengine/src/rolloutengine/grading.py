# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private grader execution and reward collection."""

import asyncio
import json
import math
import tarfile
from collections.abc import Mapping
from functools import partial
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from harbor_config.env import resolve_env_vars
from shellbox.machine import Command, ExitReason, Machine, MachineFactory
from taskcompendium.chat import chat_conversation
from taskcompendium.environment import (
    ArtifactKind,
    EnvironmentFile,
    EnvironmentKind,
    ExitCodeReward,
    FileReward,
    MissingArtifactPolicy,
    RewardFileFormat,
    ShellVerifierSpec,
    VerifierArtifact,
)
from taskcompendium.execution import TaskExecution
from taskcompendium.grading import grade_answer, validate_verifier
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import AnswerType, SkippedVerifierSpec, StageRewardStrategy, TaskSpec, TaskStage, VerifierKind
from taskcompendium.submission import Submission

from rolloutengine.cleanup import _Cleanup
from rolloutengine.machines import _install_files, _machine_command, _task_machine

MISSING_FILE_EXIT = 44


def _validate_task(task: TaskSpec, execution: TaskExecution) -> None:
    """Reject task features that this engine cannot execute."""
    if task.verifier.kind == VerifierKind.EXTERNAL and task.environment.interaction is None:
        raise ValueError("External verifiers require an interaction session")
    stage_names = {stage.name for stage in task.stages}
    missing = stage_names - execution.stages.keys()
    extra = execution.stages.keys() - stage_names
    if missing or extra:
        raise ValueError(f"Execution stages disagree with the task: missing={sorted(missing)}, extra={sorted(extra)}")
    verifiers = (task.verifier, *(stage.verifier for stage in task.stages))
    file_groups = [
        (task.environment, task.environment.files),
        *((task.environment, stage.workdir_files) for stage in execution.stages.values()),
        *((verifier.environment or task.environment, verifier.files) for verifier in verifiers),
        *((verifier.environment, verifier.environment.files) for verifier in verifiers if verifier.environment),
    ]
    for environment, files in file_groups:
        if environment.kind == EnvironmentKind.SHELLSIM and any(file.mtime_ns is not None for file in files):
            raise ValueError("ShellSim cannot preserve explicit file timestamps")
    if task.environment.interaction is None and (task.environment.tool_providers or task.interaction_tools):
        raise NotImplementedError("Native tool providers require an application-supplied task session")
    for specification in verifiers:
        validate_verifier(specification)
        if task.environment.interaction is None and specification.kind not in {
            VerifierKind.SHELL,
            VerifierKind.STAGED,
            VerifierKind.SKIPPED,
        }:
            if specification.environment is not None or task.answer_type in {
                AnswerType.FILE,
                AnswerType.STATE,
                AnswerType.WORKSPACE_STATE,
            }:
                raise NotImplementedError("Native workspace grading requires an application-supplied task session")


async def _grade_rollout(
    task: TaskSpec,
    convention: Submission,
    messages: tuple[dict[str, Any], ...],
    machine: Machine | None,
    factories: Mapping[EnvironmentKind, MachineFactory],
    cleanup: _Cleanup,
) -> GradeResult:
    """Grade the final transcript and task filesystem without model access to private files."""
    if task.verifier.kind == VerifierKind.SKIPPED:
        parameters = SkippedVerifierSpec.model_validate_json(task.verifier.parameters_json)
        return GradeResult(Outcome.SKIPPED, None, parameters.reason)
    if task.verifier.kind != VerifierKind.SHELL:
        conversation = chat_conversation(list(messages))
        return await asyncio.to_thread(grade_answer, task, convention, conversation)
    if machine is None:
        raise ValueError("Shell grading requires a task machine")
    verifier = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)
    for command in verifier.collect:
        result = await machine.run(_machine_command(command))
        if result.exit_code != 0:
            return GradeResult(
                Outcome.INFRA_ERROR, None, "Cannot collect grading inputs", failure=GradingFailure.EXECUTION
            )
    if task.verifier.environment is None:
        return await _shell_grade(verifier, messages, machine, task.verifier.files)
    async with _task_machine(task.verifier.environment, factories, cleanup) as grading_machine:
        assert grading_machine is not None
        with TemporaryDirectory(prefix="rollout-artifacts-") as directory:
            for index, artifact in enumerate(verifier.artifacts):
                path = Path(directory) / str(index)
                if await _download_artifact(machine, artifact, path, verifier.timeout, cleanup):
                    await grading_machine.upload(path, artifact.target)
        return await _shell_grade(verifier, messages, grading_machine, task.verifier.files)


async def _remove_artifact_archive(machine: Machine, path: str, timeout: float) -> None:
    result = await machine.run(Command(argv=("rm", "-f", path), timeout=timeout, user="0"))
    if result.reason == ExitReason.TIMED_OUT:
        raise TimeoutError("Artifact archive removal timed out")
    if result.exit_code != 0:
        raise RuntimeError(f"Artifact archive removal failed: {result.reason}, exit={result.exit_code}")


async def _download_artifact(
    machine: Machine, artifact: VerifierArtifact, target: Path, timeout: float, cleanup: _Cleanup
) -> bool:
    """Download an artifact. Return false only when its missing-file policy permits omission."""
    kind = artifact.kind
    if kind == ArtifactKind.AUTO or artifact.missing == MissingArtifactPolicy.SKIP:
        result = await machine.run(
            Command(
                argv=(
                    "sh",
                    "-c",
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
        if result.exit_code != 0:
            raise RuntimeError(f"Cannot inspect grading artifact {artifact.source}: exit={result.exit_code}")
        kind = ArtifactKind(result.stdout.decode())
    if kind == ArtifactKind.DIRECTORY:
        target.mkdir()
    if not artifact.exclude or kind != ArtifactKind.DIRECTORY:
        await machine.download(artifact.source, target)
        return True
    remote_archive = f"/tmp/taskcompendium-artifact-{uuid4().hex}.tar"
    try:
        result = await machine.run(
            Command(
                argv=(
                    "tar",
                    "-cf",
                    remote_archive,
                    *(f"--exclude={pattern}" for pattern in artifact.exclude),
                    "-C",
                    artifact.source,
                    ".",
                ),
                timeout=timeout,
                user="0",
            )
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Cannot archive grading artifact {artifact.source}: exit={result.exit_code}")
        archive_path = target.with_suffix(".tar")
        await machine.download(remote_archive, archive_path)
        with tarfile.open(archive_path) as archive:
            archive.extractall(target, filter="data")
    finally:
        await cleanup.run(
            "artifact_archive_remove",
            partial(_remove_artifact_archive, machine, remote_archive, timeout),
        )
    return True


async def _shell_grade(
    verifier: ShellVerifierSpec,
    messages: tuple[dict[str, Any], ...],
    machine: Machine,
    files: tuple[EnvironmentFile, ...],
) -> GradeResult:
    if isinstance(verifier.reward, FileReward):
        paths = tuple(file.path for file in verifier.reward.files)
        directories = tuple(sorted({str(PurePosixPath(path).parent) for path in paths}))
        for argv in (("mkdir", "-p", *directories), ("rm", "-f", *paths)):
            prepared = await machine.run(Command(argv=argv, timeout=verifier.timeout, user=verifier.user))
            if prepared.exit_code != 0:
                return GradeResult(
                    Outcome.INFRA_ERROR, None, "Cannot prepare private reward files", failure=GradingFailure.EXECUTION
                )
    await _install_files(machine, files)
    result = await machine.run(
        Command(
            argv=verifier.argv,
            env=resolve_env_vars(verifier.env),
            stdin=json.dumps(messages).encode(),
            timeout=verifier.timeout,
            user=verifier.user,
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
        return await _file_grade(machine, verifier.reward, verifier.timeout, diagnostics, verifier.user)
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
    machine: Machine, specification: FileReward, timeout: float, diagnostics: dict[str, Any], user: str | None
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
                user=user,
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
            rewards=(
                {
                    key: float(value)
                    for key, value in values.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                }
                | {file.key: reward}
                if isinstance(values, dict)
                else {"reward": reward}
            ),
        )
    return GradeResult(
        Outcome.INFRA_ERROR,
        None,
        "Grader did not write a reward file",
        diagnostics=diagnostics,
        failure=GradingFailure.MISSING_REWARD,
    )


async def _remove_stage_grader(stage: TaskStage, machine: Machine) -> None:
    if stage.verifier.kind != VerifierKind.SHELL:
        return
    verifier = ShellVerifierSpec.model_validate_json(stage.verifier.parameters_json)
    if stage.verifier.environment is not None:
        return
    paths = [file.path for file in stage.verifier.files]
    if isinstance(verifier.reward, FileReward):
        paths.extend(file.path for file in verifier.reward.files)
    if not paths:
        return
    result = await machine.run(Command(("rm", "-f", *paths), user="0", timeout=verifier.timeout))
    if result.exit_code != 0:
        raise RuntimeError("Cannot remove private stage verifier files")


def _combined_stage_grade(grades: list[GradeResult], strategy: StageRewardStrategy) -> GradeResult:
    final = grades[-1]
    if strategy == StageRewardStrategy.FINAL or final.status not in {Outcome.GRADED, Outcome.SKIPPED}:
        return final
    valid = [(grade, grade.reward) for grade in grades if grade.status == Outcome.GRADED and grade.reward is not None]
    if not valid:
        return final
    reward = sum(value for _, value in valid) / len(valid)
    components = [grade.rewards or {"reward": value} for grade, value in valid]
    rewards = {key: sum(values.get(key, 0.0) for values in components) / len(valid) for key in set().union(*components)}
    pass_results = [grade.passed for grade, _ in valid]
    passed = all(pass_results) if all(result is not None for result in pass_results) else None
    return GradeResult(Outcome.GRADED, reward, passed=passed, rewards=rewards)
