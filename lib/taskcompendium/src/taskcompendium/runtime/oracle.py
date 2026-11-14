# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared source OracleCommand execution and conversation answer delivery."""

from pathlib import PurePosixPath

from shellbox.machine import Command, ExitReason, Machine, Result

from taskcompendium.models import (
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    ScriptGrader,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    grader_workspace,
)
from taskcompendium.pipeline.models import OracleCommand


class OracleAnswerError(RuntimeError):
    """The original command did not leave a complete readable answer."""


class OracleAnswerTimeout(OracleAnswerError):
    """Capturing the original command's answer exceeded its deadline."""


def oracle_actor_required(task: TaskSpec) -> bool:
    return task.environment_requirements != EnvironmentRequirements()


def oracle_environment(task: TaskSpec, command: OracleCommand) -> EnvironmentRequirements:
    """Select the declared actor, or the original grader and its workspace."""
    if oracle_actor_required(task):
        return task.environment_requirements
    grader = task.grader
    if not isinstance(grader, VerifyitGrader | ScriptGrader) or grader.environment is None:
        raise ValueError("An original oracle requires its actor or grader environment")
    environment = grader.environment
    if command.answer_file is not None and environment.working_directory is None:
        return environment.model_copy(update={"working_directory": grader_workspace(grader)})
    return environment


async def run_oracle_command(
    machine: Machine, command: OracleCommand, *, cwd: str | None, timeout: float, output_limit_bytes: int
) -> Result:
    """Prepare the declared answer directory and run the unchanged source command."""
    if command.answer_file is not None:
        parent = PurePosixPath(command.answer_file).parent
        if parent != PurePosixPath("."):
            prepared = await machine.run(
                Command(("mkdir", "-p", str(parent)), cwd=cwd, timeout=timeout, output_limit_bytes=output_limit_bytes)
            )
            if prepared.reason != ExitReason.EXITED or prepared.exit_code != 0:
                raise OracleAnswerError(
                    f"Could not prepare the declared oracle answer directory {parent}: "
                    f"{prepared.stderr.decode(errors='replace')[-2000:]}"
                )
    return await machine.run(
        Command(("/bin/bash", "-lc", command.command), cwd=cwd, timeout=timeout, output_limit_bytes=output_limit_bytes)
    )


async def read_oracle_answer(
    machine: Machine, answer_file: str, *, cwd: str | None, timeout: float, output_limit_bytes: int
) -> bytes:
    """Capture the complete answer in the same workspace as the source command."""
    answer = await machine.run(
        Command(("cat", answer_file), cwd=cwd, timeout=timeout, output_limit_bytes=output_limit_bytes)
    )
    if answer.reason == ExitReason.TIMED_OUT:
        raise OracleAnswerTimeout("Original oracle answer capture timed out")
    if answer.exit_code != 0 or answer.stdout_truncated:
        raise OracleAnswerError(f"Original oracle wrote no complete readable answer file {answer_file}")
    return answer.stdout


def oracle_answer_attempt(task: TaskSpec, answer: bytes) -> GradingAttempt:
    """Deliver original answer bytes as final text under the shared UTF-8 policy."""
    reply = TextMessage(role="assistant", content=answer.decode(errors="replace"))
    return GradingAttempt(ConversationTrace(events=(*task.context.events, reply)))
