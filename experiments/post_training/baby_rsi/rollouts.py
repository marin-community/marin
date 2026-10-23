# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a policy on tasks: the rollout spec of each task, the machines, and groups of graded attempts.

Every task of the loop runs on the same ``RolloutEngine``: a pool task without tools answers in one turn, a
structural-closure task continues its assistant prefix, and a task with a shell gets a bubblewrap (``linux_process``)
or ShellSim machine. Script graders run in a bubblewrap sandbox with the host's Python environment.
"""

import asyncio
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.prefill import PrefillHarness
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import (
    GradedRollout,
    GradedRolloutInterrupted,
    MachineRuntimeSpec,
    ModelRequest,
    ModelTurn,
    RolloutLimits,
    TaskRolloutSpec,
)
from shellbox.backends.local.machine import SYSTEM_DIRECTORIES, LocalMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import (
    Backend,
    Command,
    ExecutionRequirements,
    Machine,
    MachineFactory,
    MachineSpec,
    Result,
)
from taskcompendium.grading_result import Outcome
from taskcompendium.models import CommandSemantics, ScriptGrader, TaskSpec, TextMessage, VerifyitGrader
from taskforge.agentic.tools import machine_runtime, needs_machine

from experiments.post_training.capability_driven_envs.analyze_failed_runs.rollouts import (
    FAILURE_OPERATION,
    POLICY_INTERRUPTIONS,
)
from experiments.post_training.capability_driven_envs.generate_tasks_agentic import COMMAND_TIMEOUT

logger = logging.getLogger(__name__)

SHELL_HARNESS = "shell"
PREFILL_HARNESS = "prefill"
MACHINE_BACKENDS = {CommandSemantics.LINUX_PROCESS: Backend.LOCAL, CommandSemantics.SHELL_SIMULATOR: Backend.SHELLSIM}
"""The solver machine of a task with a shell, from its command semantics: bubblewrap or ShellSim."""
PROGRESS_EVERY = 50
PASS_REWARD = 1.0
"""A rollout passes with a reward of at least this value; some graders give partial credit below it."""


def machine_factories(grader_slots: int) -> dict[str, MachineFactory]:
    """Bubblewrap with this Python environment for script graders and synthesized tasks, and ShellSim.

    The grader sandbox sees the virtual environment, the workspace's editable library sources and the base
    interpreter, read-only. It does not mount the whole checkout, because a script grader writes the answer to
    ``/app``, where an Iris task keeps the checkout. Bubblewrap needs namespaces, which an Iris task has only under
    the privileged container profile.
    """
    venv = Path(sys.prefix)
    # The sandbox sees the system directories already, and mounts inside them fail.
    interpreter = (
        ()
        if any(Path(sys.base_prefix).is_relative_to(path) for path in SYSTEM_DIRECTORIES)
        else (Path(sys.base_prefix),)
    )
    mounts = (venv, venv.parent / "lib", *interpreter)
    graders = LocalMachineFactory(read_only=mounts, bin_dirs=(venv / "bin",))
    return {
        Backend.LOCAL: BoundedFactory(graders, asyncio.Semaphore(grader_slots)),
        Backend.SHELLSIM: ShellSimMachineFactory(),
    }


@dataclass
class BoundedFactory:
    """A machine factory with at most as many open machines as ``slots`` permits.

    A script grader that runs tests measures time, so graders that wait for a CPU can fail correct answers.
    """

    factory: MachineFactory
    slots: asyncio.Semaphore

    @property
    def backend(self) -> Backend:
        return self.factory.backend

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.factory.check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        await self.slots.acquire()
        try:
            return _ReleasingMachine(await self.factory.create(spec), self.slots)
        except BaseException:
            self.slots.release()
            raise


@dataclass
class _ReleasingMachine:
    machine: Machine
    slots: asyncio.Semaphore

    async def run(self, command: Command) -> Result:
        return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        try:
            await self.machine.close()
        finally:
            self.slots.release()


def rollout_engine(grader_slots: int) -> RolloutEngine:
    """The engine of the policy's rollouts, with at most ``grader_slots`` script graders at a time, for example the
    task's CPUs."""
    harnesses = {SHELL_HARNESS: ShellHarness(command_timeout=COMMAND_TIMEOUT), PREFILL_HARNESS: PrefillHarness()}
    return RolloutEngine(machine_factories(grader_slots), harnesses)


def rollout_spec(task: TaskSpec, limits: RolloutLimits) -> TaskRolloutSpec:
    """The harness and machines of ``task``: a prefill task continues its last assistant message."""
    last = task.context.events[-1]
    prefill = isinstance(last, TextMessage) and last.role == "assistant"
    grader = task.grader
    graded_in_sandbox = isinstance(grader, ScriptGrader) or (
        isinstance(grader, VerifyitGrader) and grader.environment is not None
    )
    return TaskRolloutSpec(
        task=task,
        harness=PREFILL_HARNESS if prefill else SHELL_HARNESS,
        task_machine=_solver_machine(task),
        verifier_machine=machine_runtime(Backend.LOCAL) if graded_in_sandbox else None,
        limits=limits,
    )


def _solver_machine(task: TaskSpec) -> MachineRuntimeSpec | None:
    if not needs_machine(task):
        return None
    semantics = task.environment_requirements.command_semantics
    assert semantics is not None, f"The task {task.id} has a shell but no command semantics"
    return machine_runtime(MACHINE_BACKENDS[semantics])


def reward(graded: GradedRollout) -> float | None:
    """The training reward of an attempt, or ``None`` when a task, grader or infrastructure failure decided it.

    An attempt that the policy ran past its deadline, or that ended on a bad tool call, gets 0, as does an answer
    that the answer format rejects.
    """
    failure = graded.rollout.failure
    if failure is not None:
        return 0.0 if failure.diagnostics.get(FAILURE_OPERATION) in POLICY_INTERRUPTIONS else None
    grade = graded.grade
    if grade.status == Outcome.SUBMISSION_FAILURE:
        return 0.0
    if grade.status == Outcome.GRADED and grade.reward is not None:
        return grade.reward
    return None


@dataclass(frozen=True)
class TaskRollouts:
    """The graded attempts of one task by one policy version."""

    task: TaskSpec
    rollouts: tuple[GradedRollout, ...]
    model: str
    """The served name of the policy weights that started the attempts."""

    @property
    def rewards(self) -> list[float]:
        """The rewards of the attempts that the policy decided."""
        return [value for value in map(reward, self.rollouts) if value is not None]

    @property
    def pass_rate(self) -> float | None:
        rewards = self.rewards
        return sum(value >= PASS_REWARD for value in rewards) / len(rewards) if rewards else None


class ServedPolicy(Protocol):
    """A rollout model whose ``model`` names the weights that serve its next request, such as ``VllmPolicy``."""

    model: str

    async def __call__(self, request: ModelRequest) -> ModelTurn: ...


@dataclass
class PolicyRunner:
    """Runs the policy on tasks with one engine, at most ``concurrency`` attempts at a time."""

    engine: RolloutEngine
    policy: ServedPolicy
    limits: RolloutLimits
    concurrency: int

    def __post_init__(self) -> None:
        self._slots = asyncio.Semaphore(self.concurrency)

    async def attempt(self, task: TaskSpec) -> GradedRollout:
        async with self._slots:
            try:
                return await self.engine.run_task(rollout_spec(task, self.limits), self.policy)
            except GradedRolloutInterrupted as error:
                return GradedRollout(error.rollout, error.grade)

    async def run(self, task: TaskSpec, n: int) -> TaskRollouts:
        """``n`` graded attempts of ``task`` with the current weights of the policy."""
        served = self.policy.model
        rollouts = await asyncio.gather(*(self.attempt(task) for _ in range(n)))
        return TaskRollouts(task, tuple(rollouts), served)

    async def run_all(self, tasks: Sequence[TaskSpec], n: int) -> list[TaskRollouts]:
        """``n`` attempts of each task, in the order of ``tasks``; logs the progress of a long run."""
        done = 0

        async def run(task: TaskSpec) -> TaskRollouts:
            nonlocal done
            group = await self.run(task, n)
            done += 1
            if done % PROGRESS_EVERY == 0:
                logger.info("%d of %d tasks done with %s", done, len(tasks), group.model)
            return group

        return list(await asyncio.gather(*(run(task) for task in tasks)))
