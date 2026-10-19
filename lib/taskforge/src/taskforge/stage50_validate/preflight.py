# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Start a task's machine on its lowered backend before any trial, and fall back when it cannot.

``draft.host_requirements`` lowers a grader-base task onto the host's local backend when its shipped
files and working directories stay off that backend's read-only system directories. A setup command
can still write under one (``mkdir -p /opt/gen`` in a task's ``setup_commands``), and the lowering
cannot see that without running the command, so every trial of such a task would fail its setup on
the local backend while the same task runs in a container.

``preflight`` lowers the task, and when its grader-base task machine lands on the local backend it runs
the engine's own preparation once (machine start, resources, setup commands) by rolling out a probe that
answers at once, then grades. A failure only the task or the backend explains (``FALLBACK_CAUSES``: a
setup command that failed, or a spec the backend refuses) moves the task machine alone to the
container backend: the task is lowered again with the local backend withheld, and its verifier
machine and grader requirements are taken from the first lowering, so a grader-base verifier stays
local. The result carries a ``BackendFallback`` for the lower record. A machine that did not start or
did not start in time (``HOST_CAUSES``) is as often the host's state at probe time as the task's, and
the solver's machine must not depend on it: the probe runs once more, and a second such failure
raises ``BuildInfrastructureFailure``, so the workpiece ends as the host's and is retried later rather than
moved. Any other outcome of the probe, a grade included, keeps the first lowering: what the probe says
about the grader the trials will say again. A verifier machine never falls back: its preparation
failure is a task defect the trials report.

The fallback keeps the solver machine the proposal declared: both backends run the grader-base
CPython (``solver_machine: cpython``), and only the host decides which. A package machine runs on the
local backend alone, so it has no container to move to and is not probed; its trials report a setup
that fails there.

The probe is not evidence: it records no attempt file and no ledger span. Like every trial it runs
under the engine's messages trace (``EngineSettings.apply``), so its empty reply carries no token ids.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from rolloutengine.contracts import ModelRequest, ModelTurn, RolloutData
from rolloutengine.lowering import lower_task
from rolloutengine.spec import LoweredTaskSpec, TaskRuntimeSpec
from shellbox.machine import Backend, MachineFactory
from taskcompendium.models import TaskSpec

from taskforge.sandbox.factories import MachineHost
from taskforge.stage10_proposal.model import SolverMachine
from taskforge.stage40_build_with_program.context import BuildInfrastructureFailure
from taskforge.stage40_build_with_program.infrastructure import InfrastructureCause
from taskforge.stage40_build_with_program.sdk import solver_machine
from taskforge.stage50_validate.classify import trial_outcome
from taskforge.stage50_validate.lowering import lower
from taskforge.stage50_validate.outcome import Cause, Ungraded
from taskforge.stage50_validate.trials import Deadlines, EngineSettings

FALLBACK_CAUSES = frozenset({Cause.MACHINE_UNSUPPORTED, Cause.TASK_SETUP})
"""The probe outcomes that mean the task cannot be prepared on the local backend: the task machine moves."""
HOST_CAUSES = {
    Cause.MACHINE_START: InfrastructureCause.MACHINE_CREATE_FAILED,
    Cause.MACHINE_START_TIMEOUT: InfrastructureCause.SANDBOX_DEADLINE,
}
"""The probe outcomes the host causes as readily as the task, with the infrastructure cause a repeat raises."""
PROBE_REPLY = ""


@dataclass(frozen=True)
class BackendFallback:
    """The task machine moved off ``from_backend`` because the probe could not prepare it there."""

    from_backend: str
    to_backend: str
    cause: Cause
    detail: str


@dataclass(frozen=True)
class Preflight:
    lowered: LoweredTaskSpec
    fallback: BackendFallback | None


class _ProbeModel:
    """Answers the first request with an empty final reply, so the engine prepares the machine and grades."""

    def for_step(self, step: str) -> "_ProbeModel":
        return self

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        return ModelTurn(
            message={"role": "assistant", "content": PROBE_REPLY},
            prompt_token_ids=(),
            response_token_ids=(),
            logprobs=None,
            stop_reason="stop",
            text=PROBE_REPLY,
        )


async def preflight(
    task: TaskSpec,
    host: MachineHost,
    factories: Mapping[str, MachineFactory],
    settings: EngineSettings,
    deadlines: Deadlines,
) -> Preflight:
    """Lower ``task`` for ``host``, probing a local grader-base task machine and falling back to a container when
    the task cannot be prepared there.

    Raises:
        ValueError: RolloutEngine rejects the lowered spec (``lowering.lower``).
        BuildInfrastructureFailure: the local task machine did not start, or not in time, on two probes.
    """
    lowered = lower(task, host, factories)
    machine = lowered.runtime.task_machine
    if machine is None or machine.backend != Backend.LOCAL.value:
        return Preflight(lowered, None)
    if solver_machine(task.environment_requirements) is not SolverMachine.CPYTHON:
        return Preflight(lowered, None)
    outcome = await _probe(lowered, settings, deadlines)
    if isinstance(outcome, Ungraded) and outcome.cause in HOST_CAUSES:
        outcome = await _probe(lowered, settings, deadlines)
        if isinstance(outcome, Ungraded) and outcome.cause in HOST_CAUSES:
            raise BuildInfrastructureFailure(
                HOST_CAUSES[outcome.cause],
                f"the preflight probe's {machine.backend} task machine failed twice ({outcome.cause}): {outcome.detail}",
            )
    if not isinstance(outcome, Ungraded) or outcome.cause not in FALLBACK_CAUSES:
        return Preflight(lowered, None)
    fallback = _task_machine_fallback(task, host, factories, lowered)
    assert fallback.runtime.task_machine is not None
    return Preflight(
        fallback,
        BackendFallback(machine.backend, fallback.runtime.task_machine.backend, outcome.cause, outcome.detail),
    )


def _task_machine_fallback(
    task: TaskSpec, host: MachineHost, factories: Mapping[str, MachineFactory], lowered: LoweredTaskSpec
) -> LoweredTaskSpec:
    """``lowered`` with its task machine on the container backend and its verifier as it was."""
    without_local = {backend: factory for backend, factory in factories.items() if backend != Backend.LOCAL.value}
    container = lower(task, host, without_local)
    runtime = TaskRuntimeSpec(
        task_machine=container.runtime.task_machine, verifier_machine=lowered.runtime.verifier_machine
    )
    hosted = container.task.model_copy(update={"grader": lowered.task.grader})
    return lower_task(hosted, runtime, lowered.session, factories=factories, sessions={}, providers={})


async def _probe(lowered: LoweredTaskSpec, settings: EngineSettings, deadlines: Deadlines):
    """One engine run of ``lowered`` with the probe model, classified as a trial outcome would be."""
    engine = settings.engine(_ProbeModel())
    prepared = settings.apply(deadlines.apply(lowered))
    try:
        async with asyncio.timeout(deadlines.attempt_timeout):
            result: RolloutData | Exception = await engine.run(prepared)
    except Exception as error:
        result = error
    return trial_outcome(result, prepared.task)
