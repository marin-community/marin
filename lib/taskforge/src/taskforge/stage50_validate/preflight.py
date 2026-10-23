# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Start a task's machine on its lowered backend before any trial, and fall back when it cannot.

``draft.host_requirements`` lowers a grader-base task onto the host's local backend when its shipped
files and working directories stay off the directories that backend cannot write. On Iris the local
factory overlays the system directories copy-on-write, so that is only ``/sys`` and the resolver's
directory and a task that ships files to ``/opt`` stays local; a read-only factory keeps every system
directory off limits. A setup command can still write where the lowering did not look, and the
lowering cannot see that without running the command, so every trial of such a task would fail its
setup on the local backend while the same task runs in a container; with the overlay that is rare.

``preflight`` lowers the task, and when its grader-base task machine lands on the local backend it runs
the engine's own preparation once (machine start, resources, setup commands) by rolling out a probe that
answers at once, then grades, on the engine every trial runs on (``EngineSettings.engine``). A failure
only the task or the backend explains (``FALLBACK_CAUSES``: a setup command that failed, or a spec the
backend refuses) moves the task machine alone to the container backend: the task is lowered again with the
local backend withheld, and its verifier machine and grader requirements are taken from the first
lowering, so a grader-base verifier stays local; ``RolloutEngine.check`` accepts the combined spec on
every factory (``draft.check``). The result carries a ``BackendFallback`` for the lower record. A machine
that did not start or did not start in time (``HOST_CAUSES``) is as often the host's state at probe time
as the task's, and the solver's machine must not depend on it: the probe runs once more, and a second such
failure raises ``BuildInfrastructureFailure``, so the workpiece ends as the host's and is retried later
rather than moved. Any other outcome of the probe, a grade included, keeps the first lowering: what the
probe says about the grader the trials will say again. A verifier machine never falls back: its
preparation failure is a task defect the trials report.

The fallback keeps the solver machine the proposal declared: both backends run the grader-base
CPython (``solver_machine: cpython``), and only the host decides which. A package machine runs on the
local backend alone, so it has no container to move to and is not probed; its trials report a setup
that fails there.

The probe is not evidence: it records no attempt file and no ledger span. Like every trial it runs
under the engine's messages trace (``lowering.lower``), so its empty reply carries no token ids.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from rolloutengine.models import GradedRollout, TaskRolloutSpec
from shellbox.machine import Backend, MachineFactory
from taskcompendium.models import TaskSpec

from taskforge.sandbox.factories import MachineHost
from taskforge.spec import draft
from taskforge.spec.controls import ScriptedModel, context_assistant_turns
from taskforge.stage10_proposal.model import SolverMachine
from taskforge.stage40_build_with_program.context import BuildInfrastructureFailure
from taskforge.stage40_build_with_program.infrastructure import InfrastructureCause
from taskforge.stage40_build_with_program.sdk import solver_machine
from taskforge.stage50_validate.classify import trial_outcome
from taskforge.stage50_validate.lowering import lower
from taskforge.stage50_validate.outcome import Cause, Ungraded
from taskforge.stage50_validate.trials import EngineSettings

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
    spec: TaskRolloutSpec
    fallback: BackendFallback | None


async def preflight(
    task: TaskSpec,
    host: MachineHost,
    settings: EngineSettings,
) -> Preflight:
    """Lower ``task`` for ``host``, probing a local grader-base task machine and falling back to a container when
    the task cannot be prepared there.

    Raises:
        ValueError: RolloutEngine rejects the lowered spec (``lowering.lower``).
        BuildInfrastructureFailure: the local task machine did not start, or not in time, on two probes.
    """
    factories = settings.factories
    spec = lower(task, host, factories, settings.limits)
    machine = spec.task_machine
    if machine is None or machine.backend != Backend.LOCAL.value:
        return Preflight(spec, None)
    if solver_machine(task.environment_requirements) is not SolverMachine.CPYTHON:
        return Preflight(spec, None)
    outcome = await _probe(spec, settings)
    if isinstance(outcome, Ungraded) and outcome.cause in HOST_CAUSES:
        outcome = await _probe(spec, settings)
        if isinstance(outcome, Ungraded) and outcome.cause in HOST_CAUSES:
            raise BuildInfrastructureFailure(
                HOST_CAUSES[outcome.cause],
                f"the preflight probe's {machine.backend} task machine failed twice ({outcome.cause}): {outcome.detail}",
            )
    if not isinstance(outcome, Ungraded) or outcome.cause not in FALLBACK_CAUSES:
        return Preflight(spec, None)
    fallback = _task_machine_fallback(task, host, factories, spec)
    assert fallback.task_machine is not None
    return Preflight(
        fallback,
        BackendFallback(machine.backend, fallback.task_machine.backend, outcome.cause, outcome.detail),
    )


def _task_machine_fallback(
    task: TaskSpec, host: MachineHost, factories: Mapping[str, MachineFactory], spec: TaskRolloutSpec
) -> TaskRolloutSpec:
    """``spec`` with its task machine on the container backend and its verifier as it was."""
    without_local = {backend: factory for backend, factory in factories.items() if backend != Backend.LOCAL.value}
    container = lower(task, host, without_local, spec.limits)
    hosted = container.task.model_copy(update={"grader": spec.task.grader})
    fallback = spec.model_copy(update={"task": hosted, "task_machine": container.task_machine})
    draft.check(fallback, factories)
    return fallback


async def _probe(spec: TaskRolloutSpec, settings: EngineSettings):
    """One engine run of ``spec`` with the probe model, classified as a trial outcome would be.

    The engine holds the run to ``spec.limits.attempt_timeout`` itself, so an expired probe is an
    ``ATTEMPT_TIMEOUT`` as in a trial.
    """
    try:
        result: GradedRollout | Exception = await settings.engine().run_task(
            spec, ScriptedModel(({"role": "assistant", "content": PROBE_REPLY},), context_assistant_turns(spec.task))
        )
    except Exception as error:
        result = error
    return trial_outcome(result, spec.task)
