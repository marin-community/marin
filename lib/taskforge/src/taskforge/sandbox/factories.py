# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine factories for RolloutEngine, and the task rollout specs they refuse.

Factories are keyed by shellbox ``Backend`` value as ``MachineRuntimeSpec.backend`` refers to them;
``taskforge_adapters.backends.machine_factories`` constructs them. ``container_backend`` gives
the backend for image-backed machines on that host. Before a trial starts, ``task_refusals`` asks each
factory about the machines a ``TaskRolloutSpec`` selects on its backend through
``MachineFactory.check_environment``, so validation records a typed refusal with the factory's message
instead of failing inside ``MachineFactory.create`` after the trial started.
"""

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from rolloutengine.machines import _machine_spec
from rolloutengine.models import MachineRuntimeSpec, TaskRolloutSpec
from shellbox.backends.local.python_environment import resolve_python_environment
from shellbox.machine import Backend, MachineFactory, UnsupportedMachineSpec
from taskcompendium.models import EnvironmentRequirements, ScriptGrader, TaskSpec, VerifyitGrader, grades_with_judge
from taskcompendium.runtime.environment import validate_machine_spec

logger = logging.getLogger(__name__)

SANDBOX_SCHEDULING_TIMEOUT = 600
"""Seconds the Iris factory waits for a sandbox to start before it raises ``TimeoutError``.

A deadline the engine holds around a machine's start, such as an attempt's verifier deadline, must
outlast this budget: the factory's own timeout is what ``stage40_build_with_program.infrastructure`` classifies
as a host failure, and a shorter engine deadline cancels the wait first and reports a bare timeout.
"""


class MachineHost(StrEnum):
    """Where the factories run: a developer laptop, or inside an Iris task."""

    LAPTOP = "laptop"
    IRIS = "iris"


class MachineRole(StrEnum):
    """Which of a task rollout spec's machines a refusal concerns."""

    TASK = "task"
    VERIFIER = "verifier"


class RefusalReason(StrEnum):
    NO_FACTORY = "no_factory"
    """The host has no factory for the machine's backend."""
    UNSUPPORTED = "unsupported"
    """The backend's factory rejected the machine in ``check_environment``."""


@dataclass(frozen=True)
class Refusal:
    """A refused machine of a task rollout spec, with the factory's message in ``detail``."""

    reason: RefusalReason
    where: MachineRole
    detail: str


def container_backend(where: MachineHost) -> Backend:
    """The backend of an image-backed machine on ``where``."""
    if where is MachineHost.IRIS:
        return Backend.GVISOR
    return Backend.DOCKER


@runtime_checkable
class PythonEnvironmentFactory(Protocol):
    """A factory that resolves ``PackageSpec`` machines into a cached Python environment, as the local one does."""

    package_cache: Path
    python_version: str


WARM_THREAD = "warm-local-python"


def warm_local_python(factories: Mapping[str, MachineFactory]) -> threading.Thread | None:
    """Warm the local backend's bare CPython environment in a daemon thread.

    Returns ``None`` without a local backend. Failures are logged and leave normal machine
    creation to retry environment resolution.
    """
    factory = factories.get(Backend.LOCAL.value)
    if not isinstance(factory, PythonEnvironmentFactory):
        return None
    cache, version = factory.package_cache, factory.python_version

    def warm() -> None:
        started = time.monotonic()
        try:
            resolve_python_environment(cache, (), version)
        except Exception:
            logger.warning("Warming the local Python %s environment in %s failed", version, cache, exc_info=True)
            return
        logger.info("Warmed the local Python %s environment in %s in %.1fs", version, cache, time.monotonic() - started)

    thread = threading.Thread(target=warm, name=WARM_THREAD, daemon=True)
    thread.start()
    return thread


def _machine_refusal(
    role: MachineRole,
    selection: MachineRuntimeSpec,
    requirements: EnvironmentRequirements,
    factories: Mapping[str, MachineFactory],
) -> Refusal | None:
    factory = factories.get(selection.backend)
    if factory is None:
        return Refusal(RefusalReason.NO_FACTORY, role, f"{selection.backend}: no factory for this backend")
    try:
        validate_machine_spec(requirements, factory, _machine_spec(requirements, selection))
    except UnsupportedMachineSpec as error:
        return Refusal(RefusalReason.UNSUPPORTED, role, f"{selection.backend}: {error}")
    return None


def task_refusals(spec: TaskRolloutSpec, factories: Mapping[str, MachineFactory]) -> list[Refusal]:
    """Ask ``factories`` whether they can run the task machine and the verifier machine of ``spec``.

    For each machine, the factory of its backend receives the machine's environment requirements and
    the ``MachineSpec`` RolloutEngine creates the machine from, as ``RolloutEngine.check`` passes them
    to ``check_environment``. A machine yields at most one refusal, since ``check_environment`` raises
    at the first unsupported feature.

    The check covers declared features. Iris and ShellSim compare a command's execution user only when
    they run the command, so a trial with a user they reject fails as ``MACHINE_UNSUPPORTED`` after its
    machine starts.
    """
    task = spec.task
    selections = (
        (MachineRole.TASK, spec.task_machine, task.environment_requirements),
        (MachineRole.VERIFIER, spec.verifier_machine, grading_environment(task)),
    )
    return [
        refusal
        for role, selection, requirements in selections
        if selection is not None and requirements is not None
        if (refusal := _machine_refusal(role, selection, requirements, factories)) is not None
    ]


def grading_environment(task: TaskSpec) -> EnvironmentRequirements | None:
    """The environment of the task's verifier machine, or ``None`` when its grader takes none.

    A verifyit judge takes none: RolloutEngine grades it on the host, through the judge service.
    """
    grader = task.grader
    if isinstance(grader, ScriptGrader | VerifyitGrader) and not grades_with_judge(grader):
        return grader.environment
    return None
