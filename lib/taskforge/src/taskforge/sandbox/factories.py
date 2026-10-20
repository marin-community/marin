# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine factories for RolloutEngine, and the task rollout specs they refuse.

``machine_factories`` resolves the factories for the host Taskforge runs on, once, and keys them by
shellbox ``Backend`` value as ``MachineRuntimeSpec.backend`` refers to them. ``container_backend`` gives
the backend for image-backed machines on that host. Before a trial starts, ``task_refusals`` asks each
factory about the machines a ``TaskRolloutSpec`` selects on its backend through
``MachineFactory.check_environment``, so validation records a typed refusal with the factory's message
instead of failing inside ``MachineFactory.create`` after the trial started.
"""

import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Protocol, runtime_checkable

import shellsim
from rolloutengine.machines import _machine_spec
from rolloutengine.models import MachineRuntimeSpec, TaskRolloutSpec
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.local.python_environment import resolve_python_environment
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineFactory, UnsupportedMachineSpec
from taskcompendium.models import EnvironmentRequirements, ScriptGrader, TaskSpec, VerifyitGrader, grades_with_judge
from taskcompendium.runtime.environment import validate_machine_spec

logger = logging.getLogger(__name__)

DOCKER_PROBE_TIMEOUT = 20
SHELLSIM_CATALOG = shellsim.DEFAULT_PYPI_CATALOG
"""The ports catalog ShellSim machines install ``draft.SHELLSIM_CPYTHON`` from, over HTTPS.

shellsim's published catalog, which Marin's inference tools also read by default. Each process
fetches the catalog itself; the first on a host downloads the CPython assets into shellsim's cache
(``$XDG_CACHE_HOME/shellsim``, else ``~/.cache/shellsim``), which later processes reuse.
"""
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


@cache
def docker_unavailable() -> str | None:
    """Probe the laptop once for a reachable Docker daemon; return why Docker machines are unavailable, or None."""
    docker = shutil.which("docker")
    if docker is None:
        return "docker CLI not found on PATH"
    probe = subprocess.run(
        (docker, "info", "--format", "{{.ServerVersion}}"), capture_output=True, text=True, timeout=DOCKER_PROBE_TIMEOUT
    )
    if probe.returncode:
        return f"docker daemon unreachable: {probe.stderr.strip()[-500:]}"
    return None


@dataclass(frozen=True)
class LocalSandbox:
    """The Iris task's bubblewrap probe: a local factory for lock-only machines, solver and verifier alike, or
    why there is none.

    Exactly one of ``factory`` and ``unavailable`` is set.
    """

    factory: MachineFactory | None
    unavailable: str | None


@cache
def local_sandbox() -> LocalSandbox:
    """Probe once whether bubblewrap can sandbox commands here (it needs root or user namespaces on Linux)."""
    # The local backend needs bubblewrap_bin, which installs only on Linux.
    from shellbox.backends.local.machine import LocalMachineFactory, SandboxUnavailable  # noqa: PLC0415

    try:
        return LocalSandbox(factory=LocalMachineFactory(), unavailable=None)
    except SandboxUnavailable as error:
        return LocalSandbox(factory=None, unavailable=str(error))


def container_backend(where: MachineHost) -> Backend:
    """The backend of an image-backed machine on ``where``."""
    if where is MachineHost.IRIS:
        return Backend.GVISOR
    return Backend.DOCKER


def machine_factories(where: MachineHost, controller_url: str | None) -> Mapping[str, MachineFactory]:
    """The factories ``RolloutEngine`` takes, keyed by ``factory.backend.value``.

    On Iris the gVisor factory submits sandboxes to ``controller_url``, the controller of the task
    Taskforge runs in; the local factory, which runs grader-base and package machines for solvers and
    graders alike, is present when bubblewrap can sandbox commands in the Iris task. On a laptop
    ``controller_url`` must be
    ``None``; the Docker factory, which pulls registry images through the Docker daemon, is absent
    when the laptop has no reachable daemon. Each absent factory is logged with the probe's reason.

    The ShellSim factory's context reads ``SHELLSIM_CATALOG`` when its first machine asks for CPython
    and prepares that installation once, for every later machine the factories create.
    """
    if (where is MachineHost.IRIS) != (controller_url is not None):
        raise ValueError(f"{where} factories take a controller URL only on Iris, got {controller_url!r}")
    factories: list[MachineFactory] = [ShellSimMachineFactory(context=shellsim.Context(pypi_catalog=SHELLSIM_CATALOG))]
    if controller_url is not None:
        factories.append(
            IrisMachineFactory(controller_url=controller_url, scheduling_timeout=SANDBOX_SCHEDULING_TIMEOUT)
        )
        sandbox = local_sandbox()
        if sandbox.factory is not None:
            factories.append(sandbox.factory)
        else:
            logger.warning("No %s machine factory: %s", Backend.LOCAL, sandbox.unavailable)
    elif (docker := docker_unavailable()) is None:
        factories.append(DockerMachineFactory())
    else:
        logger.warning("No %s machine factory: %s", Backend.DOCKER, docker)
    return {factory.backend.value: factory for factory in factories}


@runtime_checkable
class PythonEnvironmentFactory(Protocol):
    """A factory that resolves ``PackageSpec`` machines into a cached Python environment, as the local one does."""

    package_cache: Path
    python_version: str


WARM_THREAD = "warm-local-python"


def warm_local_python(factories: Mapping[str, MachineFactory]) -> threading.Thread | None:
    """Build the local backend's bare CPython environment in a background thread, before the first machine asks.

    The local factory resolves a ``PackageSpec`` machine's environment once per cache (``uv`` installs the
    interpreter, makes the venv and compiles the lock) and reuses it afterwards; the first wave of
    machines in a fresh process queues on that build, about five seconds for the empty requirements of
    ``draft.LOCAL_CPYTHON``, while later machines pay nothing. A coordinator calls this once after
    ``machine_factories`` so the build overlaps its first model calls; the build children it starts share
    the cache. Returns the thread, or ``None`` when ``factories`` has no local backend. The thread is a
    daemon: a failure is logged and the first machine resolves the environment itself.
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
