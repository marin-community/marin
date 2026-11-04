# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine factories for RolloutEngine.

``machine_factories`` constructs the explicitly selected backends and keys them by
shellbox ``Backend`` value as ``MachineRuntimeSpec.backend`` refers to them. ``container_backend`` gives
the backend for image-backed machines among those selected. ``RolloutEngine.check`` (``spec.draft.check``) asks
the factories about a spec's machines before any machine starts.
"""

import logging
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence

import shellsim
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineFactory
from taskcompendium.models import EnvironmentRequirements, ScriptGrader, TaskSpec, VerifyitGrader, grades_with_judge

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


def docker_unavailable_reason() -> str | None:
    """Check the requested Docker backend for a reachable daemon; return why Docker machines are unavailable, or None."""
    docker = shutil.which("docker")
    if docker is None:
        return "docker CLI not found on PATH"
    try:
        probe = subprocess.run(
            (docker, "info", "--format", "{{.ServerVersion}}"),
            capture_output=True,
            text=True,
            timeout=DOCKER_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"docker daemon did not answer within {DOCKER_PROBE_TIMEOUT}s"
    if probe.returncode:
        return f"docker daemon unreachable: {probe.stderr.strip()[-500:]}"
    return None


def local_factory() -> MachineFactory:
    """Construct the explicitly requested local backend, raising if this host cannot sandbox it."""
    # This optional backend imports bubblewrap_bin, which is installed only on Linux.
    from shellbox.backends.local.machine import LocalMachineFactory, SystemDirectories  # noqa: PLC0415

    return LocalMachineFactory(system_directories=SystemDirectories.COPY_ON_WRITE)


CONTAINER_BACKENDS = (Backend.GVISOR, Backend.DOCKER)


def container_backend(backends: Iterable[str]) -> Backend:
    """The backend of an image-backed machine: the one ``CONTAINER_BACKENDS`` entry among the selected ``backends``.

    ``backends`` are shellbox ``Backend`` values, such as the keys of ``machine_factories``.

    Raises:
        ValueError: ``backends`` selects no container backend, or more than one.
    """
    selected = set(backends)
    containers = [backend for backend in CONTAINER_BACKENDS if backend.value in selected]
    if len(containers) != 1:
        raise ValueError(
            f"An image machine needs exactly one container backend ({', '.join(CONTAINER_BACKENDS)}) "
            f"among the selected backends, not {sorted(selected)}"
        )
    return containers[0]


def machine_factories(backends: Sequence[Backend], controller_url: str | None) -> Mapping[str, MachineFactory]:
    """Construct exactly the requested backends, failing when a selected backend is unavailable.

    ``controller_url`` is required only when ``GVISOR`` is selected. Docker availability is checked
    only when Docker is requested; local sandbox construction propagates its availability error.
    ShellSim reads ``SHELLSIM_CATALOG`` when a machine first requests catalog packages.
    """
    if Backend.GVISOR in backends and controller_url is None:
        raise ValueError("A controller URL is required when the gvisor backend is selected")
    factories: dict[str, MachineFactory] = {}
    for backend in backends:
        match backend:
            case Backend.SHELLSIM:
                factory = ShellSimMachineFactory(context=shellsim.Context(pypi_catalog=SHELLSIM_CATALOG))
            case Backend.LOCAL:
                factory = local_factory()
            case Backend.GVISOR:
                assert controller_url is not None
                factory = IrisMachineFactory(
                    controller_url=controller_url, scheduling_timeout=SANDBOX_SCHEDULING_TIMEOUT
                )
            case Backend.DOCKER:
                unavailable = docker_unavailable_reason()
                if unavailable is not None:
                    raise RuntimeError(f"Requested docker backend unavailable: {unavailable}")
                factory = DockerMachineFactory()
            case _:
                raise ValueError(f"Unsupported Taskforge backend: {backend}")
        factories[backend.value] = factory
    return factories


def grading_environment(task: TaskSpec) -> EnvironmentRequirements | None:
    """The environment of the task's verifier machine, or ``None`` when its grader takes none.

    A verifyit judge takes none: RolloutEngine grades it on the host, through the judge service.
    """
    grader = task.grader
    if isinstance(grader, ScriptGrader | VerifyitGrader) and not grades_with_judge(grader):
        return grader.environment
    return None
