# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The shellbox machine backends a host constructs: ShellSim, the local sandbox, Docker and Iris gVisor.

``machine_factories`` constructs the explicitly selected backends and keys them by shellbox ``Backend``
value, as ``taskforge.sandbox.factories`` and ``MachineRuntimeSpec.backend`` refer to them.
"""

import shutil
import subprocess
from collections.abc import Mapping, Sequence

import shellsim
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineFactory
from taskforge.sandbox.factories import SANDBOX_SCHEDULING_TIMEOUT

DOCKER_PROBE_TIMEOUT = 20
SHELLSIM_CATALOG = shellsim.DEFAULT_PYPI_CATALOG
"""The ports catalog ShellSim machines install ``draft.SHELLSIM_CPYTHON`` from, over HTTPS.

shellsim's published catalog, which Marin's inference tools also read by default. Each process
fetches the catalog itself; the first on a host downloads the CPython assets into shellsim's cache
(``$XDG_CACHE_HOME/shellsim``, else ``~/.cache/shellsim``), which later processes reuse.
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
