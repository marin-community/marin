"""Shellbox backends shared by task workers and deployment preflight checks."""

import os
import shutil
import sys
from functools import partial

from daytona import AsyncDaytona, DaytonaConfig
from omegaconf import DictConfig
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineFactory


def local_machine_factory() -> MachineFactory:
    """Probe the host sandbox before registering local task execution."""
    if sys.platform != "linux":
        raise ValueError("Local task execution requires a Linux host with working bubblewrap sandboxes")
    # The local backend's bubblewrap package is available only on Linux.
    from shellbox.backends.local.machine import LocalMachineFactory

    return LocalMachineFactory()


def task_machine_factories(task_sessions: DictConfig) -> dict[str, MachineFactory]:
    """Build the configured task backends, probing host capabilities immediately."""
    factories: dict[str, MachineFactory] = {"shellsim": ShellSimMachineFactory()}
    if task_sessions.get("local") is not None:
        factories[Backend.LOCAL] = local_machine_factory()
    daytona = task_sessions.get("daytona")
    if daytona is not None:
        factories[Backend.DAYTONA] = DaytonaMachineFactory(
            client_factory=partial(AsyncDaytona, DaytonaConfig(api_key=os.environ[daytona.api_key_env])),
            ttl_minutes=int(daytona.ttl_minutes),
        )
    if shutil.which("docker") is not None:
        factories[Backend.DOCKER] = DockerMachineFactory()
    return factories
