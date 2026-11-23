"""Shellbox backends shared by task workers and deployment preflight checks."""

import os
import shutil

from omegaconf import DictConfig
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineFactory


def task_machine_factories(task_sessions: DictConfig) -> dict[str, MachineFactory]:
    """Build the configured task backends, probing host capabilities immediately."""
    factories: dict[str, MachineFactory] = {"shellsim": ShellSimMachineFactory()}
    if task_sessions.get("local") is not None:
        factories[Backend.LOCAL] = LocalMachineFactory()
    daytona = task_sessions.get("daytona")
    if daytona is not None:
        factories[Backend.DAYTONA] = DaytonaMachineFactory(
            client_factory=DaytonaConnection(api_key=os.environ[daytona.api_key_env]).client_factory(),
            ttl_minutes=int(daytona.ttl_minutes),
        )
    if shutil.which("docker") is not None:
        factories[Backend.DOCKER] = DockerMachineFactory()
    return factories
