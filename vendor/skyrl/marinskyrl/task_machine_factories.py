"""Shellbox backends shared by task workers and deployment preflight checks."""

import os
import shutil
from functools import partial

from daytona import AsyncDaytona, DaytonaConfig
from omegaconf import DictConfig, OmegaConf
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineFactory

from shellbox.backends.iris.local_images import LocalMachineSettings, local_machine_factory
from shellbox.backends.iris.machine_service import MachineServiceBinding
from marinskyrl.local_service import local_service_factory


def task_machine_factories(task_sessions: DictConfig) -> dict[str, MachineFactory]:
    """Build the configured task backends, probing host capabilities immediately."""
    factories: dict[str, MachineFactory] = {"shellsim": ShellSimMachineFactory()}
    if task_sessions.get("local") is not None and task_sessions.get("local_service") is not None:
        raise ValueError("Configure exactly one LOCAL factory: local or local_service")
    service = task_sessions.get("local_service")
    if service is not None:
        settings = MachineServiceBinding.model_validate(OmegaConf.to_container(service, resolve=True))
        factories[Backend.LOCAL] = local_service_factory(settings)
    if task_sessions.get("local") is not None:
        local_settings = LocalMachineSettings.model_validate(OmegaConf.to_container(task_sessions.local, resolve=True))
        factories[Backend.LOCAL] = local_machine_factory(local_settings)
    daytona = task_sessions.get("daytona")
    if daytona is not None:
        factories[Backend.DAYTONA] = DaytonaMachineFactory(
            client_factory=partial(AsyncDaytona, DaytonaConfig(api_key=os.environ[daytona.api_key_env])),
            ttl_minutes=int(daytona.ttl_minutes),
        )
    if shutil.which("docker") is not None:
        factories[Backend.DOCKER] = DockerMachineFactory()
    return factories
