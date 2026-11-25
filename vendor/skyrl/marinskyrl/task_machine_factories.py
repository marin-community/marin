"""Shellbox backends shared by task workers and deployment preflight checks."""

import os
import shutil

from omegaconf import DictConfig
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.machine import DaytonaImageBuild, DaytonaMachineFactory
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineFactory
from skyrlrecipe.declared import DAYTONA_EVAL_BACKEND


def _daytona_factory(api_key: str, ttl_minutes: int, image_build: DaytonaImageBuild) -> DaytonaMachineFactory:
    connection = DaytonaConnection(api_key=api_key)
    return DaytonaMachineFactory(
        client_factory=connection.client_factory(), ttl_minutes=ttl_minutes, image_build=image_build
    )


def task_machine_factories(task_sessions: DictConfig) -> dict[str, MachineFactory]:
    """Build the configured task backends, probing host capabilities immediately."""
    factories: dict[str, MachineFactory] = {"shellsim": ShellSimMachineFactory()}
    if task_sessions.get("local") is not None:
        factories[Backend.LOCAL] = LocalMachineFactory()
    daytona = task_sessions.get("daytona")
    if daytona is not None:
        api_key = os.environ[daytona.api_key_env]
        image_build = DaytonaImageBuild(daytona.get("image_build", DaytonaImageBuild.SNAPSHOT))
        factories[Backend.DAYTONA] = _daytona_factory(api_key, int(daytona.ttl_minutes), image_build)
        evaluation_key_env = daytona.get("eval_api_key_env")
        if (
            evaluation_key_env
            or daytona.get("eval_image_build") is not None
            or daytona.get("eval_ttl_minutes") is not None
        ):
            factories[DAYTONA_EVAL_BACKEND] = _daytona_factory(
                os.environ[evaluation_key_env] if evaluation_key_env else api_key,
                int(daytona.get("eval_ttl_minutes", daytona.ttl_minutes)),
                DaytonaImageBuild(daytona.get("eval_image_build", image_build)),
            )
    if shutil.which("docker") is not None:
        factories[Backend.DOCKER] = DockerMachineFactory()
    return factories
