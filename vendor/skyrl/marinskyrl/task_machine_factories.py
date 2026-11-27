"""Shellbox backends shared by task workers and deployment preflight checks."""

import shutil
import tempfile
from functools import partial
from pathlib import Path

from daytona import AsyncDaytona, DaytonaConfig
from omegaconf import DictConfig
from rigging.secrets import require_secret
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.local.disk import LocalDiskPool
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineFactory


def registry_authfile(secret: str) -> Path:
    """Write a registry credential (a Docker config JSON secret) to a private file for LOCAL image pulls."""
    path = Path(tempfile.mkdtemp(prefix="shellbox-registry-auth-")) / "auth.json"
    path.write_text(require_secret(secret))
    path.chmod(0o600)
    return path


def task_machine_factories(task_sessions: DictConfig, *, phase: str = "train") -> dict[str, MachineFactory]:
    """Build the configured task backends, probing host capabilities immediately."""
    factories: dict[str, MachineFactory] = {"shellsim": ShellSimMachineFactory()}
    if task_sessions.get("local") is not None:
        root = task_sessions.local.get("disk_root")
        auth_secret = task_sessions.local.get("registry_auth_secret")
        factories[Backend.LOCAL] = LocalMachineFactory(
            disk_pool=None if root is None else LocalDiskPool(Path(root)),
            authfile=None if auth_secret is None else registry_authfile(auth_secret),
        )
    daytona = task_sessions.get("daytona")
    if daytona is not None:
        factories[Backend.DAYTONA] = DaytonaMachineFactory(
            client_factory=partial(AsyncDaytona, DaytonaConfig(api_key=require_secret(daytona.get("eval_api_key_env", daytona.api_key_env) if phase == "eval" else daytona.api_key_env))),
            ttl_minutes=int(daytona.ttl_minutes),
        )
    if shutil.which("docker") is not None:
        factories[Backend.DOCKER] = DockerMachineFactory()
    return factories
