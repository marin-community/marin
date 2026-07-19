# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Context manager for a single Harbor sandbox on Iris compute.

Gives Marin code a Daytona-like sandbox experience backed by Iris: enter the
context to get a running sandbox, drive it with ``exec``/``upload``/``download``,
and it is torn down on exit. For N concurrent sandboxes, open N contexts with
``asyncio.gather`` — no pool machinery needed.

Usage:
    async with iris_sandbox(docker_image="python:3.13-slim", cluster="marin") as sandbox:
        result = await sandbox.exec("echo hello")

    async with iris_sandbox(task_dir="path/to/harbor-task", cluster="marin") as sandbox:
        await sandbox.upload_file(patch, "/app/fix.patch")
"""

import contextlib
import tempfile
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

from marin.harbor.iris_environment import GVISOR_PROFILE, IrisEnvironment


@dataclass(frozen=True)
class EnvironmentSpec:
    config: EnvironmentConfig
    environment_dir: Path
    name: str


def resolve_environment_spec(
    task_dir: Path | str | None,
    docker_image: str | None,
    empty_environment_dir: Path,
) -> EnvironmentSpec:
    """Resolve the sandbox environment spec from a task dir or bare image.

    Exactly one of ``task_dir`` or ``docker_image`` must be given.
    ``empty_environment_dir`` is used as the environment dir for the bare-image
    flavor; it must be empty so nothing is built or uploaded.
    """
    if (task_dir is None) == (docker_image is None):
        raise ValueError("Pass exactly one of task_dir or docker_image.")
    if task_dir is not None:
        task = Task(task_dir, disable_verification=True)
        return EnvironmentSpec(task.config.environment, Path(task.paths.environment_dir), task.short_name)
    assert docker_image is not None
    name = docker_image.rsplit("/", 1)[-1].split(":")[0]
    return EnvironmentSpec(EnvironmentConfig(docker_image=docker_image), empty_environment_dir, name)


@contextlib.asynccontextmanager
async def iris_sandbox(
    *,
    task_dir: Path | str | None = None,
    docker_image: str | None = None,
    cluster: str | None = None,
    controller_url: str | None = None,
    name: str | None = None,
    trial_dir: Path | str | None = None,
    cpus: int | None = None,
    memory_mb: int | None = None,
    storage_mb: int | None = None,
    container_profile: str = GVISOR_PROFILE,
) -> AsyncIterator[IrisEnvironment]:
    """Run a Harbor sandbox as an Iris job for the duration of the context.

    Exactly one of ``task_dir`` (a Harbor task directory whose ``[environment]``
    section defines the sandbox) or ``docker_image`` (a bare prebuilt image)
    must be given. Only prebuilt-image environments are supported.

    Args:
        task_dir: Harbor task directory to take the environment config from.
        docker_image: Prebuilt image to run directly, without a task.
        cluster: Iris cluster name (e.g. "marin"). Mutually exclusive with
            controller_url.
        controller_url: Direct controller URL, bypassing cluster config.
        name: Base name for the sandbox job; defaults to the task or image name.
        trial_dir: Where sandbox logs/artifacts land on the host. Defaults to a
            temporary directory that is removed on exit.
        cpus: Override the environment's CPU reservation.
        memory_mb: Override the environment's memory reservation.
        storage_mb: Override the environment's disk reservation.
        container_profile: Sandbox security profile; "gvisor" (default) runs
            the image under runsc with full in-container root. See
            IrisEnvironment.

    Yields:
        The started IrisEnvironment.
    """
    with contextlib.ExitStack() as stack:
        empty_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="iris-sandbox-env-")))
        spec = resolve_environment_spec(task_dir, docker_image, empty_dir)

        if trial_dir is None:
            trial_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix="iris-sandbox-trial-"))
        trial_paths = TrialPaths(trial_dir=trial_dir)
        trial_paths.mkdir()

        env = IrisEnvironment(
            environment_dir=spec.environment_dir,
            environment_name=spec.name,
            session_id=f"{name or spec.name}__{uuid.uuid4().hex[:8]}",
            trial_paths=trial_paths,
            task_env_config=spec.config,
            override_cpus=cpus,
            override_memory_mb=memory_mb,
            override_storage_mb=storage_mb,
            cluster=cluster,
            controller_url=controller_url,
            container_profile=container_profile,
        )
        await env.start(force_build=False)
        try:
            yield env
        finally:
            await env.stop(delete=True)
