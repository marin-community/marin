# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check LOCAL resource admission before allocating a sandbox."""

import asyncio

import pytest

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ResourceMode
from shellbox.image import ImageCache
from shellbox.machine import UnsupportedMachineSpec


from tests.unit.environments.local_runtime import local_environment

@pytest.mark.parametrize(
    "resources",
    [
        {"storage_mb": 1024},
        {"gpus": 1},
        {"tpu": {"type": "v6e", "topology": "2x4"}},
    ],
    ids=["storage", "gpu", "tpu"],
)
def test_local_unsupported_resources_fail_without_allocating_a_sandbox(
    tmp_path, monkeypatch, resources
):
    def acquire(cache, source):
        pytest.fail("An unsupported resource request reached image acquisition")

    monkeypatch.setattr(ImageCache, "prepare", acquire)

    async def run():
        async with asyncio.TaskGroup() as service_tasks:
            with pytest.raises((ValueError, UnsupportedMachineSpec)):
                adapter = local_environment(
                    tmp_path,
                    task_env_config=EnvironmentConfig(
                        docker_image="offline.test/task@sha256:" + "a" * 64,
                        allow_internet=False,
                        workdir="/app",
                        **resources,
                    ),
                )
                await adapter.start(service_tasks=service_tasks, force_build=False)

    asyncio.run(run())


@pytest.mark.parametrize("resource", ["cpu", "memory"])
def test_local_rejects_ignore_policy_before_allocating_a_sandbox(
    tmp_path, monkeypatch, resource
):
    def acquire(cache, source):
        pytest.fail("An ignored resource request reached image acquisition")

    monkeypatch.setattr(ImageCache, "prepare", acquire)
    with pytest.raises(ValueError):
        local_environment(
            tmp_path, **{f"{resource}_enforcement_policy": ResourceMode.IGNORE}
        )


async def test_local_numeric_limits_do_not_bypass_required_containment(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "shellbox.backends.local.machine.delegated_cgroup_parent", lambda: None
    )

    def acquire(cache, source):
        pytest.fail("A worker without containment reached image acquisition")

    monkeypatch.setattr(ImageCache, "prepare", acquire)
    adapter = local_environment(
        tmp_path,
        task_env_config=EnvironmentConfig(
            docker_image="offline.test/task@sha256:" + "a" * 64,
            allow_internet=False,
            cpus=2,
            memory_mb=1024,
            workdir="/app",
        ),
    )
    async with asyncio.TaskGroup() as service_tasks:
        with pytest.raises(UnsupportedMachineSpec):
            await adapter.start(False, service_tasks=service_tasks)
