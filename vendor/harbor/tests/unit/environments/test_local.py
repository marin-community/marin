# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Harbor's prepared-service adapter in real local OCI sandboxes."""

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from harbor.environments.local import LocalEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import TrialPaths
from shellbox.backends.local.machine import SandboxUnavailable
from shellbox.image import ImageCache
from shellbox.machine import Command, MachineTerminated, UnsupportedMachineSpec


def local_environment(tmp_path: Path, **kwargs) -> LocalEnvironment:
    config = kwargs.pop(
        "task_env_config",
        EnvironmentConfig(
            docker_image="offline.test/task@sha256:" + "a" * 64,
            allow_internet=False,
            workdir="/app",
        ),
    )
    try:
        return LocalEnvironment(
            environment_dir=tmp_path / "environment",
            environment_name="prepared-task",
            session_id="trial",
            trial_paths=TrialPaths(tmp_path / "trial"),
            task_env_config=config,
            **kwargs,
        )
    except SandboxUnavailable as error:
        if os.environ.get("CI"):
            pytest.fail(str(error))
        pytest.skip(str(error))


@pytest.mark.parametrize(
    "resources",
    [
        {"cpus": 2},
        {"memory_mb": 1024},
        {"storage_mb": 1024},
        {"gpus": 1},
        {"tpu": {"type": "v6e", "topology": "2x4"}},
    ],
    ids=["cpu", "memory", "storage", "gpu", "tpu"],
)
def test_local_declared_resources_fail_without_allocating_a_sandbox(tmp_path, monkeypatch, resources):
    def acquire(cache, source):
        pytest.fail("An unsupported resource request reached image acquisition")

    monkeypatch.setattr(ImageCache, "prepare", acquire)

    async def run():
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
            await adapter.start(force_build=False)

    asyncio.run(run())


def test_local_prepared_services_keep_state_private_and_collect_verifier_logs(
    tmp_path, monkeypatch, prepared_busybox_image
):
    umoci = shutil.which("umoci")
    if umoci is None:
        if os.environ.get("CI"):
            pytest.fail("The prepared local runtime tests require umoci")
        pytest.skip("The prepared local runtime tests require umoci")
    image = prepared_busybox_image
    image_reference = "offline.test/busybox@" + image.manifest_digest
    monkeypatch.setattr(ImageCache, "prepare", lambda cache, source: image)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "host-storage-canary")
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "shellbox-images.json").write_text(
        json.dumps(
            {
                "main": {"image": image_reference, "workdir": "/app"},
                "runtime": {"image": image_reference, "workdir": "/app", "agent_logs": "publish"},
                "peer": {"image": image_reference, "workdir": "/app"},
            }
        )
    )
    (environment_dir / "compose.yaml").write_text(
        json.dumps(
            {
                "services": {
                    "main": {"depends_on": ["runtime", "peer"]},
                    "runtime": {
                        "expose": [8123],
                        "volumes": ["${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}"],
                    },
                    "peer": {
                        "expose": [8124],
                        "command": ["-c", "printf peer >index.html; exec httpd -f -p 8124 -h ."],
                    },
                }
            }
        )
    )
    config = EnvironmentConfig(allow_internet=False, cpus=2, memory_mb=1024)
    adapter = local_environment(
        tmp_path,
        task_env_config=config,
        cpu_enforcement_policy=ResourceMode.IGNORE,
        memory_enforcement_policy=ResourceMode.IGNORE,
        umoci=umoci,
    )

    async def run():
        await adapter.start(force_build=False)
        machine = adapter.machine
        roots = [machine.root, *(service.root for service in adapter.group.services.values())]
        try:
            # These minimal images have no Python healthcheck runner. Bound the
            # actor's first request while the real HTTP service processes start.
            response = await adapter.exec(
                "until wget -q -O /tmp/runtime http://runtime:8123 "
                "&& wget -q -O /tmp/peer http://peer:8124; do sleep 0.01; done; "
                "printf actor >index.html; cat /tmp/runtime /tmp/peer; "
                "printf '\\n%s\\n' \"${AWS_SECRET_ACCESS_KEY-unset}\"",
                timeout_sec=10,
                user="root",
            )
            assert (response.return_code, response.stdout) == (0, "from-imagepeer\nunset\n")
            verifier = await adapter.exec("cat /logs/agent/runtime_state.json; printf '\\n'; id -u; pwd", user="root")
            assert (verifier.return_code, verifier.stdout) == (0, '{"answer":42}\n0\n/app\n')
            downloaded = tmp_path / "verifier-state.json"
            await adapter.download_file("/logs/agent/runtime_state.json", downloaded)
            assert json.loads(downloaded.read_text()) == {"answer": 42}
            isolated = await adapter.exec("cat index.html", user="root")
            assert isolated.stdout == "actor"
        finally:
            await adapter.stop(delete=True)
        assert not any(root.exists() for root in roots)
        with pytest.raises(MachineTerminated):
            await machine.run(Command(("true",)))

    asyncio.run(run())
