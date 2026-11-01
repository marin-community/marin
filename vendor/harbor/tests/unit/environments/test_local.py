# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Harbor's prepared-service adapter in real local OCI sandboxes."""

import asyncio
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from harbor.environments import local, model_relay
from harbor.environments.local import LocalEnvironment
from harbor.environments.model_relay import host_model_authentication
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import TrialPaths
from shellbox.backends.local.machine import SandboxUnavailable
from shellbox.image import ImageCache
from shellbox.machine import Command, ExitReason, MachineTerminated, Result, UnsupportedMachineSpec
from shellbox.task import ReadinessProbe, Task
from tests.unit.environments.transport import ProcessMachine, available_port


class ProcessTaskFactory:
    """Substitute the sandbox process boundary while retaining task lifecycle."""

    def __init__(self, root: Path):
        self.root = root

    async def create_task(self, spec):
        machine = ProcessMachine(self.root, spec.primary_container.machine)
        return Task({spec.primary: machine}, spec, machine.close)


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
        roots = [container.root for container in adapter.execution.containers.values()]
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


async def test_local_candidate_and_judge_relays_route_independently_without_guest_credentials(tmp_path, monkeypatch):
    (tmp_path / "environment").mkdir()
    candidate_port, judge_port = available_port(), available_port()
    monkeypatch.setattr(local, "CANDIDATE_MODEL_RELAY_PORT", candidate_port)
    monkeypatch.setattr(local, "MODEL_RELAY_PORT", judge_port)
    monkeypatch.setenv("OPENAI_API_KEY", "unforwarded-host-secret")
    original_client = httpx.AsyncClient

    def provider(request):
        candidate = request.url.host == "candidate.test"
        expected_model = "candidate" if candidate else "judge"
        expected_key = "candidate-private" if candidate else "judge-private"
        assert request.headers["authorization"] == f"Bearer {expected_key}"
        payload = json.loads(request.content)
        assert payload["model"] == expected_model
        assert str(request.url) == (
            "https://candidate.test/proxy/t/private-capability/v1/chat/completions"
            if candidate else "https://judge.test/v1/chat/completions"
        )
        return httpx.Response(200, json={"model": expected_model})

    monkeypatch.setattr(
        model_relay.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(provider), **kwargs),
    )
    adapter = local_environment(
        tmp_path,
        candidate_model_endpoint="https://candidate.test/proxy/t/private-capability/v1",
        candidate_model_name="candidate",
        model_endpoint="https://judge.test/v1",
        model_name="judge",
        verifier_model_endpoint="https://judge.test/v1",
    )
    adapter.factory = ProcessTaskFactory(tmp_path / "guest")
    with host_model_authentication({
        "https://candidate.test/proxy/t/private-capability/v1": "candidate-private",
        "https://judge.test/v1": "judge-private",
    }):
        await adapter.start(force_build=False)
    machine = adapter.machine
    try:
        script = f"""
import json, os, urllib.request, urllib.error
assert os.environ['OPENAI_API_KEY'] == 'EMPTY'
assert not any('private' in value or 'host-secret' in value for value in os.environ.values())
for port, model in [({candidate_port}, 'candidate'), ({judge_port}, 'judge')]:
    request = urllib.request.Request(
        'http://127.0.0.1:' + str(port) + '/v1/chat/completions',
        data=json.dumps({{'model': 'guest-choice', 'messages': []}}).encode(),
        headers={{'Authorization': 'Bearer EMPTY'}})
    with urllib.request.urlopen(request) as response:
        assert json.load(response) == {{'model': model}}
try:
    urllib.request.urlopen('http://127.0.0.1:{candidate_port}/v1/models')
    raise AssertionError('unsupported model route accepted')
except urllib.error.HTTPError as error:
    assert error.code == 404
print('ok')
"""
        response = await machine.run(Command((sys.executable, "-c", script), env={"OPENAI_API_KEY": "EMPTY"}))
        assert response.exit_code == 0, response.stderr
        assert response.stdout == b"ok\n"
    finally:
        await adapter.stop(delete=True)
    assert machine.closed


@pytest.mark.parametrize("phase", ["exec", "readiness", "upload", "download"])
@pytest.mark.parametrize("interruption", ["cancel", "timeout"])
async def test_local_interrupted_operation_keeps_original_error_through_relay_cleanup(
    tmp_path, monkeypatch, phase, interruption
):
    (tmp_path / "environment").mkdir()
    monkeypatch.setattr(local, "CANDIDATE_MODEL_RELAY_PORT", available_port())
    io_started = asyncio.Event()
    blocked_command = Command((sys.executable, "-c", "import signal; signal.pause()"))

    class ClosingProcessMachine(ProcessMachine):
        async def run(self, command):
            session = await self.open_process(command)
            io_started.set()
            try:
                async with asyncio.timeout(command.timeout):
                    code = await session.wait()
            except BaseException as error:
                # Model LocalMachine's group teardown at the machine I/O boundary.
                await self.close()
                if isinstance(error, TimeoutError):
                    return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
                raise
            finally:
                await session.close()
            return Result(code, b"", b"", False, False, ExitReason.EXITED)

        async def upload(self, source, target):
            if target == "/interrupted-upload":
                await self.run(blocked_command)
                return
            await super().upload(source, target)

        async def download(self, source, target):
            if source == "/interrupted-download":
                await self.run(blocked_command)
                return
            await super().download(source, target)

    class ClosingTaskFactory(ProcessTaskFactory):
        async def create_task(self, spec):
            if phase == "readiness":
                spec = replace(
                    spec,
                    containers=(
                        replace(
                            spec.primary_container,
                            readiness=ReadinessProbe(
                                blocked_command, timeout=1 if interruption == "timeout" else 30, interval=0.1
                            ),
                        ),
                    ),
                )
            self.machine = ClosingProcessMachine(self.root, spec.primary_container.machine)
            return Task({spec.primary: self.machine}, spec, self.machine.close)

    adapter = local_environment(
        tmp_path,
        candidate_model_endpoint="https://candidate.test/v1",
        candidate_model_name="candidate",
    )
    factory = ClosingTaskFactory(tmp_path / "guest")
    adapter.factory = factory
    if phase != "readiness":
        # Harbor's startup phase finishes before later I/O phases use the relay.
        await asyncio.create_task(adapter.start(force_build=False))
        io_started.clear()

    async def operate():
        if phase == "readiness":
            await adapter.start(force_build=False)
        elif phase == "exec":
            await adapter.exec(
                "exec python3 -c 'import signal; signal.pause()'", timeout_sec=1 if interruption == "timeout" else None
            )
        else:
            async with asyncio.timeout(1 if interruption == "timeout" else None):
                if phase == "upload":
                    await adapter.upload_file(tmp_path / "input", "/interrupted-upload")
                else:
                    await adapter.download_dir("/interrupted-download", tmp_path / "download")

    operation = asyncio.create_task(operate())
    await asyncio.wait_for(io_started.wait(), 5)
    machine = factory.machine
    if interruption == "cancel":
        operation.cancel("original operation cancellation")
    try:
        with pytest.raises(asyncio.CancelledError if interruption == "cancel" else TimeoutError) as caught:
            await operation
        if interruption == "cancel":
            assert caught.value.args == ("original operation cancellation",)
        else:
            assert operation.cancelling() == 0
        await adapter.stop(delete=True)
        assert machine.closed
        assert adapter.machine is None
        assert all(session.process.returncode is not None for session in machine.sessions)
    finally:
        await machine.close()
