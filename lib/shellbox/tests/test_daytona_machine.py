# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Daytona adapter against a local process and filesystem fake."""

import asyncio
import json
import math
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from pathlib import Path
from ssl import SSLCertVerificationError
from types import SimpleNamespace

import pytest
from aiohttp import ClientResponseError, RequestInfo, web
from aiohttp.test_utils import TestServer

pytest.importorskip("daytona")

from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaConfig,
    DaytonaConflictError,
    DaytonaConnectionError,
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaTimeoutError,
    DaytonaValidationError,
    Image,
    Resources,
    SandboxState,
)
from daytona._utils.errors import intercept_errors
from daytona_api_client_async import SnapshotState
from multidict import CIMultiDict, CIMultiDictProxy
from rigging.task_errors import RetryDisposition
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.errors import daytona_error_identity
from shellbox.backends.daytona.machine import (
    PROVIDER_DIAGNOSTIC_CHAR_LIMIT,
    DaytonaImageBuild,
    DaytonaMachine,
    DaytonaMachineFactory,
    DaytonaNetworkMode,
    DaytonaNetworkPolicy,
    DaytonaOutputUnavailable,
    _create_sandbox,
    _delete_named_sandbox,
)
from shellbox.backends.daytona.snapshot import (
    SnapshotRequest,
    list_snapshots,
    purge_stale_snapshots,
    snapshot_name,
    snapshot_request_for_spec,
    snapshot_resources,
)
from shellbox.backends.daytona.transient import retry_transient
from shellbox.backends.docker.machine import DockerMachineFactory, docker
from shellbox.capabilities import MachineCapability
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import (
    Command,
    CommandOutputLost,
    DockerImage,
    ExitReason,
    MachineDiskFull,
    MachineSpec,
    MachineTerminated,
    MachineUnavailable,
    TaskRequirements,
    UnsupportedMachineSpec,
)
from yarl import URL


def sandbox_json(sandbox_id: str, state: str, toolbox_url: str) -> dict:
    """A Daytona API sandbox record named ``fixture``."""
    return {
        "id": sandbox_id,
        "organizationId": "fixture",
        "name": "fixture",
        "user": "root",
        "env": {},
        "labels": {},
        "public": False,
        "networkBlockAll": False,
        "target": "us",
        "cpu": 2,
        "gpu": 0,
        "memory": 4,
        "disk": 10,
        "state": state,
        "toolboxProxyUrl": toolbox_url,
    }


@pytest.mark.parametrize("failure", ["rejected_post", "late_get", "prolonged_throttle", "deadline", "cancelled"])
def test_daytona_creation_recovers_named_sandbox_after_throttling(monkeypatch, failure):
    async def scenario():
        requests = []
        delays = []
        waiting = asyncio.Event()

        async def wait(delay):
            delays.append(delay)
            waiting.set()
            if failure in {"deadline", "cancelled"}:
                await asyncio.Future()

        monkeypatch.setattr("shellbox.backends.daytona.machine.sleep", wait)

        created = False

        async def endpoint(request):
            nonlocal created
            if request.headers.get("Upgrade", "").lower() == "websocket":
                websocket = web.WebSocketResponse()
                await websocket.prepare(request)
                async for _ in websocket:
                    pass
                return websocket
            requests.append(request.method)
            if request.method == "POST" and created:
                return web.json_response({"message": "name already exists"}, status=409)
            if (request.method == "GET" and created and requests.count("GET") > 1) or (
                request.method == "POST"
                and (
                    failure == "late_get"
                    or (failure == "rejected_post" and requests.count("POST") == 2)
                    or (failure == "prolonged_throttle" and requests.count("POST") == 10)
                )
            ):
                created = True
                return web.json_response(
                    sandbox_json(
                        "created-once",
                        "started",
                        "" if failure == "late_get" and request.method == "POST" else "https://fixture.invalid/toolbox",
                    )
                )
            return web.json_response(
                {"message": "rate limited"}, status=429, headers={"Retry-After-sandbox-create": "17"}
            )

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", endpoint)
        async with TestServer(app) as server:
            async with AsyncDaytona(
                DaytonaConfig(api_key="fixture", api_url=str(server.make_url("")).rstrip("/"))
            ) as client:
                params = CreateSandboxFromSnapshotParams(name="fixture", snapshot="fixture")
                if failure in {"rejected_post", "prolonged_throttle"}:
                    sandbox = await _create_sandbox(client, params, 60)
                    assert sandbox.id == "created-once"
                    attempts = 2 if failure == "rejected_post" else 10
                    assert requests == ["POST"] * attempts
                    assert len(delays) == attempts - 1 and all(delay >= 17 for delay in delays)
                elif failure == "deadline":
                    with pytest.raises(TimeoutError):
                        async with asyncio.timeout(None) as deadline:

                            async def expire_when_waiting():
                                await waiting.wait()
                                deadline.reschedule(asyncio.get_running_loop().time())

                            expiration = asyncio.create_task(expire_when_waiting())
                            try:
                                await _create_sandbox(client, params, 60)
                            finally:
                                await expiration
                    assert requests == ["POST"]
                elif failure == "cancelled":
                    task = asyncio.create_task(_create_sandbox(client, params, 60))
                    await waiting.wait()
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert requests == ["POST"]
                else:
                    sandbox = await _create_sandbox(client, params, 60)
                    assert sandbox.id == "created-once"
                    assert requests == ["POST", "GET", "POST", "GET"]
                    assert len(delays) == 1 and delays[0] >= 17

    asyncio.run(scenario())


def test_daytona_creation_reports_a_sandbox_removed_while_starting_as_unavailable():
    async def scenario():
        async def endpoint(request):
            if request.method == "POST":
                return web.json_response(sandbox_json("removed", "starting", "https://fixture.invalid/toolbox"))
            return web.json_response({"message": "Sandbox with ID or name removed not found"}, status=404)

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", endpoint)
        async with TestServer(app) as server:
            async with AsyncDaytona(
                DaytonaConfig(api_key="fixture", api_url=str(server.make_url("")).rstrip("/"))
            ) as client:
                params = CreateSandboxFromSnapshotParams(name="fixture", snapshot="fixture")
                with pytest.raises(MachineUnavailable):
                    await _create_sandbox(client, params, 60)

    asyncio.run(scenario())


def command_script(command: str) -> str:
    script = shlex.split(command)[2].partition("export PATH; ")[2]
    argv = shlex.split(script)
    if len(argv) == 2 and argv[0] == "/bin/sh" and argv[1].endswith("/command.sh"):
        return Path(argv[1]).read_text()
    return script


class LocalFiles:
    async def upload_file_stream(self, data: bytes, target: str) -> None:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def download_file(self, source: str) -> bytes:
        return Path(source).read_bytes()


class LocalProcess:
    def __init__(self):
        self.processes = []

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None, timeout=None):
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self.processes.append(process)
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return SimpleNamespace(exit_code=process.returncode, result=stdout.decode(errors="replace"))

    async def close(self):
        for process in self.processes:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await process.wait()


class LocalSnapshots:
    def __init__(self):
        self.snapshots = {}

    async def get(self, name):
        if name not in self.snapshots:
            raise DaytonaNotFoundError("Snapshot does not exist", status_code=404)
        return self.snapshots[name]

    async def create(self, params):
        snapshot = SimpleNamespace(name=params.name, state=SnapshotState.ACTIVE, params=params)
        self.snapshots[params.name] = snapshot
        return snapshot


@pytest.mark.skipif(sys.platform != "linux", reason="Linux execve per-argument limit regression")
@pytest.mark.parametrize("shell", ["/bin/bash", "/bin/sh"])
def test_daytona_large_argument_preserves_process_contract(tmp_path: Path, shell: str) -> None:
    class RecordedFiles(LocalFiles):
        def __init__(self):
            self.staging = set()

        async def upload_file_stream(self, data: bytes, target: str) -> None:
            self.staging.add(Path(target).parent)
            await super().upload_file_stream(data, target)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.fs = RecordedFiles()
        workdir = tmp_path / "work with spaces"
        workdir.mkdir()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(source=RegistryImage("fixture@sha256:" + "a" * 64), workdir=str(workdir))
        )
        body = '/bin/cat; printf "|%s|%s|%s|%s" "$VALUE" "$PWD" "$PATH" "$0"; printf "error" >&2; exit 23'
        # The reported setup fits one Linux argument; nested control quoting did not.
        padding = 130712 - len(body.encode()) - 2
        source = "#" + "x" * padding + "\n" + body
        try:
            result = await machine.run(
                Command(
                    (shell, "-c", source),
                    stdin=b"input\x00\xff",
                    env={"VALUE": "quoted '\" value", "PATH": "/nonexistent"},
                    timeout=10,
                )
            )
            assert result.exit_code == 23
            assert result.stdout == b"input\x00\xff" + f"|quoted '\" value|{workdir}|/nonexistent|{shell}".encode()
            assert result.stderr == b"error"
            assert not result.stdout_truncated and not result.stderr_truncated
            assert all(not path.exists() for path in client.sandbox.fs.staging)
        finally:
            await machine.close()

    asyncio.run(scenario())


class LocalDaytona:
    def __init__(self):
        self.sandbox = SimpleNamespace(id="fixture", fs=LocalFiles(), process=LocalProcess())
        self.deleted = False
        self.closed = False
        self.params = None
        self.timeout = None
        self.snapshot = LocalSnapshots()

    async def create(self, params, *, timeout):
        if isinstance(params, CreateSandboxFromSnapshotParams):
            await self.snapshot.get(params.snapshot)
        else:
            assert isinstance(params, CreateSandboxFromImageParams)
        self.params = params
        self.timeout = timeout
        return self.sandbox

    async def get(self, name):
        if self.params is None or self.params.name != name:
            raise DaytonaNotFoundError("Sandbox does not exist", status_code=404)
        return self.sandbox

    async def delete(self, sandbox):
        assert sandbox is self.sandbox
        await sandbox.process.close()
        self.deleted = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        self.closed = True


class FailedDeletion(LocalDaytona):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure

    async def delete(self, sandbox):
        await sandbox.process.close()
        raise self.failure


@pytest.fixture
def lifecycle_client():
    client = LocalDaytona()

    async def prepared(_command, **_kwargs):
        return SimpleNamespace(exit_code=0, result="/work\n")

    client.sandbox.process.exec = prepared
    return client


@pytest.mark.parametrize("image_build", list(DaytonaImageBuild))
@pytest.mark.parametrize("auto_stop_interval", [None, 9])
def test_daytona_creation_sets_orphan_cleanup_and_caller_labels(lifecycle_client, image_build, auto_stop_interval):
    async def scenario():
        labels = {"owner": "agent-eval", "lane": "eval", "job": "swe-rebench"}
        options = {} if auto_stop_interval is None else {"auto_stop_interval": auto_stop_interval}
        factory = DaytonaMachineFactory(
            lambda: lifecycle_client, image_build=image_build, ttl_minutes=75, labels=labels, **options
        )
        labels["job"] = "another-job"
        machine = await factory.create(MachineSpec(source=RegistryImage("fixture:latest")))
        try:
            params = lifecycle_client.params
            assert params.auto_stop_interval == (30 if auto_stop_interval is None else 9)
            assert params.auto_delete_interval == 0
            assert params.ttl_minutes == 75
            assert params.labels == {"owner": "agent-eval", "lane": "eval", "job": "swe-rebench"}
        finally:
            await machine.close()
        assert lifecycle_client.deleted and lifecycle_client.closed

    asyncio.run(scenario())


def test_daytona_open_machine_renews_activity_after_provider_failure_and_stops_before_deletion(
    lifecycle_client, monkeypatch
):
    async def scenario():
        ticks = asyncio.Queue()
        events = []
        refreshing = asyncio.Event()

        async def wait(interval):
            tick = asyncio.get_running_loop().create_future()
            await ticks.put((interval, tick))
            await tick

        async def refresh_activity(**_kwargs):
            events.append("refresh")
            if events.count("refresh") == 1:
                raise DaytonaError("temporary gateway failure", status_code=502)
            if events.count("refresh") == 2:
                return
            refreshing.set()
            try:
                await asyncio.Future()
            finally:
                events.append("refresh stopped")

        delete = lifecycle_client.delete

        async def record_deletion(sandbox):
            events.append("delete")
            await delete(sandbox)

        lifecycle_client.sandbox.refresh_activity = refresh_activity
        lifecycle_client.delete = record_deletion
        monkeypatch.setattr("shellbox.backends.daytona.machine.sleep", wait)
        machine = await DaytonaMachineFactory(lambda: lifecycle_client, auto_stop_interval=9).create(
            MachineSpec(source=RegistryImage("fixture:latest"))
        )
        try:
            for _ in range(3):
                interval, tick = await ticks.get()
                assert interval == 180
                tick.set_result(None)
            await refreshing.wait()
        finally:
            await machine.close()
        assert events == ["refresh", "refresh", "refresh", "refresh stopped", "delete"]
        assert lifecycle_client.deleted and lifecycle_client.closed

    asyncio.run(scenario())


def test_daytona_failed_setup_cancels_activity_renewal(lifecycle_client, monkeypatch):
    async def scenario():
        sleeping = asyncio.Event()
        stopped = asyncio.Event()

        async def wait(_interval):
            sleeping.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        async def failed_setup(_command, **_kwargs):
            await sleeping.wait()
            raise DaytonaError("setup failed", status_code=500)

        lifecycle_client.sandbox.process.exec = failed_setup
        monkeypatch.setattr("shellbox.backends.daytona.machine.sleep", wait)
        with pytest.raises(DaytonaError):
            await DaytonaMachineFactory(lambda: lifecycle_client).create(
                MachineSpec(source=RegistryImage("fixture:latest"))
            )
        assert stopped.is_set()
        assert lifecycle_client.deleted and lifecycle_client.closed
        assert lifecycle_client.params.labels == {"owner": "shellbox"}

    asyncio.run(scenario())


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_daytona_failed_creation_deletes_an_accepted_sandbox(error_type):
    class AcceptedThenFailed(LocalDaytona):
        async def create(self, params, *, timeout):
            await super().create(params, timeout=timeout)
            raise error_type("SDK failed after acceptance")

    async def scenario():
        client = AcceptedThenFailed()
        factory = DaytonaMachineFactory(client_factory=lambda: client)
        with pytest.raises(error_type):
            await factory.create(MachineSpec(source=RegistryImage("fixture@sha256:" + "a" * 64), cpus=2))
        assert client.deleted
        assert client.closed

    asyncio.run(scenario())


def test_snapshot_preflight_builds_without_starting_sandbox_and_reuses_image():
    client = LocalDaytona()
    factory = DaytonaMachineFactory(client_factory=lambda: client)
    spec = MachineSpec(source=RegistryImage("fixture@sha256:" + "a" * 64), cpus=2, memory_mb=2049)

    async def scenario():
        first = await factory.prepare(spec)
        second = await factory.prepare(spec)
        assert first == second
        assert len(client.snapshot.snapshots) == 1
        snapshot = await client.snapshot.get(first)
        assert snapshot.state == SnapshotState.ACTIVE
        assert snapshot.params.resources.memory == 3
        assert client.params is None

    asyncio.run(scenario())


@pytest.mark.parametrize("policy", [None, DaytonaNetworkPolicy(DaytonaNetworkMode.DOMAIN_ALLOW_LIST, "example.org")])
@pytest.mark.requires_host_setsid
def test_daytona_binary_command_and_files(tmp_path: Path, policy) -> None:
    async def scenario() -> None:
        client = LocalDaytona()
        workdir = tmp_path / "work"
        machine = await DaytonaMachineFactory(lambda: client, network_policy=policy).create(
            MachineSpec(
                source=RegistryImage("ubuntu:24.04"),
                workdir=str(workdir),
                cpus=2,
                memory_mb=1500,
                storage_mb=1025,
                env={"TASK_VALUE": "daytona"},
                startup_timeout=900,
            )
        )
        if policy is None:
            assert client.params.network_block_all is True
            assert client.params.domain_allow_list is None
        else:
            assert client.params.network_block_all is None
            assert client.params.domain_allow_list == "example.org"
        assert client.params.ttl_minutes == 360
        resources = (await client.snapshot.get(client.params.snapshot)).params.resources
        assert (resources.cpu, resources.memory, resources.disk) == (2, 2, 2)
        assert client.params.os_user == "root"
        assert client.timeout == 900
        try:
            environment = await machine.run(Command(("sh", "-c", 'printf "%s" "$TASK_VALUE"'), user="0"))
            assert environment.stdout == b"daytona"
            result = await machine.run(
                Command(
                    ("/bin/sh", "-c", "cat; printf '\\000\\377' >&2"),
                    stdin=b"abc\x00\xff",
                    output_limit_bytes=4,
                )
            )
            assert result.exit_code == 0
            assert result.stdout == b"abc\x00"
            assert result.stdout_truncated
            assert result.stderr == b"\x00\xff"

            source = tmp_path / "input.bin"
            source.write_bytes(b"\x00\xffpayload")
            await machine.upload(source, str(workdir / "data.bin"))
            downloaded = tmp_path / "downloaded.bin"
            await machine.download(str(workdir / "data.bin"), downloaded)
            assert downloaded.read_bytes() == source.read_bytes()
        finally:
            await machine.close()
        assert client.deleted
        assert client.closed

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
@pytest.mark.parametrize("inherit_image_workdir", [False, True])
def test_daytona_recreated_workdir_reopens_path_for_next_command(tmp_path: Path, inherit_image_workdir: bool) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    original_inode = workdir.stat().st_ino

    class StaleWorkdirProcess(LocalProcess):
        def __init__(self):
            super().__init__()
            self.workdir_replaced = False

        async def exec(self, command, cwd=None, env=None, timeout=None):
            if cwd is None:
                if self.workdir_replaced:
                    raise OSError("Provider process retained the deleted working directory")
                cwd = str(workdir)
            result = await super().exec(command, cwd=cwd, env=env, timeout=timeout)
            self.workdir_replaced = workdir.stat().st_ino != original_inode
            return result

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = StaleWorkdirProcess()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="" if inherit_image_workdir else str(workdir))
        )
        try:
            replaced = await machine.run(Command(("sh", "-c", 'rm -rf "$PWD" && mkdir -p "$PWD" && printf recreated')))
            assert (replaced.exit_code, replaced.stdout) == (0, b"recreated")
            following = await machine.run(Command((sys.executable, "-c", "import os; print(os.getcwd())")))
            assert (following.exit_code, following.stdout) == (0, f"{workdir}\n".encode())
            assert workdir.stat().st_ino != original_inode
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_output_count_ignores_provider_diagnostics(tmp_path: Path) -> None:
    class NoisyCountProcess(LocalProcess):
        async def exec(self, command, **kwargs):
            result = await super().exec(command, **kwargs)
            result.result = "shell-init: getcwd failed\n" + result.result
            return result

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = NoisyCountProcess()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            result = await machine.run(Command(("printf", "output"), output_limit_bytes=3))
            assert (result.exit_code, result.stdout, result.stdout_truncated) == (0, b"out", True)
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_output_removed_by_the_guest_is_reported_as_lost(tmp_path: Path) -> None:
    class ClearedTmpProcess(LocalProcess):
        armed = False

        async def exec(self, command, **kwargs):
            if "wc -c" in command and self.armed:
                # Stands in for a guest process (e.g. systemd starting) that clears /tmp once.
                prefix = re.search(r"/tmp/\.shellbox-[0-9a-f]+", command)
                assert prefix is not None
                shutil.rmtree(prefix.group(0), ignore_errors=True)
                self.armed = False
            return await super().exec(command, **kwargs)

    async def scenario():
        client = LocalDaytona()
        process = ClearedTmpProcess()
        client.sandbox.process = process
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        process.armed = True
        try:
            with pytest.raises(CommandOutputLost):
                await machine.run(Command(("printf", "output")))
            following = await machine.run(Command(("printf", "ready")))
            assert (following.exit_code, following.stdout) == (0, b"ready")
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.skipif(shutil.which("setsid") is None, reason="Daytona command execution needs setsid")
@pytest.mark.parametrize("collection_timeout, succeeds", [(0.01, False), (1.0, True)])
def test_daytona_output_collection_uses_its_own_deadline(collection_timeout: float, succeeds: bool):
    class SlowFiles(LocalFiles):
        async def download_file(self, source: str) -> bytes:
            await asyncio.sleep(0.1)
            return await super().download_file(source)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.fs = SlowFiles()
        machine = await DaytonaMachineFactory(lambda: client, output_collection_timeout=collection_timeout).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="")
        )
        try:
            command = Command(("sh", "-c", "printf out; printf err >&2"))
            if succeeds:
                result = await machine.run(command)
                assert (result.stdout, result.stderr) == (b"out", b"err")
            else:
                with pytest.raises(DaytonaOutputUnavailable) as caught:
                    await machine.run(command)
                # Callers retry a lost provider connection.
                assert isinstance(caught.value, ConnectionError)
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("oversized", [False, True])
def test_daytona_output_capture_preserves_provider_startup_failure(tmp_path: Path, oversized: bool) -> None:
    diagnostic = "Provider rejected the command before opening its output files"
    if oversized:
        diagnostic += "x" * (2 * PROVIDER_DIAGNOSTIC_CHAR_LIMIT)

    class RejectedCommandProcess(LocalProcess):
        async def exec(self, command, cwd=None, env=None, timeout=None):
            if env is not None:
                return SimpleNamespace(exit_code=126, result=diagnostic)
            return await super().exec(command, cwd=cwd, env=env, timeout=timeout)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = RejectedCommandProcess()
        machine = DaytonaMachine(
            client.sandbox, MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path)), AsyncExitStack()
        )
        try:
            with pytest.raises(RuntimeError, match="Failed to measure command output") as caught:
                await machine.run(Command(("printf", "not executed")))
            note = next(note for note in caught.value.__notes__ if note.startswith("Daytona exec response: "))
            response = json.loads(note.removeprefix("Daytona exec response: "))
            assert response["exit_code"] == 126
            assert response["diagnostic"].startswith("Provider rejected the command")
            assert len(response["diagnostic"]) <= PROVIDER_DIAGNOSTIC_CHAR_LIMIT
            assert response["diagnostic_truncated"] is oversized
            if not oversized:
                assert response["diagnostic"] == diagnostic
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.skipif(shutil.which("setsid") is None, reason="The command boundary needs a host setsid executable")
@pytest.mark.parametrize("interruption", ["timeout", "cancel", "cancel_twice", "cancel_during_timeout_stop"])
def test_daytona_command_interruption_stops_descendants_and_preserves_next_command(tmp_path, interruption):
    async def scenario():
        stopping, release_stop = asyncio.Event(), asyncio.Event()

        class HeldStop(LocalProcess):
            def __init__(self):
                super().__init__()
                self.cancellations = []

            async def exec(self, command, **kwargs):
                try:
                    if "stop-command" in command and interruption in {"cancel_twice", "cancel_during_timeout_stop"}:
                        stopping.set()
                        await release_stop.wait()
                    return await super().exec(command, **kwargs)
                except asyncio.CancelledError as cancellation:
                    self.cancellations.append(cancellation)
                    raise

        client = LocalDaytona()
        client.sandbox.process = HeldStop()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        child = None
        try:
            await machine.run(Command(("sh", "-c", "echo 12 > answer")))
            pending = asyncio.create_task(
                machine.run(
                    Command(
                        ("sh", "-c", "sleep 3600 & echo $! > child.pid; wait"),
                        timeout=0.5 if interruption in {"timeout", "cancel_during_timeout_stop"} else None,
                    )
                )
            )
            async with asyncio.timeout(5):
                while not (tmp_path / "child.pid").exists() or not (tmp_path / "child.pid").read_text().strip():
                    # Wait for the real child to publish its PID before cancelling its parent.
                    await asyncio.sleep(0)
            child = int((tmp_path / "child.pid").read_text())
            if interruption == "timeout":
                result = await pending
                assert result.reason is ExitReason.TIMED_OUT
            else:
                if interruption == "cancel_during_timeout_stop":
                    await asyncio.wait_for(stopping.wait(), timeout=5)
                pending.cancel("agent deadline")
                if interruption == "cancel_twice":
                    await asyncio.wait_for(stopping.wait(), timeout=5)
                    pending.cancel("second cancellation")
                release_stop.set()
                with pytest.raises(asyncio.CancelledError) as caught:
                    await pending
                assert caught.value.args == ("agent deadline",)
                assert caught.value is client.sandbox.process.cancellations[-1]
            stopped = await machine.run(
                Command(
                    (
                        "sh",
                        "-c",
                        'if [ -f "/proc/$1/stat" ]; then read -r pid comm state rest < "/proc/$1/stat"; '
                        'test "$state" = Z; fi',
                        "child-state",
                        str(child),
                    )
                )
            )
            assert stopped.exit_code == 0
            graded = await machine.run(Command(("sh", "-c", 'test "$(cat answer)" = 12 && printf 1.0')))
            assert (graded.exit_code, graded.stdout) == (0, b"1.0")
            assert not client.deleted and not client.closed
        finally:
            release_stop.set()
            await machine.close()
            if child is not None:
                try:
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        assert client.deleted and client.closed

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_unanswered_command_cleanup_reports_a_lost_machine(tmp_path, monkeypatch):
    monkeypatch.setattr("shellbox.backends.daytona.machine.INTERRUPT_TIMEOUT", 0.2)

    class HungCleanup(LocalProcess):
        armed = False

        async def exec(self, command, **kwargs):
            if self.armed and "rm -rf /tmp/.shellbox-" in command:
                await asyncio.Future()
            return await super().exec(command, **kwargs)

    async def scenario():
        client = LocalDaytona()
        process = HungCleanup()
        client.sandbox.process = process
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        process.armed = True
        with pytest.raises(MachineTerminated):
            await machine.run(Command(("true",)))
        assert client.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("interrupt_failure", ["exit", "timeout", "files"])
@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
@pytest.mark.requires_host_setsid
def test_daytona_failed_interruption_cleanup_closes_machine(tmp_path, interrupt_failure, interruption):
    entered = asyncio.Event()

    class FailedStop(LocalProcess):
        async def exec(self, command, **kwargs):
            if "candidate-block" in command_script(command):
                entered.set()
                await asyncio.Future()
            if "stop-command" in command:
                if interrupt_failure == "timeout":
                    raise TimeoutError("Provider interruption timed out")
                if interrupt_failure == "files":
                    return SimpleNamespace(exit_code=0, result="")
                return SimpleNamespace(exit_code=1, result="Cannot stop the process group")
            if interrupt_failure == "files" and entered.is_set() and "rm -rf /tmp/.shellbox-" in command:
                return SimpleNamespace(exit_code=1, result="Cannot remove command files")
            return await super().exec(command, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = FailedStop()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        pending = asyncio.create_task(
            machine.run(Command(("candidate-block",), timeout=0.5 if interruption == "timeout" else None))
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        if interruption == "cancel":
            pending.cancel("agent deadline")
            with pytest.raises(asyncio.CancelledError) as failure:
                await pending
            assert failure.value.args == ("agent deadline",)
        else:
            # The failed cleanup closes the machine, which callers retry as lost infrastructure.
            with pytest.raises(MachineTerminated) as failure:
                await pending
            assert isinstance(failure.value.__cause__, TimeoutError if interrupt_failure == "timeout" else RuntimeError)
        assert client.deleted and client.closed
        # Grading after the agent stops still uses the machine; callers retry its loss as infrastructure.
        with pytest.raises(MachineTerminated, match="was closed after"):
            await machine.run(Command(("true",)))

    asyncio.run(scenario())


def test_daytona_machine_its_owner_closed_rejects_commands_as_misuse(tmp_path):
    async def scenario():
        client = LocalDaytona()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        await machine.close()
        with pytest.raises(RuntimeError, match="Machine is closed") as failure:
            await machine.run(Command(("true",)))
        assert not isinstance(failure.value, MachineTerminated)

    asyncio.run(scenario())


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
@pytest.mark.requires_host_setsid
def test_daytona_user_probe_interruption_preserves_the_machine(tmp_path, interruption):
    entered = asyncio.Event()

    class BlockedProbe(LocalProcess):
        async def exec(self, command, **kwargs):
            if "su --help" in command:
                entered.set()
                await asyncio.Future()
            return await super().exec(command, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = BlockedProbe()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            pending = asyncio.create_task(
                machine.run(
                    Command(
                        ("printf", "not-started"),
                        user="nobody",
                        timeout=0.05 if interruption == "timeout" else None,
                    )
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            if interruption == "cancel":
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                result = await asyncio.wait_for(pending, timeout=0.5)
                assert result.reason is ExitReason.TIMED_OUT
            following = await machine.run(Command(("printf", "ready")))
            assert (following.exit_code, following.stdout) == (0, b"ready")
            assert not client.closed and not client.deleted
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_cancellation_during_upload_removes_staged_command_and_preserves_machine(tmp_path):
    async def scenario():
        uploading = asyncio.Event()

        class BlockedUpload(LocalFiles):
            staged: Path | None = None

            async def upload_file_stream(self, data: bytes, target: str) -> None:
                await super().upload_file_stream(data, target)
                if self.staged is None:
                    self.staged = Path(target).parent
                    uploading.set()
                    await asyncio.Future()

        client = LocalDaytona()
        files = BlockedUpload()
        client.sandbox.fs = files
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            pending = asyncio.create_task(machine.run(Command(("sh", "-c", "touch cancelled-command"))))
            await asyncio.wait_for(uploading.wait(), timeout=5)
            assert files.staged is not None and (files.staged / "command.sh").is_file()
            pending.cancel("agent deadline")
            with pytest.raises(asyncio.CancelledError) as caught:
                await pending
            assert caught.value.args == ("agent deadline",)
            assert not files.staged.exists()
            graded = await machine.run(Command(("sh", "-c", "test ! -e cancelled-command && printf 1.0")))
            assert (graded.exit_code, graded.stdout) == (0, b"1.0")
            assert not client.closed and not client.deleted
        finally:
            await machine.close()
            if files.staged is not None and files.staged.exists():
                shutil.rmtree(files.staged)

    asyncio.run(scenario())


@pytest.mark.parametrize("command_timeout", [None, 30])
@pytest.mark.requires_host_setsid
def test_daytona_provider_timeout_remains_an_infrastructure_error(tmp_path, command_timeout):
    provider_error = TimeoutError("Provider request timed out")

    class FailedRequest(LocalProcess):
        async def exec(self, command, **kwargs):
            if "candidate-request" in command_script(command):
                raise provider_error
            return await super().exec(command, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = FailedRequest()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            with pytest.raises(TimeoutError) as caught:
                await machine.run(Command(("candidate-request",), timeout=command_timeout))
            assert caught.value is provider_error
        finally:
            await machine.close()
        assert client.closed and client.deleted

    asyncio.run(scenario())


CONNECTION_RESET = DaytonaConnectionError("Failed to execute command: [Errno 104] Connection reset by peer")
BAD_GATEWAY = DaytonaError("Failed to execute command: 502 Bad Gateway", status_code=502)


class LostConnection(LocalProcess):
    """Fail each request whose script contains ``marker`` in transit, at most ``losses`` times."""

    def __init__(self, marker: str, *, losses: int, runs_first: bool, error: DaytonaError = CONNECTION_RESET):
        super().__init__()
        self.marker = marker
        self.losses = losses
        self.runs_first = runs_first
        self.error = error

    async def exec(self, command, **kwargs):
        if self.marker not in command_script(command) or not self.losses:
            return await super().exec(command, **kwargs)
        self.losses -= 1
        if self.runs_first:
            await super().exec(command, **kwargs)
        raise self.error


class LostDownload(LocalFiles):
    lost = False

    async def download_file(self, source: str) -> bytes:
        if not self.lost:
            self.lost = True
            raise DaytonaConnectionError("Failed to download file: [Errno 104] Connection reset by peer")
        return await super().download_file(source)


# Retry backoffs stay below this; the keepalive interval is minutes.
LONGEST_RETRY_WAIT = 60


@pytest.fixture
def no_retry_wait(monkeypatch):
    """Skip the retry waits; the keepalive wait keeps its length so that it does not spin."""

    async def wait(delay):
        if delay > LONGEST_RETRY_WAIT:
            await asyncio.sleep(delay)

    monkeypatch.setattr("shellbox.backends.daytona.machine.sleep", wait)
    monkeypatch.setattr("shellbox.backends.daytona.transient.sleep", wait)


@pytest.mark.parametrize("error", [CONNECTION_RESET, BAD_GATEWAY], ids=["reset", "502"])
@pytest.mark.parametrize("command_ran", [False, True])
@pytest.mark.requires_host_setsid
def test_daytona_command_runs_once_when_its_request_fails_in_transit(tmp_path, no_retry_wait, command_ran, error):
    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = LostConnection("candidate-run", losses=1, runs_first=command_ran, error=error)
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            result = await machine.run(Command(("sh", "-c", "echo candidate-run >> runs; printf out; exit 7")))
        finally:
            await machine.close()
        assert (result.exit_code, result.stdout, result.reason) == (7, b"out", ExitReason.EXITED)
        assert (tmp_path / "runs").read_text() == "candidate-run\n"

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_command_fails_when_its_request_keeps_losing_the_connection(tmp_path, no_retry_wait):
    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = LostConnection("candidate-run", losses=math.inf, runs_first=False)
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            with pytest.raises(ConnectionError, match="Lost the connection") as failure:
                await machine.run(Command(("sh", "-c", "echo candidate-run >> runs")))
            assert isinstance(failure.value.__cause__, DaytonaConnectionError)
        finally:
            await machine.close()
        assert not (tmp_path / "runs").exists()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_repeated_command_request_keeps_its_provider_failure(tmp_path, no_retry_wait):
    class FailedStart(LostConnection):
        async def exec(self, command, **kwargs):
            if self.marker in command_script(command) and not self.losses:
                return SimpleNamespace(exit_code=-1, result="fork/exec /usr/bin/zsh: no such file or directory")
            return await super().exec(command, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = FailedStart("candidate-run", losses=1, runs_first=False)
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            with pytest.raises(RuntimeError, match="Failed to measure command output") as caught:
                await machine.run(Command(("sh", "-c", "echo candidate-run >> runs")))
        finally:
            await machine.close()
        assert any("fork/exec /usr/bin/zsh" in note for note in caught.value.__notes__)

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_command_fails_when_its_shell_dies_after_a_lost_connection(tmp_path, no_retry_wait):
    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = LostConnection("candidate-run", losses=1, runs_first=True)
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            # The command kills the shell that holds the exit trap, so no exit code is recorded.
            with pytest.raises(RuntimeError, match="without an exit code"):
                await machine.run(Command(("sh", "-c", "echo candidate-run >> runs; kill -9 $PPID")))
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("lost", ["umask 077", "measure_and_cap", "tar xzf", "download"])
@pytest.mark.requires_host_setsid
def test_daytona_repeats_a_repeatable_request_that_lost_its_connection(tmp_path, no_retry_wait, lost):
    async def scenario():
        client = LocalDaytona()
        if lost == "download":
            client.sandbox.fs = LostDownload()
        else:
            client.sandbox.process = LostConnection(lost, losses=1, runs_first=True)
        workdir = tmp_path / "work"
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(workdir))
        )
        source = tmp_path / "source"
        source.mkdir()
        (source / "data").write_text("payload")
        try:
            await machine.upload(source, str(workdir / "uploaded"))
            result = await machine.run(Command(("cat", "uploaded/data")))
        finally:
            await machine.close()
        assert (result.exit_code, result.stdout) == (0, b"payload")

    asyncio.run(scenario())


def _bad_gateway() -> ClientResponseError:
    url = URL("https://proxy.daytona.test/files/download")
    request = RequestInfo(url, "GET", CIMultiDictProxy(CIMultiDict()), url)
    return ClientResponseError(request, (), status=502, message="Bad Gateway")


def _sdk_error(error: Exception) -> DaytonaError:
    """The error the Daytona SDK raises when a request fails with ``error``."""

    @intercept_errors(message_prefix="Failed to download file: ")
    async def request():
        raise error

    with pytest.raises(DaytonaError) as caught:
        asyncio.run(request())
    return caught.value


@pytest.mark.requires_host_setsid
@pytest.mark.parametrize(
    "provider_error,state,expected",
    [
        (DaytonaConnectionError("Connection reset by peer"), SandboxState.STARTED, ConnectionError),
        (DaytonaTimeoutError("Request timed out"), SandboxState.STARTED, ConnectionError),
        (_sdk_error(_bad_gateway()), SandboxState.STARTED, ConnectionError),
        (DaytonaValidationError("Is the Sandbox started?"), SandboxState.STOPPED, MachineTerminated),
        (DaytonaValidationError("Is the Sandbox started?"), None, MachineTerminated),
        (DaytonaValidationError("Invalid request", status_code=400), SandboxState.STARTED, DaytonaValidationError),
        (DaytonaValidationError("Invalid request", status_code=400), "unreachable", ConnectionError),
    ],
    ids=["lost-connection", "timed-out", "bad-gateway", "stopped", "destroyed", "started", "probe-unreachable"],
)
def test_daytona_provider_failures_report_a_lost_machine(provider_error, state, expected):
    class FailedRequest(LocalProcess):
        async def exec(self, command, **kwargs):
            if "candidate-request" in command_script(command):
                raise provider_error
            return await super().exec(command, **kwargs)

    probe_error = _sdk_error(_bad_gateway())

    async def refresh_data(request_timeout=None):
        if state is None:
            raise DaytonaNotFoundError("Sandbox not found", status_code=404)
        if state == "unreachable":
            raise probe_error
        client.sandbox.state = state

    client = LocalDaytona()
    client.sandbox.process = FailedRequest()
    client.sandbox.refresh_data = refresh_data

    async def scenario():
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="")
        )
        try:
            with pytest.raises(expected) as caught:
                await machine.run(Command(("candidate-request",)))
            assert caught.value is provider_error or caught.value.__cause__ is provider_error
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_full_sandbox_reports_a_full_machine():
    class FullDisk(LocalProcess):
        async def exec(self, command, **kwargs):
            if "mkdir -p /tmp/.shellbox-" in command:
                return SimpleNamespace(exit_code=1, result="mkdir: cannot create directory: No space left on device")
            return await super().exec(command, **kwargs)

    client = LocalDaytona()
    client.sandbox.process = FullDisk()

    async def scenario():
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="")
        )
        try:
            with pytest.raises(MachineDiskFull):
                await machine.run(Command(("true",)))
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
@pytest.mark.parametrize("deletion_failure", [False, True])
def test_daytona_cancellation_during_failed_command_cleanup_remains_cancellation(deletion_failure):
    entered = asyncio.Event()
    deletion_error = DaytonaError("Failed to remove sandbox", status_code=500)

    class BlockedCleanup(LocalProcess):
        async def exec(self, command, **kwargs):
            if "candidate-fail" in command_script(command):
                raise OSError("Provider command failed")
            if "rm -rf /tmp/.shellbox-" in command:
                entered.set()
                await asyncio.Future()
            return await super().exec(command, **kwargs)

    async def scenario():
        client = FailedDeletion(deletion_error) if deletion_failure else LocalDaytona()
        client.sandbox.process = BlockedCleanup()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="")
        )
        pending = asyncio.create_task(machine.run(Command(("candidate-fail",))))
        await asyncio.wait_for(entered.wait(), timeout=5)
        pending.cancel("grading deadline")
        with pytest.raises(asyncio.CancelledError) as caught:
            await pending
        assert caught.value.args == ("grading deadline",)
        if deletion_failure:
            assert any(str(deletion_error) in note for note in caught.value.__notes__)
        assert client.closed and client.deleted == (not deletion_failure)

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_output_failure_retains_cleanup_note_on_the_primary_error():
    output_error = TimeoutError("Provider output request timed out")

    class FailedOutputAndCleanup(LocalProcess):
        async def exec(self, command, **kwargs):
            if "wc -c" in command:
                raise output_error
            if "rm -rf /tmp/.shellbox-" in command:
                raise OSError("Provider cleanup failed")
            return await super().exec(command, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = FailedOutputAndCleanup()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="")
        )
        with pytest.raises(ConnectionError) as caught:
            await machine.run(Command(("printf", "ready")))
        assert caught.value.__cause__ is output_error
        assert len(caught.value.__notes__) == 1
        assert client.closed and client.deleted

    asyncio.run(scenario())


@pytest.mark.parametrize("path_source", ["image", "task"])
@pytest.mark.requires_host_setsid
def test_daytona_root_cleanup_does_not_execute_candidate_path_programs(tmp_path, path_source):
    binary = tmp_path / "bin"
    binary.mkdir()
    marker = tmp_path / "root-cleanup-executed"
    for name in ("touch", "rm", "wc", "head", "mkdir", "setsid"):
        program = binary / name
        program.write_text(f'#!/bin/sh\nprintf injected > "{marker}"\nexit 99\n')
        program.chmod(0o755)
    agent_program = binary / "agent-program"
    agent_program.write_text("#!/bin/sh\nprintf agent-path\n")
    agent_program.chmod(0o755)
    candidate_path = f"{binary}:{os.environ['PATH']}"

    class ImageProcess(LocalProcess):
        async def exec(self, command, **kwargs):
            if path_source == "image":
                kwargs["env"] = {"PATH": candidate_path, **(kwargs.get("env") or {})}
            return await super().exec(command, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = ImageProcess()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(
                RegistryImage("ubuntu:24.04"),
                workdir=str(tmp_path),
                env={"PATH": candidate_path} if path_source == "task" else {},
            )
        )
        try:
            result = await machine.run(Command(("agent-program",), output_limit_bytes=4))
            assert (result.exit_code, result.stdout, result.stdout_truncated) == (0, b"agen", True)
            assert not marker.exists()
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_completion_between_probe_and_stop_preserves_the_machine(tmp_path):
    paths = []

    class CompletionDuringStop(LocalProcess):
        async def exec(self, command, **kwargs):
            if "candidate-finished" in command_script(command):
                argv = shlex.split(command_script(command))
                index = argv.index("shellbox-command")
                paths.extend(Path(path) for path in argv[index + 1 : index + 3])
                await super().exec(command, **kwargs)
                paths[1].unlink()
                paths[0].write_text("99999999\n")
                await asyncio.Future()
            if "stop-command" in command:
                # Complete after the first marker check and before PID consumption.
                interleave = f'read() {{ : > "{paths[1]}"; /bin/rm -f "{paths[0]}"; return 1; }}; '
                outer = shlex.split(command)
                control, separator, script = outer[2].partition("export PATH; ")
                argv = shlex.split(script)
                argv[2] = interleave + argv[2]
                outer[2] = control + separator + shlex.join(argv)
                command = shlex.join(outer)
            return await super().exec(command, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = CompletionDuringStop()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            result = await machine.run(Command(("printf", "candidate-finished"), timeout=1))
            assert result.reason is ExitReason.TIMED_OUT
            following = await machine.run(Command(("printf", "ready")))
            assert (following.exit_code, following.stdout) == (0, b"ready")
            assert not client.deleted
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("primary_failure", [False, True])
@pytest.mark.parametrize("cleanup_failure", ["timeout", "exit"])
@pytest.mark.parametrize("deletion_failure", [False, True])
@pytest.mark.requires_host_setsid
def test_daytona_file_cleanup_failure_closes_machine_and_preserves_primary_error(
    primary_failure, cleanup_failure, deletion_failure
):
    provider_error = OSError("Provider command failed")
    deletion_error = DaytonaError("Failed to remove sandbox", status_code=500)

    class FailedCleanup(LocalProcess):
        async def exec(self, command, **kwargs):
            if "candidate-fail" in command_script(command) and primary_failure:
                raise provider_error
            if "rm -rf /tmp/.shellbox-" in command:
                if cleanup_failure == "timeout":
                    raise TimeoutError("Provider cleanup timed out")
                return SimpleNamespace(exit_code=1, result="Cannot remove command files")
            return await super().exec(command, **kwargs)

    async def scenario():
        client = FailedDeletion(deletion_error) if deletion_failure else LocalDaytona()
        client.sandbox.process = FailedCleanup()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="")
        )
        error_type = OSError if primary_failure else RuntimeError
        with pytest.raises(error_type) as caught:
            await machine.run(Command(("printf", "candidate-fail")))
        if primary_failure:
            assert caught.value is provider_error
            assert "Daytona command cleanup failed" in caught.value.__notes__[0]
        if deletion_failure:
            assert any(str(deletion_error) in note for note in caught.value.__notes__)
        assert client.closed and client.deleted == (not deletion_failure)

    asyncio.run(scenario())


@pytest.mark.docker
@pytest.mark.parametrize("backend", ["docker", "daytona"])
@pytest.mark.parametrize("tampering", ["delete", "forge", "path"])
def test_nonroot_tampering_cannot_authorize_root_commands(backend, tampering):
    # The Daytona SDK boundary uses Docker only to test real guest UIDs and processes.
    # This is not a live Daytona service test.
    async def scenario():
        guest = await DockerMachineFactory().create(MachineSpec(DockerImage("ubuntu:24.04"), workdir="/tmp"))

        class GuestProcess:
            async def exec(self, command, cwd=None, env=None, timeout=None):
                args = ["exec", "--user", "0"]
                if cwd:
                    args.extend(("-w", cwd))
                for key, value in (env or {}).items():
                    args.extend(("-e", f"{key}={value}"))
                result = await docker(*args, guest.name, "sh", "-c", command, timeout=timeout)
                return SimpleNamespace(exit_code=result.exit_code, result=result.stdout.decode())

        class GuestFiles:
            async def upload_file_stream(self, data, target):
                result = await guest.run(
                    Command(("/bin/sh", "-c", '/bin/cat > "$1"', "upload", target), stdin=data, user="root")
                )
                assert result.exit_code == 0

            async def download_file(self, source):
                result = await docker("exec", "--user", "0", guest.name, "cat", source)
                assert result.exit_code == 0
                return result.stdout

        resources = AsyncExitStack()
        resources.push_async_callback(guest.close)
        machine = (
            guest
            if backend == "docker"
            else DaytonaMachine(SimpleNamespace(process=GuestProcess(), fs=GuestFiles()), guest.spec, resources)
        )
        try:
            await guest.run(Command(("sh", "-c", "setsid sleep 7200 >/dev/null 2>&1 & echo $! > /tmp/victim.pid")))
            if tampering == "path":
                prepared = await guest.run(Command(("sh", "-c", "mkdir -m 777 /tmp/candidate-bin")))
                assert prepared.exit_code == 0
                planted = await guest.run(
                    Command(
                        (
                            "sh",
                            "-c",
                            "for name in touch rm; do "
                            'printf \'#!/bin/sh\\nid -u > /tmp/control-user\\n/bin/%s "$@"\\n\' "$name" '
                            '> "/tmp/candidate-bin/$name"; chmod 755 "/tmp/candidate-bin/$name"; done',
                        ),
                        user="nobody",
                    )
                )
                assert planted.exit_code == 0
            mutation = (
                'rm -f "$file"'
                if tampering == "delete"
                else 'printf "%s\\n" "$victim" > "$file"' if tampering == "forge" else ":"
            )
            environment = {"PATH": "/tmp/candidate-bin:/usr/bin:/bin"} if tampering == "path" else {}
            # The deadline includes provider probes before the guest command starts.
            candidate = Command(
                (
                    "sh",
                    "-c",
                    "victim=$(cat /tmp/victim.pid); "
                    "for file in /tmp/.shellbox-*/pid; do "
                    f'[ ! -w "$file" ] || {mutation}; done; '
                    "sleep 3600 & echo $! > /tmp/candidate-child.pid; wait",
                ),
                user="nobody",
                timeout=5,
                env=environment,
            )
            result = await machine.run(candidate)
            assert result.reason is ExitReason.TIMED_OUT
            result = await guest.run(Command(("sh", "-c", 'kill -0 "$(cat /tmp/victim.pid)"')))
            assert result.exit_code == 0
            followup = await machine.run(Command(("printf", "ready"), user="nobody", env=environment))
            assert (followup.exit_code, followup.stdout) == (0, b"ready")
            if tampering == "path":
                control_user = await guest.run(Command(("test", "-f", "/tmp/control-user")))
                assert control_user.exit_code != 0
            child = await guest.run(
                Command(
                    (
                        "sh",
                        "-c",
                        'path="/proc/$(cat /tmp/candidate-child.pid)/stat"; '
                        'if [ -f "$path" ]; then read pid comm state rest < "$path"; test "$state" = Z; fi',
                    )
                )
            )
            assert child.exit_code == 0
        finally:
            await machine.close()
        disposed = await docker("inspect", guest.name)
        assert disposed.exit_code != 0

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_rejects_nonroot_execution_without_session_preserving_su(tmp_path):
    class MissingSessionOption(LocalProcess):
        async def exec(self, command, **kwargs):
            if "su --help" in command:
                return SimpleNamespace(exit_code=0, result="su -c command")
            return await super().exec(command, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = MissingSessionOption()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            with pytest.raises(UnsupportedMachineSpec, match="--session-command"):
                await machine.run(Command(("sh", "-c", "touch candidate-ran"), user="nobody"))
            assert not (tmp_path / "candidate-ran").exists()
            following = await machine.run(Command(("printf", "still-ready")))
            assert (following.exit_code, following.stdout) == (0, b"still-ready")
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode, exit_code", [(0o755, 0), (0o640, 126)])
@pytest.mark.requires_host_setsid
def test_daytona_upload_preserves_script_permissions(tmp_path: Path, mode: int, exit_code: int) -> None:
    async def scenario() -> None:
        source = tmp_path / "grader.sh"
        source.write_text("#!/bin/sh\nprintf '1.0\\n'\n")
        source.chmod(mode)
        target = tmp_path / "remote" / "private grader.sh"
        machine = await DaytonaMachineFactory(LocalDaytona).create(
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(target.parent))
        )
        try:
            await machine.upload(source, str(target))
            assert stat.S_IMODE(target.stat().st_mode) == mode
            result = await machine.run(Command((str(target),)))
            assert result.exit_code == exit_code
            assert result.stdout == (b"1.0\n" if exit_code == 0 else b"")
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_factory_owns_clients_across_worker_event_loops(tmp_path: Path) -> None:
    clients = []

    class LoopClient(LocalDaytona):
        def __init__(self):
            super().__init__()
            self.loop = asyncio.get_running_loop()
            clients.append(self)

        async def create(self, params, *, timeout):
            assert asyncio.get_running_loop() is self.loop
            if params.env_vars.get("FAIL_CREATE"):
                raise ConnectionError("Sandbox creation failed")
            return await super().create(params, timeout=timeout)

    factory = DaytonaMachineFactory(LoopClient)

    async def scenario(index):
        spec = MachineSpec(
            source=RegistryImage("ubuntu:24.04"),
            workdir=str(tmp_path / str(index)),
            env={"FAIL_CREATE": "1"} if index == 2 else {},
        )
        if index == 2:
            with pytest.raises(ConnectionError):
                await factory.create(spec)
            return
        machine = await factory.create(spec)
        try:
            result = await machine.run(Command(("printf", str(index))))
            assert result.stdout == str(index).encode()
        finally:
            await machine.close()

    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = [executor.submit(asyncio.run, scenario(index)) for index in range(3)]
        for result in pending:
            result.result()
    assert all(client.closed for client in clients)
    assert sum(client.deleted for client in clients) == 2


@pytest.mark.requires_host_setsid
def test_daytona_reuses_snapshot_until_build_inputs_or_resources_change(tmp_path: Path, monkeypatch) -> None:
    snapshots = LocalSnapshots()
    clients = []

    def client_factory():
        client = LocalDaytona()
        client.snapshot = snapshots
        clients.append(client)
        return client

    context = tmp_path / "context"
    context.mkdir()
    dockerfile = context / "Dockerfile"
    dockerfile.write_text("FROM ubuntu:24.04\nCOPY input /input\n")
    source = context / "input"
    source.write_text("first")
    monkeypatch.chdir(context)
    spec = MachineSpec(
        source=DockerfileSource(Path("."), dockerfile), workdir=str(tmp_path / "work"), cpus=1, memory_mb=1024
    )
    factory = DaytonaMachineFactory(client_factory)

    async def scenario():
        for current in (spec, spec, replace(spec, memory_mb=2048)):
            machine = await factory.create(current)
            await machine.close()
        source.write_text("second")
        machine = await factory.create(spec)
        await machine.close()

    asyncio.run(scenario())
    names = [client.params.snapshot for client in clients]
    assert names[0] == names[1]
    assert len(set(names)) == 3
    assert all(client.deleted and client.closed for client in clients)


@pytest.mark.parametrize("image_build", list(DaytonaImageBuild))
@pytest.mark.parametrize("instruction", ["ADD payload.tar /opt/", 'add ["payload.tar", "/opt/"]'])
def test_daytona_rejects_add_inputs_before_building(tmp_path, instruction, image_build):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM ubuntu:24.04\n{instruction}\n")
    (tmp_path / "payload.tar").write_bytes(b"archive fixture")
    client = LocalDaytona()
    factory = DaytonaMachineFactory(lambda: client, image_build=image_build)
    spec = MachineSpec(DockerfileSource(tmp_path, dockerfile))
    with pytest.raises(UnsupportedMachineSpec, match="ADD"):
        factory.check_environment(TaskRequirements(frozenset({MachineCapability.PROCESS})), spec)
    with pytest.raises(UnsupportedMachineSpec, match="ADD"):
        asyncio.run(factory.create(spec))
    assert not client.snapshot.snapshots
    assert client.params is None


@pytest.mark.parametrize("source_kind", ["registry", "dockerfile"])
@pytest.mark.requires_host_setsid
def test_declarative_build_starts_sandbox_from_image_without_snapshot(tmp_path, source_kind):
    context = tmp_path / "context"
    context.mkdir()
    dockerfile = context / "Dockerfile"
    dockerfile.write_text("FROM ubuntu:24.04\nRUN true\n")
    source = RegistryImage("ubuntu:24.04") if source_kind == "registry" else DockerfileSource(context, dockerfile)
    client = LocalDaytona()
    factory = DaytonaMachineFactory(lambda: client, image_build=DaytonaImageBuild.DECLARATIVE)
    spec = MachineSpec(source=source, workdir=str(tmp_path / "work"), cpus=4, memory_mb=8192, storage_mb=10240)

    async def scenario():
        machine = await factory.create(spec)
        try:
            assert (await machine.run(Command(("printf", "ok")))).stdout == b"ok"
        finally:
            await machine.close()

    asyncio.run(scenario())
    assert isinstance(client.params, CreateSandboxFromImageParams)
    if source_kind == "registry":
        assert client.params.image == "ubuntu:24.04"
    else:
        assert isinstance(client.params.image, Image)
        assert client.params.image.dockerfile() == dockerfile.read_text()
    resources = client.params.resources
    assert (resources.cpu, resources.memory, resources.disk) == (4, 8, 10)
    assert not client.snapshot.snapshots
    assert client.deleted and client.closed


@pytest.mark.parametrize("invalid_source", ["add", "outside_context"])
def test_snapshot_preflight_rejects_unsupported_build_sources(tmp_path, invalid_source):
    context = tmp_path / "context"
    context.mkdir()
    dockerfile = (tmp_path if invalid_source == "outside_context" else context) / "Dockerfile"
    dockerfile.write_text("FROM ubuntu:24.04\nADD payload /opt/\n")
    client = LocalDaytona()
    with pytest.raises(UnsupportedMachineSpec):
        asyncio.run(
            DaytonaMachineFactory(client_factory=lambda: client).prepare(
                MachineSpec(source=DockerfileSource(context, dockerfile), cpus=1)
            )
        )
    assert not client.snapshot.snapshots
    assert client.params is None


@pytest.mark.parametrize("startup_timeout", [None, 0.01])
def test_daytona_pending_snapshot_times_out_and_closes_client(startup_timeout) -> None:
    client = LocalDaytona()

    class PendingSnapshots(LocalSnapshots):
        async def create(self, params):
            snapshot = await super().create(params)
            snapshot.state = SnapshotState.BUILDING
            return snapshot

    client.snapshot = PendingSnapshots()
    factory = DaytonaMachineFactory(lambda: client, create_timeout=0.01 if startup_timeout is None else 60)
    with pytest.raises(TimeoutError):
        asyncio.run(factory.create(MachineSpec(RegistryImage("ubuntu:24.04"), startup_timeout=startup_timeout)))
    assert client.snapshot.snapshots
    assert client.params is None
    assert client.closed


def test_daytona_failed_snapshot_closes_client_without_starting_sandbox() -> None:
    client = LocalDaytona()

    class FailedSnapshots(LocalSnapshots):
        async def create(self, params):
            snapshot = await super().create(params)
            snapshot.state = SnapshotState.BUILD_FAILED
            snapshot.error_reason = "COPY input was missing"
            return snapshot

    client.snapshot = FailedSnapshots()
    with pytest.raises(RuntimeError, match="COPY input was missing"):
        asyncio.run(DaytonaMachineFactory(lambda: client).create(MachineSpec(source=RegistryImage("ubuntu:24.04"))))
    assert client.params is None
    assert client.closed


def _account_snapshot(name: str, *, last_used_at, general: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        general=general,
        last_used_at=last_used_at,
        state=SnapshotState.ACTIVE,
        cpu=2,
        mem=4,
        disk=10,
        gpu=0,
        organization_id="fixture",
        error_reason=None,
    )


class SnapshotAccount:
    """Paginated, mutable snapshot inventory for operational tests."""

    def __init__(self, snapshots=()):
        self.snapshots = {snapshot.name: snapshot for snapshot in snapshots}
        self.used_on_get = None

    async def get(self, name):
        if name not in self.snapshots:
            raise DaytonaNotFoundError("Snapshot does not exist", status_code=404)
        snapshot = self.snapshots[name]
        if self.used_on_get is not None:
            snapshot.last_used_at = self.used_on_get
        return snapshot

    async def list(self, page=None, limit=None):
        size = limit or 100
        current = page or 1
        ordered = list(self.snapshots.values())
        return SimpleNamespace(
            items=ordered[(current - 1) * size : current * size],
            total=len(ordered),
            page=current,
            total_pages=max(1, math.ceil(len(ordered) / size)),
        )

    async def delete(self, snapshot):
        del self.snapshots[snapshot.name]


class SandboxLister:
    """Async-iterable sandbox source with an injectable failure mode."""

    def __init__(self, snapshot_names=(), error=None):
        self.snapshot_names = list(snapshot_names)
        self.error = error

    def __call__(self, query=None):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.error is not None:
            raise self.error
        if not self.snapshot_names:
            raise StopAsyncIteration
        return SimpleNamespace(snapshot=self.snapshot_names.pop(0))


def test_snapshot_identity_tracks_rounded_resources_and_image_source() -> None:
    image = RegistryImage("fixture@sha256:" + "a" * 64)
    rounded = snapshot_resources(cpus=2, memory_mb=2049, storage_mb=1025, gpus=0)
    assert (rounded.cpu, rounded.memory, rounded.disk, rounded.gpu) == (2, 3, 2, None)

    name = snapshot_name(image, rounded)
    assert name == snapshot_name(
        RegistryImage(image.reference),
        snapshot_resources(cpus=2, memory_mb=3072, storage_mb=2048, gpus=0),
    )
    assert name != snapshot_name(image, snapshot_resources(cpus=2, memory_mb=3073, storage_mb=2048, gpus=0))
    assert name != snapshot_name(image, snapshot_resources(cpus=2, memory_mb=3072, storage_mb=2049, gpus=0))
    assert name != snapshot_name(image, snapshot_resources(cpus=2, memory_mb=3072, storage_mb=2048, gpus=1))


def test_machine_spec_uses_known_snapshot_identity() -> None:
    spec = MachineSpec(
        source=RegistryImage("fixture:tag"),
        cpus=2,
        memory_mb=2049,
        storage_mb=1025,
    )

    request = snapshot_request_for_spec(spec)
    assert request.name == "shellbox-63d11f0b74739ee18b90a963a6db6e078db4e468"


def test_list_snapshots_reads_every_page(monkeypatch) -> None:
    monkeypatch.setattr("shellbox.backends.daytona.snapshot.SNAPSHOT_PAGE_SIZE", 2)
    account = SnapshotAccount([_account_snapshot(f"snapshot-{index}", last_used_at=None) for index in range(5)])

    names = [snapshot.name for snapshot in asyncio.run(list_snapshots(SimpleNamespace(snapshot=account)))]

    assert names == [f"snapshot-{index}" for index in range(5)]


def test_purge_stale_snapshots_deletes_only_stale_account_entries() -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    account = SnapshotAccount(
        [
            _account_snapshot("stale", last_used_at=now - timedelta(hours=2)),
            _account_snapshot("boundary", last_used_at=now - timedelta(hours=1)),
            _account_snapshot("fresh", last_used_at=now - timedelta(minutes=10)),
            _account_snapshot("general", last_used_at=now - timedelta(hours=5), general=True),
            _account_snapshot("unknown", last_used_at=None),
        ]
    )

    report = asyncio.run(purge_stale_snapshots(SimpleNamespace(snapshot=account, list=SandboxLister()), now=now))

    assert report.deleted == ("stale",)
    assert report.unclassified == ("unknown",)
    assert report.remaining == ()
    assert set(account.snapshots) == {"boundary", "fresh", "general", "unknown"}


def test_purge_revalidates_last_use_before_deleting() -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    account = SnapshotAccount([_account_snapshot("candidate", last_used_at=now - timedelta(hours=2))])
    account.used_on_get = now
    client = SimpleNamespace(snapshot=account, list=SandboxLister())

    report = asyncio.run(purge_stale_snapshots(client, now=now))

    assert report.deleted == ()
    assert report.skipped == ("candidate",)
    assert "candidate" in account.snapshots


def test_purge_continues_when_a_target_loses_its_last_use_evidence() -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)

    class VanishingEvidence(SnapshotAccount):
        async def get(self, name):
            snapshot = await super().get(name)
            if name == "vanishing":
                snapshot.last_used_at = None
            return snapshot

    account = VanishingEvidence(
        [
            _account_snapshot("vanishing", last_used_at=now - timedelta(hours=2)),
            _account_snapshot("still-stale", last_used_at=now - timedelta(hours=2)),
        ]
    )
    client = SimpleNamespace(snapshot=account, list=SandboxLister())

    report = asyncio.run(purge_stale_snapshots(client, now=now))

    assert report.deleted == ("still-stale",)
    assert report.unclassified == ("vanishing",)
    assert set(account.snapshots) == {"vanishing"}


def test_purge_preserves_a_stale_snapshot_that_a_live_sandbox_references() -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    account = SnapshotAccount(
        [
            _account_snapshot("referenced", last_used_at=now - timedelta(hours=2)),
            _account_snapshot("unused", last_used_at=now - timedelta(hours=2)),
        ]
    )
    client = SimpleNamespace(snapshot=account, list=SandboxLister(snapshot_names=["referenced"]))

    report = asyncio.run(purge_stale_snapshots(client, now=now))

    assert report.referenced == ("referenced",)
    assert report.deleted == ("unused",)
    assert set(account.snapshots) == {"referenced"}


def test_purge_fails_closed_when_sandbox_references_cannot_be_enumerated() -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    account = SnapshotAccount([_account_snapshot("stale", last_used_at=now - timedelta(hours=2))])
    client = SimpleNamespace(
        snapshot=account,
        list=SandboxLister(error=DaytonaNotFoundError("listing failed", status_code=500)),
    )

    with pytest.raises(RuntimeError, match="Cannot enumerate sandbox references"):
        asyncio.run(purge_stale_snapshots(client, now=now))

    assert set(account.snapshots) == {"stale"}


def test_daytona_network_policy_parses_declarative_fields() -> None:
    assert DaytonaNetworkPolicy.from_config({"mode": "block_all"}).parameters() == {"network_block_all": True}
    assert DaytonaNetworkPolicy.from_config({"mode": "domain_allow_list", "value": "example.org"}).parameters() == {
        "domain_allow_list": "example.org"
    }
    with pytest.raises(ValueError, match="Unknown Daytona network policy fields"):
        DaytonaNetworkPolicy.from_config({"mode": "block_all", "extra": "value"})


def test_connection_hides_the_api_key_from_every_representation() -> None:
    connection = DaytonaConnection(api_key="synthetic-secret-key")
    factory = connection.client_factory()

    assert "synthetic-secret-key" not in repr(connection)
    assert "synthetic-secret-key" not in repr(factory)
    with pytest.raises(ValueError, match="explicit api_key"):
        DaytonaConnection().client_factory()


def test_connection_module_imports_without_the_daytona_sdk() -> None:
    script = "\n".join(
        (
            "import sys",
            "sys.modules['daytona'] = None  # the optional extra is absent",
            "import shellbox.backends.daytona.connection",
            "connection = shellbox.backends.daytona.connection.DaytonaConnection(",
            "    api_key='synthetic-secret-key'",
            ")",
            "factory = connection.client_factory()",
            "assert callable(factory)",
            "print('resolved')",
        )
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "resolved" in result.stdout


def test_snapshot_request_interchange_round_trips_the_identity() -> None:
    image = RegistryImage("fixture:tag")
    request = SnapshotRequest(
        source=image,
        resources=snapshot_resources(cpus=2, memory_mb=2049, storage_mb=1025, gpus=0),
    )

    assert SnapshotRequest.from_dict(request.as_dict()) == request


def test_snapshot_request_interchange_rejects_a_mismatched_name(tmp_path) -> None:
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    request = SnapshotRequest(source=DockerfileSource(context, context / "Dockerfile"), resources=Resources())

    spec = request.as_dict()
    spec["name"] = "shellbox-" + "0" * 40
    with pytest.raises(ValueError, match="does not match the derived name"):
        SnapshotRequest.from_dict(spec)

    del spec["name"]
    assert SnapshotRequest.from_dict(spec) == request


@pytest.mark.parametrize(
    "transient",
    [
        DaytonaError("Failed to get sandbox: 502 Bad Gateway", status_code=502),
        DaytonaConflictError("Failed to remove sandbox: Sandbox state change in progress", status_code=409),
    ],
)
def test_sandbox_deletion_retries_a_transient_error(monkeypatch, transient) -> None:
    class FlakyGateway:
        def __init__(self):
            self.attempts = 0
            self.deleted = []

        async def get(self, name):
            return SimpleNamespace(name=name)

        async def delete(self, sandbox):
            self.attempts += 1
            if self.attempts == 1:
                raise transient
            self.deleted.append(sandbox.name)

    async def no_wait(delay):
        pass

    monkeypatch.setattr("shellbox.backends.daytona.machine.sleep", no_wait)
    client = FlakyGateway()

    asyncio.run(_delete_named_sandbox(client, "fixture"))

    assert client.deleted == ["fixture"]


def test_sandbox_deletion_reports_a_conflict_that_outlasts_the_deadline(monkeypatch) -> None:
    conflict = DaytonaConflictError("Failed to remove sandbox: Sandbox state change in progress", status_code=409)

    class StuckSandbox:
        async def get(self, name):
            return SimpleNamespace(name=name)

        async def delete(self, sandbox):
            raise conflict

    async def short_wait(delay):
        await asyncio.sleep(0.01)

    monkeypatch.setattr("shellbox.backends.daytona.machine.INTERRUPT_TIMEOUT", 0.1)
    monkeypatch.setattr("shellbox.backends.daytona.machine.sleep", short_wait)

    with pytest.raises(DaytonaConflictError) as caught:
        asyncio.run(_delete_named_sandbox(StuckSandbox(), "fixture"))

    assert caught.value is conflict


@pytest.mark.parametrize("status_code", [None, HTTPStatus.CONFLICT])
def test_daytona_connection_and_delete_conflict_use_shared_retry_identity(status_code):
    reset = daytona_error_identity(DaytonaConnectionError("connection reset"), operation="create")
    assert reset.retry_disposition is RetryDisposition.TRANSIENT
    conflict = DaytonaConflictError("state change in progress", status_code=status_code)
    assert daytona_error_identity(conflict, operation="delete").retry_disposition is RetryDisposition.TRANSIENT
    expected = RetryDisposition.UNKNOWN if status_code is None else RetryDisposition.TRANSIENT
    assert daytona_error_identity(conflict, operation="create").retry_disposition is expected


@pytest.mark.parametrize(
    "status_code",
    [HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.CONFLICT, HTTPStatus.TOO_MANY_REQUESTS, HTTPStatus.SERVICE_UNAVAILABLE],
)
def test_repeatable_daytona_requests_retry_shared_transient_statuses(status_code, no_retry_wait):
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DaytonaError("temporary failure", status_code=status_code)
        return "result"

    assert asyncio.run(retry_transient(request)) == "result"
    assert attempts == 2


@pytest.mark.parametrize(
    "failure",
    [
        DaytonaError("request failed", status_code=HTTPStatus.UNAUTHORIZED),
        DaytonaError("request failed", status_code=HTTPStatus.NOT_FOUND),
        DaytonaConnectionError("TLS connection failed"),
    ],
)
def test_repeatable_daytona_requests_stop_on_nontransient_failures(failure, no_retry_wait):
    if isinstance(failure, DaytonaConnectionError):
        failure.__cause__ = SSLCertVerificationError("certificate rejected")
    attempts = 0

    async def request():
        nonlocal attempts
        attempts += 1
        raise failure

    with pytest.raises(type(failure)) as caught:
        asyncio.run(retry_transient(request))
    assert caught.value is failure
    assert attempts == 1
