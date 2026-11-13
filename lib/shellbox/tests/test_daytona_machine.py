# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Daytona adapter against a local process and filesystem fake."""

import asyncio
import json
import math
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

pytest.importorskip("daytona")

from daytona import (
    AsyncDaytona,
    CreateSandboxFromSnapshotParams,
    DaytonaAuthorizationError,
    DaytonaConfig,
    DaytonaConflictError,
    DaytonaError,
    DaytonaNotFoundError,
    Resources,
)
from daytona_api_client_async import SnapshotState
from rigging.timing import ExponentialBackoff
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.machine import (
    PROVIDER_DIAGNOSTIC_CHAR_LIMIT,
    DaytonaMachine,
    DaytonaMachineFactory,
    DaytonaNetworkMode,
    DaytonaNetworkPolicy,
    _create_sandbox,
)
from shellbox.backends.daytona.snapshot import (
    SnapshotRequest,
    ensure_snapshot,
    list_snapshots,
    purge_stale_snapshots,
    snapshot_name,
    snapshot_request_for_spec,
    snapshot_resources,
)
from shellbox.backends.docker.machine import DockerMachineFactory, docker
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import (
    Command,
    DockerImage,
    ExitReason,
    MachineSpec,
    UnsupportedMachineSpec,
)


@pytest.mark.parametrize("status", [429, 502, 503])
@pytest.mark.parametrize(
    "failure", ["rejected_post", "accepted_post", "late_get", "prolonged_throttle", "deadline", "cancelled"]
)
def test_daytona_creation_recovers_named_sandbox_after_transient_errors(monkeypatch, failure, status):
    async def scenario():
        requests = []
        names = []
        delays = []
        waiting = asyncio.Event()

        async def wait(delay):
            delays.append(delay)
            waiting.set()
            if failure in {"deadline", "cancelled"}:
                await asyncio.Future()

        monkeypatch.setattr("shellbox.backends.daytona.provider.sleep", wait)

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
            if request.method == "POST":
                names.append((await request.json())["name"])
                if failure == "accepted_post" and requests.count("POST") == 1:
                    created = True
                    return web.json_response({"message": "ambiguous creation"}, status=status)
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
                    {
                        "id": "created-once",
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
                        "state": "started",
                        "toolboxProxyUrl": (
                            ""
                            if failure == "late_get" and request.method == "POST"
                            else "https://fixture.invalid/toolbox"
                        ),
                    }
                )
            return web.json_response(
                {"message": "transient provider error"}, status=status, headers={"Retry-After-sandbox-create": "17"}
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
                    assert requests == (
                        ["POST", "POST", "GET", "POST", "GET"]
                        if failure == "accepted_post"
                        else ["POST", "GET", "POST", "GET"]
                    )
                    assert len(delays) == (2 if failure == "accepted_post" else 1)
                    assert all(0 < delay <= 120 for delay in delays)
                    assert delays[-1] >= 17
                assert set(names) == {"fixture"}

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["rejected_create", "accepted_create", "metadata", "list", "transport", "forbidden"])
def test_daytona_snapshot_recovers_transient_http_without_duplicate_images(monkeypatch, failure):
    async def scenario():
        # Advance the existing backoff clock without real waiting in either version.
        monkeypatch.setattr(ExponentialBackoff, "next_interval", lambda self: 0)
        request = SnapshotRequest(
            RegistryImage("fixture.invalid/image@sha256:" + "0" * 64), Resources(cpu=2, memory=4, disk=10)
        )
        snapshots = {}
        attempts = Counter()

        def snapshot():
            return {
                "id": request.name,
                "name": request.name,
                "general": False,
                "state": SnapshotState.ACTIVE.value,
                "size": None,
                "entrypoint": None,
                "cpu": 2,
                "gpu": 0,
                "mem": 4,
                "disk": 10,
                "errorReason": None,
                "createdAt": "2026-11-13T15:00:00Z",
                "updatedAt": "2026-11-13T15:00:00Z",
                "lastUsedAt": None,
            }

        async def endpoint(http):
            operation = "create" if http.method == "POST" else "list" if http.path.endswith("/snapshots") else "get"
            attempts[operation] += 1
            if failure == "forbidden":
                return web.json_response({"message": "forbidden"}, status=403)
            if failure == "transport" and operation == "get" and attempts[operation] == 1:
                assert http.transport is not None
                http.transport.close()
                return web.Response()
            if (failure == "metadata" and operation == "get" and attempts[operation] <= 5) or (
                failure == "list" and operation == "list" and attempts[operation] <= 5
            ):
                return web.Response(text="<html>Bad Gateway</html>", status=502)
            if operation == "create":
                name = (await http.json())["name"]
                if failure == "rejected_create" and attempts[operation] == 1:
                    return web.Response(text="Unavailable", status=503)
                if name in snapshots:
                    return web.json_response({"message": "name already exists"}, status=409)
                snapshots[name] = snapshot()
                if failure == "accepted_create" and attempts[operation] == 1:
                    return web.Response(text="<html>Bad Gateway</html>", status=502)
                return web.json_response(snapshots[name])
            if operation == "list":
                return web.json_response(
                    {"items": list(snapshots.values()), "total": len(snapshots), "page": 1, "totalPages": 1}
                )
            name = http.path.rsplit("/", 1)[-1]
            if name not in snapshots:
                return web.json_response({"message": "missing"}, status=404)
            return web.json_response(snapshots[name])

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", endpoint)
        async with TestServer(app) as server:
            async with AsyncDaytona(
                DaytonaConfig(api_key="fixture", api_url=str(server.make_url("")).rstrip("/"))
            ) as client:
                if failure == "forbidden":
                    with pytest.raises(DaytonaAuthorizationError):
                        await ensure_snapshot(client, request)
                    assert not snapshots
                    return
                assert await ensure_snapshot(client, request) == request.name
                assert [row.name for row in await list_snapshots(client)] == [request.name]
        assert set(snapshots) == {request.name}
        if failure == "accepted_create":
            assert attempts["create"] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["prepare", "create_snapshot", "create_sandbox"])
@pytest.mark.parametrize("status", [429, 502])
def test_daytona_startup_retry_deadline_preserves_provider_diagnostics(operation, status):
    async def scenario():
        request = SnapshotRequest(RegistryImage("ubuntu:24.04"), Resources(cpu=1, memory=1, disk=1))

        async def endpoint(http):
            if http.path.startswith("/snapshots") and operation == "create_sandbox":
                return web.json_response(
                    {
                        "id": request.name,
                        "name": request.name,
                        "general": False,
                        "state": "active",
                        "cpu": 1,
                        "gpu": 0,
                        "mem": 1,
                        "disk": 1,
                        "createdAt": "2026-11-13T15:00:00Z",
                        "updatedAt": "2026-11-13T15:00:00Z",
                    }
                )
            if http.method == "GET" and not http.path.startswith("/snapshots"):
                return web.json_response({"message": "not created"}, status=404)
            return web.json_response(
                {"message": "provider outage"},
                status=status,
                headers={"Retry-After": "120", "X-RateLimit-Remaining": "0"},
            )

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", endpoint)
        async with TestServer(app) as server:
            factory = DaytonaMachineFactory(
                lambda: AsyncDaytona(
                    DaytonaConfig(
                        api_key="fixture",
                        api_url=str(server.make_url("")).rstrip("/"),
                    )
                )
            )
            spec = MachineSpec(request.source, startup_timeout=0.05)
            with pytest.raises(DaytonaError) as failure:
                if operation == "prepare":
                    await factory.prepare(spec)
                else:
                    await factory.create(spec)
            assert failure.value.status_code == status
            assert failure.value.headers["X-RateLimit-Remaining"] == "0"
            assert failure.value.headers["Retry-After"] == "120"

    asyncio.run(scenario())


class LocalFiles:
    async def upload_file_stream(self, data: bytes, target: str) -> None:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def download_file(self, source: str) -> bytes:
        return Path(source).read_bytes()


def command_script(command: str) -> str:
    script = shlex.split(command)[2].partition("export PATH; ")[2]
    argv = shlex.split(script)
    if len(argv) == 2 and argv[0] == "/bin/sh" and argv[1].endswith("/command.sh"):
        return Path(argv[1]).read_text()
    return script


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


class LocalDaytona:
    def __init__(self):
        self.sandbox = SimpleNamespace(fs=LocalFiles(), process=LocalProcess())
        self.deleted = False
        self.closed = False
        self.params = None
        self.timeout = None
        self.snapshot = LocalSnapshots()

    async def create(self, params, *, timeout):
        assert isinstance(params, CreateSandboxFromSnapshotParams)
        await self.snapshot.get(params.snapshot)
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


def test_daytona_recreated_workdir_preserves_command_output(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    original_inode = workdir.stat().st_ino

    class StaleWorkdirProcess(LocalProcess):
        def __init__(self):
            super().__init__()
            self.workdir_replaced = False

        async def exec(self, command, cwd=None, env=None, timeout=None):
            if self.workdir_replaced and cwd is None:
                raise OSError("Provider process retained the deleted working directory")
            result = await super().exec(command, cwd=cwd, env=env, timeout=timeout)
            self.workdir_replaced = workdir.stat().st_ino != original_inode
            return result

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = StaleWorkdirProcess()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(workdir))
        )
        try:
            replaced = await machine.run(Command(("sh", "-c", 'rm -rf "$PWD" && mkdir -p "$PWD" && printf recreated')))
            assert (replaced.exit_code, replaced.stdout) == (0, b"recreated")
            following = await machine.run(Command(("printf", "ready")))
            assert (following.exit_code, following.stdout) == (0, b"ready")
            assert workdir.stat().st_ino != original_inode
        finally:
            await machine.close()

    asyncio.run(scenario())


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


@pytest.mark.parametrize("interrupt_failure", ["exit", "timeout", "files"])
@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
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
            with pytest.raises(RuntimeError) as failure:
                await pending
            assert isinstance(failure.value.__cause__, TimeoutError if interrupt_failure == "timeout" else RuntimeError)
        assert client.deleted and client.closed
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(("true",)))

    asyncio.run(scenario())


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_daytona_user_probe_is_bounded_and_cancellation_closes_the_machine(tmp_path, interruption):
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
                assert client.closed and client.deleted
            else:
                result = await asyncio.wait_for(pending, timeout=0.5)
                assert result.reason is ExitReason.TIMED_OUT
                following = await machine.run(Command(("printf", "ready")))
                assert (following.exit_code, following.stdout) == (0, b"ready")
                assert not client.deleted
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("command_timeout", [None, 30])
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


@pytest.mark.parametrize("deletion_failure", [False, True])
def test_daytona_cancellation_during_failed_command_cleanup_remains_cancellation(deletion_failure):
    entered = asyncio.Event()
    deletion_error = DaytonaConflictError("Sandbox state change in progress", status_code=409)

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
        with pytest.raises(RuntimeError) as caught:
            await machine.run(Command(("printf", "ready")))
        assert caught.value.__cause__ is output_error
        assert len(caught.value.__notes__) == 1
        assert client.closed and client.deleted

    asyncio.run(scenario())


@pytest.mark.parametrize("path_source", ["image", "task"])
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
def test_daytona_file_cleanup_failure_closes_machine_and_preserves_primary_error(
    primary_failure, cleanup_failure, deletion_failure
):
    provider_error = OSError("Provider command failed")
    deletion_error = DaytonaConflictError("Sandbox state change in progress", status_code=409)

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


@pytest.mark.parametrize("instruction", ["ADD payload.tar /opt/", 'add ["payload.tar", "/opt/"]'])
def test_daytona_rejects_add_inputs_before_snapshot_creation(tmp_path, instruction):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM ubuntu:24.04\n{instruction}\n")
    (tmp_path / "payload.tar").write_bytes(b"archive fixture")
    client = LocalDaytona()
    with pytest.raises(UnsupportedMachineSpec, match="ADD"):
        asyncio.run(DaytonaMachineFactory(lambda: client).create(MachineSpec(DockerfileSource(tmp_path, dockerfile))))
    assert not client.snapshot.snapshots


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
    connection = DaytonaConnection(
        api_key="synthetic-secret-key", api_url="https://fixture.invalid/api", target="fixture"
    )
    factory = connection.client_factory()

    assert "synthetic-secret-key" not in repr(connection)
    assert "synthetic-secret-key" not in repr(factory)
    with pytest.raises(ValueError, match="explicit api_key, api_url and target"):
        DaytonaConnection(api_url="https://fixture.invalid/api", target="fixture").client_factory()


def test_connection_module_imports_without_the_daytona_sdk() -> None:
    script = "\n".join(
        (
            "import sys",
            "sys.modules['daytona'] = None  # the optional extra is absent",
            "import shellbox.backends.daytona.connection",
            "connection = shellbox.backends.daytona.connection.DaytonaConnection(",
            "    api_key='synthetic-secret-key', api_url='https://fixture.invalid/api', target='fixture'",
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
