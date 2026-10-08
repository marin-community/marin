# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Daytona adapter against a local process and filesystem fake."""

import asyncio
import os
import shutil
import signal
import stat
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from daytona import CreateSandboxFromSnapshotParams, DaytonaNotFoundError
from daytona_api_client_async import SnapshotState
from rigging.timing import ExponentialBackoff
from shellbox.backends.daytona.machine import (
    DaytonaMachine,
    DaytonaMachineFactory,
    DaytonaNetworkMode,
    DaytonaNetworkPolicy,
)
from shellbox.backends.docker.machine import DockerMachineFactory, docker
from shellbox.file_transfer import DOWNLOAD_CHUNK_BYTES
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import Command, DockerImage, DownloadLimitExceeded, ExitReason, MachineSpec, UnsupportedMachineSpec


class LocalFiles:
    def __init__(self):
        self.downloaded_bytes = 0
        self.download_closed = False

    async def upload_file_stream(self, data: bytes, target: str) -> None:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def download_file(self, source: str) -> bytes:
        return Path(source).read_bytes()

    async def download_file_stream(self, source: str):
        async def chunks():
            try:
                with Path(source).open("rb") as file:
                    while data := file.read(DOWNLOAD_CHUNK_BYTES):
                        self.downloaded_bytes += len(data)
                        yield data
            finally:
                self.download_closed = True

        return chunks()


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

    async def delete(self, sandbox):
        assert sandbox is self.sandbox
        await sandbox.process.close()
        self.deleted = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        self.closed = True


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


@pytest.mark.skipif(shutil.which("setsid") is None, reason="The command boundary needs a host setsid executable")
def test_daytona_command_timeout_stops_descendants_and_preserves_next_command(tmp_path):
    async def scenario():
        client = LocalDaytona()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            await machine.run(Command(("sh", "-c", "echo 12 > answer")))
            result = await machine.run(Command(("sh", "-c", "sleep 3600 & echo $! > child.pid; wait"), timeout=0.5))
            assert result.reason is ExitReason.TIMED_OUT
            child = int((tmp_path / "child.pid").read_text())
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
            await machine.close()
        assert client.deleted and client.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("interrupt_failure", ["exit", "timeout"])
def test_daytona_failed_timeout_cleanup_is_infrastructure_failure(tmp_path, interrupt_failure):
    class FailedStop(LocalProcess):
        async def exec(self, command, **kwargs):
            if "candidate-block" in command:
                raise TimeoutError("Command deadline expired")
            if "stop-command" in command:
                if interrupt_failure == "timeout":
                    raise TimeoutError("Provider interruption timed out")
                return SimpleNamespace(exit_code=1, result="Cannot stop the process group")
            return await super().exec(command, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = FailedStop()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        with pytest.raises(RuntimeError) as failure:
            await machine.run(Command(("candidate-block",), timeout=0.01))
        assert isinstance(failure.value.__cause__, TimeoutError if interrupt_failure == "timeout" else RuntimeError)
        assert client.deleted and client.closed
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(("true",)))

    asyncio.run(scenario())


def test_daytona_completion_at_pid_removal_preserves_the_sandbox(tmp_path):
    armed, removed, released = (tmp_path / name for name in ("armed", "removed", "released"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    removal = bin_dir / "rm"
    # Pause after the trap's PID-file removal, before its remaining work can finish.
    removal.write_text(
        '#!/bin/sh\n/bin/rm "$@"\ncase "$2" in */pid)\n'
        f'if [ -f "{armed}" ] && [ ! -f "{removed}" ]; then touch "{removed}"; '
        f'while [ ! -f "{released}" ]; do /bin/sleep 0.01; done; fi;; esac\n'
    )
    removal.chmod(0o755)
    executions = []

    class DeadlineAtRemoval(LocalProcess):
        async def exec(self, command, **kwargs):
            if "candidate-finished" in command:
                execution = asyncio.create_task(super().exec(command, **kwargs))
                executions.append(execution)
                backoff = ExponentialBackoff(initial=0.001, maximum=0.01)
                async with asyncio.timeout(5):
                    while not removed.exists():
                        await asyncio.sleep(backoff.next_interval())
                raise TimeoutError("The command deadline expired after PID removal")
            return await super().exec(command, **kwargs)

        async def close(self):
            released.touch()
            await super().close()

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = DeadlineAtRemoval()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(
                RegistryImage("ubuntu:24.04"), workdir=str(tmp_path), env={"PATH": f"{bin_dir}:{os.environ['PATH']}"}
            )
        )
        try:
            result = await machine.run(Command(("sh", "-c", f"touch {armed}; printf candidate-finished"), timeout=5))
            assert result.reason is ExitReason.TIMED_OUT
            following = await machine.run(Command(("printf", "still-ready")))
            assert (following.exit_code, following.stdout) == (0, b"still-ready")
            assert not client.deleted
        finally:
            released.touch()
            await asyncio.gather(*executions)
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.docker
@pytest.mark.parametrize("backend", ["docker", "daytona"])
@pytest.mark.parametrize("tampering", ["delete", "forge"])
def test_nonroot_pid_tampering_cannot_authorize_a_root_kill(backend, tampering):
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
            mutation = 'rm -f "$file"' if tampering == "delete" else 'printf "%s\\n" "$victim" > "$file"'
            candidate = Command(
                (
                    "sh",
                    "-c",
                    "victim=$(cat /tmp/victim.pid); "
                    "for file in /tmp/.shellbox-command-* /tmp/.shellbox-*/pid; do "
                    f'[ ! -w "$file" ] || {mutation}; done; '
                    "sleep 3600 & echo $! > /tmp/candidate-child.pid; wait",
                ),
                user="nobody",
                timeout=1,
            )
            result = await machine.run(candidate)
            assert result.reason is ExitReason.TIMED_OUT
            result = await guest.run(Command(("sh", "-c", 'kill -0 "$(cat /tmp/victim.pid)"')))
            assert result.exit_code == 0
            followup = await machine.run(Command(("printf", "ready"), user="nobody"))
            assert (followup.exit_code, followup.stdout) == (0, b"ready")
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


def test_daytona_download_caps_a_file_that_grows_after_the_regular_file_probe(tmp_path):
    source, target = tmp_path / "archive.tar", tmp_path / "download"
    source.write_bytes(b"archive")
    target.write_bytes(b"existing")
    limit = 1024**2

    class GrowingFiles(LocalFiles):
        def grow(self, remote):
            if Path(remote) != source:
                return
            with Path(remote).open("ab") as file:
                for _ in range(512):
                    file.write(b"x" * DOWNLOAD_CHUNK_BYTES)

        async def download_file(self, remote):
            self.grow(remote)
            return await super().download_file(remote)

        async def download_file_stream(self, remote):
            self.grow(remote)
            return await super().download_file_stream(remote)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.fs = GrowingFiles()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            tracemalloc.start()
            try:
                with pytest.raises(DownloadLimitExceeded):
                    await machine.download(str(source), target, max_bytes=limit)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            assert source.stat().st_size > 32 * 1024**2
            assert target.read_bytes() == b"existing"
            assert client.sandbox.fs.downloaded_bytes <= limit + DOWNLOAD_CHUNK_BYTES
            assert client.sandbox.fs.download_closed
            assert peak < 8 * 1024**2
            assert not list(tmp_path.glob(".shellbox-download-*"))
            following = await machine.run(Command(("printf", "still-ready")))
            assert (following.exit_code, following.stdout) == (0, b"still-ready")
        finally:
            await machine.close()

    asyncio.run(scenario())


def test_daytona_rejects_nonroot_execution_without_session_preserving_su(tmp_path):
    class MissingSessionOption(LocalProcess):
        async def exec(self, command, **kwargs):
            if command == "su --help":
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
