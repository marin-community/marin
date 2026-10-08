# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound candidate file transfers through public machine APIs."""

import asyncio
import sys
import tracemalloc
from contextlib import AsyncExitStack
from types import SimpleNamespace

import pytest
from shellbox.backends.daytona.machine import DaytonaMachine, DaytonaMachineFactory
from shellbox.backends.docker.machine import DockerCommandResult, DockerMachine, DockerMachineFactory, docker
from shellbox.backends.gvisor.machine import GvisorMachine
from shellbox.backends.qemu.machine import Acceleration, QemuMachine
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import (
    Command,
    DockerImage,
    DownloadLimitExceeded,
    ExitReason,
    MachineSpec,
    QemuBundle,
    ShellSimBuiltins,
    UnsupportedMachineSpec,
)

from .test_daytona_machine import LocalDaytona


@pytest.mark.parametrize("backend", ["docker", "gvisor", "daytona", "shellsim", "qemu"])
@pytest.mark.parametrize("oversized", [False, True])
def test_bounded_download_preserves_binary_files_or_existing_target(tmp_path, monkeypatch, backend, oversized):
    create_process = asyncio.create_subprocess_exec

    async def local_process(*args, **kwargs):
        if args[0] == "docker":
            args = args[args.index("download-fixture") + 1 :]
        return await create_process(*args, **kwargs)

    async def local_docker(*args, **kwargs):
        if args[0] == "rm":
            return DockerCommandResult(0, b"", b"")
        process = await local_process("docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await process.communicate()
        return DockerCommandResult(process.returncode, stdout, stderr)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", local_process)
    monkeypatch.setattr("shellbox.backends.docker.machine.docker", local_docker)
    payload = b"\x00\xffanswer" * 32768
    source = tmp_path / "source"
    source.write_bytes(payload)
    target = tmp_path / "download"
    target.write_bytes(b"existing")
    limit = len(payload) - 1 if oversized else len(payload)

    async def scenario():
        client = None
        remote = str(source)
        if backend in {"docker", "gvisor"}:
            machine = (DockerMachine if backend == "docker" else GvisorMachine)(
                "download-fixture", MachineSpec(DockerImage("fixture"))
            )
        elif backend == "daytona":
            client = LocalDaytona()
            machine = await DaytonaMachineFactory(lambda: client).create(
                MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
            )
        elif backend == "qemu":
            machine = QemuMachine(MachineSpec(QemuBundle(tmp_path), workdir=str(tmp_path)), Acceleration.TCG)
            # This process consumes real serial framing without booting a VM.
            script = (
                "import base64, subprocess, sys\n"
                "data = bytearray()\n"
                "for line in sys.stdin.buffer:\n"
                " if line == b'BEGIN\\n': data.clear()\n"
                " elif line.startswith(b'DATA|'): data.extend(line[5:].strip())\n"
                " elif line == b'END\\n':\n"
                "  command = base64.b64decode(data).decode().replace('/harbor/busybox ', '')\n"
                "  result = subprocess.run(['/bin/sh'], input=command.encode(), capture_output=True)\n"
                "  print('RESULT|' + str(result.returncode), flush=True)\n"
                "  for prefix, output in [('OUT|', result.stdout), ('ERR|', result.stderr)]:\n"
                "   for offset in range(0, len(output), 4096):\n"
                "    print(prefix + base64.b64encode(output[offset:offset+4096]).decode(), flush=True)\n"
                "  print('ENDRESULT', flush=True)\n"
            )
            machine.process = await create_process(
                sys.executable,
                "-c",
                script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            machine = await ShellSimMachineFactory().create(MachineSpec(ShellSimBuiltins()))
            remote = "/workspace/source"
            await machine.upload(source, remote)
        try:
            if oversized:
                with pytest.raises(DownloadLimitExceeded):
                    await machine.download(remote, target, max_bytes=limit)
                assert target.read_bytes() == b"existing"
            else:
                await machine.download(remote, target, max_bytes=limit)
                assert target.read_bytes() == payload
            # A large supported limit must not allocate a buffer of that size.
            tracemalloc.start()
            try:
                await machine.download(remote, target, max_bytes=1024**3)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            assert peak < 8 * 1024**2
            assert target.read_bytes() == payload
            directory = "/workspace" if backend == "shellsim" else str(tmp_path)
            with pytest.raises(UnsupportedMachineSpec, match="regular file"):
                await machine.download(directory, target, max_bytes=limit)
            assert target.read_bytes() == payload
            assert not list(tmp_path.glob(".shellbox-download-*"))
            if client is not None:
                assert client.sandbox.fs.download_closed
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.docker
@pytest.mark.parametrize("backend", ["docker", "daytona"])
def test_nonroot_pid_tampering_cannot_authorize_a_root_kill(backend, monkeypatch):
    # The Daytona SDK boundary uses Docker only to test real guest UIDs and processes.
    # This is not a live Daytona service test.
    async def scenario():
        guest = await DockerMachineFactory().create(MachineSpec(DockerImage("ubuntu:24.04"), workdir="/tmp"))
        victim_survived = False

        async def checked_docker(*args, **kwargs):
            nonlocal victim_survived
            if args[:2] == ("rm", "-f"):
                result = await docker("exec", "--user", "0", guest.name, "sh", "-c", 'kill -0 "$(cat /tmp/victim.pid)"')
                victim_survived = result.exit_code == 0
            return await docker(*args, **kwargs)

        monkeypatch.setattr("shellbox.backends.docker.machine.docker", checked_docker)

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
            candidate = Command(
                (
                    "sh",
                    "-c",
                    "victim=$(cat /tmp/victim.pid); "
                    "for file in /tmp/.shellbox-command-* /tmp/.shellbox-*/pid; do "
                    '[ ! -w "$file" ] || printf "%s\\n" "$victim" > "$file"; done; '
                    "sleep 3600 & echo $! > /tmp/candidate-child.pid; wait",
                ),
                user="nobody",
                timeout=1,
            )
            if backend == "docker":
                with pytest.raises(TimeoutError):
                    await machine.run(candidate)
                assert victim_survived
                disposed = await docker("inspect", guest.name)
                assert disposed.exit_code != 0
            else:
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

    asyncio.run(scenario())


@pytest.mark.parametrize("interruption", ["provider", "cancel"])
def test_interrupted_daytona_download_closes_stream_and_preserves_target(tmp_path, interruption):
    async def scenario():
        started = asyncio.Event()
        closed = asyncio.Event()
        client = LocalDaytona()
        target = tmp_path / "download"
        target.write_bytes(b"existing")

        async def stream(_source):
            async def chunks():
                try:
                    yield b"partial"
                    started.set()
                    if interruption == "provider":
                        raise ConnectionError("Provider transfer failed")
                    await asyncio.Future()
                finally:
                    closed.set()

            return chunks()

        client.sandbox.fs.download_file_stream = stream
        source = tmp_path / "regular-file"
        source.write_text("input")
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            pending = asyncio.create_task(machine.download(str(source), target, max_bytes=1024))
            await asyncio.wait_for(started.wait(), timeout=5)
            if interruption == "cancel":
                pending.cancel()
            with pytest.raises(ConnectionError if interruption == "provider" else asyncio.CancelledError):
                await pending
            assert closed.is_set()
            assert target.read_bytes() == b"existing"
            assert not list(tmp_path.glob(".shellbox-download-*"))
            assert not client.deleted
        finally:
            await machine.close()

    asyncio.run(scenario())
