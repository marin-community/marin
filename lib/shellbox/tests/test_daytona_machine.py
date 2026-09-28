# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Daytona adapter against a local process and filesystem fake."""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec


class LocalFiles:
    async def upload_file_stream(self, data: bytes, target: str) -> None:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def download_file(self, source: str) -> bytes:
        return Path(source).read_bytes()


class LocalProcess:
    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None, timeout=None):
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return SimpleNamespace(exit_code=process.returncode, result=stdout.decode(errors="replace"))


class LocalDaytona:
    def __init__(self):
        self.sandbox = SimpleNamespace(fs=LocalFiles(), process=LocalProcess())
        self.deleted = False
        self.params = None

    async def create(self, params):
        self.params = params
        return self.sandbox

    async def delete(self, sandbox):
        assert sandbox is self.sandbox
        self.deleted = True


def test_daytona_binary_command_and_files(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = LocalDaytona()
        workdir = tmp_path / "work"
        machine = await DaytonaMachineFactory(client).create(
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(workdir))
        )
        assert client.params.network_block_all is True
        assert client.params.ttl_minutes == 360
        try:
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

    asyncio.run(scenario())
