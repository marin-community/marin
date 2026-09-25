# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Iris exec wire path with a local subprocess provider."""

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

from shellbox.backends.iris.machine import IrisMachine
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, NetworkPolicy


class LocalRpc:
    def exec_in_container(self, request, timeout_ms):
        del timeout_ms
        result = subprocess.run(request.command, capture_output=True, text=True, timeout=30)
        return SimpleNamespace(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr, error="")


class LocalJob:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


class LocalClient:
    def shutdown(self):
        pass


class LocalEndpoint:
    def close(self):
        pass


def test_iris_binary_command_and_file_round_trip(tmp_path: Path) -> None:
    async def scenario() -> None:
        job = LocalJob()
        machine = IrisMachine(
            LocalEndpoint(),
            LocalClient(),
            LocalRpc(),
            job,
            "task",  # type: ignore[arg-type]
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path), network=NetworkPolicy.ALLOW),
        )
        try:
            result = await machine.run(
                Command(("/bin/sh", "-c", "cat; printf '\\000\\377' >&2"), stdin=b"abc\x00\xff", output_limit_bytes=4)
            )
            assert result.exit_code == 0
            assert result.stdout == b"abc\x00"
            assert result.stdout_truncated
            assert result.stderr == b"\x00\xff"

            source = tmp_path / "input.bin"
            source.write_bytes(b"\x00\xffpayload")
            await machine.upload(source, str(tmp_path / "remote.bin"))
            target = tmp_path / "download.bin"
            await machine.download(str(tmp_path / "remote.bin"), target)
            assert target.read_bytes() == source.read_bytes()
        finally:
            await machine.close()
        assert job.terminated

    asyncio.run(scenario())
