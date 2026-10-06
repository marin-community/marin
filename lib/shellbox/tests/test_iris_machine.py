# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Iris exec wire path with a local subprocess provider."""

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from iris.rpc import job_pb2
from shellbox.backends.iris import machine as iris_machine
from shellbox.backends.iris.machine import IrisMachine, IrisMachineFactory
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


@pytest.mark.parametrize("resource", ["cpus", "storage_mb"])
def test_zero_resource_requests_cannot_silently_select_iris_defaults(resource):
    with pytest.raises(ValueError, match=resource):
        MachineSpec(source=RegistryImage("ubuntu:24.04"), **{resource: 0})


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


class _SubmitRejected(Exception):
    pass


class RecordingClient(LocalClient):
    def __init__(self):
        self.submitted: dict = {}

    def submit(self, **kwargs):
        self.submitted = kwargs
        raise _SubmitRejected


@pytest.mark.parametrize(
    ("network", "egress"),
    [
        (NetworkPolicy.ALLOW, job_pb2.EGRESS_POLICY_INTERNET),
        (NetworkPolicy.DENY, job_pb2.EGRESS_POLICY_NONE),
    ],
)
def test_network_policy_selects_the_egress_policy(monkeypatch, network, egress):
    client = RecordingClient()
    endpoint = SimpleNamespace(url="http://controller:10000", credentials=None, close=lambda: None)
    monkeypatch.setattr(iris_machine, "connect_controller", lambda **_: endpoint)
    monkeypatch.setattr(iris_machine.IrisClient, "remote", lambda *_, **__: client)

    factory = IrisMachineFactory(controller_url="http://controller:10000")
    with pytest.raises(_SubmitRejected):
        asyncio.run(factory.create(MachineSpec(source=RegistryImage("ubuntu:24.04"), network=network)))

    assert client.submitted["container_profile"] == job_pb2.CONTAINER_PROFILE_SANDBOX
    assert client.submitted["egress_policy"] == egress
