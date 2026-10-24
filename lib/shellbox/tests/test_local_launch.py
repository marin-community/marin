# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent launcher behavior at the namespace and exec boundaries."""

import asyncio
import tempfile

import pytest
from shellbox.backends.local import machine as local
from shellbox.machine import Command, HostImage, MachineSpec

# Inject a failing kernel boundary in the separately executed launcher. Keeper
# creation needs no setns; entering the already-created machine does.
DENIED_NAMESPACE_ENTRY = """
import os
def denied_setns(descriptor, flags):
    raise PermissionError(1, "x" * 65536)
os.setns = denied_setns
"""


@pytest.fixture
def factory() -> local.LocalMachineFactory:
    try:
        return local.LocalMachineFactory()
    except local.SandboxUnavailable as error:
        pytest.skip(str(error))


def test_factory_rejects_namespace_entry_failure_and_removes_probe_files(factory, monkeypatch, tmp_path):
    original_mkdtemp = tempfile.mkdtemp

    def probe_directory(*args, **kwargs):
        return original_mkdtemp(*args, **{**kwargs, "dir": tmp_path})

    monkeypatch.setattr(tempfile, "mkdtemp", probe_directory)
    monkeypatch.setattr(local, "LAUNCH_SOURCE", DENIED_NAMESPACE_ENTRY + local.LAUNCH_SOURCE)
    with pytest.raises(local.SandboxUnavailable):
        local.LocalMachineFactory(bwrap=factory.bwrap)
    assert list(tmp_path.iterdir()) == []


def test_namespace_setup_failure_is_infrastructure_error_and_machine_remains_usable(factory, monkeypatch):
    async def check():
        machine = await factory.create(MachineSpec(HostImage(), workdir="/"))
        try:
            with monkeypatch.context() as patch:
                patch.setattr(local, "LAUNCH_SOURCE", DENIED_NAMESPACE_ENTRY + local.LAUNCH_SOURCE)
                with pytest.raises(RuntimeError):
                    await machine.run(Command(("true",), timeout=5))
            result = await machine.run(Command(("/bin/sh", "-c", "exit 1")))
            assert result.exit_code == 1
        finally:
            await machine.close()

    asyncio.run(check())


def test_guest_exec_inherits_no_private_launcher_descriptors(factory):
    async def check():
        machine = await factory.create(MachineSpec(HostImage(), workdir="/"))
        try:
            result = await machine.run(
                Command(("/bin/sh", "-c", 'for fd in /proc/$$/fd/*; do printf "%s\\n" "${fd##*/}"; done'))
            )
            assert result.exit_code == 0
            assert set(result.stdout.splitlines()) == {b"0", b"1", b"2"}
        finally:
            await machine.close()

    asyncio.run(check())


def test_commands_join_the_keeper_cgroup_namespace(factory):
    async def check():
        machine = await factory.create(MachineSpec(HostImage(), workdir="/"))
        try:
            result = await machine.run(
                Command(("/bin/sh", "-c", "readlink /proc/self/ns/cgroup; readlink /proc/1/ns/cgroup"))
            )
            assert result.exit_code == 0
            command, keeper = result.stdout.splitlines()
            assert command == keeper
        finally:
            await machine.close()

    asyncio.run(check())
