# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the ShellSim adapter through a complete local Harbor trial."""

import asyncio
import importlib
from pathlib import Path

import pytest
from shellbox.machine import MachineSpec, ShellSimBuiltins


def test_shellsim_harbor_trial(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    pytest.importorskip("shellsim")
    smoke = importlib.import_module("harbor_smoke")

    asyncio.run(
        smoke.main(
            tmp_path / "jobs",
            Path(__file__).parent / "manual/task",
            ["echo 'hello from qemu' > /workspace/answer.txt"],
            "shellbox.backends.shellsim.environment:ShellSimEnvironment",
            {"network_policy": "deny"},
        )
    )


def test_shellsim_downloads_to_remote_paths(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    pytest.importorskip("shellsim")
    StoragePath = pytest.importorskip("rigging.filesystem.storage_path").StoragePath
    environment_module = importlib.import_module("shellbox.backends.shellsim.environment")
    machine_module = importlib.import_module("shellbox.backends.shellsim.machine")

    async def check() -> None:
        machine = await machine_module.ShellSimMachineFactory().create(MachineSpec(ShellSimBuiltins()))
        environment = object.__new__(environment_module.ShellSimEnvironment)
        environment.machine = machine
        try:
            source = tmp_path / "logs"
            (source / "nested").mkdir(parents=True)
            (source / "agent.log").write_text("agent failed")
            (source / "nested" / "trace.txt").write_text("trace")
            await machine.upload(source, "/logs/agent")

            remote = StoragePath(f"memory://shellbox-{tmp_path.name}")
            await environment.download_file("/logs/agent/agent.log", remote / "agent.log")
            await environment.download_dir("/logs/agent", remote / "downloaded")
            await environment.download_file("/logs/agent/agent.log", tmp_path / "local-agent.log")
            await environment.download_dir("/logs/agent", f"file://{tmp_path}/local-download")

            assert (remote / "agent.log").read_text() == "agent failed"
            assert (remote / "downloaded" / "agent.log").read_text() == "agent failed"
            assert (remote / "downloaded" / "nested" / "trace.txt").read_text() == "trace"
            assert (tmp_path / "local-agent.log").read_text() == "agent failed"
            assert (tmp_path / "local-download" / "nested" / "trace.txt").read_text() == "trace"
        finally:
            await machine.close()

    asyncio.run(check())
