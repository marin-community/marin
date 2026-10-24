# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the ShellSim adapter through a complete local Harbor trial."""

import asyncio
from pathlib import Path

from harbor.environments.shellsim import ShellSimEnvironment
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from tests import shellbox_smoke
from shellbox.machine import MachineSpec, PackageSpec


def test_shellsim_harbor_trial(tmp_path: Path) -> None:
    asyncio.run(
        shellbox_smoke.main(
            tmp_path / "jobs",
            Path(__file__).parents[2] / "fixtures/shellbox/task",
            ["echo 'hello from qemu' > /workspace/answer.txt"],
            "harbor.environments.shellsim:ShellSimEnvironment",
            {"network_policy": "deny"},
        )
    )


def test_shellsim_downloads_to_remote_paths(tmp_path: Path) -> None:
    async def check() -> None:
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        environment = object.__new__(ShellSimEnvironment)
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
