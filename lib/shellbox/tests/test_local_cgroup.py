# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real aggregate resource enforcement inside an explicitly delegated cgroup."""

import asyncio
from pathlib import Path

import pytest
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import Command, ExitReason, HostImage, MachineSpec, MachineTerminated

pytestmark = pytest.mark.manual


@pytest.fixture
def delegated_parent(pytestconfig) -> Path:
    value = pytestconfig.getoption("--local-cgroup-parent")
    if value is None:
        pytest.skip("Pass --local-cgroup-parent inside a delegated worker service")
    return Path(value)


def test_local_cpu_limit_throttles_all_forked_children_and_reclaims_cgroup(delegated_parent):
    async def check():
        before = set(delegated_parent.glob("shellbox-*"))
        machine = await LocalMachineFactory(cgroup_parent=delegated_parent).create(
            MachineSpec(HostImage(), workdir="/", cpus=1, memory_mb=256)
        )
        [group] = set(delegated_parent.glob("shellbox-*")) - before

        def throttled():
            fields = (line.split() for line in (group / "cpu.stat").read_text().splitlines())
            return int(dict(fields)["nr_throttled"])

        initial = throttled()
        script = (
            "import os,time\nchildren=[]\nfor _ in range(4):\n"
            " child=os.fork()\n if child == 0:\n"
            "  start=time.process_time()\n  while time.process_time()-start < 0.3: pass\n"
            "  os._exit(0)\n children.append(child)\n"
            "for child in children: assert os.waitpid(child,0)[1] == 0\n"
        )
        try:
            result = await machine.run(Command(("python3", "-c", script), timeout=10))
            assert result.exit_code == 0, result.stderr.decode(errors="replace")
            assert throttled() > initial
        finally:
            await machine.close()
        assert set(delegated_parent.glob("shellbox-*")) == before

    asyncio.run(check())


def test_local_memory_limit_kills_a_forked_group_whose_combined_memory_exceeds_the_budget(delegated_parent):
    async def check():
        before = set(delegated_parent.glob("shellbox-*"))
        machine = await LocalMachineFactory(cgroup_parent=delegated_parent).create(
            MachineSpec(HostImage(), workdir="/", cpus=1, memory_mb=128)
        )
        script = (
            "import os\nready_read,ready_write=os.pipe()\nstop_read,stop_write=os.pipe()\n"
            "for _ in range(2):\n"
            " if os.fork() == 0:\n"
            "  payload=bytearray(90*1024*1024)\n  os.write(ready_write,b'1')\n"
            "  os.read(stop_read,1)\n  os._exit(0)\n"
            "assert len(os.read(ready_read,2)) > 0\n"
            "os.read(stop_read,1)\n"
        )
        try:
            try:
                result = await machine.run(Command(("python3", "-c", script), timeout=10))
            except MachineTerminated:
                pass
            else:
                assert result.reason is ExitReason.EXITED and result.exit_code in (137, -9)
        finally:
            await machine.close()
        assert set(delegated_parent.glob("shellbox-*")) == before

    asyncio.run(check())
