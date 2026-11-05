# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real resource enforcement and command containment inside delegated cgroups."""

import asyncio
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import cast

import pytest
import pytest_asyncio
from shellbox.backends.local.cgroup import CGROUP_ROOT, delegated_cgroup_parent
from shellbox.backends.local.machine import LocalMachine, LocalMachineFactory
from shellbox.http_service import HttpResponse, serve_http
from shellbox.machine import Command, ExitReason, HostImage, MachineSpec, MachineTerminated
from shellbox.task import ContainerSpec, ReadinessProbe, Task, TaskSpec, single_container_task

pytestmark = [pytest.mark.manual, pytest.mark.asyncio]
HTTP_CLIENT = "import sys,urllib.request; print(urllib.request.urlopen(sys.argv[1]).read().decode())"


@pytest.fixture
def delegated_parent(pytestconfig) -> Path:
    value = pytestconfig.getoption("--local-cgroup-parent")
    parent = Path(value) if value is not None else delegated_cgroup_parent()
    if parent is None:
        pytest.skip("Run inside a delegated worker service")
    return parent


@pytest_asyncio.fixture
async def create_task(delegated_parent: Path, pytestconfig) -> AsyncIterator[Callable[[TaskSpec], Awaitable[Task]]]:
    override = pytestconfig.getoption("--local-cgroup-parent")
    factory = LocalMachineFactory(cgroup_parent=Path(override) if override is not None else None)
    before = set(delegated_parent.glob("shellbox-*"))
    async with AsyncExitStack() as cleanup:

        async def create(spec: TaskSpec) -> Task:
            task = await factory.create_task(spec)
            cleanup.push_async_callback(task.close)
            return task

        yield create
    assert set(delegated_parent.glob("shellbox-*")) == before


@pytest.fixture
def create_machine(create_task) -> Callable[[MachineSpec], Awaitable[LocalMachine]]:
    async def create(spec: MachineSpec) -> LocalMachine:
        return cast(LocalMachine, (await create_task(single_container_task(spec))).primary)

    return create


@pytest.fixture
def unavailable_limit(request, monkeypatch):
    if request.param is None:
        return
    write_text = Path.write_text

    def write(path, data, *args, **kwargs):
        if path.name == request.param:
            raise PermissionError("Controller limit write denied")
        return write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)


async def test_local_discovery_keeps_allocations_inside_the_worker_unit(create_machine):
    membership = Path("/proc/self/cgroup").read_text().strip().removeprefix("0::/")
    current = CGROUP_ROOT / membership
    boundary = next(
        (path for path in (current, *current.parents) if path.name.endswith((".service", ".scope"))), current
    )
    before = set(boundary.rglob("shellbox-*"))
    machine = await create_machine(MachineSpec(HostImage(), workdir="/app"))
    assert len(set(boundary.rglob("shellbox-*")) - before) == 1
    result = await machine.run(Command(("sh", "-c", "printf answer > answer; cat answer")))
    assert (result.exit_code, result.stdout) == (0, b"answer")


@pytest.mark.parametrize("unavailable_limit", [None, "memory.max"], indirect=True)
async def test_local_cpu_limit_throttles_all_forked_children_and_reclaims_cgroup(
    delegated_parent, create_machine, unavailable_limit
):
    before = set(delegated_parent.glob("shellbox-*"))
    machine = await create_machine(MachineSpec(HostImage(), workdir="/", cpus=1, memory_mb=256))
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


@pytest.mark.parametrize("unavailable_limit", [None, "cpu.max"], indirect=True)
async def test_local_memory_limit_kills_a_forked_group_whose_combined_memory_exceeds_the_budget(
    delegated_parent, create_machine, unavailable_limit
):
    before = set(delegated_parent.glob("shellbox-*"))
    machine = await create_machine(MachineSpec(HostImage(), workdir="/", cpus=1, memory_mb=128))
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


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
async def test_local_command_interruption_reaps_detached_descendants_and_retains_task(create_task, interruption):
    spec = MachineSpec(HostImage(), workdir="/app")
    task = await create_task(
        TaskSpec(
            (ContainerSpec("actor", spec), ContainerSpec("runtime", spec, entrypoint=("sleep", "3600"))),
            primary="actor",
        )
    )
    neighbor = cast(LocalMachine, (await create_task(single_container_task(spec))).primary)
    script = (
        "import os,pathlib,signal\n"
        "if os.fork() == 0:\n"
        " os.setsid()\n pathlib.Path('/app/detached').write_text(str(os.getpid()))\n signal.pause()\n"
        "pathlib.Path('/app/answer').write_text('partial answer')\n"
        "signal.pause()\n"
    )
    pending = None
    try:
        background = await task.primary.run(Command(("sh", "-c", "sleep 3600 & echo $!")))
        pending = asyncio.create_task(
            task.primary.run(Command(("python3", "-c", script), timeout=1 if interruption == "timeout" else None))
        )
        ready = await task.primary.run(
            Command(("sh", "-c", "until test -s /app/detached && test -s /app/answer; do sleep 0.01; done"), timeout=5)
        )
        assert ready.exit_code == 0
        if interruption == "cancel":
            pending.cancel("original command cancellation")
            with pytest.raises(asyncio.CancelledError) as caught:
                await pending
            assert caught.value.args == ("original command cancellation",)
        else:
            assert (await pending).reason is ExitReason.TIMED_OUT
        answer = await task.primary.run(Command(("cat", "/app/answer")))
        assert answer.stdout == b"partial answer"
        reaped = await task.primary.run(
            Command(
                ("sh", "-c", 'pid=$(cat /app/detached); until ! test -e "/proc/$pid"; do sleep 0.01; done'),
                timeout=5,
            )
        )
        assert reaped.exit_code == 0
        previous = await task.primary.run(Command(("kill", "-0", background.stdout.decode().strip())))
        assert previous.exit_code == 0
        assert (await task.containers["runtime"].run(Command(("true",)))).exit_code == 0
        assert (await neighbor.run(Command(("true",)))).exit_code == 0
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize("retained_stream", ["stdout", "stderr"])
async def test_local_process_close_reaps_detached_helper_without_deleting_task(create_machine, retained_stream):
    machine = await create_machine(MachineSpec(HostImage(), workdir="/app"))
    session = None
    try:
        assert (await machine.run(Command(("sh", "-c", "echo answer > /app/answer")))).exit_code == 0
        redirect = "2>/dev/null" if retained_stream == "stdout" else ">/dev/null"
        session = await machine.open_process(Command(("setsid", "sh", "-c", f"echo ready; exec sleep 3600 {redirect}")))
        assert await asyncio.wait_for(session.read(6), 5) == b"ready\n"
        await asyncio.wait_for(session.close(), 5)
        assert await session.wait() != 0
        answer = await machine.run(Command(("cat", "/app/answer")))
        assert (answer.exit_code, answer.stdout) == (0, b"answer\n")
    finally:
        if session is not None:
            await session.close()


@pytest.mark.parametrize("operation", ["run", "open_process"])
async def test_local_cancelled_acquisition_reclaims_late_command_and_retains_files(
    create_machine, monkeypatch, operation
):
    machine = await create_machine(MachineSpec(HostImage(), workdir="/app"))
    connection_made = asyncio.subprocess.SubprocessStreamProtocol.connection_made

    def cancel_before_handle_returns(protocol, transport):
        connection_made(protocol, transport)
        monkeypatch.setattr(asyncio.subprocess.SubprocessStreamProtocol, "connection_made", connection_made)
        # The child exists, but asyncio has not handed its handle to Shellbox.
        pending.cancel("original acquisition cancellation")
        asyncio.get_running_loop().call_soon(pending.cancel, "repeated acquisition cancellation")

    pending = None
    try:
        assert (await machine.run(Command(("sh", "-c", "echo answer > /app/answer")))).exit_code == 0
        monkeypatch.setattr(asyncio.subprocess.SubprocessStreamProtocol, "connection_made", cancel_before_handle_returns)
        command = Command(("setsid", "sleep", "3600"))
        pending = asyncio.create_task(machine.run(command) if operation == "run" else machine.open_process(command))
        with pytest.raises(asyncio.CancelledError) as caught:
            await pending
        assert caught.value.args == ("original acquisition cancellation",)
        answer = await machine.run(Command(("cat", "/app/answer")))
        assert (answer.exit_code, answer.stdout) == (0, b"answer\n")
        remaining = await machine.run(Command(("sh", "-c", "ps -eo args")))
        assert remaining.exit_code == 0
        assert b"sleep 3600" not in remaining.stdout
    finally:
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)


async def test_local_quiesce_stops_prior_tmux_agent_and_preserves_http_helper_and_peer(create_task):
    if shutil.which("tmux") is None:
        pytest.skip("A tmux executable is required for persistent-agent cancellation")

    spec = MachineSpec(HostImage(), workdir="/app")
    task = await create_task(
        TaskSpec(
            (ContainerSpec("actor", spec), ContainerSpec("runtime", spec, entrypoint=("sleep", "3600"))),
            primary="actor",
        )
    )
    machine = cast(LocalMachine, task.primary)

    async def handler(request):
        return HttpResponse(200, b"helper alive", "text/plain")

    async with (
        asyncio.TaskGroup() as service_tasks,
        serve_http(machine, handler, task_group=service_tasks, port=8787) as url,
    ):
        started = await machine.run(
            Command(
                (
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    "agent",
                    "echo $$ > /app/agent-pid; echo complete > /app/answer; exec sleep 3600",
                )
            )
        )
        assert started.exit_code == 0, started.stderr
        ready = await machine.run(Command(("sh", "-c", "until test -s /app/answer; do sleep 0.01; done"), timeout=5))
        assert ready.exit_code == 0
        await machine.quiesce_commands()
        stopped = await machine.run(
            Command(
                ("sh", "-c", 'pid=$(cat /app/agent-pid); until ! test -e "/proc/$pid"; do sleep 0.01; done'),
                timeout=5,
            )
        )
        assert stopped.exit_code == 0
        verifier = await machine.run(Command(("sh", "-c", 'test "$(cat /app/answer)" = complete')))
        assert verifier.exit_code == 0
        helper = await machine.run(
            Command(
                (
                    "python3",
                    "-c",
                    HTTP_CLIENT,
                    url,
                ),
                timeout=5,
            )
        )
        assert (helper.exit_code, helper.stdout) == (0, b"helper alive\n")
        assert (await task.containers["runtime"].run(Command(("true",)))).exit_code == 0


async def test_local_cancelled_readiness_preserves_helper_until_owner_closes_it(create_task):
    spec = TaskSpec(
        (
            ContainerSpec(
                "actor",
                MachineSpec(HostImage(), workdir="/app"),
                readiness=ReadinessProbe(
                    Command(("sh", "-c", "touch /app/probe-started; exec sleep 3600")),
                    timeout=30,
                    interval=0.01,
                ),
            ),
        ),
        primary="actor",
    )
    task = await create_task(spec)
    waiting = None

    async def handler(request):
        return HttpResponse(200, b"helper alive", "text/plain")

    try:
        async with (
            asyncio.TaskGroup() as service_tasks,
            serve_http(task.primary, handler, task_group=service_tasks, port=8787) as url,
        ):
            waiting = asyncio.create_task(task.wait_ready())
            started = await task.primary.run(
                Command(("sh", "-c", "until test -f /app/probe-started; do sleep 0.01; done"), timeout=5)
            )
            assert started.exit_code == 0
            waiting.cancel("cancelled readiness")
            with pytest.raises(asyncio.CancelledError) as caught:
                await waiting
            assert caught.value.args == ("cancelled readiness",)
            helper = await task.primary.run(
                Command(
                    (
                        "python3",
                        "-c",
                        HTTP_CLIENT,
                        url,
                    ),
                    timeout=5,
                )
            )
            assert (helper.exit_code, helper.stdout) == (0, b"helper alive\n")
    finally:
        if waiting is not None and not waiting.done():
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
