# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host command execution and namespace isolation."""

import asyncio
import os
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from shellbox.backends.local.launch import COMMAND_NICENESS
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.backends.local.runtime import LocalMachine
from shellbox.backends.local.sandbox import NPROC_HEADROOM, SandboxUnavailable
from shellbox.machine import (
    Command,
    DockerImage,
    ExitReason,
    HostImage,
    MachineSpec,
    NetworkPolicy,
    Result,
    UnsupportedMachineSpec,
)
from shellbox.task import ContainerSpec, TaskSpec

from lib.shellbox.tests.local_machine_support import (
    ACCESS_PROBE,
    SPEC,
    THREAD_BOMB,
    access_outcomes,
    make_factory,
    on_machine,
    run_command,
)


def test_local_factory_reports_unsupported_host(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(SandboxUnavailable):
        LocalMachineFactory()


def test_local_numeric_limits_allow_execution_without_delegation(undelegated_factory):
    async def check():
        machine = await undelegated_factory.create(MachineSpec(HostImage(), workdir="/app", cpus=2, memory_mb=256))
        try:
            result = await machine.run(Command(("sh", "-c", "printf answer > answer; cat answer")))
            assert (result.exit_code, result.stdout) == (0, b"answer")
        finally:
            await machine.close()

    asyncio.run(check())


def test_command_reports_output_status_and_runs_in_its_cwd_with_its_env(factory):
    spec = MachineSpec(HostImage(), workdir="/app", env={"SPEC": "from-spec", "ANSWER": "overridden"})
    script = 'pwd; printf "%s %s\\n" "$SPEC" "$ANSWER"; cat; echo diagnostic >&2; exit 3'

    async def scenario(machine: LocalMachine) -> tuple[Result, Result]:
        explicit = await machine.run(Command(("sh", "-c", script), cwd="/tmp", env={"ANSWER": "42"}, stdin=b"input\n"))
        default = await machine.run(Command(("pwd",)))
        return explicit, default

    explicit, default = on_machine(factory, spec, scenario)
    assert explicit == Result(3, b"/tmp\nfrom-spec 42\ninput\n", b"diagnostic\n", False, False, ExitReason.EXITED)
    assert default.stdout == b"/app\n"


@pytest.mark.parametrize(
    ("argv", "exit_code"),
    [(("shellbox-no-such-program",), 127), (("sh", "-c", "kill -KILL $$"), 128 + 9)],
)
def test_failures_report_shell_exit_statuses(factory, argv, exit_code):
    assert run_command(factory, Command(argv)).exit_code == exit_code


def test_bin_dirs_win_path_lookup_and_host_environment_is_not_inherited(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELLBOX_HOST_SECRET", "leaked")
    bin_dir = tmp_path / "venv-bin"
    bin_dir.mkdir()
    fake = bin_dir / "python3"
    fake.write_text("#!/bin/sh\necho fake-python\n")
    fake.chmod(0o755)
    factory = make_factory(read_only=(bin_dir,), bin_dirs=(bin_dir,))

    async def scenario(machine: LocalMachine) -> tuple[Result, Result]:
        return await machine.run(Command(("python3",))), await machine.run(Command(("env",)))

    python, environment = on_machine(factory, SPEC, scenario)
    assert python.stdout == b"fake-python\n"
    assert b"SHELLBOX_HOST_SECRET" not in environment.stdout


@pytest.mark.parametrize("environment_source", ["spec", "command"])
def test_task_loader_environment_takes_effect_only_inside_sandbox(tmp_path, environment_source):
    # A preloaded library can write this host directory before isolation. Inside the sandbox it is read-only.
    library = tmp_path / "preload.so"
    marker = tmp_path / "host-marker"
    factory = make_factory(read_only=(tmp_path,))
    source = b"""
    #include <fcntl.h>
    #include <stdlib.h>
    #include <unistd.h>
    __attribute__((constructor)) static void record(void) {
        const char *path = getenv("SHELLBOX_HOST_MARKER");
        if (path) {
            int fd = open(path, O_WRONLY | O_CREAT, 0600);
            if (fd >= 0) close(fd);
        }
    }
    """
    subprocess.run(["cc", "-shared", "-fPIC", "-x", "c", "-", "-o", str(library)], input=source, check=True)
    environment = {"LD_PRELOAD": str(library), "SHELLBOX_HOST_MARKER": str(marker)}
    spec = replace(SPEC, env=environment if environment_source == "spec" else {})
    command = Command(
        ("sh", "-c", 'printf "%s" "$LD_PRELOAD"'),
        env=environment if environment_source == "command" else {},
    )
    result = run_command(factory, command, spec)
    assert (result.exit_code, result.stdout) == (0, str(library).encode()), result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("environment_source", ["spec", "command"])
def test_task_environment_is_absent_from_host_process_arguments(factory, monkeypatch, environment_source):
    secret = "canary with spaces, 'quotes', and\na newline"
    environment = {"SHELLBOX_TEST_SECRET": secret}
    spec = replace(SPEC, env=environment if environment_source == "spec" else {})
    command = Command(
        ("sh", "-c", 'read release; printf "%s" "$SHELLBOX_TEST_SECRET"'),
        env=environment if environment_source == "command" else {},
        stdin=b"release\n",
    )
    create_subprocess = asyncio.create_subprocess_exec
    host_arguments = []

    async def observe_launcher(*args, **kwargs):
        process = await create_subprocess(*args, **kwargs)
        # Inspect the actual launcher rather than only the supplied subprocess arguments.
        host_arguments.append(Path(f"/proc/{process.pid}/cmdline").read_bytes())
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", observe_launcher)
    result = run_command(factory, command, spec)
    assert (result.exit_code, result.stdout) == (0, secret.encode()), result.stderr
    assert host_arguments and all(secret.encode() not in arguments for arguments in host_arguments)


def test_task_environment_cannot_inject_arguments_through_nul_bytes(factory):
    # Embedded NULs must remain invalid environment data rather than becoming extra mounts.
    command = Command(("true",), env={"TASK_VALUE": "value\0--ro-bind\0/etc\0/host-etc"})
    with pytest.raises(ValueError, match="NUL"):
        run_command(factory, command)


def test_timeout_kills_the_command_and_its_descendants(factory):
    # run() returns only once the output pipes close, so a surviving grandchild would hold it past the test timeout.
    command = Command(("sh", "-c", "sleep 3600 & wait"), timeout=1, stdin=b"unread")
    assert run_command(factory, command) == Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)


def test_background_processes_survive_successful_commands(factory):
    async def scenario(machine):
        started = await machine.run(
            Command(("sh", "-c", "sleep 3600 <&0 >/dev/null 2>&1 & echo $!"), timeout=5, stdin=b"unused" * 200_000)
        )
        assert started.exit_code == 0
        child = started.stdout.decode().strip()
        observed = await machine.run(Command(("cat", f"/proc/{child}/cmdline")))
        assert observed.exit_code == 0
        assert observed.stdout.split(b"\0")[:2] == [b"sleep", b"3600"]

    on_machine(factory, SPEC, scenario)


def test_output_beyond_the_limit_is_truncated(factory):
    result = run_command(factory, Command(("sh", "-c", "yes | head -c 1000000; printf ab >&2"), output_limit_bytes=4))
    assert (result.exit_code, result.stdout, result.stdout_truncated) == (0, b"y\ny\n", True)
    assert (result.stderr, result.stderr_truncated) == (b"ab", False)


@pytest.mark.parametrize(
    "spec",
    [
        MachineSpec(DockerImage("python:3.12")),
        MachineSpec(HostImage(), workdir="", gpus=1),
        MachineSpec(HostImage(), workdir="/usr/lib/shellbox-test"),
    ],
)
def test_specs_the_host_cannot_honor_are_rejected(factory, spec):
    with pytest.raises(UnsupportedMachineSpec):
        asyncio.run(factory.create(spec))


def test_the_filesystem_root_as_workdir_runs_commands_there(factory):
    # Grader scripts that read absolute paths run with cwd "/".
    assert run_command(factory, Command(("pwd",)), MachineSpec(HostImage(), workdir="/")).stdout == b"/\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="Root may run commands as any user")
def test_a_non_root_host_runs_only_its_own_user(factory):
    async def scenario(machine: LocalMachine) -> Result:
        with pytest.raises(UnsupportedMachineSpec):
            await machine.run(Command(("id", "-u"), user="0"))
        return await machine.run(Command(("id", "-u"), user=str(os.geteuid())))

    assert on_machine(factory, SPEC, scenario).stdout == f"{os.geteuid()}\n".encode()


@pytest.mark.skipif(os.geteuid() != 0, reason="Switching users requires a root host process")
def test_a_root_host_runs_commands_as_the_requested_user_without_capabilities(factory):
    script = "id -u; grep CapEff /proc/self/status; echo written > /tmp/user-file && echo wrote"
    result = run_command(factory, Command(("sh", "-c", script), user="65534"))
    assert result.stdout == b"65534\nCapEff:\t0000000000000000\nwrote\n"


def test_commands_cannot_gain_privileges(factory):
    status = run_command(factory, Command(("cat", "/proc/self/status"))).stdout.decode().splitlines()
    assert "NoNewPrivs:\t1" in status
    assert "CapEff:\t0000000000000000" in status


def test_commands_run_below_the_host_process_cpu_priority(factory):
    # A CPU-bound command must not starve the host process that drives its machine.
    niceness = run_command(factory, Command(("sh", "-c", "cut -d' ' -f19 /proc/self/stat"))).stdout
    assert int(niceness) == min(os.nice(0) + COMMAND_NICENESS, 19)


@pytest.mark.skipif(os.geteuid() == 0, reason="RLIMIT_NPROC does not limit root")
def test_a_command_cannot_start_unbounded_tasks(python_factory):
    host_tasks = int(Path("/proc/loadavg").read_text().split()[3].split("/")[1])
    # Past the limit even if the host gains tasks before the command starts.
    attempts = host_tasks + 4 * NPROC_HEADROOM
    result = run_command(python_factory, Command(("python3", "-c", THREAD_BOMB, str(attempts)), timeout=30))
    started = int(result.stdout)
    # The limit leaves the command room to work however many tasks this user already runs.
    assert NPROC_HEADROOM // 2 < started < attempts


def test_commands_see_only_their_private_root_and_processes(python_factory, outside_dir):
    (outside_dir / "secret.txt").write_text("host")
    outcomes = access_outcomes(
        python_factory,
        ("read", outside_dir / "secret.txt"),
        ("write", outside_dir / "created.txt"),
        ("read", f"/proc/{os.getpid()}/environ"),
        ("signal", os.getpid()),
    )
    assert "ok" not in outcomes
    assert not (outside_dir / "created.txt").exists()


@pytest.mark.parametrize(("network", "connects"), [(NetworkPolicy.DENY, False), (NetworkPolicy.ALLOW, True)])
@pytest.mark.parametrize("explicit_task", [False, True], ids=["machine-api", "task-api"])
def test_singleton_network_policy_controls_host_connections(python_factory, network, connects, explicit_task):
    spec = MachineSpec(HostImage(), workdir="/app", network=network)

    async def scenario(port):
        if explicit_task:
            task = await python_factory.create_task(TaskSpec((ContainerSpec("actor", spec),), primary="actor"))
            machine = task.primary
        else:
            machine = await python_factory.create(spec)
        try:
            return await machine.run(Command(("python3", "-c", ACCESS_PROBE, "connect", str(port))))
        finally:
            await machine.close()

    with socket.create_server(("127.0.0.1", 0)) as server:
        result = asyncio.run(scenario(server.getsockname()[1]))
    assert result.exit_code == 0
    assert (result.stdout.split() == [b"ok"]) == connects
