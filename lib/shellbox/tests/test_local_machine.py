# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local machines: host subprocesses that own a set of host directories."""

import asyncio
import os
import shutil
import socket
import sys
import tarfile
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from shellbox.backends.local.machine import NPROC_HEADROOM, LocalMachine, LocalMachineFactory
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

# Prints "ok" or the errno name for each (operation, target) pair in argv, so one command reports several checks.
ACCESS_PROBE = """
import errno, os, socket, sys

def attempt(operation, target):
    if operation == "read":
        open(target).close()
    elif operation == "write":
        open(target, "w").close()
    elif operation == "signal":
        os.kill(int(target), 0)
    else:
        socket.create_connection(("127.0.0.1", int(target)), timeout=10).close()

for operation, target in zip(sys.argv[1::2], sys.argv[2::2]):
    try:
        attempt(operation, target)
        print("ok")
    except OSError as error:
        print(errno.errorcode[error.errno])
"""
# Starts up to argv[1] threads, which RLIMIT_NPROC counts as it counts processes, and prints how many started.
# The threads end with the command, so a test leaves no stray processes behind.
THREAD_BOMB = """
import sys, threading

threading.stack_size(1 << 16)
release = threading.Event()
started = 0
try:
    while started < int(sys.argv[1]):
        threading.Thread(target=release.wait, daemon=True).start()
        started += 1
except RuntimeError:
    pass
release.set()
print(started)
"""


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "tests", tmp_path / "app"


@pytest.fixture
def factory(tmp_path: Path, roots: tuple[Path, Path]) -> LocalMachineFactory:
    return LocalMachineFactory(tuple(map(str, roots)), lock_path=tmp_path / "local.lock")


@pytest.fixture
def python_factory(tmp_path: Path, roots: tuple[Path, Path]) -> LocalMachineFactory:
    """A factory whose commands find this test's Python, from a venv whose interpreter may lie anywhere."""
    return LocalMachineFactory(
        tuple(map(str, roots)), bin_dirs=(Path(sys.executable).parent,), lock_path=tmp_path / "local.lock"
    )


@pytest.fixture
def spec(roots: tuple[Path, Path]) -> MachineSpec:
    return MachineSpec(HostImage(), workdir=str(roots[1] / "work"))


@pytest.fixture
def outside_dir() -> Iterator[Path]:
    """A directory the test user may write, outside every path that local commands may read or write."""
    path = Path(tempfile.mkdtemp(dir="/var/tmp"))
    yield path
    shutil.rmtree(path)


def on_machine[T](
    factory: LocalMachineFactory, spec: MachineSpec, scenario: Callable[[LocalMachine], Awaitable[T]]
) -> T:
    async def run() -> T:
        machine = await factory.create(spec)
        try:
            return await scenario(machine)
        finally:
            await machine.close()

    return asyncio.run(run())


def run_command(factory: LocalMachineFactory, spec: MachineSpec, command: Command) -> Result:
    return on_machine(factory, spec, lambda machine: machine.run(command))


def access_outcomes(factory: LocalMachineFactory, spec: MachineSpec, *checks: tuple[str, object]) -> list[str]:
    argv = [str(part) for check in checks for part in check]
    return run_command(factory, spec, Command(("python3", "-c", ACCESS_PROBE, *argv))).stdout.decode().split()


def test_command_reports_output_status_and_runs_in_its_cwd_with_its_env(factory, spec, roots):
    spec = MachineSpec(HostImage(), workdir=spec.workdir, env={"SPEC": "from-spec", "ANSWER": "overridden"})
    script = 'pwd; printf "%s %s\\n" "$SPEC" "$ANSWER"; cat; echo diagnostic >&2; exit 3'

    async def scenario(machine: LocalMachine) -> tuple[Result, Result]:
        explicit = await machine.run(
            Command(("sh", "-c", script), cwd=str(roots[0]), env={"ANSWER": "42"}, stdin=b"input\n")
        )
        default = await machine.run(Command(("pwd",)))
        return explicit, default

    explicit, default = on_machine(factory, spec, scenario)
    assert explicit == Result(
        3, f"{roots[0]}\nfrom-spec 42\ninput\n".encode(), b"diagnostic\n", False, False, ExitReason.EXITED
    )
    assert default.stdout == f"{spec.workdir}\n".encode()


@pytest.mark.parametrize(
    ("argv", "exit_code"),
    [(("shellbox-no-such-program",), 127), (("sh", "-c", "kill -KILL $$"), 128 + 9)],
)
def test_failures_report_shell_exit_statuses(factory, spec, argv, exit_code):
    assert run_command(factory, spec, Command(argv)).exit_code == exit_code


def test_bin_dirs_win_path_lookup_and_host_environment_is_not_inherited(tmp_path, roots, spec, monkeypatch):
    monkeypatch.setenv("SHELLBOX_HOST_SECRET", "leaked")
    bin_dir = tmp_path / "venv-bin"
    bin_dir.mkdir()
    fake = bin_dir / "python3"
    fake.write_text("#!/bin/sh\necho fake-python\n")
    fake.chmod(0o755)
    factory = LocalMachineFactory(tuple(map(str, roots)), bin_dirs=(bin_dir,), lock_path=tmp_path / "local.lock")

    async def scenario(machine: LocalMachine) -> tuple[Result, Result]:
        return await machine.run(Command(("python3",))), await machine.run(Command(("env",)))

    python, environment = on_machine(factory, spec, scenario)
    assert python.stdout == b"fake-python\n"
    assert b"SHELLBOX_HOST_SECRET" not in environment.stdout


def _running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def test_timeout_kills_the_command_and_its_descendants(factory, spec, roots):
    pid_file = roots[0] / "child.pid"

    async def scenario(machine: LocalMachine) -> tuple[Result, int]:
        result = await machine.run(
            # A surviving grandchild would hold the output pipes open past the test timeout.
            Command(("sh", "-c", f"sleep 3600 & echo $! > {pid_file}; wait"), timeout=1, stdin=b"unread")
        )
        return result, int(pid_file.read_text())

    result, child = on_machine(factory, spec, scenario)
    assert result == Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
    assert not _running(child)


def test_output_beyond_the_limit_is_truncated(factory, spec):
    result = run_command(
        factory, spec, Command(("sh", "-c", "yes | head -c 1000000; printf ab >&2"), output_limit_bytes=4)
    )
    assert (result.exit_code, result.stdout, result.stdout_truncated) == (0, b"y\ny\n", True)
    assert (result.stderr, result.stderr_truncated) == (b"ab", False)


def test_files_round_trip_between_host_paths_and_commands(tmp_path, factory, spec, roots):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/data.txt").write_text("from directory")
    single = tmp_path / "single.txt"
    single.write_text("from file")
    downloaded = tmp_path / "downloaded"

    async def scenario(machine: LocalMachine) -> Result:
        await machine.upload(single, f"{roots[0]}/inputs/single.txt")
        await machine.upload(source, f"{roots[0]}/tree")
        read = await machine.run(Command(("cat", "inputs/single.txt", "tree/nested/data.txt"), cwd=str(roots[0])))
        await machine.run(Command(("sh", "-c", f"mkdir -p {roots[1]}/out && echo 0.5 > {roots[1]}/out/reward.txt")))
        await machine.download(f"{roots[1]}/out/reward.txt", downloaded / "reward.txt")
        with pytest.raises(RuntimeError):
            await machine.download(f"{roots[1]}/out/missing.txt", downloaded / "missing.txt")
        return read

    assert on_machine(factory, spec, scenario).stdout == b"from filefrom directory"
    assert (downloaded / "reward.txt").read_text() == "0.5\n"


def test_owned_roots_start_empty_and_are_removed_with_scratch_uploads_on_close(tmp_path, factory, spec, roots):
    (roots[0] / "stale").mkdir(parents=True)
    (roots[0] / "stale/left-over.txt").write_text("previous machine")
    archive = tmp_path / "archive.tar"
    archive.write_bytes(b"archive")
    scratch = Path(f"/tmp/shellbox-local-test-{uuid.uuid4().hex}")

    async def scenario(machine: LocalMachine) -> list[list[str]]:
        await machine.upload(archive, f"{scratch}/staging/archive.tar")
        assert (scratch / "staging/archive.tar").read_bytes() == b"archive"
        with pytest.raises(UnsupportedMachineSpec):
            await machine.upload(archive, "/var/lib/shellbox-local-test/archive.tar")
        return [sorted(path.name for path in root.iterdir()) for root in roots]

    assert on_machine(factory, spec, scenario) == [[], ["work"]]
    assert not any(path.exists() for path in (*roots, scratch))


def test_a_second_machine_waits_for_the_first_to_close(tmp_path, spec, roots):
    first_factory, second_factory = (
        LocalMachineFactory(tuple(map(str, roots)), lock_path=tmp_path / "local.lock") for _ in range(2)
    )

    async def scenario() -> list[str]:
        first = await first_factory.create(spec)
        await first.run(Command(("touch", f"{roots[0]}/first.txt")))
        pending = asyncio.create_task(second_factory.create(spec))
        # Without the lock, the second create would reset the roots well within several lock polls.
        done, _ = await asyncio.wait({pending}, timeout=0.5)
        assert not done
        assert (roots[0] / "first.txt").exists()
        await first.close()
        second = await asyncio.wait_for(pending, timeout=5)
        try:
            return sorted(path.name for path in roots[0].iterdir())
        finally:
            await second.close()

    assert asyncio.run(scenario()) == []


@pytest.mark.parametrize(
    "spec",
    [
        MachineSpec(DockerImage("python:3.12")),
        MachineSpec(HostImage(), workdir="", gpus=1),
        MachineSpec(HostImage(), workdir="/workspace-outside-owned-roots"),
    ],
)
def test_specs_the_host_cannot_honor_are_rejected(factory, spec):
    with pytest.raises(UnsupportedMachineSpec):
        asyncio.run(factory.create(spec))


def test_a_shared_root_keeps_its_files_and_loses_only_the_machine_uploads(tmp_path, roots):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "kept.txt").write_text("host")
    upload = tmp_path / "answer.txt"
    upload.write_text("7")
    factory = LocalMachineFactory(tuple(map(str, roots)), shared_roots=(str(shared),), lock_path=tmp_path / "lock")

    async def scenario() -> Result:
        machine = await factory.create(MachineSpec(HostImage(), workdir=str(shared)))
        try:
            await machine.upload(upload, str(shared / "staged" / "answer.txt"))
            return await machine.run(Command(("sh", "-c", "cat kept.txt staged/answer.txt && echo written > made.txt")))
        finally:
            await machine.close()

    assert asyncio.run(scenario()).stdout == b"host7"
    assert (shared / "kept.txt").exists() and (shared / "made.txt").exists()
    assert not (shared / "staged").exists()


def test_an_archive_extracts_at_the_filesystem_root_into_an_owned_root(tmp_path, roots, factory):
    # The grading runtime unpacks its inputs with ``tar -C /``; tar opens "/" to extract relative members.
    archive = tmp_path / "inputs.tar"
    with tarfile.open(archive, "w") as tar:
        payload = tmp_path / "payload"
        payload.write_text("7")
        tar.add(payload, arcname=str(roots[0]).lstrip("/") + "/answer.txt")

    async def scenario() -> Result:
        machine = await factory.create(MachineSpec(HostImage(), workdir=""))
        try:
            await machine.upload(archive, "/tmp/shellbox-test-inputs.tar")
            return await machine.run(
                Command(("sh", "-c", f"tar -xf /tmp/shellbox-test-inputs.tar -C / && cat {roots[0]}/answer.txt"))
            )
        finally:
            await machine.close()

    result = asyncio.run(scenario())
    assert (result.exit_code, result.stdout) == (0, b"7"), result.stderr


def test_the_filesystem_root_as_workdir_runs_commands_there(factory):
    # Grader scripts that read absolute paths run with cwd "/", which no machine owns.
    assert run_command(factory, MachineSpec(HostImage(), workdir="/"), Command(("pwd",))).stdout == b"/\n"


def test_a_factory_cannot_own_the_directory_its_process_runs_from(tmp_path, monkeypatch):
    bundle = tmp_path / "app"
    (bundle / "lib").mkdir(parents=True)
    monkeypatch.chdir(bundle / "lib")
    with pytest.raises(ValueError, match="runs from"):
        LocalMachineFactory((str(bundle),), lock_path=tmp_path / "local.lock")
    assert (bundle / "lib").is_dir()


@pytest.mark.skipif(os.geteuid() == 0, reason="Root may run commands as any user")
def test_a_non_root_host_runs_only_its_own_user(factory, spec):
    async def scenario(machine: LocalMachine) -> Result:
        with pytest.raises(UnsupportedMachineSpec):
            await machine.run(Command(("id", "-u"), user="0"))
        return await machine.run(Command(("id", "-u"), user=str(os.geteuid())))

    assert on_machine(factory, spec, scenario).stdout == f"{os.geteuid()}\n".encode()


@pytest.mark.skipif(os.geteuid() != 0, reason="Switching users requires a root host process")
def test_a_root_host_runs_commands_as_the_requested_user(factory, spec):
    # Other users cannot enter pytest's private temporary directory, which holds the workdir.
    assert run_command(factory, spec, Command(("id", "-u"), cwd="/", user="65534")).stdout == b"65534\n"


def test_commands_cannot_gain_privileges(factory, spec):
    if not factory.lockdown.no_new_privs:
        pytest.skip("The kernel does not support no_new_privs")
    status = run_command(factory, spec, Command(("cat", "/proc/self/status"))).stdout.decode()
    assert "NoNewPrivs:\t1" in status.splitlines()


@pytest.mark.skipif(os.geteuid() == 0, reason="RLIMIT_NPROC does not limit root")
def test_a_command_cannot_start_unbounded_tasks(python_factory, spec):
    host_tasks = int(Path("/proc/loadavg").read_text().split()[3].split("/")[1])
    # Past the limit even if the host gains tasks before the command starts.
    attempts = host_tasks + 4 * NPROC_HEADROOM
    result = run_command(python_factory, spec, Command(("python3", "-c", THREAD_BOMB, str(attempts)), timeout=30))
    started = int(result.stdout)
    # The limit leaves the command room to work however many tasks this user already runs.
    assert NPROC_HEADROOM // 2 < started < attempts


def test_landlock_confines_file_access_to_the_machine(python_factory, spec, roots, outside_dir):
    if not python_factory.lockdown.filesystem:
        pytest.skip("The kernel or a seccomp filter refuses Landlock")
    (outside_dir / "secret.txt").write_text("host")
    outcomes = access_outcomes(
        python_factory,
        spec,
        ("read", outside_dir / "secret.txt"),
        ("write", outside_dir / "created.txt"),
        ("read", f"/proc/{os.getpid()}/environ"),
        ("write", roots[0] / "created.txt"),
        ("read", roots[0] / "created.txt"),
    )
    assert outcomes == ["EACCES", "EACCES", "EACCES", "ok", "ok"]
    assert not (outside_dir / "created.txt").exists()


def test_commands_cannot_signal_processes_outside_the_command(python_factory, spec):
    if not python_factory.lockdown.scopes:
        pytest.skip("The kernel's Landlock cannot scope signals")
    assert access_outcomes(python_factory, spec, ("signal", os.getpid())) == ["EPERM"]


@pytest.mark.parametrize(("network", "outcome"), [(NetworkPolicy.DENY, "EACCES"), (NetworkPolicy.ALLOW, "ok")])
def test_deny_network_policy_refuses_tcp_connections(python_factory, roots, network, outcome):
    if not python_factory.lockdown.tcp:
        pytest.skip("The kernel's Landlock cannot restrict TCP")
    spec = MachineSpec(HostImage(), workdir=str(roots[1]), network=network)
    with socket.create_server(("127.0.0.1", 0)) as server:
        assert access_outcomes(python_factory, spec, ("connect", server.getsockname()[1])) == [outcome]
