# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local machines: bubblewrap sandboxes over a private root, with the host's system directories read-only."""

import asyncio
import os
import shutil
import socket
import subprocess
import tarfile
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from shellbox.backends.local.machine import NPROC_HEADROOM, LocalMachine, LocalMachineFactory, SandboxUnavailable
from shellbox.backends.local.python_environment import PythonEnvironment, build_python_environment
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
SPEC = MachineSpec(HostImage(), workdir="/app")
PYTHON_VERSION = "3.12.13"
# A hash lock as ``uv pip compile --generate-hashes`` writes it, for a small pure-Python package.
LOCK = """\
six==1.17.0 \\
    --hash=sha256:4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274 \\
    --hash=sha256:ff70335d468e7eb6ec65b95b99d3a2836546063f63acc5171de367e834932a81
"""


def make_factory(**options) -> LocalMachineFactory:
    try:
        return LocalMachineFactory(**options)
    except SandboxUnavailable as error:
        pytest.skip(str(error))


@pytest.fixture
def factory() -> LocalMachineFactory:
    return make_factory()


@pytest.fixture(scope="session")
def python_environment(tmp_path_factory) -> PythonEnvironment:
    lock = tmp_path_factory.mktemp("lock") / "requirements.lock"
    lock.write_text(LOCK)
    try:
        return build_python_environment(tmp_path_factory.mktemp("environment") / "root", lock, PYTHON_VERSION)
    except subprocess.CalledProcessError as error:
        if error.cmd[1:3] != ["python", "install"]:
            raise
        pytest.skip(f"uv could not download a managed CPython {PYTHON_VERSION}: {error}")


@pytest.fixture
def python_factory(python_environment) -> LocalMachineFactory:
    """A factory whose commands find python3 in a self-contained environment that it mounts read-only."""
    return make_factory(read_only=(python_environment.root,), bin_dirs=(python_environment.bin_dir,))


@pytest.fixture
def outside_dir() -> Iterator[Path]:
    """A host directory the test user may write, which local commands cannot see."""
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


def run_command(factory: LocalMachineFactory, command: Command, spec: MachineSpec = SPEC) -> Result:
    return on_machine(factory, spec, lambda machine: machine.run(command))


def access_outcomes(factory: LocalMachineFactory, *checks: tuple[str, object], spec: MachineSpec = SPEC) -> list[str]:
    argv = [str(part) for check in checks for part in check]
    return run_command(factory, Command(("python3", "-c", ACCESS_PROBE, *argv)), spec).stdout.decode().split()


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


def test_timeout_kills_the_command_and_its_descendants(factory):
    # run() returns only once the output pipes close, so a surviving grandchild would hold it past the test timeout.
    command = Command(("sh", "-c", "sleep 3600 & wait"), timeout=1, stdin=b"unread")
    assert run_command(factory, command) == Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)


def test_background_processes_end_with_their_command(factory):
    result = run_command(factory, Command(("sh", "-c", "sleep 3600 & echo started"), timeout=30))
    assert (result.exit_code, result.stdout) == (0, b"started\n")


def test_output_beyond_the_limit_is_truncated(factory):
    result = run_command(factory, Command(("sh", "-c", "yes | head -c 1000000; printf ab >&2"), output_limit_bytes=4))
    assert (result.exit_code, result.stdout, result.stdout_truncated) == (0, b"y\ny\n", True)
    assert (result.stderr, result.stderr_truncated) == (b"ab", False)


def test_files_round_trip_between_host_paths_and_commands(tmp_path, factory):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/data.txt").write_text("from directory")
    single = tmp_path / "single.txt"
    single.write_text("from file")
    downloaded = tmp_path / "downloaded"

    async def scenario(machine: LocalMachine) -> Result:
        await machine.upload(single, "/tests/inputs/single.txt")
        await machine.upload(source, "/tmp/tree")
        read = await machine.run(Command(("cat", "inputs/single.txt", "/tmp/tree/nested/data.txt"), cwd="/tests"))
        await machine.run(Command(("sh", "-c", "mkdir -p /logs/verifier && echo 0.5 > /logs/verifier/reward.txt")))
        await machine.download("/logs/verifier/reward.txt", downloaded / "reward.txt")
        await machine.download("/etc/passwd", downloaded / "passwd")
        with pytest.raises(RuntimeError):
            await machine.download("/logs/verifier/missing.txt", downloaded / "missing.txt")
        with pytest.raises(UnsupportedMachineSpec):
            await machine.upload(single, "/usr/local/single.txt")
        return read

    assert on_machine(factory, SPEC, scenario).stdout == b"from filefrom directory"
    assert (downloaded / "reward.txt").read_text() == "0.5\n"
    assert (downloaded / "passwd").read_bytes() == Path("/etc/passwd").read_bytes()


def test_machines_run_side_by_side_on_private_filesystems_that_close_removes(factory):
    # Paths that the host may use itself, such as an Iris task's /app, as well as /tmp, HOME and a new directory.
    marker = f"shellbox-test-{uuid.uuid4().hex}"
    paths = (f"/app/{marker}", f"/tmp/{marker}", f"$HOME/{marker}", f"/srv/{marker}")
    write = "mkdir /srv && " + " && ".join(f'echo "$1" > {path}' for path in paths)
    read = "cat " + " ".join(paths)

    async def scenario() -> tuple[list[bytes], list[Path]]:
        machines = await asyncio.gather(factory.create(SPEC), factory.create(SPEC))
        try:
            for machine, name in zip(machines, "ab", strict=True):
                await machine.run(Command(("sh", "-c", write, "sh", name)))
            results = await asyncio.gather(*(machine.run(Command(("sh", "-c", read))) for machine in machines))
            return [result.stdout for result in results], [machine.root for machine in machines]
        finally:
            await asyncio.gather(*(machine.close() for machine in machines))

    outputs, roots = asyncio.run(scenario())
    assert outputs == [b"a\n" * len(paths), b"b\n" * len(paths)]
    assert not any(Path(path.replace("$HOME", "/home/shellbox")).exists() for path in paths)
    assert not any(root.exists() for root in roots)


def test_an_archive_extracted_at_the_root_stays_in_its_machine(tmp_path, factory):
    # The grading runtime unpacks its inputs with ``tar -xf ... -C /``, which writes into the workspace.
    payload = tmp_path / "payload"
    payload.write_text("7")
    archive = tmp_path / "inputs.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(payload, arcname="app/solution.py")

    async def scenario(machine: LocalMachine) -> Result:
        await machine.upload(archive, "/tmp/inputs.tar")
        return await machine.run(Command(("sh", "-c", "tar -xf /tmp/inputs.tar -C / && cat /app/solution.py")))

    extracted = on_machine(factory, MachineSpec(HostImage(), workdir="/"), scenario)
    assert (extracted.exit_code, extracted.stdout) == (0, b"7"), extracted.stderr
    later = run_command(factory, Command(("ls", "-A", "/app")))
    assert (later.exit_code, later.stdout) == (0, b"")


def test_host_system_directories_are_read_only(factory):
    script = "touch /usr/shellbox-test 2>/dev/null || echo refused; touch /etc/shellbox-test 2>/dev/null || echo refused"
    assert run_command(factory, Command(("sh", "-c", script))).stdout == b"refused\nrefused\n"


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
def test_deny_network_policy_refuses_tcp_connections(python_factory, network, connects):
    spec = MachineSpec(HostImage(), workdir="/app", network=network)
    with socket.create_server(("127.0.0.1", 0)) as server:
        outcomes = access_outcomes(python_factory, ("connect", server.getsockname()[1]), spec=spec)
    assert (outcomes == ["ok"]) == connects


def test_a_transfer_does_not_follow_a_symlink_out_of_the_machine_root(tmp_path, factory, outside_dir):
    (outside_dir / "secret.txt").write_text("host")
    source = tmp_path / "payload"
    source.write_text("x")

    async def scenario() -> None:
        machine = await factory.create(MachineSpec(HostImage(), workdir="/app"))
        try:
            await machine.run(Command(("ln", "-s", str(outside_dir), "/app/escape")))
            with pytest.raises(RuntimeError, match="symlink"):
                await machine.upload(source, "/app/escape/planted.txt")
            with pytest.raises(RuntimeError, match="symlink"):
                await machine.download("/app/escape/secret.txt", tmp_path / "stolen.txt")
        finally:
            await machine.close()

    asyncio.run(scenario())
    assert not (outside_dir / "planted.txt").exists() and not (tmp_path / "stolen.txt").exists()


def test_a_python_environment_runs_its_own_interpreter_and_locked_packages(python_environment, python_factory):
    script = "import sys, six; print(sys.prefix); print(sys.base_prefix)"
    result = run_command(python_factory, Command(("python3", "-c", script)))
    assert result.exit_code == 0, result.stderr.decode()
    prefix, base_prefix = map(Path, result.stdout.decode().split())
    assert prefix == python_environment.bin_dir.parent
    assert base_prefix.is_relative_to(python_environment.root)


def test_every_symlink_in_a_python_environment_resolves_inside_its_root(python_environment):
    root = python_environment.root
    links = [
        Path(directory, name)
        for directory, directories, files in os.walk(root)
        for name in (*directories, *files)
        if os.path.islink(Path(directory, name))
    ]
    assert python_environment.python in links
    assert [link for link in links if not Path(os.path.realpath(link)).is_relative_to(root)] == []
