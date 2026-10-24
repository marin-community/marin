# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local machines: bubblewrap sandboxes over a private root, with the host's system directories read-only."""

import asyncio
import contextlib
import errno
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
from shellbox.backends.local.launch import COMMAND_NICENESS
from shellbox.backends.local.machine import (
    NPROC_HEADROOM,
    LocalMachine,
    LocalMachineFactory,
    SandboxUnavailable,
    SystemDirectories,
    _archive,
    _submounts,
)
from shellbox.backends.local.python_environment import (
    PythonEnvironment,
    UvError,
    build_python_environment,
    resolve_python_environment,
)
from shellbox.backends.local.transfers import unpack_local_archive
from shellbox.http_service import HttpRequest, HttpResponse, serve_http
from shellbox.machine import (
    Command,
    DockerImage,
    ExitReason,
    HostImage,
    MachineSpec,
    MachineTerminated,
    Mount,
    NetworkPolicy,
    PackageSpec,
    Result,
    UnsupportedMachineSpec,
)
from shellbox.service import ReadinessProbe, ServiceEndpoint, ServiceGroupSpec, ServiceSpec
from shellbox.transfer import download_files

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


def test_local_factory_reports_unsupported_host(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(SandboxUnavailable):
        LocalMachineFactory()


def test_recursive_download_preserves_hardlinks_as_files_and_literal_names(factory, tmp_path):
    async def check():
        machine = await factory.create(SPEC)
        try:
            source = tmp_path / "source"
            source.write_bytes(b"answer\x00")
            name = "/app/submission/nested/colon:\nλ\\file"
            await machine.upload(source, name)
            linked = await machine.run(Command(("ln", name, "/app/submission/alias")))
            assert linked.exit_code == 0
            symlink = await machine.run(Command(("ln", "-s", "/etc/passwd", "/app/submission/private")))
            assert symlink.exit_code == 0
            files = await download_files(
                machine,
                ("/missing", "/app/submission", name),
                timeout=10,
                max_files=2,
                max_bytes=1024,
                max_file_bytes=1024,
            )
            assert files == {name: b"answer\x00", "/app/submission/alias": b"answer\x00"}
            with pytest.raises(RuntimeError, match="links"):
                await download_files(
                    machine,
                    ("/app/submission/private",),
                    timeout=10,
                    max_files=2,
                    max_bytes=1024,
                    max_file_bytes=1024,
                )
        finally:
            await machine.close()

    asyncio.run(check())


@pytest.fixture(scope="session")
def python_environment(tmp_path_factory) -> PythonEnvironment:
    lock = tmp_path_factory.mktemp("lock") / "requirements.lock"
    lock.write_text(LOCK)
    try:
        return build_python_environment(
            Path.home() / ".cache/shellbox/python/test-locked-six" / PYTHON_VERSION, lock, PYTHON_VERSION
        )
    except UvError as error:
        # uv reports an unreachable or failed fetch this way; any other failure is a bug in the builder.
        if not str(error).startswith("uv python install") or "Failed to download" not in str(error):
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


@pytest.mark.parametrize("system_directories", list(SystemDirectories))
def test_host_configuration_is_hidden_from_commands_and_downloads(tmp_path, system_directories):
    factory = make_factory(system_directories=system_directories)
    # These host paths must not become visible even when commands run as the host's root user.
    hidden = ("/etc/profile", "/etc/environment", "/etc/shadow", "/etc/gshadow", "/etc/ssl/private", "/sys/kernel")

    async def scenario(machine: LocalMachine):
        result = await machine.run(Command(("sh", "-c", 'for path do test ! -e "$path" || exit 1; done', "sh", *hidden)))
        assert result.exit_code == 0, result.stderr
        for path in hidden:
            with pytest.raises(RuntimeError, match="No such file"):
                await machine.download(path, tmp_path / "download")
        configuration = await machine.run(Command(("cat", "/etc/resolv.conf")))
        assert (configuration.exit_code, configuration.stdout) == (0, Path("/etc/resolv.conf").read_bytes())
        await machine.download("/etc", tmp_path / "etc")

    on_machine(factory, SPEC, scenario)
    assert not (tmp_path / "etc/profile").exists()
    assert (tmp_path / "etc/passwd").read_bytes() == Path("/etc/passwd").read_bytes()


def test_host_image_adds_read_only_mount_and_program_path(tmp_path, factory):
    bin_dir = tmp_path / "venv-bin"
    bin_dir.mkdir()
    command = bin_dir / "grader-python"
    command.write_text("#!/bin/sh\necho mounted-runtime\n")
    command.chmod(0o755)
    spec = MachineSpec(HostImage(read_only=(bin_dir,), bin_dirs=(bin_dir,)), workdir="/app")

    result = run_command(factory, Command(("grader-python",)), spec)

    assert (result.exit_code, result.stdout) == (0, b"mounted-runtime\n")


def test_host_image_mounts_a_directory_read_only_at_another_guest_path(tmp_path, factory):
    data = tmp_path / "english"
    data.mkdir()
    (data / "abbrev.tab").write_text("etc\n")
    guest = PurePosixPath("/srv/fixture/data")
    spec = MachineSpec(HostImage(mounts=(Mount(data, guest),)), workdir="/app")
    script = f"cat {guest}/abbrev.tab; echo changed > {guest}/abbrev.tab 2>/dev/null || echo refused"

    result = run_command(factory, Command(("sh", "-c", script)), spec)

    assert (result.exit_code, result.stdout) == (0, b"etc\nrefused\n"), result.stderr
    assert (data / "abbrev.tab").read_text() == "etc\n"


def test_a_mount_inside_a_host_system_directory_is_refused(tmp_path, factory):
    spec = MachineSpec(HostImage(mounts=(Mount(tmp_path, PurePosixPath("/usr/local/share/fixture")),)), workdir="/app")
    with pytest.raises(UnsupportedMachineSpec, match="system directory"):
        asyncio.run(factory.create(spec))


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


def test_losing_the_host_launcher_reclaims_its_still_running_guest(factory):
    async def scenario(machine):
        # The guest keeper still holds stderr open after this host-side fault.
        signal.pidfd_send_signal(cast(int, machine._keeper_pidfd), signal.SIGKILL)
        with pytest.raises(MachineTerminated):
            await asyncio.wait_for(machine.run(Command(("sleep", "3600"))), 5)
        await machine.close()
        assert not machine.root.exists()

    on_machine(factory, SPEC, scenario)


def test_cancelled_creation_reclaims_a_process_before_its_handle_is_returned(factory, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    create_subprocess = asyncio.create_subprocess_exec

    async def scenario():
        spawned, release = asyncio.Event(), asyncio.Event()
        processes = []

        async def hold_real_process(*args, **kwargs):
            process = await create_subprocess(*args, **kwargs)
            processes.append(process)
            spawned.set()
            await release.wait()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", hold_real_process)
        creation = asyncio.create_task(factory.create(SPEC))
        try:
            await asyncio.wait_for(spawned.wait(), 5)
            roots = tuple(tmp_path.iterdir())
            assert roots
            creation.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(creation, 10)
            assert all(process.returncode is not None for process in processes)
            assert not any(root.exists() for root in roots)
        finally:
            release.set()
            if not creation.done():
                creation.cancel()
            await asyncio.gather(creation, return_exceptions=True)

    asyncio.run(scenario())


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


@pytest.fixture
def copy_on_write_factory() -> LocalMachineFactory:
    return make_factory(system_directories=SystemDirectories.COPY_ON_WRITE)


# Copy-on-write keeps file permissions, so only a root command may write the root-owned system directories.
root_commands = pytest.mark.skipif(os.geteuid() != 0, reason="Writing root-owned system directories needs root")


@root_commands
def test_copy_on_write_system_directories_keep_each_machines_writes_to_itself(copy_on_write_factory):
    name = f"shellbox-test-{uuid.uuid4().hex}"
    paths = (f"/opt/{name}/file", f"/usr/local/bin/{name}", f"/etc/{name}")
    write = f"mkdir /opt/{name} && " + " && ".join(f'echo "$1" > {path}' for path in paths)
    read = "cat " + " ".join(paths)

    async def scenario() -> tuple[list[bytes], list[Path]]:
        machines = await asyncio.gather(copy_on_write_factory.create(SPEC), copy_on_write_factory.create(SPEC))
        try:
            for machine, label in zip(machines, "ab", strict=True):
                written = await machine.run(Command(("sh", "-c", write, "sh", label)))
                assert written.exit_code == 0, written.stderr
            results = await asyncio.gather(*(machine.run(Command(("sh", "-c", read))) for machine in machines))
            return [result.stdout for result in results], [machine.layers for machine in machines]
        finally:
            await asyncio.gather(*(machine.close() for machine in machines))

    outputs, layers = asyncio.run(scenario())
    assert outputs == [b"a\n" * len(paths), b"b\n" * len(paths)]
    assert not any(Path(path).exists() for path in (f"/opt/{name}", *paths[1:]))
    assert not any(layer.exists() for layer in layers)


@root_commands
def test_copy_on_write_transfers_see_the_machines_own_system_directories(tmp_path, copy_on_write_factory):
    name = f"shellbox-test-{uuid.uuid4().hex}"
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/data.txt").write_text("from directory")
    single = tmp_path / "single.txt"
    single.write_text("from file")
    downloaded = tmp_path / "downloaded"

    async def scenario(machine: LocalMachine) -> Result:
        await machine.upload(source, f"/opt/{name}/tree")
        await machine.upload(single, f"/usr/local/share/{name}/single.txt")
        read = await machine.run(
            Command(("cat", "tree/nested/data.txt", f"/usr/local/share/{name}/single.txt"), cwd=f"/opt/{name}")
        )
        await machine.run(Command(("sh", "-c", f"echo changed > /opt/{name}/tree/nested/data.txt")))
        await machine.download(f"/opt/{name}/tree", downloaded / "tree")
        await machine.download(f"/usr/local/share/{name}/single.txt", downloaded / "single.txt")
        await machine.download("/etc/passwd", downloaded / "passwd")
        with pytest.raises(RuntimeError, match="No such file"):
            await machine.download(f"/opt/{name}/missing", downloaded / "missing")
        with pytest.raises(UnsupportedMachineSpec):
            await machine.upload(single, f"/sys/{name}")
        # A link to a path that exists only on the host does not resolve in the sandbox.
        await machine.run(Command(("ln", "-s", str(tmp_path), f"/usr/local/share/{name}/host")))
        with pytest.raises(RuntimeError, match="No such file"):
            await machine.download(f"/usr/local/share/{name}/host", downloaded / "host")
        return read

    read = on_machine(copy_on_write_factory, SPEC, scenario)
    assert (read.exit_code, read.stdout) == (0, b"from directoryfrom file"), read.stderr
    assert (downloaded / "tree/nested/data.txt").read_text() == "changed\n"
    assert (downloaded / "single.txt").read_text() == "from file"
    assert (downloaded / "passwd").read_bytes() == Path("/etc/passwd").read_bytes()
    assert not Path(f"/opt/{name}").exists() and not Path(f"/usr/local/share/{name}").exists()


@root_commands
def test_copy_on_write_uploads_to_the_root_reach_the_layer_whatever_the_machines_environment(
    tmp_path, copy_on_write_factory
):
    name = f"shellbox-test-{uuid.uuid4().hex}"
    tree = tmp_path / "tree"
    (tree / "opt" / name).mkdir(parents=True)
    (tree / "opt" / name / "data.txt").write_text("in the layer")
    (tree / "app").mkdir()
    (tree / "app" / "data.txt").write_text("in the root")
    # The backend's own transfer commands ignore the PATH and tar options meant for the task's commands.
    spec = MachineSpec(HostImage(), workdir="/app", env={"PATH": "/nonexistent", "TAR_OPTIONS": "--no-such-option"})

    async def scenario(machine: LocalMachine) -> Result:
        await machine.upload(tree, "/")
        with pytest.raises(UnsupportedMachineSpec, match="contains its system directories"):
            await machine.download("/", tmp_path / "everything")
        return await machine.run(Command(("/bin/cat", f"/opt/{name}/data.txt", "/app/data.txt")))

    read = on_machine(copy_on_write_factory, spec, scenario)
    assert (read.exit_code, read.stdout) == (0, b"in the layerin the root"), read.stderr
    assert not Path(f"/opt/{name}").exists()


def _stream(*members: tuple[tarfile.TarInfo, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for member, data in members:
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return stream.getvalue()


def _member(name: str, kind: bytes = tarfile.REGTYPE, linkname: str = "") -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type, member.linkname, member.mode = kind, linkname, 0o755 if kind == tarfile.DIRTYPE else 0o644
    return member


def test_a_downloaded_archive_keeps_inner_symlinks_as_links_and_refuses_host_reads(tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("host only")
    tree = _stream(
        (_member(".", tarfile.DIRTYPE), b""),
        (_member("data.txt"), b"data"),
        (_member("link", tarfile.SYMTYPE, str(secret)), b""),
    )
    unpack_local_archive(tree, tmp_path / "tree")
    assert (tmp_path / "tree/data.txt").read_text() == "data"
    assert os.readlink(tmp_path / "tree/link") == str(secret)

    refused = [
        _stream((_member("secret", tarfile.SYMTYPE, str(secret)), b"")),
        _stream((_member(".", tarfile.DIRTYPE), b""), (_member("copy", tarfile.LNKTYPE, str(secret)), b"")),
        _stream((_member(".", tarfile.DIRTYPE), b""), (_member("disk", tarfile.BLKTYPE), b"")),
        _stream((_member(str(secret)), b"guest")),
        _stream(),
    ]
    for stream in refused:
        with pytest.raises((ValueError, tarfile.FilterError)):
            unpack_local_archive(stream, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_an_uploaded_archive_holds_what_a_symlinked_source_leads_to(tmp_path):
    (tmp_path / "directory").mkdir()
    (tmp_path / "directory/data.txt").write_text("data")
    (tmp_path / "directory/inner").symlink_to("data.txt")
    (tmp_path / "linked-directory").symlink_to(tmp_path / "directory")
    (tmp_path / "linked-file").symlink_to(tmp_path / "directory/data.txt")

    with tarfile.open(fileobj=io.BytesIO(_archive(tmp_path / "linked-directory", "."))) as archive:
        members = {member.name: member for member in archive.getmembers()}
    assert members["."].isdir() and members["./data.txt"].isfile()
    assert members["./inner"].issym() and members["./inner"].linkname == "data.txt"
    with tarfile.open(fileobj=io.BytesIO(_archive(tmp_path / "linked-file", "copy.txt"))) as archive:
        assert [(member.name, member.isfile()) for member in archive.getmembers()] == [("copy.txt", True)]


@root_commands
def test_a_copy_on_write_machine_may_work_in_a_system_directory(copy_on_write_factory):
    workdir = f"/opt/shellbox-test-{uuid.uuid4().hex}/work"
    spec = MachineSpec(HostImage(), workdir=workdir)
    result = run_command(copy_on_write_factory, Command(("sh", "-c", "pwd && touch made && ls")), spec)
    assert (result.exit_code, result.stdout) == (0, f"{workdir}\nmade\n".encode()), result.stderr
    assert not Path(workdir).parent.exists()


def test_files_mounted_inside_overlaid_directories_are_found_parents_first():
    # A container runtime mounts files such as /etc/resolv.conf over the image; an overlay of /etc hides them.
    mountinfo = "\n".join(
        [
            "21 1 0:31 / / rw,relatime - overlay overlay rw",
            "22 21 9:127 /pods/resolv.conf /etc/resolv.conf rw - xfs /dev/md127 rw",
            "23 21 9:127 /pods/host /etc/host\\040name rw - xfs /dev/md127 rw",
            "24 21 0:40 / /usr/lib/modules ro - tmpfs tmpfs rw",
            "25 21 0:41 / /usr ro - tmpfs tmpfs rw",
            "26 21 0:42 / /usrlocal/data rw - tmpfs tmpfs rw",
            "27 21 9:127 /pods/app /app rw - xfs /dev/md127 rw",
        ]
    )
    assert _submounts(mountinfo, ("/etc", "/usr")) == ("/etc/host name", "/etc/resolv.conf", "/usr/lib/modules")


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
            with pytest.raises(RuntimeError):
                await machine.upload(source, "/app/escape/planted.txt")
            with pytest.raises(RuntimeError):
                await machine.download("/app/escape/secret.txt", tmp_path / "stolen.txt")
        finally:
            await machine.close()

    asyncio.run(scenario())
    assert not (outside_dir / "planted.txt").exists() and not (tmp_path / "stolen.txt").exists()


def test_a_python_environment_runs_its_own_interpreter_and_locked_packages(
    python_environment, python_factory, monkeypatch
):
    script = "import sys, six; print(sys.prefix); print(sys.base_prefix)"
    result = run_command(python_factory, Command(("python3", "-c", script)))
    assert result.exit_code == 0, result.stderr.decode()
    prefix, base_prefix = map(Path, result.stdout.decode().split())
    assert prefix == python_environment.bin_dir.parent
    assert base_prefix.is_relative_to(python_environment.root)

    packages = PackageSpec(pypi_packages=("six==1.17.0",), tools=("python3",))
    factory = make_factory(python_version=PYTHON_VERSION, hash_seed="23")
    spec = MachineSpec(packages, workdir="/app")

    async def check():
        first = await factory.create(spec)
        try:
            result = await first.run(
                Command(
                    (
                        "python3",
                        "-c",
                        "import six,sys,os; print(sys.prefix); print(sys.base_prefix); print(six.text_type(42)); "
                        "print(os.environ['PYTHONHASHSEED'])",
                    )
                )
            )
            assert result.exit_code == 0, result.stderr.decode()
            prefix, base_prefix, answer, seed = result.stdout.decode().splitlines()
            assert prefix != base_prefix
            assert Path(base_prefix).is_relative_to(factory.package_cache)
            assert (answer, seed) == ("42", "23")
            assert (await first.run(Command(("touch", "private")))).exit_code == 0
        finally:
            await first.close()
        # A fresh factory must reuse the resolved installation without index access.
        monkeypatch.setenv("UV_OFFLINE", "1")
        second = await make_factory(python_version=PYTHON_VERSION).create(spec)
        try:
            result = await second.run(Command(("python3", "-c", "import six,sys; print(sys.prefix)")))
            assert (result.exit_code, result.stdout.decode().strip()) == (0, prefix)
            assert (await second.run(Command(("test", "-e", "private")))).exit_code == 1
        finally:
            await second.close()

    asyncio.run(check())


def test_package_resolution_cannot_run_build_hooks_on_the_host(tmp_path, monkeypatch):
    marker = tmp_path / "host-hook"
    files = {
        "pyproject.toml": b'[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["."]\n',
        "backend.py": (
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\nraise RuntimeError('hook')\n".encode()
        ),
    }
    source = tmp_path / "unsafe-1.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        for name, body in files.items():
            entry = tarfile.TarInfo("unsafe-1/" + name)
            entry.size = len(body)
            archive.addfile(entry, io.BytesIO(body))
    monkeypatch.setenv("UV_OFFLINE", "1")
    with pytest.raises(UvError):
        resolve_python_environment(tmp_path / "cache", (f"unsafe @ {source.as_uri()}",), PYTHON_VERSION)
    assert not marker.exists()


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


HTTP_CLIENT = """import sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
    sys.stdout.buffer.write(response.read())
"""
SERVICE_SERVER = """import http.server, os, pathlib, sys
pathlib.Path('/app/member-secret').write_text(sys.argv[1])
pathlib.Path('/app/service-pid').write_text(str(os.getpid()))
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = pathlib.Path('/app/member-secret').read_bytes()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
http.server.ThreadingHTTPServer(('127.0.0.1', 8123), Handler).serve_forever()
"""
CONNECT_PROBE = """import socket, sys
try:
    socket.create_connection((sys.argv[2], int(sys.argv[1])), timeout=1).close()
except OSError:
    print('blocked')
else:
    print('connected')
"""


def private_group_spec(marker="service", readiness=None):
    return ServiceGroupSpec(
        task=MachineSpec(HostImage(), workdir="/app", network=NetworkPolicy.DENY),
        services=(
            ServiceSpec(
                name="runtime",
                machine=MachineSpec(HostImage(), workdir="/app", network=NetworkPolicy.DENY),
                endpoints=(ServiceEndpoint("http", 8123),),
                entrypoint=("python3", "-c", SERVICE_SERVER),
                arguments=(marker,),
                readiness=readiness
                or ReadinessProbe(
                    command=Command(
                        ("python3", "-c", "import socket; socket.create_connection(('127.0.0.1', 8123)).close()")
                    ),
                    timeout=10,
                    interval=0.05,
                ),
            ),
        ),
    )


def test_persistent_machine_hides_host_processes_and_drops_all_privilege(factory):
    script = """import json, os, pathlib
status = dict(line.split(':', 1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)
print(json.dumps({key: status[key].strip() for key in ('NoNewPrivs', 'CapEff', 'CapPrm', 'CapInh', 'CapAmb', 'CapBnd')}))
print(pathlib.Path('/proc/' + os.environ['HOST_PROCESS']).exists())
links = []
for descriptor in pathlib.Path('/proc/self/fd').iterdir():
    try:
        links.append(os.readlink(descriptor))
    except FileNotFoundError:
        pass
print(json.dumps(links))
"""
    result = run_command(factory, Command(("python3", "-c", script), env={"HOST_PROCESS": str(os.getpid())}))
    assert result.exit_code == 0, result.stderr
    status, sees_host, descriptors = result.stdout.decode().splitlines()
    assert json.loads(status) == {
        "NoNewPrivs": "1",
        **{key: "0000000000000000" for key in ("CapEff", "CapPrm", "CapInh", "CapAmb", "CapBnd")},
    }
    assert sees_host == "False"
    assert not any(
        target.startswith(("user:[", "mnt:[", "pid:[", "net:[")) or "shellbox-config" in target
        for target in json.loads(descriptors)
    )


def test_real_local_http_relay_serves_concurrent_requests(factory):
    async def scenario(machine):
        blocked, release = asyncio.Event(), asyncio.Event()

        async def handler(request: HttpRequest):
            if request.path == "/slow":
                blocked.set()
                await release.wait()
            return HttpResponse(200, request.path.encode(), "text/plain")

        async with serve_http(machine, handler, port=8787) as endpoint:
            slow = asyncio.create_task(
                machine.run(Command(("python3", "-c", HTTP_CLIENT, endpoint + "/slow"), timeout=15))
            )
            try:
                await asyncio.wait_for(blocked.wait(), 10)
                fast = await machine.run(Command(("python3", "-c", HTTP_CLIENT, endpoint + "/fast"), timeout=10))
                assert (fast.exit_code, fast.stdout) == (0, b"/fast")
                release.set()
                result = await slow
                assert (result.exit_code, result.stdout) == (0, b"/slow")
            finally:
                release.set()
                if not slow.done():
                    slow.cancel()
                    await asyncio.gather(slow, return_exceptions=True)

    on_machine(factory, MachineSpec(HostImage(), workdir="/app", network=NetworkPolicy.DENY), scenario)


@pytest.mark.parametrize("family,host", [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")], ids=["ipv4", "ipv6"])
def test_local_group_peers_have_private_filesystems_and_no_host_network(factory, family, host):
    async def scenario(host_port):
        groups = await asyncio.gather(
            factory.create_group(private_group_spec("first")),
            factory.create_group(private_group_spec("second")),
        )
        roots = [machine.root for group in groups for machine in (group.task, *group.services.values())]
        try:
            await asyncio.gather(*(group.wait_ready() for group in groups))
            for group, expected in zip(groups, (b"first", b"second"), strict=True):
                written = await group.task.run(Command(("sh", "-c", "echo actor >/app/member-secret")))
                assert written.exit_code == 0
                own = await group.task.run(Command(("cat", "/app/member-secret")))
                assert own.stdout == b"actor\n"
                answer = await group.task.run(
                    Command(("python3", "-c", HTTP_CLIENT, "http://" + group.endpoint("runtime", "http")))
                )
                assert (answer.exit_code, answer.stdout) == (0, expected)
                connection = await group.task.run(Command(("python3", "-c", CONNECT_PROBE, str(host_port), host)))
                assert connection.stdout == b"blocked\n"
        finally:
            await asyncio.gather(*(group.close() for group in groups))
        assert not any(root.exists() for root in roots)

    with contextlib.ExitStack() as cleanup:
        try:
            listener = cleanup.enter_context(socket.socket(family))
            listener.bind((host, 0))
        except OSError as error:
            if family == socket.AF_INET6 and error.errno in (
                errno.EAFNOSUPPORT,
                errno.EPROTONOSUPPORT,
                errno.EADDRNOTAVAIL,
            ):
                pytest.skip("Host IPv6 loopback is unavailable")
            raise
        listener.listen()
        asyncio.run(scenario(listener.getsockname()[1]))


def test_local_group_retries_a_timed_readiness_probe_without_destroying_the_service(factory):
    probe = """import pathlib, signal, socket
marker = pathlib.Path('/app/probe-attempted')
if not marker.exists():
    marker.touch()
    signal.pause()
socket.create_connection(('127.0.0.1', 8123), timeout=1).close()
"""

    async def scenario():
        spec = private_group_spec(
            readiness=ReadinessProbe(
                command=Command(("python3", "-c", probe), timeout=0.15),
                timeout=10,
                interval=0.05,
                attempts=5,
            )
        )
        group = await factory.create_group(spec)
        try:
            await group.wait_ready()
            result = await group.services["runtime"].run(Command(("test", "-f", "/app/probe-attempted")))
            assert result.exit_code == 0
            response = await group.task.run(Command(("python3", "-c", HTTP_CLIENT, "http://runtime:8123")))
            assert response.stdout == b"service"
        finally:
            await group.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
def test_local_group_command_deadline_closes_every_member(factory, cancel):
    async def scenario():
        group = await factory.create_group(private_group_spec())
        roots = [group.task.root, group.services["runtime"].root]
        try:
            await group.wait_ready()
            command = Command(
                ("sh", "-c", "touch /app/command-started; exec sleep 3600"), timeout=None if cancel else 0.2
            )
            pending = asyncio.create_task(group.task.run(command))
            if cancel:
                started = await group.task.run(
                    Command(("sh", "-c", "until test -f /app/command-started; do sleep 0.01; done"), timeout=5)
                )
                assert started.exit_code == 0
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                result = await pending
                assert result.reason is ExitReason.TIMED_OUT
            for machine in (group.task, *group.services.values()):
                with pytest.raises(MachineTerminated):
                    await machine.run(Command(("true",)))
        finally:
            await group.close()
        assert not any(root.exists() for root in roots)

    asyncio.run(scenario())


def test_cancelled_command_start_closes_the_machine(factory, monkeypatch):
    async def scenario():
        machine = await factory.create(SPEC)
        connection_made = asyncio.subprocess.SubprocessStreamProtocol.connection_made

        def cancel_before_handle_returns(protocol, transport):
            connection_made(protocol, transport)
            monkeypatch.setattr(asyncio.subprocess.SubprocessStreamProtocol, "connection_made", connection_made)
            # asyncio calls this after spawning the process but before returning
            # its handle. Cancel without adding an artificial await to startup.
            pending.cancel()

        try:
            monkeypatch.setattr(
                asyncio.subprocess.SubprocessStreamProtocol, "connection_made", cancel_before_handle_returns
            )
            pending = asyncio.create_task(machine.run(Command(("sleep", "3600"))))
            with pytest.raises(asyncio.CancelledError):
                await pending
            with pytest.raises(MachineTerminated):
                await machine.run(Command(("true",)))
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("service_exit", [False, True], ids=["ttl", "service-exit"])
def test_local_group_failure_and_ttl_reclaim_all_members(service_exit):
    factory = make_factory(ttl=None if service_exit else 3)

    async def scenario():
        group = await factory.create_group(private_group_spec())
        roots = [group.task.root, group.services["runtime"].root]
        try:
            await group.wait_ready()
            waiting = asyncio.create_task(group.task.run(Command(("sleep", "3600"))))
            if service_exit:
                with contextlib.suppress(MachineTerminated):
                    await group.services["runtime"].run(Command(("sh", "-c", "kill -TERM $(cat /app/service-pid)")))
            with pytest.raises(MachineTerminated):
                await asyncio.wait_for(waiting, 8)
            for machine in (group.task, *group.services.values()):
                with pytest.raises(MachineTerminated):
                    await machine.run(Command(("true",)))
        finally:
            await group.close()
        assert not any(root.exists() for root in roots)

    asyncio.run(scenario())


def test_absolute_guest_symlinks_resolve_inside_the_private_root(factory, tmp_path):
    source = tmp_path / "payload"
    source.write_text("updated")
    target = tmp_path / "downloaded"

    async def scenario(machine):
        prepared = await machine.run(
            Command(("sh", "-c", "mkdir -p /app/actual && echo guest >/app/actual/file && ln -s /app/actual /app/link"))
        )
        assert prepared.exit_code == 0
        await machine.upload(source, "/app/link/file")
        await machine.download("/app/link/file", target)

    on_machine(factory, SPEC, scenario)
    assert target.read_text() == "updated"


def test_repeated_download_cannot_follow_an_artifact_link_into_the_host(factory, tmp_path, outside_dir):
    host_marker = outside_dir / "marker"
    host_marker.write_text("host-private")
    downloaded = tmp_path / "artifacts"

    async def scenario(machine):
        prepared = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    'mkdir /app/artifacts && ln -s "$1" /app/artifacts/nested '
                    '&& ln -s "$1/marker" /app/artifacts/linked-file',
                    "sh",
                    str(outside_dir),
                )
            )
        )
        assert prepared.exit_code == 0
        await machine.download("/app/artifacts", downloaded)
        (downloaded / "unrelated").write_text("retained")
        replaced = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    "rm /app/artifacts/nested /app/artifacts/linked-file && mkdir /app/artifacts/nested "
                    "&& echo changed >/app/artifacts/nested/marker && echo replaced >/app/artifacts/linked-file",
                )
            )
        )
        assert replaced.exit_code == 0
        await machine.download("/app/artifacts", downloaded)
        assert (downloaded / "nested/marker").read_text() == "changed\n"
        assert (downloaded / "linked-file").read_text() == "replaced\n"
        assert (downloaded / "unrelated").read_text() == "retained"
        target = tmp_path / "file-link"
        target.symlink_to(host_marker)
        with pytest.raises(ValueError):
            await machine.download("/app/artifacts/nested/marker", target)
        target.unlink()
        target.symlink_to(outside_dir, target_is_directory=True)
        with pytest.raises(OSError):
            await machine.download("/app/artifacts/nested/marker", target / "marker")
        target.unlink()
        target = outside_dir / "file-hardlink"
        target.hardlink_to(host_marker)
        await machine.download("/app/artifacts/nested/marker", target)
        assert target.read_text() == "changed\n"

    on_machine(factory, SPEC, scenario)
    assert host_marker.read_text() == "host-private"


@pytest.mark.parametrize("first_mode,second_mode,new_mode", [(0o500, 0o777, 0o700), (0o777, 0o500, 0o755)])
def test_repeated_download_preserves_existing_host_directory_permissions(
    factory, tmp_path, first_mode, second_mode, new_mode
):
    downloaded = tmp_path / "artifacts"
    existing = downloaded / "existing"
    existing.mkdir(parents=True)
    downloaded.chmod(0o750)
    existing.chmod(0o710)
    (downloaded / "unrelated").write_text("host-owned")

    async def scenario(machine):
        first = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    "mkdir -p /app/artifacts/existing /app/artifacts/new; "
                    "printf first >/app/artifacts/existing/state; printf first >/app/artifacts/new/state; "
                    f"chmod {first_mode:o} /app/artifacts /app/artifacts/existing /app/artifacts/new",
                )
            )
        )
        assert first.exit_code == 0, first.stderr
        await machine.download("/app/artifacts", downloaded)
        assert downloaded.stat().st_mode & 0o777 == 0o750
        assert existing.stat().st_mode & 0o777 == 0o710
        assert (downloaded / "new").stat().st_mode & 0o777 == new_mode
        # The caller's permissions also survive later guest metadata changes.
        (downloaded / "new").chmod(0o750)
        second = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    "chmod 700 /app/artifacts /app/artifacts/existing /app/artifacts/new; "
                    "printf second >/app/artifacts/existing/state; printf second >/app/artifacts/new/state; "
                    f"chmod {second_mode:o} /app/artifacts /app/artifacts/existing /app/artifacts/new",
                )
            )
        )
        assert second.exit_code == 0, second.stderr
        await machine.download("/app/artifacts", downloaded)

    on_machine(factory, SPEC, scenario)
    assert downloaded.stat().st_mode & 0o777 == 0o750
    assert existing.stat().st_mode & 0o777 == 0o710
    assert (downloaded / "new").stat().st_mode & 0o777 == 0o750
    assert (existing / "state").read_text() == "second"
    assert (downloaded / "new/state").read_text() == "second"
    assert (downloaded / "unrelated").read_text() == "host-owned"


@pytest.fixture
def executable_oci_image(oci_image):
    busybox = shutil.which("busybox")
    if busybox is None:
        pytest.skip("The offline executable image requires busybox-static; the Shellbox CI lane installs it")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as output:
        binary = Path(busybox).read_bytes()
        entry = tarfile.TarInfo("bin/busybox")
        entry.mode, entry.size = 0o755, len(binary)
        output.addfile(entry, io.BytesIO(binary))
        for applet in ("sh", "sleep", "mkdir", "cat", "tar", "httpd", "wget", "dirname", "basename", "id"):
            entry = tarfile.TarInfo(f"bin/{applet}")
            entry.type, entry.linkname = tarfile.SYMTYPE, "busybox"
            output.addfile(entry)
        entry = tarfile.TarInfo("etc")
        entry.type, entry.mode = tarfile.DIRTYPE, 0o755
        output.addfile(entry)
    return oci_image(
        (archive.getvalue(),),
        {
            "Env": ["PATH=/bin", "IMAGE_VALUE=from-image"],
            "WorkingDir": "/app",
            "Entrypoint": ["/bin/sh", "-c"],
            "Cmd": ['printf "%s" "$IMAGE_VALUE" >index.html; exec httpd -f -p 8123 -h .'],
            "User": "0:0",
        },
    )


@pytest.mark.parametrize(
    "entrypoint,arguments,expected",
    [
        (None, None, b"from-image"),
        (
            (
                "/bin/sh",
                "-c",
                'test "$#" -eq 0 || exit 23; printf entrypoint >index.html; exec httpd -f -p 8123 -h .',
                "entrypoint",
            ),
            None,
            b"entrypoint",
        ),
        (None, ("printf arguments >index.html; exec httpd -f -p 8123 -h .",), b"arguments"),
        (
            ("/bin/sh", "-c", 'printf "%s" "$1" >index.html; exec httpd -f -p 8123 -h .', "entrypoint"),
            ("both",),
            b"both",
        ),
    ],
    ids=["image-defaults", "entrypoint-only", "arguments-only", "both-overridden"],
)
def test_prepared_local_group_applies_startup_overrides_with_private_writable_roots(
    executable_oci_image, tmp_path, entrypoint, arguments, expected
):
    umoci = shutil.which("umoci")
    if umoci is None:
        pytest.skip("OCI preparation requires umoci; the Shellbox CI lane installs it")
    factory = make_factory(umoci=Path(umoci))
    image = executable_oci_image
    original = {path.relative_to(image.layout): path.read_bytes() for path in image.layout.rglob("*") if path.is_file()}
    task = MachineSpec(image, workdir="", network=NetworkPolicy.DENY)
    service = ServiceSpec(
        "runtime",
        task,
        endpoints=(ServiceEndpoint("http", 8123),),
        readiness=ReadinessProbe(Command(("wget", "-q", "-O", "-", "http://runtime:8123")), timeout=10, interval=0.05),
        entrypoint=entrypoint,
        arguments=arguments,
    )

    async def scenario():
        group = await factory.create_group(ServiceGroupSpec(task, (service,)))
        roots = [group.task.root, group.services["runtime"].root]
        try:
            await group.wait_ready()
            identity = await group.task.run(Command(("sh", "-c", 'pwd; printf "%s\\n" "$IMAGE_VALUE"; id -u')))
            assert (identity.exit_code, identity.stdout) == (0, b"/app\nfrom-image\n0\n")
            written = await group.task.run(Command(("sh", "-c", "printf actor >index.html")))
            assert written.exit_code == 0
            response = await group.task.run(Command(("wget", "-q", "-O", "-", "http://runtime:8123")))
            assert (response.exit_code, response.stdout) == (0, expected)
            await group.task.download("/app/index.html", tmp_path / "actor")
            await group.services["runtime"].download("/app/index.html", tmp_path / "runtime")
            assert (tmp_path / "actor").read_bytes() == b"actor"
            assert (tmp_path / "runtime").read_bytes() == expected
        finally:
            await group.close()
        assert not any(root.exists() for root in roots)

    asyncio.run(scenario())
    assert {
        path.relative_to(image.layout): path.read_bytes() for path in image.layout.rglob("*") if path.is_file()
    } == original
