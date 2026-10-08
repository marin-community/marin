# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local machines: host subprocesses that own a set of host directories."""

import asyncio
import os
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from shellbox.backends.local.machine import LocalMachine, LocalMachineFactory
from shellbox.machine import Command, DockerImage, ExitReason, HostImage, MachineSpec, Result, UnsupportedMachineSpec


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "tests", tmp_path / "app"


@pytest.fixture
def factory(tmp_path: Path, roots: tuple[Path, Path]) -> LocalMachineFactory:
    return LocalMachineFactory(tuple(map(str, roots)), lock_path=tmp_path / "local.lock")


@pytest.fixture
def spec(roots: tuple[Path, Path]) -> MachineSpec:
    return MachineSpec(HostImage(), workdir=str(roots[1] / "work"))


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
