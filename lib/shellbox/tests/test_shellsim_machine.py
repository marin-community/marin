# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior checks for the simulated shell and verifier boundary."""

import asyncio
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("shellsim")

import shellsim
from shellbox.backends.shellsim.machine import INTERRUPT_GRACE, ShellSimMachineFactory
from shellbox.machine import Command, ExitReason, MachineSpec, PackageSpec, ShellStatus
from shellbox.transfer import download_files

UNSUPPORTED_COMMAND = "unsupported-native-command-with-a-long-name-for-output-truncation-testing"
UNSUPPORTED_OUTPUT_LIMIT_BYTES = 64


class _ControlledAction:
    """Hold the external simulator boundary open until native cleanup is released."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.started = asyncio.Event()
        self.requested = asyncio.Event()
        self.release = threading.Event()
        self._cancelled = threading.Event()
        self.result = shellsim.run(":")
        self.error: Exception | None = None

    def cancel(self):
        self._cancelled.set()
        self.loop.call_soon_threadsafe(self.requested.set)

    @property
    def cancelled(self):
        return self._cancelled.is_set()

    def run(self, _source, _stdin=b"", *, cancellation):
        self.loop.call_soon_threadsafe(self.started.set)
        self.release.wait()
        if self.error is not None:
            raise self.error
        interrupted = cancellation.cancelled
        return replace(self.result, returncode=130 if interrupted else 0, interrupted=interrupted)


@contextmanager
def _controlled_action(monkeypatch, runner):
    action = _ControlledAction()
    with monkeypatch.context() as patch:
        patch.setattr(shellsim, "CancellationToken", lambda: action)
        patch.setattr(runner, "run", action.run)
        try:
            yield action
        finally:
            action.release.set()


@pytest.mark.parametrize("interruption", ["timeout", "cancel", "timeout-error"])
def test_interrupted_command_waits_for_native_cleanup_and_preserves_machine(monkeypatch, tmp_path, caplog, interruption):
    async def check():
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        monkeypatch.setattr(loop, "time", lambda: clock[0])
        source = tmp_path / "after"
        source.write_text("reused")
        try:
            with _controlled_action(monkeypatch, machine.simulation) as action:
                if interruption == "timeout-error":
                    action.error = shellsim.SimulationError("native cleanup failed")
                running = asyncio.create_task(machine.run(Command(("fixture",), timeout=10)))
                await action.started.wait()
                if interruption != "cancel":
                    clock[0] += 10
                else:
                    running.cancel()
                await action.requested.wait()
                queued_upload = asyncio.create_task(machine.upload(source, "/workspace/after"))
                await asyncio.sleep(0)
                assert not running.done()
                assert not queued_upload.done()
                if interruption == "cancel":
                    # A second caller cancellation must not release the serialization lock.
                    running.cancel()
                    await asyncio.sleep(0)
                    assert not queued_upload.done()
                action.release.set()
                if interruption != "cancel":
                    result = await running
                    assert result.reason is ExitReason.TIMED_OUT
                else:
                    with pytest.raises(asyncio.CancelledError):
                        await running
                await queued_upload
                if interruption == "timeout-error":
                    assert any(
                        record.exc_info and record.exc_info[0] is shellsim.SimulationError for record in caplog.records
                    )
            result = await machine.run(Command(("cat", "/workspace/after")))
            assert (result.exit_code, result.stdout) == (0, b"reused")
        finally:
            await machine.close()

    asyncio.run(check())


@pytest.mark.parametrize("interruption", ["timeout", "cancel", "interrupt", "close"])
def test_uncooperative_native_work_closes_the_machine_after_a_bounded_grace(monkeypatch, interruption):
    async def check():
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        shell = await machine.open_shell()
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        monkeypatch.setattr(loop, "time", lambda: clock[0])
        shell_action = interruption in ("interrupt", "close")
        try:
            with _controlled_action(monkeypatch, shell.shell if shell_action else machine.simulation) as action:
                running = asyncio.create_task(
                    shell.execute("fixture") if shell_action else machine.run(Command(("fixture",), timeout=10))
                )
                await action.started.wait()
                if interruption == "timeout":
                    clock[0] += 10
                elif interruption == "cancel":
                    running.cancel()
                else:
                    stopping = asyncio.create_task(shell.interrupt() if interruption == "interrupt" else machine.close())
                await action.requested.wait()
                clock[0] += INTERRUPT_GRACE
                if interruption == "cancel":
                    running.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await running
                elif interruption == "timeout":
                    result = await running
                    assert result.reason is ExitReason.TIMED_OUT
                elif interruption == "interrupt":
                    with pytest.raises(RuntimeError):
                        await stopping
                else:
                    await stopping
                # Reject admission without waiting for the still-running native worker's lock.
                with pytest.raises(RuntimeError):
                    await machine.run(Command(("true",)))
                action.release.set()
                if shell_action:
                    result = await running
                    assert result.exit_code == 130
        finally:
            await machine.close()

    asyncio.run(check())


def test_close_bounds_native_file_transfer_cleanup(monkeypatch, tmp_path):
    async def check():
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        monkeypatch.setattr(loop, "time", lambda: clock[0])
        started = asyncio.Event()
        release = threading.Event()

        original_mount = machine.simulation.mount

        def mount(source, target):
            loop.call_soon_threadsafe(started.set)
            release.wait()
            original_mount(source, target)

        monkeypatch.setattr(machine.simulation, "mount", mount)
        upload = asyncio.create_task(machine.upload(tmp_path, "/workspace/input"))
        try:
            await started.wait()
            closing = asyncio.create_task(machine.close())
            await asyncio.sleep(0)
            clock[0] += INTERRUPT_GRACE
            await closing
            with pytest.raises(RuntimeError):
                await machine.run(Command(("true",)))
        finally:
            release.set()
            await upload
            await machine.close()

    asyncio.run(check())


@pytest.mark.parametrize("interruption", ["interrupt", "close"])
def test_shell_interruption_and_machine_close_drain_active_work(monkeypatch, interruption):
    async def check():
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        shell = await machine.open_shell()
        try:
            with _controlled_action(monkeypatch, shell.shell) as action:
                running = asyncio.create_task(shell.execute("fixture", wait=10, output_limit_bytes=4))
                await action.started.wait()
                if interruption == "close":
                    queued = asyncio.create_task(machine.run(Command(("touch", "/workspace/after-close"))))
                    await asyncio.sleep(0)
                stopping = asyncio.create_task(shell.interrupt() if interruption == "interrupt" else machine.close())
                await action.requested.wait()
                assert not stopping.done()
                action.release.set()
                await stopping
                result = await running
                assert (result.status, result.exit_code) == (ShellStatus.COMPLETED, 130)
            if interruption == "close":
                with pytest.raises(RuntimeError):
                    await queued
            else:
                followup = await shell.execute("printf reusable")
                assert (followup.exit_code, followup.output) == (0, b"reusable")
        finally:
            await machine.close()

    asyncio.run(check())


def test_queued_timeout_and_shell_interrupt_leave_an_unrelated_command_running(monkeypatch):
    async def check():
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        shell = await machine.open_shell()
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        monkeypatch.setattr(loop, "time", lambda: clock[0])
        try:
            with _controlled_action(monkeypatch, machine.simulation) as action:
                running = asyncio.create_task(machine.run(Command(("fixture",))))
                await action.started.wait()
                await shell.interrupt()
                queued = asyncio.create_task(machine.run(Command(("queued",), timeout=1)))
                await asyncio.sleep(0)
                clock[0] += 1
                result = await queued
                assert result.reason is ExitReason.TIMED_OUT
                assert not action.requested.is_set()
                action.release.set()
                result = await running
                assert (result.reason, result.exit_code) == (ExitReason.EXITED, 0)
            result = await machine.run(Command(("printf", "usable")))
            assert (result.exit_code, result.stdout) == (0, b"usable")
        finally:
            await machine.close()

    asyncio.run(check())


def test_command_timeout_reuses_the_real_shellsim_machine(monkeypatch):
    async def check():
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        monkeypatch.setattr(loop, "time", lambda: clock[0])
        started = asyncio.Event()
        native_run = machine.simulation.run

        def run(source, stdin=b"", *, cancellation):
            loop.call_soon_threadsafe(started.set)
            return native_run(source, stdin, cancellation=cancellation)

        monkeypatch.setattr(machine.simulation, "run", run)
        try:
            running = asyncio.create_task(machine.run(Command(("sh", "-c", "while :; do :; done"), timeout=10)))
            await started.wait()
            # Expire the host deadline after the native worker starts, without a host timer.
            clock[0] += 10
            result = await running
            assert result.reason is ExitReason.TIMED_OUT
            followup = await machine.run(Command(("printf", "reused")))
            assert (followup.exit_code, followup.stdout) == (0, b"reused")
        finally:
            await machine.close()

    asyncio.run(check())


@pytest.mark.parametrize(
    "limits",
    [
        {"max_files": 1, "max_bytes": 1024, "max_file_bytes": 1024},
        {"max_files": 8, "max_bytes": 3, "max_file_bytes": 1024},
        {"max_files": 8, "max_bytes": 1024, "max_file_bytes": 1},
    ],
)
def test_download_rejects_file_and_byte_overflow(limits):
    async def check():
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        try:
            result = await machine.run(Command(("sh", "-c", "printf aa > first; printf bb > second")))
            assert result.exit_code == 0
            with pytest.raises(RuntimeError):
                await download_files(machine, ("/workspace",), timeout=10, **limits)
        finally:
            await machine.close()

    asyncio.run(check())


def test_shell_state_and_one_shot_command() -> None:
    async def check() -> None:
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec(), env={"INITIAL": "base"}))
        try:
            shell = await machine.open_shell()
            first = await shell.execute(
                'cd /tmp; export ANSWER=hello; answer() { echo "$ANSWER"; }; echo file >/workspace/result', wait=0
            )
            assert first.status is ShellStatus.COMPLETED
            second = await shell.execute("pwd; answer")
            assert second.output == b"/tmp\nhello\n"

            verifier = await machine.run(Command(("sh", "-c", 'pwd; cat /workspace/result; echo "$INITIAL:$ANSWER"')))
            assert verifier.stdout == b"/workspace\nfile\nbase:\n"
            after = await shell.execute("pwd; answer")
            assert after.output == b"/tmp\nhello\n"
        finally:
            await machine.close()

    asyncio.run(check())


def test_file_transfer_and_bounded_output(tmp_path: Path) -> None:
    async def check() -> None:
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        try:
            source = tmp_path / "source"
            source.mkdir()
            source.chmod(0o700)
            (source / "data.txt").write_text("payload")
            (source / "data.txt").chmod(0o751)
            await machine.upload(source, "/workspace/imported")
            permissions = await machine.run(Command(("stat", "-c", "%a", "/workspace/imported")))
            assert permissions.stdout.strip() == b"700"
            shell = await machine.open_shell()
            result = await shell.execute("cat /workspace/imported/data.txt; printf 'x'", output_limit_bytes=4)
            assert result.output == b"payl"
            assert result.truncated

            target = tmp_path / "downloaded"
            target.mkdir()
            await machine.download("/workspace/imported", target)
            assert (target / "data.txt").read_text() == "payload"
            assert (target / "data.txt").stat().st_mode & 0o777 == 0o751
        finally:
            await machine.close()

    asyncio.run(check())


def test_run_unsupported_command_returns_failed_result(tmp_path: Path) -> None:
    async def check() -> None:
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        try:
            source = tmp_path / "numbers.txt"
            source.write_text("1\n2\n")
            await machine.upload(source, "/workspace/numbers.txt")

            result = await machine.run(
                Command(("sh", "-c", UNSUPPORTED_COMMAND), output_limit_bytes=UNSUPPORTED_OUTPUT_LIMIT_BYTES)
            )
            assert result.reason is ExitReason.EXITED
            assert result.exit_code == 127
            assert result.stdout == b""
            assert len(result.stderr) == UNSUPPORTED_OUTPUT_LIMIT_BYTES
            assert result.stderr_truncated

            followup = await machine.run(Command(("cat", "/workspace/numbers.txt")))
            assert (followup.exit_code, followup.stdout) == (0, b"1\n2\n")
        finally:
            await machine.close()

    asyncio.run(check())


def test_shell_session_unsupported_command_returns_failed_update(tmp_path: Path) -> None:
    async def check() -> None:
        machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
        try:
            source = tmp_path / "numbers.txt"
            source.write_text("1\n2\n")
            await machine.upload(source, "/workspace/numbers.txt")
            shell = await machine.open_shell()

            update = await shell.execute(UNSUPPORTED_COMMAND, output_limit_bytes=UNSUPPORTED_OUTPUT_LIMIT_BYTES)
            assert update.status is ShellStatus.COMPLETED
            assert update.exit_code == 127
            assert len(update.output) == UNSUPPORTED_OUTPUT_LIMIT_BYTES
            assert update.truncated

            followup = await shell.execute("cat /workspace/numbers.txt")
            assert (followup.exit_code, followup.output) == (0, b"1\n2\n")
        finally:
            await machine.close()

    asyncio.run(check())
