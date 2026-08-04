# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded subprocess execution for hardware probes."""

import logging
import math
import os
import selectors
import signal
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from rigging.timing import Deadline

MAX_OUTPUT_BYTES = 256 * 1024
_PROCESS_REAP_TIMEOUT = 0.1
_DEFAULT_INTERVAL = 10 * 60.0
_MAX_SHUTDOWN_TIMEOUT = 5.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandOutput:
    stdout: bytes
    returncode: int


class CommandStatus(StrEnum):
    """Terminal outcomes for a bounded probe command."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    START_FAILED = "start_failed"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    OUTPUT_LIMIT = "output_limit"
    OUTPUT_FAILED = "output_failed"
    INVALID_TIMEOUT = "invalid_timeout"


@dataclass(frozen=True)
class CommandResult:
    """One bounded command result with an explicit non-completion reason."""

    status: CommandStatus
    output: CommandOutput | None = None


class BoundedCommandRunner:
    """Run probe commands with bounded output, deadlines, and cancellation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: set[subprocess.Popen[bytes]] = set()
        self._cancelled = False

    def run_result(self, argv: tuple[str, ...], timeout: float) -> CommandResult:
        """Return completed output or the reason collection did not complete."""
        if not argv:
            raise ValueError("argv must not be empty")
        if not math.isfinite(timeout) or timeout <= 0:
            return CommandResult(CommandStatus.INVALID_TIMEOUT)

        with self._lock:
            if self._cancelled:
                return CommandResult(CommandStatus.CANCELLED)
            try:
                process = subprocess.Popen(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    start_new_session=os.name == "posix",
                )
            except OSError as error:
                logger.debug("could not start telemetry probe %s: %s", argv[0], error)
                return CommandResult(CommandStatus.START_FAILED)
            self._active.add(process)

        assert process.stdout is not None
        deadline = Deadline.from_seconds(timeout)
        output = bytearray()
        try:
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        remaining = deadline.remaining_seconds()
                        if remaining <= 0 or not selector.select(remaining):
                            self._terminate(process)
                            return CommandResult(CommandStatus.DEADLINE_EXCEEDED)
                        chunk = os.read(process.stdout.fileno(), min(65_536, MAX_OUTPUT_BYTES + 1 - len(output)))
                        if not chunk:
                            selector.unregister(process.stdout)
                            break
                        output.extend(chunk)
                        if len(output) > MAX_OUTPUT_BYTES:
                            self._terminate(process)
                            return CommandResult(CommandStatus.OUTPUT_LIMIT)
            except OSError as error:
                logger.warning("telemetry probe output failed for process %s: %s", process.pid, error)
                self._terminate(process)
                return CommandResult(CommandStatus.OUTPUT_FAILED)

            remaining = deadline.remaining_seconds()
            if remaining <= 0:
                self._terminate(process)
                return CommandResult(CommandStatus.DEADLINE_EXCEEDED)
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                self._terminate(process)
                return CommandResult(CommandStatus.DEADLINE_EXCEEDED)
            with self._lock:
                if self._cancelled:
                    return CommandResult(CommandStatus.CANCELLED)
            return CommandResult(CommandStatus.COMPLETED, CommandOutput(bytes(output), returncode))
        finally:
            process.stdout.close()
            with self._lock:
                self._active.discard(process)

    def cancel(self, timeout: float) -> None:
        """Prevent new commands and reap active process groups within one budget."""
        deadline = Deadline.from_seconds(max(0.0, timeout))
        with self._lock:
            self._cancelled = True
            active = tuple(self._active)
        for process in active:
            self._kill(process)
        for process in active:
            try:
                process.wait(timeout=deadline.remaining_seconds())
            except subprocess.TimeoutExpired:
                logger.warning("telemetry probe process %s was not reaped during shutdown", process.pid)
                break

    @staticmethod
    def _kill(process: subprocess.Popen[bytes]) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return
        except OSError:
            try:
                process.kill()
            except OSError as error:
                logger.warning("could not terminate telemetry probe process %s: %s", process.pid, error)
                return

    @classmethod
    def _terminate(cls, process: subprocess.Popen[bytes]) -> None:
        cls._kill(process)
        try:
            process.wait(timeout=_PROCESS_REAP_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning("telemetry probe process %s was not reaped after termination", process.pid)
            return


class PeriodicProbe:
    """Run one concrete collector periodically with bounded shutdown."""

    def __init__(
        self,
        name: str,
        collect: Callable[[BoundedCommandRunner], None],
        *,
        interval: float = _DEFAULT_INTERVAL,
    ) -> None:
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("probe interval must be positive and finite")
        self._name = name
        self._collect = collect
        self._interval = interval
        self._stop = threading.Event()
        self._runner = BoundedCommandRunner()
        self._thread = threading.Thread(target=self._run, name=f"rigging-probe-{name}", daemon=True)
        self._shutdown_lock = threading.Lock()
        self._shutdown = False
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._collect(self._runner)
            except Exception:
                logger.warning("%s telemetry probe failed", self._name, exc_info=True)
            if self._stop.wait(self._interval):
                return

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop scheduling and reap the active subprocess within one budget."""
        try:
            budget = float(timeout)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("shutdown timeout must be nonnegative and finite") from None
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("shutdown timeout must be nonnegative and finite")
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
            deadline = Deadline.from_seconds(min(budget, _MAX_SHUTDOWN_TIMEOUT))
            self._stop.set()
            self._runner.cancel(deadline.remaining_seconds())
            self._thread.join(deadline.remaining_seconds())

    def __enter__(self) -> "PeriodicProbe":
        return self

    def __exit__(self, *_args: object) -> None:
        self.shutdown()
