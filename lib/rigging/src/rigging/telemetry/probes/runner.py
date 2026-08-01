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
from dataclasses import dataclass

from rigging.timing import Deadline

MAX_OUTPUT_BYTES = 256 * 1024
_PROCESS_REAP_TIMEOUT = 0.1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandOutput:
    stdout: bytes
    returncode: int


class BoundedCommandRunner:
    """Run probe commands with bounded output, deadlines, and cancellation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: set[subprocess.Popen[bytes]] = set()
        self._cancelled = False

    def run(self, argv: tuple[str, ...], timeout: float) -> CommandOutput | None:
        """Return completed output, or None when collection cannot finish safely."""
        if not argv:
            raise ValueError("argv must not be empty")
        if not math.isfinite(timeout) or timeout <= 0:
            return None

        with self._lock:
            if self._cancelled:
                return None
            try:
                process = subprocess.Popen(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    start_new_session=os.name == "posix",
                )
            except OSError as error:
                logger.debug("could not start telemetry probe %s: %s", argv[0], error)
                return None
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
                            return None
                        chunk = os.read(process.stdout.fileno(), min(65_536, MAX_OUTPUT_BYTES + 1 - len(output)))
                        if not chunk:
                            selector.unregister(process.stdout)
                            break
                        output.extend(chunk)
                        if len(output) > MAX_OUTPUT_BYTES:
                            self._terminate(process)
                            return None
            except OSError as error:
                logger.warning("telemetry probe output failed for process %s: %s", process.pid, error)
                self._terminate(process)
                return None

            try:
                returncode = process.wait(timeout=deadline.remaining_seconds())
            except subprocess.TimeoutExpired:
                self._terminate(process)
                return None
            with self._lock:
                if self._cancelled:
                    return None
            return CommandOutput(bytes(output), returncode)
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
