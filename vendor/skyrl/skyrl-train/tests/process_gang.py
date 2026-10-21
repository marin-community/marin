"""Synchronize, control, and bound the lifecycle of disposable process gangs."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ReapOutcome(StrEnum):
    REQUESTED_SIGNAL = "reaped-after-requested-signal"
    SIGKILL_ESCALATION = "reaped-after-sigkill-escalation"
    NOT_REAPED = "not-reaped"


class SignalWaitOutcome(StrEnum):
    EXITED = "exited"
    TIMED_OUT = "timed-out"


@dataclass(frozen=True)
class ProcessGangResult:
    returncode: int
    output: str


class ProcessGangTimeoutError(TimeoutError):
    def __init__(self, timeout_seconds: float, reap_outcome: ReapOutcome, output: str) -> None:
        super().__init__(f"process gang exceeded {timeout_seconds}s; cleanup={reap_outcome.value}; output:\n{output}")


@dataclass
class ProcessGang:
    process: subprocess.Popen[str]
    control_directory: Path
    log_path: Path
    reap_timeout_seconds: float
    termination_signal: signal.Signals = signal.SIGKILL

    def output(self) -> str:
        """Read captured output, tolerating partial non-UTF-8 writes."""

        return self.log_path.read_text(errors="replace")

    def _signal_and_wait(self, requested_signal: signal.Signals) -> SignalWaitOutcome:
        """Signal the gang and classify its leader's bounded wait."""

        try:
            os.killpg(self.process.pid, requested_signal)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=self.reap_timeout_seconds)
        except subprocess.TimeoutExpired:
            return SignalWaitOutcome.TIMED_OUT
        return SignalWaitOutcome.EXITED

    def kill_and_reap(self) -> ReapOutcome:
        """Terminate the subprocess group and report how its leader was reaped."""

        if self._signal_and_wait(self.termination_signal) is SignalWaitOutcome.EXITED:
            return ReapOutcome.REQUESTED_SIGNAL
        if self.termination_signal == signal.SIGKILL:
            return ReapOutcome.NOT_REAPED
        if self._signal_and_wait(signal.SIGKILL) is SignalWaitOutcome.EXITED:
            return ReapOutcome.SIGKILL_ESCALATION
        return ReapOutcome.NOT_REAPED

    def wait(self, timeout_seconds: float) -> ProcessGangResult:
        """Wait for completion or kill and reap the gang at the deadline."""

        try:
            returncode = self.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            reap_outcome = self.kill_and_reap()
            raise ProcessGangTimeoutError(timeout_seconds, reap_outcome, self.output()) from error
        return ProcessGangResult(returncode, self.output())


@contextmanager
def launch_process_gang(
    *,
    command: Sequence[str],
    working_directory: Path,
    environment: Mapping[str, str],
    control_directory: Path,
    log_path: Path,
    reap_timeout_seconds: float,
    process_description: str,
    termination_signal: signal.Signals = signal.SIGKILL,
) -> Iterator[ProcessGang]:
    """Launch a process gang and guarantee bounded cleanup of its session."""

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        gang = ProcessGang(process, control_directory, log_path, reap_timeout_seconds, termination_signal)
        try:
            yield gang
        finally:
            if gang.kill_and_reap() is ReapOutcome.NOT_REAPED:
                raise RuntimeError(f"failed to reap {process_description} {process.pid}")
