# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared, JAX-free types for the recovery package.

This module holds pure, JAX-free data types shared by the supervisor parent and
the trainer child, so either side can import it without pulling in JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


# Exit-code contract between the trainer subprocess and the supervisor.
#
# 124 matches ``levanter.callbacks.progress_watchdog.STALLED_TRAINING_EXIT_CODE`` so
# the existing watchdog and the new sentinel agree on one number for "stalled".
EXIT_OK = 0
EXIT_STALL = 124
EXIT_STICKY_FAULT = 125


class FaultClass(StrEnum):
    """How the supervisor interpreted a trainer subprocess exit."""

    NONE = "none"  # clean exit (EXIT_OK)
    STALL = "stall"  # watchdog / deadman fired (EXIT_STALL or supervisor-initiated kill)
    STICKY = "sticky"  # poisoned CUDA context sentinel (EXIT_STICKY_FAULT)
    CRASH = "crash"  # killed by a signal: XLA LOG(FATAL) SIGABRT, SIGSEGV, OOM SIGKILL
    HARD = "hard"  # non-sentinel nonzero exit: application error, not recoverable


# Exit interpretations the supervisor will warm-restart instead of propagate.
RECOVERABLE = frozenset({FaultClass.STALL, FaultClass.STICKY, FaultClass.CRASH})


def classify_returncode(returncode: int, *, deadman: bool = False) -> FaultClass:
    """Map a ``subprocess`` returncode to a fault class.

    ``deadman`` marks a kill the supervisor initiated because the child stopped
    heart-beating; the returncode is then the supervisor's own signal, so the
    stall classification is forced regardless of what the signal was.
    """
    if deadman:
        return FaultClass.STALL
    if returncode == EXIT_OK:
        return FaultClass.NONE
    if returncode == EXIT_STALL:
        return FaultClass.STALL
    if returncode == EXIT_STICKY_FAULT:
        return FaultClass.STICKY
    if returncode < 0:
        # Popen reports signal deaths as the negated signal number. XLA's hang
        # watchdog and clique-abort paths end in LOG(FATAL) -> SIGABRT (-6); a
        # poisoned context or OOM shows up as SIGSEGV/SIGKILL.
        return FaultClass.CRASH
    return FaultClass.HARD


class RunOutcome(StrEnum):
    COMPLETED = "completed"  # reached the step target (possibly after recoveries)
    FAILED = "failed"  # unrecoverable, or restart budget exhausted
    UNHEALTHY_GPU = "unhealthy_gpu"  # health gate refused restart; escalate to scheduler


@dataclass(frozen=True)
class FaultRecord:
    """One trainer subprocess death within a single run()."""

    attempt: int
    fault_class: FaultClass
    returncode: int
    wall_time: float
    last_step: int | None = None
    detail: str = ""


@dataclass
class RunResult:
    """Outcome of one ``supervisor.run(...)`` invocation (one ablation)."""

    label: str
    outcome: RunOutcome
    attempts: int
    final_step: int | None
    total_wall_time: float
    faults: list[FaultRecord] = field(default_factory=list)
    restored_from_snapshot: bool = False

    @property
    def recovered(self) -> bool:
        return self.outcome is RunOutcome.COMPLETED and len(self.faults) > 0


@dataclass(frozen=True)
class AblationSpec:
    """One entry in an ablation sweep run under a single warm supervisor."""

    name: str
    env: dict[str, str] = field(default_factory=dict)
    num_steps: int | None = None  # override the entrypoint default when set
    deadman_timeout: float | None = None  # override the supervisor deadman for this run
    notes: str = ""


class FaultKind(StrEnum):
    """Fault-injection modes exercised by ``levanter.recovery.faults``."""

    NONE = "none"
    SIGKILL = "sigkill"  # os.kill(self, SIGKILL): abrupt process death (peer-death drill)
    STICKY = "sticky"  # raise a fake sticky CUDA error through the sentinel path
    SPIN = "spin"  # launch an unterminating device kernel (true silent GPU wedge)
    SIGSTOP = "sigstop"  # freeze the process (host-side stall; exercises the deadman)
