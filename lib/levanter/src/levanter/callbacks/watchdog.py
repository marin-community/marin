# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward-progress watchdog callback.

Native hangs inside ``jax.jit``-compiled steps (for example, XLA command-buffer
or CUDA-graph teardown corruption on B200/FP8 runs) leave the orchestrator
believing the job is healthy: pods stay up, GPUs stay hot, and only forward
progress stops. This callback converts those silent native hangs into actionable
failures.

A heartbeat is recorded on every ``on_step`` call. A background daemon thread
wakes up periodically and, if no heartbeat has been recorded within
``timeout`` seconds *after the first successful step*, dumps the Python stacks
of every thread (including the one parked inside native execution) and then
escalates via ``SIGABRT`` so the orchestrator marks the pod failed instead of
keeping the slot occupied.

The first-step gate is important: warmup/compilation can legitimately take many
minutes, and we do not want the watchdog firing before training has actually
begun.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
import traceback
from typing import Callable

from levanter.callbacks._core import Callback, StepInfo


logger = logging.getLogger(__name__)


TimeoutHandler = Callable[[float, int], None]


def _default_on_timeout(elapsed: float, last_step: int) -> None:
    """Dump Python stacks for every live thread, then SIGABRT the process.

    ``SIGABRT`` (vs. SIGTERM/SIGKILL) is intentional: it produces a coredump on
    systems where one is configured, which is the most useful artifact for
    debugging native hangs in XLA/NCCL/CUDA. It also reliably propagates a
    non-zero exit so Iris/K8s liveness sees a real failure.
    """
    header = f"\n=== forward-progress watchdog fired (elapsed={elapsed:.1f}s, last_completed_step={last_step}) ===\n"
    sys.stderr.write(header)
    sys.stderr.write("Thread stacks at hang time:\n")
    frames = sys._current_frames()
    for tid, frame in frames.items():
        sys.stderr.write(f"\n--- Thread {tid} ---\n")
        sys.stderr.write("".join(traceback.format_stack(frame)))
    sys.stderr.write("=== end watchdog dump ===\n")
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGABRT)


class ForwardProgressWatchdog(Callback):
    """Callback that fails loudly when training stops making forward progress.

    Args:
        timeout: Seconds without a completed step before the watchdog fires.
            Should be set comfortably larger than the slowest expected step
            (including periodic evals/checkpoints that run inline on the main
            thread) so it does not produce false positives.
        check_interval: How often the daemon thread wakes up to compare the
            wall clock against the heartbeat. A small fraction of ``timeout``
            is appropriate; the default of 30s is a good compromise.
        on_timeout: Called once if the watchdog fires. The default dumps thread
            stacks and SIGABRTs the process. Tests override this.

    The watchdog daemon thread starts at construction time but only begins
    enforcing the deadline after the first heartbeat (i.e. the first
    ``on_step`` call). Warmup and JIT compilation therefore do not count
    against the deadline.
    """

    def __init__(
        self,
        timeout: float,
        *,
        check_interval: float = 30.0,
        on_timeout: TimeoutHandler | None = None,
    ):
        if timeout <= 0:
            raise ValueError(f"watchdog timeout must be positive, got {timeout}")
        if check_interval <= 0:
            raise ValueError(f"watchdog check_interval must be positive, got {check_interval}")
        if check_interval >= timeout:
            # Not strictly wrong, but check_interval >= timeout means the
            # watchdog can sleep through a full timeout window before noticing.
            logger.warning(
                "watchdog check_interval (%.1fs) >= timeout (%.1fs); detection latency may exceed timeout",
                check_interval,
                timeout,
            )

        self._timeout = timeout
        self._check_interval = check_interval
        self._on_timeout = on_timeout if on_timeout is not None else _default_on_timeout
        self._lock = threading.Lock()
        self._last_heartbeat: float | None = None
        self._last_step: int = -1
        self._stop = threading.Event()
        self._triggered = False
        self._thread = threading.Thread(target=self._run, name="forward-progress-watchdog", daemon=True)
        self._thread.start()

    def on_step(self, info: StepInfo, force: bool = False) -> None:
        with self._lock:
            self._last_heartbeat = time.monotonic()
            # info.step is the most recently completed step.
            self._last_step = int(info.step)

    def stop(self) -> None:
        """Signal the watchdog thread to exit. Safe to call multiple times."""
        self._stop.set()

    @property
    def triggered(self) -> bool:
        return self._triggered

    def _run(self) -> None:
        while not self._stop.wait(self._check_interval):
            with self._lock:
                last = self._last_heartbeat
                step = self._last_step
            if last is None:
                # No heartbeat yet — still in warmup/compilation. Keep waiting.
                continue
            elapsed = time.monotonic() - last
            if elapsed < self._timeout:
                continue
            if self._triggered:
                # Defensive: handler should have terminated the process already.
                return
            self._triggered = True
            logger.error(
                "Forward-progress watchdog: no step completed in %.1fs "
                "(last completed step=%d, timeout=%.1fs). Dumping stacks and aborting.",
                elapsed,
                step,
                self._timeout,
            )
            try:
                self._on_timeout(elapsed, step)
            except Exception:
                logger.exception("Watchdog on_timeout handler raised; the process may not exit.")
            return


__all__ = ["ForwardProgressWatchdog", "TimeoutHandler"]
