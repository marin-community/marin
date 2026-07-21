# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-place crash respawner: restart the task command when it dies from a crash signal.

``python -m iris.cluster.hooks.respawn_main --max-restarts N -- <argv>`` runs
``<argv>`` and, when it dies from a crash signal, starts it again inside the same
task — same container, same synced venv, same workdir — instead of surfacing the
failure to iris.

Built for fate-sharing gangs (JAX distributed): when one task's process crashes,
XLA's coordination service propagates the error and every peer process aborts
with ``LOG(FATAL)`` (SIGABRT). Without the respawner, iris sees one FAILED task,
cascades the other tasks, and a retry pays scheduling + container + setup for
the whole gang. With it, each task's respawner restarts its own dead child; the
restarted processes re-run ``jax.distributed.initialize``, whose registration
retries (including the "duplicate task registration" abort from a restarted
task) ride out the window until the restarted task-0 child brings a fresh
coordination service up. The workload then resumes from its own checkpoints.

Respawn policy — only deaths from a crash signal qualify:

- SIGABRT (``LOG(FATAL)``/``abort()``, the JAX fate-sharing exit), SIGSEGV,
  SIGBUS, SIGILL, SIGFPE, SIGTRAP → respawn.
- A nonzero exit code (e.g. a Python exception) is a deterministic application
  failure and propagates immediately.
- SIGKILL is external (kernel OOM kill, container teardown) and propagates, so
  the worker's OOM annotation (exit 137) and iris's retry budgets keep working.

Two brakes bound crash loops: ``--max-restarts`` caps total respawns for the
task attempt, and three consecutive deaths within ``_MIN_UPTIME`` of launch give
up early (a healthy gang crash happens hours in; a crash-at-startup loop does
not deserve the full budget). Exhausting either propagates the exit, handing
recovery back to iris's ``max_retries_failure``/``max_task_failures`` machinery.

The child runs in its own session: a crashed attempt's orphaned descendants may
still hold accelerator devices, so the whole process group is SIGKILLed before
the next attempt starts. The respawner forwards SIGINT/SIGTERM (preemption,
job stop) to the group and exits ``128 + signum`` without respawning. Child
signal deaths propagate as the conventional ``128 + signum`` exit code.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

from rigging.timing import Deadline, Duration, Timer

from iris.cluster.hooks.respawn import IRIS_RESPAWN_ATTEMPT_ENV

logger = logging.getLogger("iris.respawn")

_SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM)
# How often to re-poll child liveness while waiting for it (seconds).
_REAP_POLL_INTERVAL = 1.0
# Grace period after a forwarded SIGTERM before escalating to SIGKILL, so a child
# that traps or ignores SIGTERM cannot wedge the respawner (and hence the task).
_TERMINATE_GRACE = Duration.from_seconds(10.0)
# Deaths that qualify for a respawn: runtime faults, not external kills or
# deliberate exits. SIGABRT is the JAX fate-sharing path (LOG(FATAL) → abort()).
_RESPAWN_SIGNALS = frozenset(
    {signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS, signal.SIGILL, signal.SIGFPE, signal.SIGTRAP}
)
# Pause before each respawn, so a crash-at-exec loop cannot spin hot.
_RESPAWN_DELAY = Duration.from_seconds(5.0)
# A death within this many seconds of launch counts as "rapid"; three consecutive
# rapid deaths give up early. A death after it resets the consecutive count.
_MIN_UPTIME = 600.0
_MAX_RAPID_DEATHS = 3


def _crash_signal(code: int) -> int | None:
    """The respawn-qualifying signal behind a child exit code, or ``None``.

    A direct child killed by a signal reports the negative signal number; an
    intermediate shell/launcher (``bash -c`` without ``exec``, ``uv run``)
    reports the conventional ``128 + signum`` instead, so both encodings are
    recognized.
    """
    signum = -code if code < 0 else code - 128 if 128 < code < 160 else None
    if signum in _RESPAWN_SIGNALS:
        return signum
    return None


def _normalize_exit(code: int) -> int:
    """Map a child's raw wait() code to a process exit code (``-N`` → ``128 + N``)."""
    return 128 - code if code < 0 else code


def _signal_group(child: subprocess.Popen, sig: int) -> None:
    """Send ``sig`` to the child's process group (it runs in its own session)."""
    try:
        os.killpg(child.pid, sig)
    except ProcessLookupError:
        pass


def run(max_restarts: int, child_argv: list[str]) -> int:
    """Run ``child_argv``, respawning on crash-signal deaths; return the exit code.

    ``128 + signum`` if SIGINT/SIGTERM was delivered to the respawner (external
    termination/preemption — never respawned); 0 on a clean child exit; otherwise
    the child's final (normalized) exit code once it is non-respawnable or the
    restart budget is exhausted.
    """
    if max_restarts < 0:
        raise ValueError(f"--max-restarts must be >= 0, got {max_restarts}")
    if not child_argv:
        raise ValueError("no child command given after '--'")

    child: subprocess.Popen | None = None
    shutdown_signum: int | None = None
    # Deadline after which the group is SIGKILLed; None until a teardown begins.
    # Shared between the signal handler and the reap loop, both in the main thread.
    kill_deadline: Deadline | None = None

    def _forward_signal(signum: int, _frame) -> None:
        nonlocal shutdown_signum, kill_deadline
        if shutdown_signum is None:
            shutdown_signum = signum
        logger.warning("received signal %d; forwarding to child group", signum)
        if child is not None:
            _signal_group(child, signum)
            if kill_deadline is None:
                kill_deadline = Deadline.from_now(_TERMINATE_GRACE)

    previous_handlers = {sig: signal.getsignal(sig) for sig in _SHUTDOWN_SIGNALS}
    for sig in _SHUTDOWN_SIGNALS:
        signal.signal(sig, _forward_signal)

    try:
        restarts = 0
        rapid_deaths = 0
        while True:
            uptime = Timer()
            child_env = {**os.environ, IRIS_RESPAWN_ATTEMPT_ENV: str(restarts)}
            # Own session: lets a crashed attempt's whole descendant group be killed
            # before the next attempt, so no orphan holds the accelerator.
            child = subprocess.Popen(child_argv, env=child_env, start_new_session=True)
            if shutdown_signum is not None:
                # Shutdown raced the spawn: the handler saw child=None and signalled
                # nothing, so deliver to the fresh group here.
                _signal_group(child, shutdown_signum)
                kill_deadline = Deadline.from_now(_TERMINATE_GRACE)

            while True:
                if kill_deadline is not None and kill_deadline.expired():
                    logger.error("SIGTERM grace expired; SIGKILL child group")
                    _signal_group(child, signal.SIGKILL)
                    kill_deadline = None  # SIGKILL is final; do not re-fire each tick
                try:
                    code = child.wait(timeout=_REAP_POLL_INTERVAL)
                    break
                except subprocess.TimeoutExpired:
                    continue

            # Reap any descendants the dead child left behind before deciding what
            # to do next — a respawned attempt must not race them for devices.
            _signal_group(child, signal.SIGKILL)

            # An external signal wins over the child's exit code: the code is an
            # artifact of the forwarded teardown, not a task failure.
            if shutdown_signum is not None:
                return 128 + shutdown_signum
            if code == 0:
                return 0

            exit_code = _normalize_exit(code)
            signum = _crash_signal(code)
            if signum is None:
                logger.error("child exited %d (not a crash signal); propagating", exit_code)
                return exit_code

            elapsed = uptime.elapsed_seconds()
            rapid_deaths = rapid_deaths + 1 if elapsed < _MIN_UPTIME else 0
            if rapid_deaths >= _MAX_RAPID_DEATHS:
                logger.error(
                    "child died from signal %d after %.0fs — %d consecutive rapid deaths; giving up",
                    signum,
                    elapsed,
                    rapid_deaths,
                )
                return exit_code
            if restarts >= max_restarts:
                logger.error(
                    "child died from signal %d after %.0fs; restart budget (%d) exhausted",
                    signum,
                    elapsed,
                    max_restarts,
                )
                return exit_code

            restarts += 1
            logger.error(
                "child died from signal %d after %.0fs; respawning (%d/%d)",
                signum,
                elapsed,
                restarts,
                max_restarts,
            )
            # Short sleep slices so a shutdown signal during the pause is honored
            # within one reap interval rather than after the full delay.
            deadline = Deadline.from_now(_RESPAWN_DELAY)
            while not deadline.expired() and shutdown_signum is None:
                time.sleep(min(_REAP_POLL_INTERVAL, max(deadline.remaining_seconds(), 0.0)))
            if shutdown_signum is not None:
                return 128 + shutdown_signum
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw:
        raise SystemExit("usage: python -m iris.cluster.hooks.respawn_main --max-restarts N -- <command...>")
    split = raw.index("--")
    own_args, child_argv = raw[:split], raw[split + 1 :]

    parser = argparse.ArgumentParser(prog="python -m iris.cluster.hooks.respawn_main")
    parser.add_argument(
        "--max-restarts", type=int, required=True, help="total in-place restarts allowed for this task attempt"
    )
    args = parser.parse_args(own_args)
    return run(args.max_restarts, child_argv)


if __name__ == "__main__":
    sys.exit(main())
