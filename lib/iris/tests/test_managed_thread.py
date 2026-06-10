# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ManagedThread lifecycle.

Focus: on_stop callback must run both when stop() is called externally AND
when the thread target returns on its own. A missed on_stop on the natural-
completion path left task containers un-reaped in production — the container
process stayed wedged on the TPU vfio/iommu group, poisoning the VM for
subsequent tasks.
"""

import threading
import time

from iris.managed_thread import ManagedThread


def test_on_stop_runs_when_stop_is_called():
    stopped = threading.Event()
    released = threading.Event()

    def target(stop_event: threading.Event) -> None:
        stop_event.wait(timeout=5.0)

    def on_stop() -> None:
        stopped.set()
        released.set()

    t = ManagedThread(target=target, name="stop-called", on_stop=on_stop)
    t.start()
    t.stop()
    t.join()
    assert stopped.is_set()


def test_on_stop_runs_when_target_returns_naturally():
    """Regression: target returning on its own must still fire on_stop.

    Before the fix, on_stop was only invoked when an explicit stop() set the
    stop event. When the target returned naturally (e.g. a task container
    exited and the monitoring loop finished), the watcher stayed parked on
    stop_event.wait() and the finally block timed out silently, skipping
    on_stop. For task threads this meant docker kill + docker rm never ran,
    leaving wedged containers holding TPU vfio groups.
    """
    on_stop_ran = threading.Event()

    def target(_stop_event: threading.Event) -> None:
        # Return immediately without touching the stop event.
        return

    def on_stop() -> None:
        on_stop_ran.set()

    t = ManagedThread(target=target, name="natural-return", on_stop=on_stop)
    t.start()
    t.join()
    assert on_stop_ran.is_set(), "on_stop must run when target completes naturally"


def test_on_stop_runs_when_target_raises():
    """on_stop must also fire when the target raises — exception path."""
    on_stop_ran = threading.Event()

    class _Boom(Exception):
        pass

    def target(_stop_event: threading.Event) -> None:
        raise _Boom("task blew up")

    def on_stop() -> None:
        on_stop_ran.set()

    t = ManagedThread(target=target, name="raising-target", on_stop=on_stop)
    t.start()
    t.join()
    assert on_stop_ran.is_set(), "on_stop must run even when target raises"


def test_on_stop_runs_only_once():
    """on_stop must not double-fire when both stop() and natural return occur."""
    calls = []
    lock = threading.Lock()

    def target(stop_event: threading.Event) -> None:
        stop_event.wait(timeout=0.2)

    def on_stop() -> None:
        with lock:
            calls.append(time.monotonic())

    t = ManagedThread(target=target, name="no-double-fire", on_stop=on_stop)
    t.start()
    t.stop()
    t.join()
    assert len(calls) == 1, f"on_stop fired {len(calls)} times, expected 1"
