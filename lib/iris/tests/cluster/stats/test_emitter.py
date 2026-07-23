# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior test for the shared periodic-emitter scaffold: a raising step must
not kill the thread — every collector relies on this for failure isolation."""

import threading

from iris.cluster.stats.emitter import PeriodicEmitter


def test_step_failure_does_not_kill_the_emitter():
    calls = 0
    survived = threading.Event()

    def step() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        survived.set()

    emitter = PeriodicEmitter(step, interval=0.01, name="test-emitter")
    try:
        assert survived.wait(timeout=5), "emitter thread died after a raising step"
    finally:
        emitter.close()
