# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""xprof lifecycle: capacity cap (incl. the concurrent-launch reservation) + prepare."""

from __future__ import annotations

import dataclasses
import threading

import pytest
from buoy.xprof import XprofCapacityError, XprofManager


class DummyProc:
    def __init__(self) -> None:
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self) -> None:
        self._alive = False


@pytest.fixture
def mgr(cfg, monkeypatch):
    manager = XprofManager(dataclasses.replace(cfg, xprof_bin="/bin/true", max_xprof_procs=1))
    monkeypatch.setattr(XprofManager, "_materialize", lambda self, key, logdir: "/tmp/logdir")
    return manager


def test_cap_rejects_second_active_session(mgr, monkeypatch):
    monkeypatch.setattr(XprofManager, "_spawn", lambda self, logdir: (DummyProc(), 30001))
    assert mgr.ensure("a", "ld") == 30001
    # cap=1 and "a" was just used (within grace) — it can't be evicted.
    with pytest.raises(XprofCapacityError):
        mgr.ensure("b", "ld")


def test_concurrent_launch_reserves_slot(mgr, monkeypatch):
    # Force "a" to sit mid-launch (slot reserved, not yet registered) while "b" arrives.
    entered = threading.Event()
    release = threading.Event()

    def gated_spawn(self, logdir):
        entered.set()
        assert release.wait(5)
        return DummyProc(), 30002

    monkeypatch.setattr(XprofManager, "_spawn", gated_spawn)
    result: dict[str, int] = {}
    worker = threading.Thread(target=lambda: result.__setitem__("a", mgr.ensure("a", "ld")))
    worker.start()
    assert entered.wait(5)  # "a" has reserved its slot and is blocked in spawn

    with pytest.raises(XprofCapacityError):
        mgr.ensure("b", "ld")  # the in-flight launch counts against the cap

    release.set()
    worker.join(5)
    assert result["a"] == 30002
    assert set(mgr._procs) == {"a"}


def test_prepare_status(mgr, monkeypatch):
    monkeypatch.setattr(XprofManager, "_spawn", lambda self, logdir: (DummyProc(), 30003))
    assert mgr.prepare_status("a")["state"] == "absent"
    mgr.ensure("a", "ld")
    assert mgr.prepare_status("a")["state"] == "ready"
