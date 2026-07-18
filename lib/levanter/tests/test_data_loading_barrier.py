# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the host-side data-loading rendezvous barrier.

These exercise the real rendezvous/timeout/cleanup logic against an in-memory fake of
JAX's coordination key-value store. The fake mirrors the observable contract of the
coordination client that was verified against a live service: ``key_value_dir_get``
returns ``(key, value)`` pairs under a prefix, and ``key_value_delete`` removes either a
single key or, when the argument ends in ``/``, every key under that directory.
"""

import jax
import jax._src.distributed as jax_distributed
import pytest

from levanter.trainer import _barrier_active
from levanter.utils.jax_utils import data_loading_barrier


class _FakeKVClient:
    """In-memory stand-in for JAX's coordination key-value store.

    ``on_dir_get`` is an optional hook invoked before each directory read, letting a test
    reveal a lagging peer after a set number of polls without racing on wall-clock time.
    """

    def __init__(self, on_dir_get=None):
        self.store: dict[str, str] = {}
        self._on_dir_get = on_dir_get

    def key_value_set(self, key: str, value: str) -> None:
        self.store[key] = value

    def key_value_dir_get(self, prefix: str):
        if self._on_dir_get is not None:
            self._on_dir_get(self.store)
        return [(k, v) for k, v in self.store.items() if k.startswith(prefix)]

    def key_value_delete(self, key: str) -> None:
        if key.endswith("/"):
            for k in [k for k in self.store if k.startswith(key)]:
                del self.store[k]
        else:
            self.store.pop(key, None)


@pytest.fixture
def coordination(monkeypatch):
    """Install a fake coordination client for the duration of a test."""
    client = _FakeKVClient()
    monkeypatch.setattr(jax_distributed.global_state, "client", client)
    return client


def _no_sleep(_interval: float) -> None:
    """Inter-poll sleep replacement so the barrier loop runs on no real wall-clock time."""


def _set_world(monkeypatch, *, num_processes: int, process_id: int) -> None:
    monkeypatch.setattr(jax, "process_count", lambda: num_processes)
    monkeypatch.setattr(jax, "process_index", lambda: process_id)


def test_returns_when_all_processes_ready(coordination, monkeypatch):
    _set_world(monkeypatch, num_processes=3, process_id=0)
    # peers 1 and 2 have already published; this process publishes its own key on entry
    coordination.store["levanter/data_ready/7/1"] = "1"
    coordination.store["levanter/data_ready/7/2"] = "1"

    data_loading_barrier(7, timeout=5.0)

    assert coordination.store["levanter/data_ready/7/0"] == "1"


def test_waits_for_a_lagging_process(monkeypatch):
    _set_world(monkeypatch, num_processes=2, process_id=0)

    # peer 1 is absent for the first two polls, then appears — deterministic, no wall clock.
    polls = 0

    def reveal_peer_on_third_poll(store):
        nonlocal polls
        polls += 1
        if polls >= 3:
            store["levanter/data_ready/3/1"] = "1"

    client = _FakeKVClient(on_dir_get=reveal_peer_on_third_poll)
    monkeypatch.setattr(jax_distributed.global_state, "client", client)

    data_loading_barrier(3, timeout=5.0, sleep=_no_sleep)

    # returned only after polling until the lagging peer showed up
    assert polls >= 3
    assert client.store["levanter/data_ready/3/0"] == "1"


def test_timeout_names_the_missing_process(coordination, monkeypatch):
    _set_world(monkeypatch, num_processes=3, process_id=0)
    coordination.store["levanter/data_ready/9/1"] = "1"
    # process 2 never publishes

    with pytest.raises(TimeoutError) as exc:
        data_loading_barrier(9, timeout=0.3, warn_interval=0.05, sleep=_no_sleep)

    assert "2" in str(exc.value)


def test_single_process_is_a_noop(coordination, monkeypatch):
    _set_world(monkeypatch, num_processes=1, process_id=0)

    data_loading_barrier(0, timeout=5.0)

    # a single-process job returns before touching the coordination store
    assert coordination.store == {}


def test_leader_cleans_up_previous_step(coordination, monkeypatch):
    _set_world(monkeypatch, num_processes=2, process_id=0)

    coordination.store["levanter/data_ready/10/1"] = "1"
    data_loading_barrier(10, timeout=5.0)
    assert "levanter/data_ready/10/0" in coordination.store

    coordination.store["levanter/data_ready/11/1"] = "1"
    data_loading_barrier(11, timeout=5.0, previous_step=10)

    # entering step 11 with previous_step=10 deletes step 10's directory
    assert not any(k.startswith("levanter/data_ready/10/") for k in coordination.store)
    assert "levanter/data_ready/11/0" in coordination.store


@pytest.mark.parametrize(
    "step, expected",
    [
        (100, True),  # run start: within the first active_steps
        (104, True),  # last step of the cold-start window
        (105, False),  # cold-start window closed, no recent save
        (140, False),  # steady state, far from any save
        (201, True),  # first step after a save at 200
        (205, True),  # last step of the post-save window
        (206, False),  # post-save window closed
    ],
)
def test_barrier_active_windows(step, expected):
    # run resumed at step 100, most recent checkpoint saved at step 200, active_steps=5
    assert _barrier_active(step, run_start_step=100, last_save_step=200, active_steps=5) is expected
