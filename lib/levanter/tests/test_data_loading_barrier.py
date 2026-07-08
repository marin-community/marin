# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the host-side data-loading rendezvous barrier.

These exercise the real rendezvous/timeout/cleanup logic against an in-memory fake of
JAX's coordination key-value store. The fake mirrors the observable contract of the
coordination client that was verified against a live service: ``key_value_dir_get``
returns ``(key, value)`` pairs under a prefix, and ``key_value_delete`` removes either a
single key or, when the argument ends in ``/``, every key under that directory.
"""

import threading

import jax
import jax._src.distributed as jax_distributed
import pytest

import levanter.utils.jax_utils as jax_utils
from levanter.utils.jax_utils import data_loading_barrier


class _FakeKVClient:
    def __init__(self):
        self.store: dict[str, str] = {}

    def key_value_set(self, key: str, value: str) -> None:
        self.store[key] = value

    def key_value_dir_get(self, prefix: str):
        return [(k, v) for k, v in self.store.items() if k.startswith(prefix)]

    def key_value_delete(self, key: str) -> None:
        if key.endswith("/"):
            for k in [k for k in self.store if k.startswith(key)]:
                del self.store[k]
        else:
            self.store.pop(key, None)


@pytest.fixture
def coordination(monkeypatch):
    """Install a fake coordination client and reset the module's cleanup cursor."""
    client = _FakeKVClient()
    monkeypatch.setattr(jax_distributed.global_state, "client", client)
    monkeypatch.setattr(jax_utils, "_last_data_ready_step", None)
    return client


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


def test_waits_for_a_lagging_process(coordination, monkeypatch):
    _set_world(monkeypatch, num_processes=2, process_id=0)

    def publish_late():
        # peer 1 shows up after this process is already polling
        coordination.store["levanter/data_ready/3/1"] = "1"

    timer = threading.Timer(0.2, publish_late)
    timer.start()
    try:
        # small warn_interval exercises the leader's periodic still-waiting log path
        data_loading_barrier(3, timeout=5.0, warn_interval=0.05)
    finally:
        timer.cancel()


def test_timeout_names_the_missing_process(coordination, monkeypatch):
    _set_world(monkeypatch, num_processes=3, process_id=0)
    coordination.store["levanter/data_ready/9/1"] = "1"
    # process 2 never publishes

    with pytest.raises(TimeoutError) as exc:
        data_loading_barrier(9, timeout=0.3, warn_interval=0.05)

    assert "2" in str(exc.value)


def test_single_process_is_a_noop(monkeypatch):
    # No coordination client installed: a single-process job must not touch it.
    monkeypatch.setattr(jax_distributed.global_state, "client", None)
    _set_world(monkeypatch, num_processes=1, process_id=0)

    data_loading_barrier(0, timeout=5.0)


def test_leader_cleans_up_previous_step(coordination, monkeypatch):
    _set_world(monkeypatch, num_processes=1, process_id=0)
    # single-process short-circuits, so drive with a 1-process world but a real client by
    # calling with a 2-process world where the peer is pre-published.
    monkeypatch.setattr(jax, "process_count", lambda: 2)

    coordination.store["levanter/data_ready/10/1"] = "1"
    data_loading_barrier(10, timeout=5.0)
    assert "levanter/data_ready/10/0" in coordination.store

    coordination.store["levanter/data_ready/11/1"] = "1"
    data_loading_barrier(11, timeout=5.0)

    # entering step 11 deletes step 10's directory
    assert not any(k.startswith("levanter/data_ready/10/") for k in coordination.store)
    assert "levanter/data_ready/11/0" in coordination.store
