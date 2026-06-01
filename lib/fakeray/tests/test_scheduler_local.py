# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 tests: fakeray over Fray LocalClient, no smallpond, no Iris.

Proves the Ray-compatible surface against a real DAG:
- remote/get round-trip and auto-deref of ObjectRef args
- a diamond DAG (A -> {B, C} -> D) executes in dependency order
- put() refs are usable as deps
- wait(timeout=0) is a non-blocking poll
- a failing task propagates its exception to get() and to descendants
"""

from __future__ import annotations

import os

import fakeray as ray
import pytest
from fakeray._scheduler import FakeRayConfig


@pytest.fixture(autouse=True)
def _fresh_cluster():
    ray.set_config(FakeRayConfig(pool_size=4))
    ray.init()
    yield
    ray.shutdown()


@ray.remote
def add(a, b):
    return a + b


@ray.remote
def inc(x):
    return x + 1


def test_remote_get_roundtrip():
    assert ray.get(add.remote(2, 3)) == 5


def test_auto_deref_of_dependency_refs():
    a = inc.remote(10)  # 11
    b = inc.remote(a)  # depends on a -> 12
    assert ray.get(b) == 12


def test_diamond_dag():
    a = inc.remote(0)  # 1
    b = add.remote(a, 10)  # 11
    c = add.remote(a, 100)  # 101
    d = add.remote(b, c)  # 112
    assert ray.get(d) == 112


def test_put_ref_as_dependency():
    base = ray.put(41)
    assert ray.get(inc.remote(base)) == 42


def test_get_list_preserves_order():
    refs = [inc.remote(i) for i in range(5)]
    assert ray.get(refs) == [1, 2, 3, 4, 5]


def test_wait_nonblocking_poll():
    ready, not_ready = ray.wait([inc.remote(1)], num_returns=1, timeout=0)
    assert len(ready) + len(not_ready) == 1


@ray.remote
def boom(_x):
    raise ValueError("kaboom")


def test_failure_propagates_to_get():
    with pytest.raises(ValueError, match="kaboom"):
        ray.get(boom.remote(1))


def test_failure_poisons_descendants():
    bad = boom.remote(1)
    downstream = inc.remote(bad)  # never runnable; must not hang
    with pytest.raises(ValueError, match="kaboom"):
        ray.get(downstream, timeout=10)


# Simulate an actor that dies (raises) the first few calls then recovers — i.e.
# preemption + Fray restart. The attempt counter must live OUTSIDE the pickled
# function (a module global is captured by-value through cloudpickle and would
# reset every dispatch), so we count via an on-disk file the remote fn updates.
def flaky_via_file(x, counter_path, fail_times):
    n = 0
    if os.path.exists(counter_path):
        with open(counter_path) as f:
            n = int(f.read() or "0")
    n += 1
    with open(counter_path, "w") as f:
        f.write(str(n))
    if n <= fail_times:
        raise RuntimeError(f"simulated preemption #{n}")
    return x * 10


def test_redispatch_recovers_from_transient_failure(tmp_path):
    """A task whose first attempts fail is re-dispatched and eventually succeeds."""
    counter = str(tmp_path / "attempts")
    # default max_task_retries=3 → up to 4 attempts; fail 2, succeed on the 3rd.
    fn = ray.remote(flaky_via_file)
    assert ray.get(fn.remote(5, counter, 2), timeout=30) == 50
    with open(counter) as f:
        assert int(f.read()) == 3


def test_redispatch_gives_up_after_max_retries(tmp_path):
    """A persistently failing task settles as failed once retries are exhausted."""
    counter = str(tmp_path / "attempts")
    fn = ray.remote(flaky_via_file)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        ray.get(fn.remote(5, counter, 999), timeout=30)
