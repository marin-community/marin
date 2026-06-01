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
