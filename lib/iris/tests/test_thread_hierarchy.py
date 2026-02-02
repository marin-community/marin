# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for hierarchical ThreadContainer behavior."""

import threading
import time

import pytest

from iris.managed_thread import ThreadContainer


def _busy_wait(stop_event: threading.Event) -> None:
    stop_event.wait()


def _counter(stop_event: threading.Event, results: list) -> None:
    results.append(1)
    stop_event.wait()


def test_create_child_and_stop_propagates():
    parent = ThreadContainer(name="parent")
    child = parent.create_child("child")

    parent_results: list[int] = []
    child_results: list[int] = []

    parent.spawn(target=_counter, name="parent-thread", args=(parent_results,))
    child.spawn(target=_counter, name="child-thread", args=(child_results,))

    time.sleep(0.1)
    assert len(parent_results) == 1
    assert len(child_results) == 1

    parent.stop(timeout=2.0)

    assert parent_results == [1]
    assert child_results == [1]


def test_alive_threads_includes_children():
    parent = ThreadContainer(name="parent")
    child = parent.create_child("child")

    parent.spawn(target=_busy_wait, name="p1")
    child.spawn(target=_busy_wait, name="c1")
    child.spawn(target=_busy_wait, name="c2")

    time.sleep(0.05)
    alive = parent.alive_threads()
    assert len(alive) == 3

    parent.stop(timeout=2.0)
    assert parent.alive_threads() == []


def test_spawn_executor_shutdown_on_stop():
    container = ThreadContainer(name="test")
    executor = container.spawn_executor(max_workers=2, prefix="test-exec")

    future = executor.submit(lambda: 42)
    assert future.result() == 42

    container.stop(timeout=2.0)

    with pytest.raises(RuntimeError):
        executor.submit(lambda: 1)


def test_nested_children():
    root = ThreadContainer(name="root")
    mid = root.create_child("mid")
    leaf = mid.create_child("leaf")

    leaf.spawn(target=_busy_wait, name="leaf-thread")

    time.sleep(0.05)
    assert len(root.alive_threads()) == 1

    root.stop(timeout=2.0)
    assert root.alive_threads() == []


def test_child_stop_before_parent_threads():
    """Verify children are stopped before parent's own threads are joined."""
    order: list[str] = []

    def child_thread(stop_event: threading.Event) -> None:
        stop_event.wait()
        order.append("child_exited")

    def parent_thread(stop_event: threading.Event) -> None:
        stop_event.wait()
        order.append("parent_exited")

    parent = ThreadContainer(name="parent")
    child = parent.create_child("child")
    child.spawn(target=child_thread, name="child-t")
    parent.spawn(target=parent_thread, name="parent-t")

    time.sleep(0.05)
    parent.stop(timeout=2.0)

    # Child threads should be stopped (and joined) before parent threads
    assert order.index("child_exited") < order.index("parent_exited")
