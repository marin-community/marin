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

"""Tests for the contextvar-based ThreadContainer."""

import threading

from iris.managed_thread import get_thread_container, thread_container_scope
from tests.test_utils import wait_for_condition


def _busy_wait(stop_event: threading.Event) -> None:
    stop_event.wait()


def _counter(stop_event: threading.Event, results: list) -> None:
    results.append(1)
    stop_event.wait()


def test_container_scope_spawns_and_cleans_up():
    """Threads spawned inside a scope are stopped when the scope exits."""
    results: list[int] = []
    with thread_container_scope("test-cleanup") as container:
        container.spawn(target=_counter, name="t1", args=(results,))
        wait_for_condition(lambda: len(results) == 1, timeout=1.0)
        assert container.is_alive

    # After exiting the scope, threads have been stopped
    assert not container.is_alive


def test_container_scope_isolates_registries():
    """Nested scopes get independent registries."""
    with thread_container_scope("outer") as outer:
        outer.spawn(target=_busy_wait, name="outer-t")

        with thread_container_scope("inner") as inner:
            inner.spawn(target=_busy_wait, name="inner-t")
            assert get_thread_container() is inner

        # Inner scope cleaned up, outer is restored
        assert get_thread_container() is outer
        assert len(outer.alive_threads()) == 1


def test_container_scope_restores_previous():
    """After a scope exits, the previous container is restored."""
    with thread_container_scope("first") as first:
        first_ref = get_thread_container()
        assert first_ref is first

        with thread_container_scope("second"):
            pass

        assert get_thread_container() is first
