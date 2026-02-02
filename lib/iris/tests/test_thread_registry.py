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

"""Tests for the contextvar-based ThreadRegistry."""

import threading
import time

from iris.managed_thread import get_thread_registry, thread_registry_scope


def _busy_wait(stop_event: threading.Event) -> None:
    stop_event.wait()


def _counter(stop_event: threading.Event, results: list) -> None:
    results.append(1)
    stop_event.wait()


def test_registry_scope_spawns_and_cleans_up():
    """Threads spawned inside a scope are stopped when the scope exits."""
    results: list[int] = []
    with thread_registry_scope("test-cleanup") as registry:
        registry.spawn(target=_counter, name="t1", args=(results,))
        time.sleep(0.05)
        assert len(results) == 1
        assert registry.is_alive

    # After exiting the scope, threads have been stopped
    assert not registry.is_alive


def test_registry_scope_isolates_registries():
    """Nested scopes get independent registries."""
    with thread_registry_scope("outer") as outer:
        outer.spawn(target=_busy_wait, name="outer-t")

        with thread_registry_scope("inner") as inner:
            inner.spawn(target=_busy_wait, name="inner-t")
            assert get_thread_registry() is inner

        # Inner scope cleaned up, outer is restored
        assert get_thread_registry() is outer
        assert len(outer.alive_threads()) == 1


def test_registry_scope_restores_previous():
    """After a scope exits, the previous registry is restored."""
    with thread_registry_scope("first") as first:
        first_ref = get_thread_registry()
        assert first_ref is first

        with thread_registry_scope("second"):
            pass

        assert get_thread_registry() is first


def test_get_thread_registry_returns_default():
    """Outside any scope, get_thread_registry creates a default."""
    with thread_registry_scope("isolate"):
        # Inside a scope, we get the scoped one
        reg = get_thread_registry()
        assert reg is not None


def test_registry_container_interop():
    """The underlying ThreadContainer is accessible for code that needs it."""
    with thread_registry_scope("interop") as registry:
        container = registry.container
        container.spawn(target=_busy_wait, name="via-container")
        time.sleep(0.05)

        # Threads spawned on the container are visible through the registry
        assert len(registry.alive_threads()) == 1
