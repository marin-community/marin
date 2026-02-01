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

"""Tests for ManagedThread and ThreadRegistry."""

import threading
import time


from iris.managed_thread import ManagedThread, ThreadRegistry


def test_managed_thread_start_stop():
    """Thread starts, runs until stop is signaled, then exits."""
    entered = threading.Event()

    def target(stop_event: threading.Event):
        entered.set()
        stop_event.wait()

    t = ManagedThread(target=target, name="test-thread")
    t.start()
    assert entered.wait(timeout=2.0)
    assert t.is_alive

    t.stop()
    t.join(timeout=2.0)
    assert not t.is_alive


def test_managed_thread_passes_args():
    """Extra args are forwarded after stop_event."""
    result = []

    def target(stop_event: threading.Event, a: int, b: str):
        result.extend([a, b])

    t = ManagedThread(target=target, name="args-thread", args=(42, "hello"))
    t.start()
    t.join(timeout=2.0)
    assert result == [42, "hello"]


def test_managed_thread_name():
    t = ManagedThread(target=lambda stop_event: None, name="my-thread")
    assert t.name == "my-thread"


def test_registry_spawn_and_shutdown():
    """Registry spawns threads and shuts them all down."""
    entered = threading.Event()

    def target(stop_event: threading.Event):
        entered.set()
        stop_event.wait()

    registry = ThreadRegistry()
    t = registry.spawn(target=target, name="reg-thread")
    assert entered.wait(timeout=2.0)
    assert t.is_alive

    stuck = registry.shutdown(timeout=5.0)
    assert stuck == []
    assert not t.is_alive


def test_registry_reports_stuck_threads():
    """Registry returns names of threads that don't exit within timeout."""

    def stubborn(stop_event: threading.Event):
        # Ignores stop event, sleeps forever
        time.sleep(100)

    registry = ThreadRegistry()
    registry.spawn(target=stubborn, name="stuck-thread")

    stuck = registry.shutdown(timeout=0.1)
    assert stuck == ["stuck-thread"]


def test_registry_context_manager():
    """Context manager calls shutdown on exit."""
    entered = threading.Event()

    def target(stop_event: threading.Event):
        entered.set()
        stop_event.wait()

    with ThreadRegistry() as registry:
        t = registry.spawn(target=target, name="ctx-thread")
        assert entered.wait(timeout=2.0)

    assert not t.is_alive


def test_registry_multiple_threads():
    """Registry handles multiple threads."""
    barrier = threading.Barrier(4)  # 3 threads + main
    started_count = [0]
    lock = threading.Lock()

    def target(stop_event: threading.Event, idx: int):
        with lock:
            started_count[0] += 1
        barrier.wait(timeout=2.0)
        stop_event.wait()

    with ThreadRegistry() as registry:
        for i in range(3):
            registry.spawn(target=target, name=f"thread-{i}", args=(i,))
        barrier.wait(timeout=2.0)
        assert started_count[0] == 3
