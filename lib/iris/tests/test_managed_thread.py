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

    registry.shutdown()
    assert not t.is_alive


def test_registry_shutdown_with_timeout():
    """Registry with timeout doesn't block forever on stuck threads."""

    def stubborn(stop_event: threading.Event):
        time.sleep(100)

    registry = ThreadRegistry()
    t = registry.spawn(target=stubborn, name="stuck-thread")

    registry.shutdown(timeout=0.1)
    # Thread is still alive because it ignores stop_event
    assert t.is_alive


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
