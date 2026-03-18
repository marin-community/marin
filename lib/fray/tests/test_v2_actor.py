# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for fray v2 actor support via LocalClient."""

import threading

import pytest

from fray.v2 import LocalClient, current_actor


class Counter:
    def __init__(self, start: int = 0):
        self._value = start

    def increment(self, amount: int = 1) -> int:
        self._value += amount
        return self._value

    def get(self) -> int:
        return self._value


class Adder:
    def add(self, a: int, b: int) -> int:
        return a + b


class SelfAwareActor:
    def actor_identity(self) -> tuple[int, str]:
        ctx = current_actor()
        return (ctx.index, ctx.group_name)

    def terminate(self) -> str:
        current_actor().terminate()
        return "terminating"


class ConcurrentTerminateActor:
    def __init__(self):
        self._terminate_requested = threading.Event()
        self._allow_finish = threading.Event()

    def request_terminate(self) -> str:
        current_actor().terminate()
        self._terminate_requested.set()
        assert self._allow_finish.wait(timeout=5.0)
        return "terminating"

    def ping(self) -> str:
        assert self._terminate_requested.wait(timeout=5.0)
        return "pong"

    def allow_finish(self) -> None:
        self._allow_finish.set()


class ProbeActor:
    def ping(self) -> str:
        return "pong"

    def shutdown(self) -> None:
        pass


@pytest.fixture
def client():
    c = LocalClient(max_threads=4)
    yield c
    c.shutdown(wait=True)


def test_create_actor_and_call_remote(client: LocalClient):
    actor = client.create_actor(Counter, name="counter")
    result = actor.increment.remote(5).result()
    assert result == 5
    assert actor.get.remote().result() == 5


def test_create_actor_synchronous_call(client: LocalClient):
    actor = client.create_actor(Counter, start=10, name="counter")
    assert actor.get() == 10
    actor.increment(3)
    assert actor.get() == 13


def test_create_actor_with_args(client: LocalClient):
    actor = client.create_actor(Counter, 42, name="counter")
    assert actor.get.remote().result() == 42


def test_actor_group_create_and_wait_ready(client: LocalClient):
    group = client.create_actor_group(Counter, name="counters", count=3)
    assert group.ready_count == 3
    handles = group.wait_ready()
    assert len(handles) == 3

    for i, h in enumerate(handles):
        h.increment.remote(i + 1).result()

    values = [h.get.remote().result() for h in handles]
    assert values == [1, 2, 3]


def test_actor_group_wait_ready_partial(client: LocalClient):
    group = client.create_actor_group(Counter, name="counters", count=5)
    handles = group.wait_ready(count=2)
    assert len(handles) == 2


def test_actor_group_shutdown(client: LocalClient):
    group = client.create_actor_group(Counter, name="counters", count=2)
    handles = group.wait_ready()
    assert len(handles) == 2
    group.shutdown()
    # Shutdown completes without error - no statuses() in protocol


def test_concurrent_remote_calls_thread_safety(client: LocalClient):
    """Multiple threads calling .remote() on the same actor should be safe."""
    actor = client.create_actor(Counter, name="counter")
    num_threads = 10
    results: list[int] = [0] * num_threads
    barrier = threading.Barrier(num_threads)

    def worker(idx: int):
        barrier.wait()
        results[idx] = actor.increment.remote(1).result()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each increment adds 1, so final value should be num_threads.
    # Individual results should be unique values 1..num_threads (serialized by lock).
    assert actor.get() == num_threads
    assert sorted(results) == list(range(1, num_threads + 1))


def test_actor_method_with_kwargs(client: LocalClient):
    actor = client.create_actor(Adder, name="adder")
    assert actor.add.remote(a=3, b=4).result() == 7
    assert actor.add(a=10, b=20) == 30


def test_actor_methods_run_with_current_actor_context(client: LocalClient):
    actor = client.create_actor(SelfAwareActor, name="self-aware")
    assert actor.actor_identity.remote().result() == (0, "self-aware")


def test_actor_can_request_backend_termination(client: LocalClient):
    actor = client.create_actor(SelfAwareActor, name="self-aware")
    assert actor.terminate.remote().result() == "terminating"
    with pytest.raises(RuntimeError, match="Actor not found"):
        actor.actor_identity.remote()


def test_actor_termination_is_owned_by_requesting_call(client: LocalClient):
    actor = client.create_actor(ConcurrentTerminateActor, name="self-aware")
    future = actor.request_terminate.remote()

    assert actor.ping() == "pong"
    actor.allow_finish()

    assert future.result() == "terminating"
    with pytest.raises(RuntimeError, match="Actor not found"):
        actor.ping.remote()


def test_stale_group_shutdown_does_not_kill_same_endpoint_replacement(client: LocalClient):
    original_group = client.create_actor_group(ProbeActor, name="probe", count=1)
    original_group.wait_ready()

    replacement_group = client.create_actor_group(ProbeActor, name="probe", count=1)
    replacement = replacement_group.wait_ready()[0]

    original_group.shutdown()

    assert replacement.ping.remote().result() == "pong"
