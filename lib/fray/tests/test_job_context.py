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

"""Tests for execution contexts."""

import pytest
from fray.v1.job import RayContext, SimpleActor, SyncContext, ThreadContext, create_job_ctx


@pytest.fixture(params=["sync", "threadpool", "ray"])
def job_context(request, ray_cluster):
    if request.param == "sync":
        return SyncContext()
    elif request.param == "threadpool":
        return ThreadContext(max_workers=2)

    return RayContext()


def test_context_put_get(job_context):
    obj = {"key": "value"}
    ref = job_context.put(obj)
    assert job_context.get(ref) == obj


def test_context_run(job_context):
    future = job_context.run(lambda x: x * 2, 5)
    assert job_context.get(future) == 10


def test_context_wait(job_context):
    futures = [job_context.run(lambda x: x, i) for i in range(5)]
    ready, pending = job_context.wait(futures, num_returns=2)
    assert len(ready) == 2
    assert len(pending) == 3


def test_fray_job_ctx_invalid():
    with pytest.raises(ValueError, match="Unknown context type"):
        create_job_ctx("invalid")  # type: ignore


def test_actor_named_get_if_exists(job_context):
    actor1 = job_context.create_actor(SimpleActor, 100, name="test_actor", get_if_exists=True)
    future1 = actor1.increment.remote(10)
    job_context.get(future1)

    actor2 = job_context.create_actor(SimpleActor, 999, name="test_actor", get_if_exists=True)
    future2 = actor2.increment.remote(0)
    assert job_context.get(future2) == 110


def test_actor_thread_safety(job_context):
    actor = job_context.create_actor(SimpleActor, 0)

    futures = [actor.increment.remote(1) for _ in range(100)]
    [job_context.get(f) for f in futures]

    final_value = actor.get_value.remote()
    assert job_context.get(final_value) == 100


def test_actor_integration_with_put_get_wait(job_context):
    actor = job_context.create_actor(SimpleActor, 0)

    futures = [actor.increment.remote(1) for _ in range(10)]
    ready, pending = job_context.wait(futures, num_returns=5)
    assert len(ready) == 5
    assert len(pending) == 5
    results = [job_context.get(f) for f in ready]
    assert all(isinstance(r, int) for r in results)

    # Wait for remaining
    ready2, pending2 = job_context.wait(pending, num_returns=len(pending))
    assert len(ready2) == 5
    assert len(pending2) == 0


def test_actor_unnamed_isolation(job_context):
    """Test that unnamed actors are isolated instances."""
    actor1 = job_context.create_actor(SimpleActor, 10)
    actor2 = job_context.create_actor(SimpleActor, 20)

    job_context.get(actor1.increment.remote(5))
    job_context.get(actor2.increment.remote(3))

    assert job_context.get(actor1.get_value.remote()) == 15
    assert job_context.get(actor2.get_value.remote()) == 23


def test_actor_named_without_get_if_exists(job_context):
    """Test that named actors without get_if_exists create new instances."""
    actor1 = job_context.create_actor(SimpleActor, 10, name="actor", get_if_exists=False)
    job_context.get(actor1.increment.remote(5))

    with pytest.raises(ValueError):
        job_context.create_actor(SimpleActor, 20, name="actor", get_if_exists=False)
