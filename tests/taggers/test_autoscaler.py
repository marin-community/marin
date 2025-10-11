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

import time

import pytest
import ray
from ray.util.queue import Queue

from marin.processing.classification.autoscaler import AutoscalingActorPool
from marin.processing.classification.classifier import AutoClassifierRayActor


@pytest.fixture
def ray_runtime():
    ray.shutdown()
    ray.init(num_cpus=2, ignore_reinit_error=True)
    try:
        yield
    finally:
        ray.shutdown()


def test_autoscaler_requeues_failed_tasks(ray_runtime):
    task_queue = Queue()
    result_queue = Queue()

    pool = AutoscalingActorPool(
        AutoClassifierRayActor,
        model_name_or_path="model",
        attribute_name="attr",
        model_type="dummy",
        task_queue=task_queue,
        result_queue=result_queue,
        min_actors=1,
        max_actors=2,
        target_queue_size=1,
        actor_health_check_interval=2.0,
        actor_health_check_timeout=1.0,
    )

    try:
        original_task = {"id": "task-1", "text": "test"}
        task_queue.put(original_task)

        with pool.lock:
            initial_actor = pool.actors[0]
            initial_actor_id = initial_actor._actor_id.hex()

        deadline = time.time() + 5
        task_assigned = False
        while time.time() < deadline:
            with pool.lock:
                pending = pool.actor_futures.get(initial_actor, [])
            if pending:
                task_assigned = True
                break
            time.sleep(0.05)

        assert task_assigned, "Task was not dispatched to the initial actor"

        ray.kill(initial_actor)
        print(f"Killed initial actor: {initial_actor_id}")

        time.sleep(3.0)  # Greater than the actor health check interval so that we see that the initial actor is dead

        with pool.lock:
            print(f"Actor futures: {pool.actor_futures}")

        start = time.time()
        while True:
            try:
                processed_task = result_queue.get(timeout=30)
                break
            except Exception as e:
                print(f"Hit error: {e}")
                print(f"Time that has passed: {time.time() - start}")

        assert processed_task["id"] == original_task["id"]
        # assert processed_task["processed"] is True
        # assert processed_task["actor_id"] != initial_actor_id

        with pytest.raises(ray.exceptions.RayTaskError):
            result_queue.get(timeout=30)  # next one should not succeed since only one task
    finally:
        pool.shutdown()
