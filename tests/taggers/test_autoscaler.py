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


def test_autoscaler_requeues_failed_tasks(ray_tpu_cluster):
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
    )

    try:
        original_task = {"id": "task-1", "text": "test"}
        task_queue.put(original_task)

        with pool.actor_task_metadata_lock:
            initial_actor = pool.actors[0]
            initial_actor_id = initial_actor._actor_id.hex()

        deadline = time.time() + 15
        task_assigned = False
        while time.time() < deadline:
            with pool.actor_task_metadata_lock:
                pending = pool.actor_futures.get(initial_actor, [])
            if pending or result_queue.qsize() > 0:  # pending task or finished task
                task_assigned = True
                break
            time.sleep(0.05)

        assert task_assigned, "Task was not dispatched to the initial actor"

        ray.kill(initial_actor)
        print(f"Killed initial actor: {initial_actor_id}")

        time.sleep(3.0)  # Greater than the actor health check interval so that we see that the initial actor is dead

        with pool.actor_task_metadata_lock:
            print(f"Actor futures: {pool.actor_futures}")

        while True:  # Grab the task done
            processed_task = result_queue.get(timeout=30)
            break

        print("Finished processing first task")

        assert processed_task["id"] == original_task["id"]

        with pytest.raises(ray.exceptions.RayTaskError):
            result_queue.get(timeout=5)  # next one should not succeed since only one task
    finally:
        pool.shutdown()
