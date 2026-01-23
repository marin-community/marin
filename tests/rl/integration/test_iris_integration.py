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

"""Integration test: Iris actors running under Ray infrastructure.

Validates the hybrid approach from issue #2365:
- Ray handles task launching (scheduling, resource management)
- Iris handles actor communication (gRPC via ActorServer/ActorClient)

Uses the real Curriculum class to ensure representative-ish coverage.
"""
import itertools
import logging
import os

import pytest
import ray

from iris.actor import ActorClient, ActorServer
from iris.actor.resolver import FixedResolver
from marin.rl.curriculum import Curriculum, CurriculumConfig, LessonConfig
from marin.rl.environments.base import EnvConfig
from marin.rl.types import RolloutStats

pytestmark = pytest.mark.skipif(os.environ.get("CI"), reason="Skipping integration tests on CI environment")


def _create_test_curriculum_config() -> CurriculumConfig:
    """Minimal curriculum config with 3 independent lessons."""
    lessons = {
        name: LessonConfig(lesson_id=name, env_config=EnvConfig(env_class="test.FakeEnv", env_args={}))
        for name in ("easy", "medium", "hard")
    }
    return CurriculumConfig(lessons=lessons, max_seq_len=42)


@ray.remote
class IrisActorHost:
    """Ray actor that hosts an Iris ActorServer with a Curriculum instance."""

    def __init__(self, config: CurriculumConfig):
        self._server = ActorServer(host="localhost")
        self._server.register("curriculum", Curriculum(config))
        self._server.serve_background()  # Runs and assing port

    def get_address(self) -> str:
        return f"http://{self._server.address}"


@ray.remote
def iris_worker_task(actor_address: str, worker_id: int, num_iterations: int) -> dict:
    """Ray task that communicates with the Curriculum via Iris gRPC."""
    resolver = FixedResolver({"curriculum": [actor_address]})
    client = ActorClient(resolver, "curriculum")

    lessons_sampled = []
    for i in range(num_iterations):
        step = worker_id * 1000 + i  # unique
        lesson = client.sample_lesson(seed=step)
        lessons_sampled.append(lesson)

        stats = [
            RolloutStats(
                episode_reward=0.42, env_example_id=f"ex:{worker_id}/{i}", lesson_id=lesson, temperature=1.0, top_k=None
            )
        ]
        client.update_lesson_stats(stats, mode="training", current_step=step)

    metrics = client.get_metrics()
    return {"worker_id": worker_id, "lessons": lessons_sampled, "metrics": metrics}


@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        logging.info("Initializing Ray cluster")
        ray.init(
            address="local",
            num_cpus=8,
            ignore_reinit_error=True,
            logging_level=logging.INFO,
            log_to_driver=True,
            resources={"head_node": 1},
        )
        yield


def test_curriculum_via_iris_under_ray(ray_cluster):
    """Multiple Ray tasks communicate with Iris-hosted Curriculum."""
    # Launch Iris actor host (Ray actor with gRPC server)
    host = IrisActorHost.remote(_create_test_curriculum_config())
    address = ray.get(host.get_address.remote())

    # Launch workers that use Iris ActorClient
    num_workers = 10
    num_iterations = 4
    worker_refs = [
        iris_worker_task.remote(address, worker_id=i, num_iterations=num_iterations) for i in range(num_workers)
    ]
    assert len(worker_refs) == num_workers
    results = ray.get(worker_refs)
    assert len(results) == num_workers

    # Verify: all workers got valid lesson IDs
    all_lessons = list(itertools.chain.from_iterable(r["lessons"] for r in results))
    assert len(all_lessons) == num_workers * num_iterations
    assert set(all_lessons) == {"easy", "medium", "hard"}

    # Verify: final metrics show the curriculum is functional
    final_metrics = results[-1]["metrics"]
    assert final_metrics["active_lessons"] == 3
    assert final_metrics["mean_success"] is not None
