# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression test for Ray task_manager assertion during fuzzy dedup.

The crash (before fix):
  task_manager.cc:983: Check failed: it != submissible_tasks_.end()
  Tried to complete task that was not pending ...

Root cause: RayActorGroup.shutdown() fell back to ray.kill() which races
with task completion callbacks (ray-project/ray#54260).  Workers' daemon
threads (heartbeat, polling) kept sending tasks to the coordinator during
termination, preventing __ray_terminate__ from completing in time.

Fix: (1) removed ray.kill() fallback, (2) reordered shutdown so coordinator
signals SHUTDOWN before workers receive __ray_terminate__.
"""

import logging
import os
import tempfile

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
os.environ["MARIN_CI_DISABLE_RUNTIME_ENVS"] = "1"

import pytest
import ray
from fray.v2 import ResourceConfig
from fray.v2.client import set_current_client
from fray.v2.ray_backend.backend import RayClient
from zephyr import write_jsonl_file

from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(
            address="local",
            num_cpus=8,
            ignore_reinit_error=True,
            logging_level="info",
            log_to_driver=True,
            resources={"head_node": 1},
        )
    yield


@pytest.fixture(scope="module")
def ray_client(ray_cluster):
    client = RayClient()
    yield client
    client.shutdown(wait=True)


@pytest.fixture
def fox_corpus_jsonl():
    """Minimal corpus with duplicate documents to trigger multi-iteration CC."""
    test = [
        {
            "id": "gray_dup_1",
            "text": (
                "Gray climbing specialists ascend vegetation using retractable talons.\n"
                "They frequently perch on elevated branches throughout daylight hours."
            ),
        },
        {
            "id": "gray_dup_2",
            "text": (
                "Gray climbing specialists ascend vegetation using retractable talons.\n"
                "They frequently perch on elevated branches throughout daylight hours."
            ),
        },
        {
            "id": "gray_dup_3",
            "text": (
                "Gray climbing specialists ascend vegetation using retractable talons.\n"
                "They frequently perch on elevated branches throughout daylight hours."
            ),
        },
        {
            "id": "contaminated_1",
            "text": "Arctic predators have superior auditory capabilities for hunting beneath snow.",
        },
        {
            "id": "contaminated_2",
            "text": "Arctic predators have superior auditory capabilities for hunting beneath thick snow.",
        },
        {
            "id": "unique_1",
            "text": "Desert mammals possess oversized pinnae for thermal regulation.",
        },
        {
            "id": "unique_2",
            "text": "Rapid runners represent the most diminutive wild dogs on the planet.",
        },
    ]

    with tempfile.TemporaryDirectory() as test_dir, tempfile.TemporaryDirectory() as output_dir:
        for i, shard_docs in enumerate([test[:4], test[4:]]):
            write_jsonl_file(shard_docs, os.path.join(test_dir, f"shard_{i}.jsonl.gz"))
        yield {"test_dir": test_dir, "output_dir": output_dir}


@pytest.mark.timeout(120)
def test_fuzzy_dedup_on_ray(ray_client, fox_corpus_jsonl):
    """Run fuzzy dedup with Ray backend — exercises multi-pipeline CC flow.

    Connected components calls ctx.execute() multiple times, each creating
    and tearing down coordinator + worker actors.  Before the fix, this
    triggered ray-project/ray#54260 due to ray.kill() racing with task
    completion callbacks during actor shutdown.
    """
    with set_current_client(ray_client):
        result = dedup_fuzzy_document(
            input_paths=fox_corpus_jsonl["test_dir"],
            output_path=fox_corpus_jsonl["output_dir"],
            max_parallelism=4,
            worker_resources=ResourceConfig(cpu=1, ram="1g"),
        )
    assert result["success"]
