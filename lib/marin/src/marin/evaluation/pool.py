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

import logging
import os
import tempfile
from pathlib import Path

from fray import Cluster, Entrypoint, EnvironmentConfig, FileQueue, JobId, JobRequest, LocalCluster

from marin.evaluation.types import EvaluationConfig, InferenceRequest, InferenceResult

logger = logging.getLogger(__name__)

# An inference pool manages a set of workers, which each run an inference server and listen on a request/response queue
# The inference pool creates the initial queues and worker pool, then provides a map() api to submit tasks to the workers
# Individual tasks create an inference pool object and then iterate over their tasks in batches as needed.


class InferencePool:
    workers: list[JobId]
    queue_dir: tempfile.TemporaryDirectory
    cluster: Cluster
    task_queue: FileQueue
    result_queue: FileQueue

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.workers = []

    def __enter__(self) -> "InferencePool":
        self.queue_dir = tempfile.TemporaryDirectory("marin-eval-queues")
        self.cluster = LocalCluster(working_dir=Path(os.getcwd()), queue_dir=Path(self.queue_dir.name))
        self.task_queue = FileQueue(path=str(Path(self.queue_dir.name) / "tasks"))
        self.result_queue = FileQueue(path=str(Path(self.queue_dir.name) / "results"))

        if self.config.evaluator == "levanter":
            from marin.evaluation.levanter import LevanterWorker

            worker_impl = LevanterWorker
        elif self.config.evaluator == "vllm":
            from marin.evaluation.vllm import VllmWorker

            worker_impl = VllmWorker
        elif self.config.evaluator == "transformers":
            from marin.evaluation.transformers import TransformersWorker

            worker_impl = TransformersWorker
        else:
            raise ValueError(f"Unknown evaluator: {self.config.evaluator}")

        def _worker_closure():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
            worker_impl(self.config.model_config)

        job_request = JobRequest(
            name="marin-inference-worker",
            resources=self.config.worker_resources,
            entrypoint=Entrypoint(callable=_worker_closure),
            environment=EnvironmentConfig(workspace="."),
        )
        self.workers.append(self.cluster.launch(job_request))
        return self

    def map(self, tasks: list[InferenceRequest]) -> list[InferenceResult]:
        for task in tasks:
            logger.info(f"Pushing: {task}")
            self.task_queue.push(task)

        results = []
        for _ in range(len(tasks)):
            results.append(self.result_queue.pop())
            logger.info(f"Popped: {results[-1]}")
        return results

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.queue_dir.cleanup()
        self.cluster.shutdown()
