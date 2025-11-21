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
import time
from pathlib import Path

from fray.queues.file import FileQueue
from fray.types import InferenceRequest, InferenceResult
from fray.worker_pool import WorkerPool, WorkerPoolConfig

from marin.evaluation.evaluation_config import EvaluationConfig, ModelConfig

logger = logging.getLogger(__name__)


class Controller:
    """
    Controller for running evaluations using a distributed worker pool.
    """

    def __init__(self, config: EvaluationConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config

    def run(self):
        """Run the evaluation controller."""
        from fray.cluster.local_cluster import LocalCluster

        # Create a temporary directory for queues
        self.temp_dir = tempfile.mkdtemp(prefix="marin-eval-")
        queue_dir = Path(self.temp_dir) / "queues"
        queue_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Using temporary directory: {self.temp_dir}")

        # Setup cluster
        self.cluster = LocalCluster(working_dir=Path(os.getcwd()), queue_dir=queue_dir)

        # 2. Setup Queues
        # We use FileQueue for robust local/remote handling
        self.task_queue = FileQueue(path=str(queue_dir / "tasks"))
        self.result_queue = FileQueue(path=str(queue_dir / "results"))

        # 3. Setup Worker Pool
        # Select worker implementation based on config
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

        # Configure worker pool
        pool_config = WorkerPoolConfig(
            worker_impl=worker_impl,
            num_workers=1,  # Default to 1 for local testing, maybe scale up later
            worker_args=(self.model_config,),  # Pass model config
            worker_kwargs={},
        )

        self.pool = WorkerPool(
            cluster=self.cluster,
            config=pool_config,
            task_queue=self.task_queue,
            result_queue=self.result_queue,
        )

        # 4. Run Evaluation Loop
        try:
            results = self._run_dispatch_loop()
            return results
        finally:
            logger.info("Shutting down worker pool...")
            self.pool.shutdown()

    def _run_dispatch_loop(self) -> dict[str, InferenceResult]:
        """Dispatch tasks and collect results with backpressure."""
        pending_requests = set()
        results = {}

        # Flatten all tasks into a list of requests
        all_requests = []
        for eval_task in self.config.evals:
            if hasattr(eval_task, "prompts"):
                for i, prompt in enumerate(eval_task.prompts):
                    req_id = f"{eval_task.name}_{i}"
                    request = InferenceRequest(request_id=req_id, prompt=prompt)
                    all_requests.append(request)
            else:
                logger.warning(f"Task {eval_task.name} has no prompts to evaluate.")

        total_tasks = len(all_requests)
        logger.info(f"Total tasks to process: {total_tasks}")

        task_idx = 0
        max_pending = 10  # Backpressure limit

        while len(results) < total_tasks:
            # 1. Submit tasks up to max_pending
            while task_idx < total_tasks and len(pending_requests) < max_pending:
                request = all_requests[task_idx]
                self.task_queue.push(request)
                pending_requests.add(request.request_id)
                task_idx += 1
                logger.debug(f"Submitted task {request.request_id}. Pending: {len(pending_requests)}")

            # 2. Collect results
            # We use a short timeout to allow loop to check for new submissions
            lease = self.result_queue.pop(lease_timeout=30.0)

            if lease:
                result = lease.item
                self.result_queue.done(lease)

                if result.request_id in pending_requests:
                    results[result.request_id] = result
                    pending_requests.remove(result.request_id)
                    logger.info(f"Received result for {result.request_id}. Progress: {len(results)}/{total_tasks}")
                else:
                    logger.warning(f"Received unexpected result: {result.request_id}")
            else:
                # No results yet, sleep briefly to avoid busy loop if queue is empty but pending is full
                if len(pending_requests) == max_pending:
                    time.sleep(0.1)

        logger.info("All tasks completed.")
        return results
