#!/usr/bin/env python3
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

"""Demo: Math inference with worker pool and autoscaling.

This example demonstrates how to use WorkerPool to scale LLM inference across
multiple workers. It uses fake LLM workers to simulate solving math problems,
showcasing the autoscaling and queue-based task distribution.

Example:
    $ python -m fray.examples.inference_pool
"""

import logging
import time

from fray.cluster.local_cluster import LocalCluster
from fray.examples.fake_llm_worker import worker_loop
from fray.worker_pool import WorkerPool, WorkerPoolConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run math inference demo with worker pool."""
    logger.info("Starting math inference demo")

    # Sample math problems to solve
    problems = [
        {"id": 1, "problem": "What is 2 + 2?"},
        {"id": 2, "problem": "What is 15 * 3?"},
        {"id": 3, "problem": "What is 100 - 47?"},
        {"id": 4, "problem": "What is 144 / 12?"},
        {"id": 5, "problem": "What is 8^2?"},
        {"id": 6, "problem": "What is sqrt(256)?"},
        {"id": 7, "problem": "What is 7! (factorial)?"},
        {"id": 8, "problem": "What is the sum of integers from 1 to 100?"},
        {"id": 9, "problem": "What is 2^10?"},
        {"id": 10, "problem": "What is the greatest common divisor of 48 and 18?"},
    ]

    # Create local cluster
    logger.info("Creating local cluster")
    cluster = LocalCluster()

    # Configure worker pool with autoscaling
    config = WorkerPoolConfig(
        worker_func=worker_loop,
        min_workers=2,  # Start with 2 workers
        max_workers=5,  # Scale up to 5 workers
        scale_up_threshold=0.8,  # Scale up when >0.8 tasks per worker
        scale_down_threshold=0.2,  # Scale down when <0.2 tasks per worker
        scale_check_interval=2.0,  # Check every 2 seconds
    )

    logger.info(f"Creating worker pool: min={config.min_workers}, max={config.max_workers}")
    pool = WorkerPool(
        cluster=cluster,
        config=config,
    )

    try:
        # Process problems in batches to demonstrate autoscaling
        logger.info(f"Processing {len(problems)} math problems")

        # Submit all problems (should trigger scale up)
        start_time = time.time()
        for problem in problems:
            pool.submit(problem)

        results = pool.collect(num_results=len(problems), timeout=60.0)
        duration = time.time() - start_time

        # Display results
        logger.info(f"\nResults (processed in {duration:.2f}s):")
        for result in sorted(results, key=lambda r: r["problem_id"]):
            problem = next(p for p in problems if p["id"] == result["problem_id"])
            logger.info(f"  Problem {result['problem_id']}: {problem['problem']}")
            logger.info(f"  Answer: {result['answer']}")

        # Show pool metrics
        logger.info("\nPool metrics:")
        logger.info(f"  Total problems processed: {len(results)}")
        logger.info(f"  Average time per problem: {duration / len(results):.2f}s")
        logger.info(f"  Final worker count: {pool.num_workers()}")

    finally:
        logger.info("\nShutting down worker pool")
        pool.shutdown(timeout=10.0)
        logger.info("Demo complete")


if __name__ == "__main__":
    main()
