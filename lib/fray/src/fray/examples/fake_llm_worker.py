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

"""Example worker that simulates LLM inference with fake responses.

This worker is used for testing the WorkerPool without requiring actual LLM APIs.
It processes math problems from a task queue and returns mock answers.
"""

import logging
import random
import time

import click

from fray.cluster import current_cluster

logger = logging.getLogger(__name__)


def process_task(task: dict) -> dict:
    # Simulate processing time
    time.sleep(random.uniform(0.1, 0.5))

    # Generate a fake answer
    problem_id = task["id"]
    problem = task["problem"]

    # Simple mock answer generation
    answer = f"The answer to '{problem}' is 42 (simulated)"

    return {
        "problem_id": problem_id,
        "answer": answer,
    }


def worker_loop(task_queue_name: str, result_queue_name: str) -> None:
    """Main worker loop that processes tasks from queue.

    Args:
        task_queue_name: Name of the task queue to consume from
        result_queue_name: Name of the result queue to publish to
    """
    logger.info(f"Worker starting: task_queue={task_queue_name}, result_queue={result_queue_name}")

    # Get queue connections from current cluster
    cluster = current_cluster()
    task_queue = cluster.create_queue(name=task_queue_name)
    result_queue = cluster.create_queue(name=result_queue_name)

    # Process tasks until interrupted
    while True:
        # Try to get a task
        lease = task_queue.pop()
        logger.info("Popped task lease from queue")

        if lease is None:
            # No tasks available, sleep and retry
            time.sleep(0.1)
            continue

        try:
            # Process the task
            task = lease.item
            logger.info(f"Processing task: {task}")
            result = process_task(task)

            # Publish result
            result_queue.push(result)
            logger.info(f"Published result: {result}")

            # Mark task as complete
            task_queue.done(lease)

        except Exception as e:
            logger.error(f"Error processing task: {e}")
            # Release the lease so task can be retried
            task_queue.release(lease)


@click.command()
@click.option("--task-queue", required=True, help="Name of task queue")
@click.option("--result-queue", required=True, help="Name of result queue")
def main(task_queue: str, result_queue: str):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        worker_loop(task_queue, result_queue)
    except KeyboardInterrupt:
        logger.info("Worker shutting down")


if __name__ == "__main__":
    main()
