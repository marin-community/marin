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

"""
Script to run an evaluator on a model checkpoint using the Fray-based inference pool.

Usage:

python3 run.py <Name of evaluator> --model <Path to model or Hugging Face model name> \
--evals <List of evals to run> --output-path <Where to output logs and results>
"""

import logging
import os
import time

import draccus
from fray.cluster.local_cluster import LocalCluster
from fray.queue.file import FileQueue

from marin.evaluation.evaluation_config import EvaluationConfig, ModelConfig
from marin.evaluation.evaluators.evaluator import Evaluator
from marin.evaluation.evaluators.evaluator_factory import get_evaluator
from marin.evaluation.inference_pool import InferencePool
from marin.evaluation.utils import discover_hf_checkpoints
from marin.evaluation.vllm import InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)


def evaluate(config: EvaluationConfig) -> None:
    """Run evaluation using the Fray-based inference pool.

    This function:
    1. Creates a Fray cluster and queues
    2. Starts the inference pool with VLLM servers
    3. Runs the evaluator with the pool's OpenAI-compatible API
    4. Shuts down the pool and cluster
    """
    logger.info(f"Running evals with args: {config}")

    # Create model config
    model: ModelConfig = _impute_model_config(config)
    logger.info(f"Evaluating {model.name} with {config.evals}")

    # Create Fray cluster
    logger.info("Creating Fray cluster")
    cluster = LocalCluster()

    # Create queues for request/response
    queue_dir = os.path.join("/tmp", f"inference-pool-{int(time.time())}")
    logger.info(f"Creating queues in {queue_dir}")
    request_queue = FileQueue[InferenceRequest](path=os.path.join(queue_dir, "requests"))
    response_queue = FileQueue[InferenceResponse](path=os.path.join(queue_dir, "responses"))

    # Create and start inference pool
    logger.info("Creating inference pool")
    pool = InferencePool(
        config=config.pool_config,
        cluster=cluster,
        request_queue=request_queue,
        response_queue=response_queue,
    )

    try:
        start_time = time.time()

        logger.info("Starting inference pool")
        pool.start()

        logger.info("Waiting for pool to be healthy")
        pool.wait_for_healthy()

        # Get OpenAI base URL from pool
        openai_base_url = pool.base_url()
        logger.info(f"Pool ready at {openai_base_url}")

        # Create and run evaluator
        evaluator: Evaluator = get_evaluator(config)
        evaluator.evaluate(
            model=model,
            evals=config.evals,
            openai_base_url=openai_base_url,
            output_path=config.evaluation_path,
            max_eval_instances=config.max_eval_instances,
            wandb_tags=config.wandb_tags,
        )

        logger.info(f"Evaluation complete (total time: {time.time() - start_time:.1f}s)")

    finally:
        logger.info("Shutting down inference pool")
        pool.shutdown()
        cluster.shutdown()


def _impute_model_config(config):
    """Create ModelConfig from EvaluationConfig.

    Imputes model name from path if not provided and extracts engine_kwargs
    from pool_config.model_config.
    """
    model_path = config.model_path

    if config.model_path is None:
        raise ValueError("model_name or model_path must be provided")

    if config.discover_latest_checkpoint:
        model_path = discover_hf_checkpoints(model_path)[-1]

    if config.model_name is None and "gcsfuse" in model_path:
        model_name = model_path.split("/")[-1]
    elif config.model_name is None:
        # have to impute the model name from the path
        model_name_parts = model_path.split("/")
        # we're looking for something that looks like a run name and something that looks like a step
        # e.g. $RUN/hf/step-$STEP
        step_part = model_name_parts[-1]
        if step_part.startswith("step-"):
            step_part = step_part.split("-")[1]

        # don't assume there's an hf. look for a run name, which probably has a - in it
        for part in reversed(model_name_parts[:-1]):
            if "-" in part:
                model_name = part
                break
        else:
            # just use the penultimate part
            model_name = model_name_parts[-2]

        model_name = f"{model_name}-{step_part}"
    else:
        model_name = config.model_name

    # Use model config from pool_config
    return config.pool_config.model_config


@draccus.wrap()
def main(config: EvaluationConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
