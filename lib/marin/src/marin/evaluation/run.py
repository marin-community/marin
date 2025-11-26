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
import time

import draccus
from fray.cluster import current_cluster
from fray.queue.http import HttpQueueServer

from marin.evaluation.backends.inference_pool import InferencePool
from marin.evaluation.evaluation_config import EvaluationConfig, ModelConfig
from marin.evaluation.evaluators.evaluator import Evaluator
from marin.evaluation.evaluators.evaluator_factory import get_evaluator
from marin.evaluation.utils import discover_hf_checkpoints

logger = logging.getLogger(__name__)


def evaluate(config: EvaluationConfig) -> None:
    """Run evaluation using the Fray-based inference pool."""
    logger.info(f"Running evals with args: {config}")
    model: ModelConfig = config.pool_config.model_config

    # Handle checkpoint discovery if needed
    if config.discover_latest_checkpoint and config.model_path:
        discovered_path = discover_hf_checkpoints(config.model_path)[-1]
        logger.info(f"Discovered latest checkpoint: {discovered_path}")
        model = ModelConfig(
            name=model.name,
            path=discovered_path,
            engine_kwargs=model.engine_kwargs,
            device=model.device,
            generation_params=model.generation_params,
            apply_chat_template=model.apply_chat_template,
        )
        # Update pool config with new model config
        from dataclasses import replace

        config = replace(config, pool_config=replace(config.pool_config, model_config=model))

    logger.info(f"Evaluating {model.name} with {config.evals}")
    cluster = current_cluster()
    logger.info(f"Using cluster: {cluster.__class__.__name__}")

    # Use an HTTP queue server to communicate with vllm inference servers
    with HttpQueueServer(port=9999) as queue_server:
        request_queue = queue_server.new_queue("requests")
        response_queue = queue_server.new_queue("responses")
        logger.info(f"Queue server ready at {queue_server.get_client_host()}:{queue_server.port}")

        pool = InferencePool(
            config=config.pool_config,
            cluster=cluster,
            request_queue=request_queue,
            response_queue=response_queue,
        )

        start_time = time.time()
        with pool:
            openai_base_url = pool.base_url()
            logger.info(f"Pool ready at {openai_base_url}")
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


@draccus.wrap()
def main(config: EvaluationConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
