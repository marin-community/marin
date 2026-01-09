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
Script to run an evaluator on a model checkpoint.

Usage:

python3 run.py <Name of evaluator> --model <Path to model or Hugging Face model name> \
--evals <List of evals to run> --output-path <Where to output logs and results>
"""

import logging
import time

import draccus

from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.evaluators.levanter_lm_eval_evaluator import LevanterLmEvalEvaluator
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import LMEvaluationHarnessEvaluator
from marin.evaluation.evaluators.simple_evaluator import SimpleEvaluator
from marin.evaluation.utils import discover_hf_checkpoints

logger = logging.getLogger(__name__)

EVALUATORS = {
    "lm_evaluation_harness": LMEvaluationHarnessEvaluator,
    "levanter_lm_evaluation_harness": LevanterLmEvalEvaluator,
    "debug": SimpleEvaluator,
}


def get_evaluator(config: EvaluationConfig) -> Evaluator:
    if config.evaluator not in EVALUATORS:
        raise ValueError(f"Unknown evaluator: {config.evaluator}. Available: {list(EVALUATORS.keys())}")
    return EVALUATORS[config.evaluator]()


def evaluate(config: EvaluationConfig) -> None:
    logger.info(f"Running evals with args: {config}")
    evaluator: Evaluator = get_evaluator(config)

    model: ModelConfig = _impute_model_config(config)
    logger.info(f"Evaluating {model.name} with {config.evals}")

    start_time: float = time.time()
    if config.launch_with_ray:
        # Pass generation_kwargs only for Levanter evaluator
        if config.evaluator == "levanter_lm_evaluation_harness":
            evaluator.launch_evaluate_with_ray(
                model,
                evals=config.evals,
                output_path=config.evaluation_path,
                max_eval_instances=config.max_eval_instances,
                resource_config=config.resource_config,
                wandb_tags=config.wandb_tags,
                max_length=config.max_length,
                generation_kwargs=config.generation_params,
            )
        else:
            evaluator.launch_evaluate_with_ray(
                model,
                evals=config.evals,
                output_path=config.evaluation_path,
                max_eval_instances=config.max_eval_instances,
                resource_config=config.resource_config,
                wandb_tags=config.wandb_tags,
                max_length=config.max_length,
                generation_params=config.generation_params,
            )
    else:
        # Pass generation_kwargs only for Levanter evaluator
        if config.evaluator == "levanter_lm_evaluation_harness":
            evaluator.evaluate(
                model,
                evals=config.evals,
                output_path=config.evaluation_path,
                max_eval_instances=config.max_eval_instances,
                wandb_tags=config.wandb_tags,
                max_length=config.max_length,
                generation_kwargs=config.generation_params,
            )
        else:
            evaluator.evaluate(
                model,
                evals=config.evals,
                output_path=config.evaluation_path,
                max_eval_instances=config.max_eval_instances,
                wandb_tags=config.wandb_tags,
                max_length=config.max_length,
            )

    logger.info(f"Done (total time: {time.time() - start_time} seconds)")


def _impute_model_config(config):
    model_path = config.model_path

    if config.model_path is None:
        raise ValueError("model_name or model_path must be provided")

    if config.discover_latest_checkpoint:
        discovered_checkpoints = discover_hf_checkpoints(model_path)
        if discovered_checkpoints:
            model_path = discovered_checkpoints[-1]
        else:
            # If no HF checkpoints found, assume the path is already a valid model path
            logger.info(f"No HF checkpoints found in {model_path}, using path as-is")

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
    generation_params = {}
    engine_kwargs = {}
    if config.generation_params is None:
        logger.warning(f"No generation params provided for {model_name}, using default params")
    else:
        generation_params = config.generation_params
    if config.engine_kwargs is None:
        logger.warning(f"No engine kwargs provided for {model_name}, using default params")
    else:
        engine_kwargs = config.engine_kwargs

    return ModelConfig(
        name=model_name,
        path=model_path,
        engine_kwargs=engine_kwargs,
        generation_params=generation_params,
        apply_chat_template=config.apply_chat_template,
    )


@draccus.wrap()
def main(config: EvaluationConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
