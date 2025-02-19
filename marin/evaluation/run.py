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
from marin.evaluation.evaluators.evaluator_factory import get_evaluator
from marin.evaluation.utils import discover_hf_checkpoints

logger = logging.getLogger(__name__)


def evaluate(config: EvaluationConfig) -> None:
    logger.info(f"Running evals with args: {config}")
    evaluator: Evaluator = get_evaluator(config)

    model: ModelConfig = _impute_model_config(config)
    logger.info(f"Evaluating {model.name} with {config.evals}")

    start_time: float = time.time()
    if config.launch_with_ray:
        evaluator.launch_evaluate_with_ray(
            model,
            evals=config.evals,
            output_path=config.evaluation_path,
            max_eval_instances=config.max_eval_instances,
            resource_config=config.resource_config,
        )
    else:
        evaluator.evaluate(
            model,
            evals=config.evals,
            output_path=config.evaluation_path,
            max_eval_instances=config.max_eval_instances,
        )

    logger.info(f"Done (total time: {time.time() - start_time} seconds)")


def _impute_model_config(config):
    model_path = config.model_path

    if config.model_name is None:
        # have to impute the model name from the path
        if config.model_path is None:
            raise ValueError("model_name or model_path must be provided")

        if config.discover_latest_checkpoint:
            model_path = discover_hf_checkpoints(model_path)[-1]

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

    return ModelConfig(name=model_name, path=model_path, engine_kwargs=config.engine_kwargs)


@draccus.wrap()
def main(config: EvaluationConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
