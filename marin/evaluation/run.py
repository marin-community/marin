"""
Script to run an evaluator on a model checkpoint.

Usage:

python3 run.py <Name of evaluator> --model <Path to model or Hugging Face model name> \
--evals <List of evals to run> --output-path <Where to output logs and results>
"""

import time

import draccus

from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.evaluators.evaluator_factory import get_evaluator


def evaluate(config: EvaluationConfig) -> None:
    print(f"Running evals with args: {config}")
    evaluator: Evaluator = get_evaluator(config)

    model: ModelConfig = ModelConfig(name=config.model_name, path=config.model_path)
    print(f"Evaluating {model.name} with {config.evals}")

    start_time: float = time.time()
    if config.launch_with_ray:
        evaluator.launch_evaluate_with_ray(
            model,
            evals=config.evals,
            output_path=config.evaluation_path,
            max_eval_instances=config.max_eval_instances,
        )
    else:
        evaluator.evaluate(
            model, evals=config.evals, output_path=config.evaluation_path, max_eval_instances=config.max_eval_instances
        )
    print(f"Done (total time: {time.time() - start_time} seconds)")


@draccus.wrap()
def main(config: EvaluationConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
