"""
Script to run evals on GPUs.
"""

import time

import draccus

from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.evaluators.evaluator_factory import get_evaluator


@draccus.wrap()
def main(config: EvaluationConfig) -> None:
    print(f"Running evals with args: {config}")
    evaluator: Evaluator = get_evaluator(config)
    model: ModelConfig = ModelConfig(name=config.model_name, path=config.model_path)

    start_time: float = time.time()
    evaluator.run(
        model, evals=config.evals, output_path=config.evaluation_path, max_eval_instances=config.max_eval_instances
    )
    print(f"Done (total time: {time.time() - start_time} seconds)")


if __name__ == "__main__":
    main()
