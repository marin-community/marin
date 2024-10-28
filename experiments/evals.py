from dataclasses import dataclass, replace
import logging
import os

import draccus

from marin.execution.executor import ExecutorMainConfig
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

"""
Canonical set of evals.

How to run:
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python3 experiments/evals.py
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_steps(prefix: str, model_path: str, model_name: str, run_lm_eval_harness: bool) -> list[ExecutorStep]:
    full_model_path: str = os.path.join(prefix, model_path, model_name)
    logger.info(f"Evaluating model at {full_model_path}")

    steps: list[ExecutorStep] = []
    if run_lm_eval_harness:
        # LM Eval Harness
        lm_eval_step = ExecutorStep(
            name=os.path.join(prefix, "lm_eval"),
            fn=evaluate,
            config=EvaluationConfig(
                evaluator="eleuther",
                model_name=model_name,
                model_path=full_model_path,
                evaluation_path=this_output_path(),
                evals=["mmlu"],
                launch_with_ray=False,
            ),
        )
        steps.append(lm_eval_step)

    # HELM MMLU
    helm_evaluate_step = ExecutorStep(
        name=os.path.join(prefix, "helm"),
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="helm",
            model_name=model_name,
            model_path=full_model_path,
            evaluation_path=this_output_path(),
            evals=["mmlu"],
        ),
    )
    steps.append(helm_evaluate_step)

    # AlpacaEval
    alpaca_evaluate_step = ExecutorStep(
        name=os.path.join(prefix, "alpaca"),
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="alpaca",
            model_name=model_name,
            model_path=full_model_path,
            evaluation_path=this_output_path(),
        ),
    )
    steps.append(alpaca_evaluate_step)

    return steps


@dataclass(frozen=True)
class EvalExecutorConfig:
    prefix: str = "gs://marin-us-central2"

    model_path: str = "checkpoints/quickstart_single_script_docker_test_09_18/pf5pe4ut/hf"

    model_name: str = "pf5pe4ut/step-600"

    run_lm_eval_harness: bool = False


@draccus.wrap()
def main(config: EvalExecutorConfig) -> None:
    try:
        prefix = config.prefix
        executor_main_config = ExecutorMainConfig(prefix=prefix)
        executor_main_config = replace(
            executor_main_config, prefix=prefix, executor_info_base_path=os.path.join(prefix, "evals")
        )

        steps: list[ExecutorStep] = create_steps(
            prefix, config.model_path, config.model_name, config.run_lm_eval_harness
        )
        executor_main(executor_main_config, steps=steps)
        logger.info(f"Execution completed successfully. All outputs are in {prefix}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise e


if __name__ == "__main__":
    main()

