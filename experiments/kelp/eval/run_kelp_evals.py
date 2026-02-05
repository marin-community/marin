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

"""ExecutorStep entrypoint for running Kelp evaluations."""

import logging

from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path

from experiments.kelp.eval.config import (
    HUMANEVAL_EVAL,
    MBPP_EVAL,
    VALIDITY_EVAL,
    KelpEvalTaskConfig,
    KelpEvaluationConfig,
)

logger = logging.getLogger(__name__)


def run_kelp_evaluation(config: KelpEvaluationConfig) -> dict:
    """Run Kelp tree diffusion evaluation.

    This function is called by the executor framework.

    Args:
        config: Evaluation configuration.

    Returns:
        Dictionary with evaluation results.
    """
    from experiments.kelp.eval.evaluator import TreeDiffusionEvaluator, save_results
    from experiments.kelp.model.model import load_model

    logger.info(f"Loading model from {config.model_path}")
    model = load_model(config.model_path)

    evaluator = TreeDiffusionEvaluator(model, config)
    results = evaluator.run_all_evals()

    output_file = f"{config.output_path}/results.json"
    save_results(results, output_file)

    return results.to_dict()


def kelp_eval_step(
    model_step: ExecutorStep | InputName | str,
    evals: list[KelpEvalTaskConfig] | None = None,
    max_eval_instances: int | None = None,
    name_suffix: str = "",
) -> ExecutorStep:
    """Create an ExecutorStep for Kelp model evaluation.

    Args:
        model_step: The training step that produced the model, or path to model.
        evals: List of evaluation tasks to run. Defaults to validity + MBPP.
        max_eval_instances: Maximum instances per task (for debugging).
        name_suffix: Optional suffix for the step name.

    Returns:
        ExecutorStep that runs the evaluation.
    """
    if evals is None:
        evals = [VALIDITY_EVAL, MBPP_EVAL]

    if isinstance(model_step, ExecutorStep):
        model_path = output_path_of(model_step, "checkpoints")
        model_name = model_step.name
    elif isinstance(model_step, InputName):
        if model_step.step is None:
            model_path = model_step.name
            model_name = model_step.name.split("/")[-1] if model_step.name else "unknown"
        else:
            model_path = output_path_of(model_step.step, "checkpoints")
            model_name = model_step.step.name
    else:
        model_path = model_step
        model_name = model_step.split("/")[-1]

    step_name = f"evaluation/kelp/{model_name}"
    if name_suffix:
        step_name = f"{step_name}/{name_suffix}"

    return ExecutorStep(
        name=step_name,
        fn=run_kelp_evaluation,
        config=KelpEvaluationConfig(
            model_path=model_path,  # type: ignore
            output_path=this_output_path(),
            evals=evals,
            max_eval_instances=max_eval_instances,
        ),
    )


def default_kelp_eval(
    model_step: ExecutorStep | InputName | str,
    max_eval_instances: int | None = None,
) -> list[ExecutorStep]:
    """Create default evaluation steps for a Kelp model.

    Runs validity, MBPP, and HumanEval evaluations.

    Args:
        model_step: The training step that produced the model.
        max_eval_instances: Maximum instances per task.

    Returns:
        List of ExecutorSteps for each evaluation.
    """
    return [
        kelp_eval_step(
            model_step,
            evals=[VALIDITY_EVAL],
            max_eval_instances=max_eval_instances,
            name_suffix="validity",
        ),
        kelp_eval_step(
            model_step,
            evals=[MBPP_EVAL],
            max_eval_instances=max_eval_instances,
            name_suffix="mbpp",
        ),
        kelp_eval_step(
            model_step,
            evals=[HUMANEVAL_EVAL],
            max_eval_instances=max_eval_instances,
            name_suffix="humaneval",
        ),
    ]


if __name__ == "__main__":
    from marin.execution.executor import executor_main

    # Example: evaluate a checkpoint
    example_model_path = "gs://marin-us-central2/experiments/kelp/toy/checkpoints"

    executor_main(
        steps=[
            kelp_eval_step(example_model_path, max_eval_instances=10),
        ]
    )
