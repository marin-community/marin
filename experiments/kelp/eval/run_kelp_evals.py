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

"""ExecutorStep entrypoint for running Kelp evaluations.

Can be run in two modes:
1. Local mode (default): Direct execution with CLI arguments
2. Executor mode: Via Marin executor framework (--use-executor)

Usage:
    # Local evaluation on a checkpoint
    uv run python experiments/kelp/eval/run_kelp_evals.py \\
        --checkpoint checkpoints/kelp/step-010001 \\
        --output-dir checkpoints/kelp/eval

    # With specific evals
    uv run python experiments/kelp/eval/run_kelp_evals.py \\
        --checkpoint checkpoints/kelp/step-010001 \\
        --evals validity mbpp \\
        --max-instances 10

    # Via executor framework
    uv run python experiments/kelp/eval/run_kelp_evals.py --use-executor
"""

import argparse
import logging
import sys

from experiments.kelp.eval.config import (
    HUMANEVAL_EVAL,
    MBPP_EVAL,
    VALIDITY_EVAL,
    KelpEvalTaskConfig,
    KelpEvaluationConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

EVAL_PRESETS = {
    "validity": VALIDITY_EVAL,
    "mbpp": MBPP_EVAL,
    "humaneval": HUMANEVAL_EVAL,
}


def run_kelp_evaluation(config: KelpEvaluationConfig) -> dict:
    """Run Kelp tree diffusion evaluation.

    This function is called by the executor framework or directly.

    Args:
        config: Evaluation configuration.

    Returns:
        Dictionary with evaluation results.
    """
    import jax

    from experiments.kelp.eval.evaluator import TreeDiffusionEvaluator, save_results
    from experiments.kelp.model.model import load_model

    logger.info("=" * 60)
    logger.info("Kelp Tree Diffusion Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model path: {config.model_path}")
    logger.info(f"Output path: {config.output_path}")
    logger.info(f"Evals: {[e.name for e in config.evals]}")
    logger.info(f"Max instances: {config.max_eval_instances}")
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info("=" * 60)

    logger.info(f"Loading model from {config.model_path}")
    model = load_model(config.model_path)
    logger.info(f"Loaded model: {model.config.hidden_dim}d, {model.config.num_layers}L, vocab={model.config.vocab_size}")

    evaluator = TreeDiffusionEvaluator(model, config)
    results = evaluator.run_all_evals()

    output_file = f"{config.output_path}/results.json"
    save_results(results, output_file)

    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    if results.validity:
        logger.info(f"Validity rate: {results.validity.validity_rate:.2%}")
    if results.mbpp_pass_at_1:
        logger.info(f"MBPP pass@1: {results.mbpp_pass_at_1.pass_rate:.2%}")
    if results.humaneval_pass_at_1:
        logger.info(f"HumanEval pass@1: {results.humaneval_pass_at_1.pass_rate:.2%}")
    logger.info("=" * 60)

    return results.to_dict()


def kelp_eval_step(
    model_step,  # ExecutorStep | InputName | str
    evals: list[KelpEvalTaskConfig] | None = None,
    max_eval_instances: int | None = None,
    name_suffix: str = "",
):
    """Create an ExecutorStep for Kelp model evaluation.

    Args:
        model_step: The training step that produced the model, or path to model.
        evals: List of evaluation tasks to run. Defaults to validity + MBPP.
        max_eval_instances: Maximum instances per task (for debugging).
        name_suffix: Optional suffix for the step name.

    Returns:
        ExecutorStep that runs the evaluation.
    """
    from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path

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
    model_step,  # ExecutorStep | InputName | str
    max_eval_instances: int | None = None,
) -> list:
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Kelp tree diffusion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate a local checkpoint
    uv run python experiments/kelp/eval/run_kelp_evals.py \\
        --checkpoint checkpoints/kelp/step-010001 \\
        --output-dir checkpoints/kelp/eval

    # Run only validity evaluation with limited instances
    uv run python experiments/kelp/eval/run_kelp_evals.py \\
        --checkpoint checkpoints/kelp/step-010001 \\
        --evals validity --max-instances 10

    # Run via executor framework
    uv run python experiments/kelp/eval/run_kelp_evals.py --use-executor
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint directory (local or GCS)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--evals",
        type=str,
        nargs="+",
        choices=["validity", "mbpp", "humaneval"],
        default=["validity"],
        help="Evaluation tasks to run (default: validity)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Maximum evaluation instances per task (for quick testing)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--use-executor",
        action="store_true",
        help="Run via Marin executor framework (requires MARIN_PREFIX)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.use_executor:
        from marin.execution.executor import executor_main

        model_path = args.checkpoint or "gs://marin-us-central2/experiments/kelp/toy/checkpoints"
        executor_main(
            steps=[
                kelp_eval_step(model_path, max_eval_instances=args.max_instances),
            ]
        )
    else:
        if not args.checkpoint:
            logger.error("Must specify --checkpoint for local evaluation")
            sys.exit(1)

        output_dir = args.output_dir
        if output_dir is None:
            output_dir = f"{args.checkpoint}/eval"

        evals = [EVAL_PRESETS[name] for name in args.evals]

        config = KelpEvaluationConfig(
            model_path=args.checkpoint,
            output_path=output_dir,
            evals=evals,
            max_eval_instances=args.max_instances,
        )

        run_kelp_evaluation(config)


if __name__ == "__main__":
    main()
