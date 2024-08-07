"""
Script to run an evaluation harnesses on a model checkpoint.

Usage:

python3 run.py <Name of eval harness> --model_gcs_path <GCS path to model> --evals <List of evals to run> \
--output-path <Where to output logs and results>
"""

import argparse
import time

from scripts.evaluation.evaluation_harness_factory import get_evaluation_harness, NAME_TO_EVALUATION_HARNESS
from scripts.evaluation.evaluation_harness import EvaluationHarness


def main():
    print(f"Evaluating {args.model_gcs_path} with {args.eval_harness}")
    start_time: float = time.time()
    harness: EvaluationHarness = get_evaluation_harness(args.eval_harness)
    harness.evaluate(
        model_gcs_path=args.model_gcs_path,
        evals=args.evals,
        output_path=args.output_path,
    )
    print(f"Done ({time.time() - start_time} seconds)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an evaluation harness")
    parser.add_argument(
        "eval_harness",
        type=str,
        help="Which eval harness to run",
        choices=list(NAME_TO_EVALUATION_HARNESS.keys()),
    )
    parser.add_argument("--model-gcs-path", type=str, help="GCS path to the model to evaluate", required=True)
    parser.add_argument(
        "-e",
        "--evals",
        nargs="*",
        help="Which specific evals within an evaluation harness to run. This would be a list of "
        "tasks in for EleutherAI's lm-evaluation-harness or a list of run_entries_*.py files from HELM.",
        default=[],
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="The location of the output path (filesystem path or URL)",
        required=True,
    )
    args = parser.parse_args()

    main()
