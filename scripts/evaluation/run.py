"""
Script to run an evaluator on a model checkpoint.

Usage:

python3 run.py <Name of evaluator> --model <Path to model or Hugging Face model name> \
--evals <List of evals to run> --output-path <Where to output logs and results>
"""

import argparse
import time

from scripts.evaluation.evaluator_factory import get_evaluator, NAME_TO_EVALUATOR
from scripts.evaluation.evaluator import Evaluator, EvaluatorConfig


def main():
    config = EvaluatorConfig(
        name=args.evaluator,
        output_path=args.output_path,
        credentials_path=args.credentials_path,
    )
    print(f"Creating an evaluator with config: {config}")
    evaluator: Evaluator = get_evaluator(config)

    print(f"Evaluating {args.model} with {args.evals}")
    start_time: float = time.time()
    evaluator.evaluate(model_name_or_path=args.model, evals=args.evals)
    print(f"Done (total time: {time.time() - start_time} seconds)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an evaluator on a model checkpoint.")
    parser.add_argument(
        "evaluator",
        type=str,
        help="Which evaluator to run",
        choices=list(NAME_TO_EVALUATOR.keys()),
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Can be the name of the model in Hugging Face (e.g, google/gemma-2b) or "
        "a path to the model to evaluate (can be a GCS path)",
        required=True,
    )
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
        default="output",
    )
    parser.add_argument(
        "--credentials-path",
        type=str,
        help="Path to the JSON file containing credentials.",
        required=False,
        default="credentials.json",
    )
    args = parser.parse_args()

    main()
