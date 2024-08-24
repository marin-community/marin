"""
Script to run an evaluator on a model checkpoint.

Usage:

python3 run.py <Name of evaluator> --model <Path to model or Hugging Face model name> \
--evals <List of evals to run> --output-path <Where to output logs and results>
"""

import argparse
import time

from scripts.evaluation.evaluator_factory import get_evaluator, NAME_TO_EVALUATOR
from scripts.evaluation.evaluator import Evaluator, EvaluatorConfig, Model


def main():
    config = EvaluatorConfig(name=args.evaluator, credentials_path=args.credentials_path)
    print(f"Creating an evaluator with config: {config}")
    evaluator: Evaluator = get_evaluator(config)

    model: Model = Model(name=args.model_name, path=args.model_path)
    print(f"Evaluating {model.name} with {args.evals}")

    start_time: float = time.time()
    evaluator.evaluate(model, evals=args.evals, output_path=args.results_path)
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
        "--model-name",
        type=str,
        help="Can be a name of the model in Hugging Face (e.g, google/gemma-2b) or "
        "a name given to the model checkpoint (e.g., $RUN/$CHECKPOINT).",
        required=True,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Optional: Path to the model. Can be a path on GCS.",
        default=None,
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
        "--results-path",
        type=str,
        help="Where to write results to. Can be a local path (e.g., /path/to/output) or "
        "a path on GCS (e.g., gs://bucket/path/to/output).",
        default="results",
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
