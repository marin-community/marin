# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a pinned deterministic evaluator against replayed output files only."""

import argparse
import json
import runpy
from pathlib import Path

OUTPUT_ERRORS = (ValueError, TypeError, IndexError, KeyError, AssertionError, ZeroDivisionError)


def scored(success: bool, detail: dict) -> dict:
    return {"status": "scored", "reward": float(success), "detail": detail}


def evaluate_output(evaluator: Path, primary_output: Path, candidate_directory: Path) -> dict:
    """Preserve upstream task success after environment/reference validation.

    Source data/shape errors count as failed submissions, as upstream does.
    Missing gold files, import failures, and other runtime failures propagate
    so the outer runtime leaves them ungraded. Reference validation must reject
    a package whose gold data or evaluator produces output/shape errors itself.
    """
    namespace = runpy.run_path(str(evaluator))
    evaluate = namespace["eval"]
    if not callable(evaluate):
        raise TypeError("The pinned evaluator must expose a callable eval")
    if not primary_output.is_file():
        return scored(False, {"reason": "missing_primary_output"})
    try:
        result = evaluate()
    except FileNotFoundError as error:
        if error.filename is None or not Path(error.filename).resolve().is_relative_to(candidate_directory.resolve()):
            raise
        return scored(False, {"reason": "missing_candidate_output"})
    except OUTPUT_ERRORS as error:
        return scored(False, {"reason": "candidate_output_rejected", "exception_type": type(error).__name__})
    if result is False:
        return scored(False, {"reason": "upstream_early_failure"})
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("Unexpected result from the pinned evaluator")
    success, details = result
    return scored(bool(success == 1), {"upstream": str(details)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluator", type=Path)
    parser.add_argument("primary_output", type=Path)
    parser.add_argument("candidate_directory", type=Path)
    parser.add_argument("verdict", type=Path)
    args = parser.parse_args()
    verdict = evaluate_output(args.evaluator, args.primary_output, args.candidate_directory)
    args.verdict.parent.mkdir(parents=True, exist_ok=True)
    args.verdict.write_text(json.dumps(verdict, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
