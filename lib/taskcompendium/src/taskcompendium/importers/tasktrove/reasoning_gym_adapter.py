#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt private shared Reasoning Gym scoring to the isolated script protocol."""

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Optional scorer dependencies belong to the private verifier image.
try:
    from tasktrove_verify.grade import InvalidTask
    from tasktrove_verify.modes.grade_reasoning_gym import grade_reasoning_gym_candidate, load_entry
    from tasktrove_verify.spec import ReasoningGymSpec
except ImportError as dependency_error:
    DEPENDENCY_ERROR = str(dependency_error)
else:
    DEPENDENCY_ERROR = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", type=Path)
    parser.add_argument("verifier", type=Path)
    args = parser.parse_args()
    result_path = args.verifier / "result.json"
    try:
        scoring = json.loads((args.tests / "scoring.json").read_text())
        submission = json.loads((args.verifier / "submission.json").read_text())
        if submission["protocol_version"] != 1 or submission["answer_type"] != "text":
            raise ValueError("Unsupported Reasoning Gym submission protocol")
        candidate = submission["answer"]
        if not isinstance(candidate, str):
            raise ValueError("Reasoning Gym script requires an extracted text answer")
        dataset = scoring["dataset"]
        required_version = scoring["reasoning_gym_version"]
        if not isinstance(dataset, str) or not isinstance(required_version, str):
            raise ValueError("Invalid pinned Reasoning Gym scoring configuration")
    except (OSError, ValueError, KeyError, TypeError) as error:
        result_path.write_text(json.dumps({"status": "invalid_task", "error": str(error)}) + "\n")
        return

    if DEPENDENCY_ERROR is not None:
        result_path.write_text(json.dumps({"status": "infra_error", "error": DEPENDENCY_ERROR}) + "\n")
        return
    try:
        if version("reasoning-gym") != required_version:
            raise ImportError(f"Verifier image requires reasoning-gym=={required_version}")
    except (ImportError, PackageNotFoundError) as error:
        result_path.write_text(json.dumps({"status": "infra_error", "error": str(error)}) + "\n")
        return

    try:
        entry = load_entry(args.tests / "entry.json")
        result = grade_reasoning_gym_candidate(ReasoningGymSpec(dataset=dataset), entry, candidate)
        record = {"status": result.status.value, "reward": result.reward}
    except InvalidTask as error:
        record = {"status": "invalid_task", "error": str(error)}
    except Exception as error:
        record = {"status": "infra_error", "error": f"Reasoning Gym scorer failed: {type(error).__name__}"}
    result_path.write_text(json.dumps(record, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
