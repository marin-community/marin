# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NeMo's string-only terminal-action contract, without a terminal actor or judge."""

import json
from difflib import SequenceMatcher
from pathlib import Path

from jsonschema.exceptions import ValidationError
from openapi_schema_validator import validate
from terminus_schemas import TERMINUS_1_SCHEMA, TERMINUS_2_SCHEMA

SCHEMAS = {"terminus_1": TERMINUS_1_SCHEMA, "terminus_2": TERMINUS_2_SCHEMA}


def score(expected: dict, candidate_text: str, harness: str, threshold: float) -> float:
    """Apply NeMo's JSON/schema, completion and concatenated-keystroke checks."""
    schema = SCHEMAS[harness]
    validate(expected, schema)
    candidate_text = candidate_text.split("</think>")[-1].strip()
    try:
        candidate = json.loads(candidate_text)
        validate(candidate, schema)
    except (json.JSONDecodeError, ValidationError):
        return 0.0
    if expected.get("task_complete"):
        if not candidate.get("task_complete"):
            return 0.0
    elif expected.get("is_task_complete") and not candidate.get("is_task_complete"):
        return 0.0
    gold = [command["keystrokes"] for command in expected.get("commands", []) if "keystrokes" in command]
    predicted = [command["keystrokes"] for command in candidate.get("commands", []) if "keystrokes" in command]
    if not gold and not predicted:
        similarity = 1.0
    elif not gold or not predicted:
        similarity = 0.0
    else:
        similarity = SequenceMatcher(None, "".join(gold), "".join(predicted)).ratio()
    return float(similarity >= threshold)


def main() -> None:
    config = json.loads(Path("/tests/config.json").read_text(encoding="utf-8"))
    candidate = Path("/app/answer.txt").read_text(encoding="utf-8")
    print(score(config["expected"], candidate, config["harness"], config["threshold"]))


if __name__ == "__main__":
    main()
