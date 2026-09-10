# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers for the per-task files every Nemotron-Gym adapter template ships."""

import json

from experiments.post_training.tasktrove.taskbinary import TaskFiles

VERIFIER_DATA = "tests/verifier_data.json"
METADATA = "metadata.json"


def verifier_data(task: TaskFiles) -> dict:
    """``tests/verifier_data.json``: the grader's per-task inputs (expected answers, schema, cases)."""
    return json.loads(task.text(VERIFIER_DATA))


def metadata(task: TaskFiles) -> dict:
    """Top-level ``metadata.json`` when the template ships one, else empty."""
    raw = task.get_text(METADATA)
    return json.loads(raw) if raw else {}
