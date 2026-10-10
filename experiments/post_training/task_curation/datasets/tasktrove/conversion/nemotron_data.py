# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers for the per-task files every Nemotron-Gym task template ships."""

import json

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import TaskFiles

VERIFIER_DATA = "tests/verifier_data.json"


def verifier_data(task: TaskFiles) -> dict:
    """``tests/verifier_data.json``: the grader's per-task inputs (expected answers, schema, cases)."""
    return json.loads(task.text(VERIFIER_DATA))
