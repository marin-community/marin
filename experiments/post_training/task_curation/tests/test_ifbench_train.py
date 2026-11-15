# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from taskcompendium.models import format_conversation, grades_in_process
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.ifbench.tasks import sources
from experiments.post_training.task_curation.tests.conversion import converted_task


def test_released_python_literal_constraints_stay_private_and_keep_all_messages():
    row = {
        "key": "synthetic-fixture",
        "messages": [
            {"role": "system", "content": "Write a short answer."},
            {"role": "user", "content": "Use lowercase and avoid commas."},
        ],
        "ground_truth": repr(
            [
                {
                    "instruction_id": ["change_case:english_lowercase", "punctuation:no_comma"],
                    "kwargs": [None, {}],
                }
            ]
        ),
    }
    task = converted_task(sources()[0].config, row)
    public = format_conversation(task.context.events)
    assert "Write a short answer." in public
    assert "Use lowercase and avoid commas." in public
    assert "instruction_id" not in public
    contract = next(resource for resource in task.resources.verifier if resource.path == "config.json")
    assert json.loads(resource_bytes(contract)) == {
        "instruction_id": ["change_case:english_lowercase", "punctuation:no_comma"],
        "kwargs": [None, {}],
    }
    assert not grades_in_process(task.grader)
