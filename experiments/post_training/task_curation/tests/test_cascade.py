# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from taskcompendium.models import format_conversation
from taskcompendium.pipeline.models import ImportRejection
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.cascade.tasks import sources
from experiments.post_training.task_curation.tests.conversion import convert_row, converted_task


def test_cascade_mcqa_keeps_the_released_conversation_and_private_regex_contract():
    row = {
        "agent_ref": {"name": "mcqa_simple_agent"},
        "responses_create_params": {"input": [{"role": "user", "content": "Choose A or B."}], "tools": []},
        "expected_answer": "A",
        "options": [{"A": "one"}, {"B": "two"}],
        "template_metadata": {"output_regex": r"Final: ([AB])"},
    }
    task = converted_task(sources()[1].config, row)
    public = format_conversation(task.context.events)
    assert "Choose A or B." in public
    assert "Final:" not in public
    contract = next(resource for resource in task.resources.verifier if resource.path == "config.json")
    payload = json.loads(resource_bytes(contract))
    assert payload["expected_answer"] == "A"
    assert payload["template_metadata"]["output_regex"] == r"Final: ([AB])"


@pytest.mark.parametrize(
    ("agent", "reason"),
    [
        ("instruction_following_simple_agent", "grader_package_license_f2"),
        ("swe_agents_train", "swe_workspace_reconstruction_required"),
        ("workplace_assistant_simple_agent", "workplace_live_provider_required"),
    ],
)
def test_missing_source_runtimes_are_not_flattened_into_terminal_answer_tasks(agent, reason):
    row = {
        "agent_ref": {"name": agent},
        "responses_create_params": {"input": [], "metadata": {"instance_id": "synthetic"}},
    }
    result = convert_row(sources()[1].config, row)
    assert isinstance(result, ImportRejection)
    assert result.reason == reason
