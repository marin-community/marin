# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from verifyit.grade import InvalidTask

from taskcompendium.importers.evalchemy.scirecipe import (
    SYSTEM_PROMPT,
    convert_scirecipe,
    scirecipe_item,
    scirecipe_prompt,
)
from taskcompendium.importers.evalchemy.scorers.scirecipe import prediction_blocks, structured_metrics
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import resource_bytes

KEY = (
    'Step 1: {"action": "mix", "objects": ["water"], "parameters": ["2 ml"]}\n'
    'Step 2: {"action": "wait", "objects": ["water"], "parameters": ["5 min"]}'
)


def test_conversion_keeps_reference_private_and_preserves_groups():
    document = {
        "exp_goal": "Prepare a sample.",
        "action": ["mix", "wait"],
        "key": KEY,
        "orc": "Private reference",
        "think": "Private reasoning",
        "note": "Private note",
        "type": "Planning",
        "level": "level2",
    }
    source = Source(dataset="synthetic", revision="fixture", row="test.parquet:0", importer_revision="fixture")
    environment = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="fixture.lock")
    task = convert_scirecipe(RawRow("synthetic-0", source, document), ConversionContext({}, environment))
    assert [event.content for event in task.context.events] == [SYSTEM_PROMPT, scirecipe_prompt(document)]
    config = json.loads(resource_bytes(next(r for r in task.resources.verifier if r.path == "config.json")))
    assert config == {"key": KEY, "orc": "Private reference"}
    item = scirecipe_item(task)
    assert item.groups == {"type": "Planning", "level": "level2"}
    assert item.operation.response_selection == "final_only"


def test_prompt_matches_reference_shuffle_and_action_new():
    # Rendered with the pinned reference's build_user_prompt, SEED=42.
    document = {"exp_goal": "  Prepare a sample.  ", "action": [" Mix ", "WAIT", "read", "", None]}
    prefix = "Prepare a sample. Please provide the experimental protocol to solve this problem. "
    assert scirecipe_prompt(document) == (
        prefix + 'Please note that you can only use the following actions: "wait", "mix", "read".'
    )
    assert scirecipe_prompt({**document, "action_new": ["READ", " Mix "]}) == (
        prefix + 'Please note that you can only use the following actions: "read", "mix".'
    )
    assert document["action"] == [" Mix ", "WAIT", "read", "", None]


@pytest.mark.parametrize("count", [2, 15, 25])
def test_structured_exact_match_preserves_unnormalized_content_score(count):
    # Dividing each pair before summing can push a perfect 15-step reward above 1.
    lines = KEY.splitlines()
    key = "\n".join(f"Step {i + 1}: {lines[i % 2].split(': ', 1)[1]}" for i in range(count))
    assert structured_metrics(key, key) == {
        "step_match": 1,
        "order_strict": 1,
        "order_lcs": 1,
        "order_tau": 1,
        "content_score": 1.5,
        "avg_obj": 1,
        "avg_para": 1,
        "avg_mx": 1,
    }


def test_malformed_reference_is_an_invalid_task_instead_of_a_model_zero():
    with pytest.raises(InvalidTask, match="reference"):
        structured_metrics(KEY, 'Step 1: {"action": null}')


def test_structured_reversed_actions_and_greedy_alignment():
    result = structured_metrics("\n".join(reversed(KEY.splitlines())), KEY)
    assert result["step_match"] == 1
    assert result["order_strict"] == 0
    assert result["order_lcs"] == 0.5
    assert result["order_tau"] == 0
    assert result["content_score"] == pytest.approx(1.5 * (1 - 0.5**1.5))
    # An unknown leading action exhausts the reference's greedy alignment cursor.
    unknown = 'Step 0: {"action": "unknown"}\n' + KEY
    assert structured_metrics(unknown, KEY)["content_score"] == 0


@pytest.mark.parametrize("response", ["", "I cannot answer.", "Step 1: {broken}", 'Step 1: {"action": []}'])
def test_unparseable_answer_has_no_content_credit(response):
    result = structured_metrics(response, KEY)
    assert result["content_score"] == 0
    assert result["order_lcs"] == 0


def test_object_token_fallback_and_parameter_gate():
    prediction = 'Step 1: {"action": "mix", "objects": ["water buffer"], "parameters": ["2 ml"]}'
    result = structured_metrics(prediction, KEY.splitlines()[0])
    assert result["avg_obj"] == 0.5
    assert result["avg_para"] == 1
    assert result["content_score"] == 1


def test_final_blocks_survive_separate_reasoning_channel():
    final = f"<key>{KEY}</key><orc>Mix and wait.</orc><note>Check labels.</note>"
    assert prediction_blocks(final) == (KEY, "Mix and wait.")
    assert prediction_blocks("<think>Plan.</think>" + final) == (KEY, "Mix and wait.")
    example = final.replace("Mix and wait.", "Discarded example.")
    assert prediction_blocks("<think>Example: " + example + "</think>" + final) == (KEY, "Mix and wait.")
    assert prediction_blocks("<think>Unfinished " + final) == ("<think>Unfinished " + final, "")
    assert prediction_blocks(KEY) == (KEY, "")
    assert prediction_blocks(final + final.replace("Mix and wait.", "Contradiction.")) == (KEY, "Mix and wait.")
