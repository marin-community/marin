# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from taskcompendium.models import Source, TaskSpec, judge_spec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets import hierarchy_intake, instruction_hierarchy
from experiments.post_training.task_curation.datasets.instruction_hierarchy import REPO, REVISION, convert_hierarchy
from experiments.post_training.task_curation.source import SourceInfo
from experiments.post_training.task_curation.tests.admission_fixtures import actor_census_input

ROWS = json.loads((Path(__file__).parent / "fixtures/instruction_hierarchy.json").read_text())


def raw_row(subset, native):
    return RawRow(
        id=f"hierarchy-{subset}",
        source=Source(dataset=REPO, revision=REVISION, row=f"data/{subset}/rubrics.jsonl:0", importer_revision="test"),
        data=native,
    )


@pytest.mark.parametrize("subset", ROWS)
def test_original_role_messages_and_private_criteria_survive_conversion(subset):
    native = ROWS[subset]
    task = convert_hierarchy(raw_row(subset, native), ConversionContext({}, None))
    assert isinstance(task, TaskSpec)
    messages = [{"role": event.role, "content": event.content} for event in task.context.events]
    expected = [] if subset == "user_only" else [{"role": "system", "content": native["system_prompt"]}]
    assert messages == [*expected, {"role": "user", "content": native["user_query"]}]
    assert not task.resources.worker and not task.resources.oracle
    resources = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    assert json.loads(resources["source-row.json"]) == native
    assert json.loads(resources["conversation.json"]) == messages
    spec = judge_spec(TaskSpec.model_validate_json(task.model_dump_json()).grader)
    assert spec is not None
    assert spec.criteria == tuple(entry["criterion"] for entry in native["evaluation_criteria"])
    assert spec.rubric == "checklist" and spec.model == "MiniMaxAI/MiniMax-M3-MXFP8"
    assert spec.context == "conversation.json"
    assert spec.system_prompt == "" and spec.prompt_template == "" and spec.label_scores == {}


@pytest.mark.parametrize("defect", ["empty_criteria", "non_object_criterion", "wrong_system_sentinel"])
def test_incomplete_native_contracts_are_typed_instead_of_filled_in(defect):
    native = deepcopy(ROWS["conflicting"])
    if defect == "empty_criteria":
        native["evaluation_criteria"] = []
    elif defect == "non_object_criterion":
        native["evaluation_criteria"][0]["raw_value_json"] = '"unhandled source value"'
    else:
        native["system_prompt"] = "N/A"
    result = convert_hierarchy(raw_row("conflicting", native), ConversionContext({}, None))
    assert isinstance(result, ImportRejection)
    assert (
        result.reason
        == {
            "empty_criteria": "missing_evaluation_criteria",
            "non_object_criterion": "unhandled_evaluation_criterion",
            "wrong_system_sentinel": "invalid_system_instruction",
        }[defect]
    )


@pytest.fixture
def measured_hierarchy():
    original = instruction_hierarchy.source()
    actor = actor_census_input(REPO, REVISION, instruction_hierarchy.VERSION, 3)
    info = replace(original.info, actor_census_input=actor)
    return {
        "original_info": TypeAdapter(SourceInfo).dump_python(info, mode="json"),
        "original_converted_tasks": 3,
        "original_task_specs_sha256": actor.task_specs_sha256,
    }


def test_measured_intake_keeps_the_original_converter_before_retained_selection(measured_hierarchy, monkeypatch):
    monkeypatch.setattr(hierarchy_intake, "read_registry_json", lambda path: measured_hierarchy)
    (source,) = hierarchy_intake.sources()
    assert source.config == instruction_hierarchy.source().config
    assert source.info.actor_census_input.measured_tasks == 3


@pytest.mark.parametrize("field,value", [("dataset_revision", "b" * 40), ("measured_tasks", 4)])
def test_measured_intake_refuses_a_foreign_census(measured_hierarchy, monkeypatch, field, value):
    measured_hierarchy["original_info"]["actor_census_input"][field] = value
    monkeypatch.setattr(hierarchy_intake, "read_registry_json", lambda path: measured_hierarchy)
    with pytest.raises(ValueError, match="measured source conversion"):
        hierarchy_intake.sources()
