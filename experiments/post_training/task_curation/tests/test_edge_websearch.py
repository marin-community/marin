# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from taskcompendium.models import Source, judge_spec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets import edge_websearch

NATIVE = json.loads((Path(__file__).parent / "fixtures/edge_preparation.json").read_text())


def converted(native):
    row = RawRow(
        id="original-edge-row",
        source=Source(
            dataset=edge_websearch.REPO, revision=edge_websearch.REVISION, row="fixture", importer_revision="test"
        ),
        data=native,
    )
    result = edge_websearch.convert_plan(row, ConversionContext({}, None))
    assert isinstance(result, NormalizedTask)
    return result


def test_planning_contract_is_explicit_and_source_request_remains_unchanged():
    result = converted(NATIVE)
    task = result.task
    system, user = task.context.events
    assert system.role == "system" and user.role == "user" and user.content == NATIVE["user_prompt"]
    context = json.loads(system.content.split("\n\nSource context:\n", 1)[1])
    assert context == {field: NATIVE[field] for field in ("role", "industry", "os")}
    assert "rather than carrying out" in system.content
    assert len(result.changes) == 1 and result.changes[0].replacement == system.content
    assert edge_websearch.CARD_URL in result.changes[0].reason
    resources = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    assert json.loads(resources["source-row.json"]) == NATIVE
    assert not task.resources.worker and not task.resources.oracle
    assert edge_websearch.source(("native.parquet",)).info.id == edge_websearch.SOURCE_ID


def test_generated_reasoning_does_not_become_a_judge_reference_or_public_prompt():
    original = converted(NATIVE).task
    changed = converted({**NATIVE, "agent_reasoning": "A different synthetic answer, not trusted gold."}).task
    assert original.context == changed.context
    first, second = judge_spec(original.grader), judge_spec(changed.grader)
    assert first == second and first is not None
    assert not first.references and not first.exact_gate
    assert first.rubric == "checklist" and len(first.criteria) == 5
    assert first.model == "MiniMaxAI/MiniMax-M3-MXFP8"


def test_literal_escape_sequences_are_preserved_in_the_original_user_message():
    native = {**NATIVE, "user_prompt": NATIVE["user_prompt"] + r"\nKeep this literal escape."}
    task = converted(native).task
    assert task.context.events[-1].content == native["user_prompt"]
