# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, Source, TaskSpec, judge_spec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets import qa_formatting
from experiments.post_training.task_curation.datasets.qa_formatting_grade import format_verification, score_abstention

ROWS = json.loads((Path(__file__).parent / "fixtures/instruction_sources/qa-formatting.json").read_text())


@pytest.mark.parametrize(
    "alias,family", [("qa-abstention", "abstention"), ("freeform", "freeform"), ("citation", "citation")]
)
def test_native_roles_and_private_scoring_fields_survive_conversion(alias, family):
    original = ROWS[alias][0]
    source = qa_formatting.source(family)
    row = RawRow(
        id="real-source-row",
        source=Source(
            dataset=source.dataset.name, revision=source.dataset.revision, row="fixture", importer_revision="test"
        ),
        data=original,
    )
    task = qa_formatting.convert_native(
        row,
        ConversionContext(
            {},
            EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=qa_formatting.GRADER_IMAGE
            ),
        ),
    )
    assert isinstance(task, TaskSpec)
    assert [{"role": event.role, "content": event.content} for event in task.context.events] == original[
        "responses_create_params"
    ]["input"]
    files = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    assert json.loads(files["source-row.json"]) == original
    assert not task.resources.worker and not task.resources.oracle
    config = json.loads(files["config.json"])
    if family == "abstention":
        assert config["answer"] == original["answer"] and config["question"] == original["question"]
        judge = judge_spec(task.grader)
        assert judge is not None and judge.max_completion_tokens == 64 and judge.temperature == 0.0
    else:
        assert config["verifier"] == original["verifier"]
        assert judge_spec(task.grader) is None


def test_normalized_boxed_abstention_bypasses_judge_but_never_receives_correct_reward():
    def unexpected_judge(prompt):
        raise AssertionError("Explicit abstention must not invoke the judge")

    reward, detail = score_abstention("Question", "Target", r"<think>reasoning</think>\boxed{ [IDK] }", unexpected_judge)
    assert reward == 0.5 and detail["verdict"] == "abstain" and detail["judge_output"] is None


@pytest.mark.parametrize("reply,reward", [("A", 1.0), ("B", 0.0), ("C", 0.5), ("?", 0.0)])
def test_semantic_judge_uses_native_prompt_and_three_way_reward(reply, reward):
    def judge(prompt):
        assert "Gold target: CQ Press" in prompt and "Predicted answer: Congressional Quarterly" in prompt
        return reply

    actual, detail = score_abstention("Publisher?", "CQ Press", r"\boxed{Congressional Quarterly}", judge)
    assert actual == reward and detail["extracted_answer"] == "Congressional Quarterly"


def test_formatter_counts_lines_once_and_rejects_spurious_citations():
    verifier = {"type": "regex", "verify_regex": [r"^\d+\.", r"^1\."], "verify_min_matches": 2}
    assert format_verification.grade_format("1. one line", verifier)[0] == 0
    assert format_verification.grade_format("1. one line\n2. another", verifier)[0] == 1
    verifier = {"type": "string_match", "patterns": [r"\[ref:\d+\]"], "expected_markers": ["[ref:2]"]}
    assert format_verification.grade_format("[ref:2]", verifier)[0] == 1
    assert format_verification.grade_format("[ref:2] [ref:3]", verifier)[0] == 0
