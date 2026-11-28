# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public prompts and private source checks survive the two instruction converters."""

import json
from pathlib import Path

import pytest
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, Source, TaskSpec, judge_spec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets import dolci_precise_if, inverse_ifeval
from experiments.post_training.task_curation.datasets.inverse_ifeval_grade import score_checks

FIXTURES = Path(__file__).parent / "fixtures/instruction_sources"


def raw_row(module, native):
    return RawRow(
        id="native-1",
        source=Source(
            dataset=module.REPO, revision=module.REVISION, row="native-fixture", importer_revision=module.VERSION
        ),
        data=native,
    )


def test_dolci_reference_is_private_provenance_and_never_a_gold_gate():
    native = json.loads((FIXTURES / "dolci.json").read_text())
    task = dolci_precise_if.convert_precise_if(raw_row(dolci_precise_if, native), ConversionContext({}, None))
    assert isinstance(task, TaskSpec)
    assert [event.content for event in task.context.events] == [native["messages"][0]["content"]]
    files = {file.path: resource_bytes(file) for file in task.resources.verifier}
    assert json.loads(files["source-row.json"]) == native
    assert json.loads(files["conversation.json"])[0]["content"] == native["messages"][0]["content"]
    spec = judge_spec(task.grader)
    assert spec is not None and not spec.references and not spec.exact_gate
    assert len(spec.criteria) == 1 and spec.context == "conversation.json"
    assert not task.resources.worker and not task.resources.oracle


def test_inverse_keeps_per_check_polarity_and_private_native_rubric():
    native = json.loads((FIXTURES / "inverse.json").read_text())
    env = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=inverse_ifeval.GRADER_IMAGE
    )
    task = inverse_ifeval.convert_inverse(raw_row(inverse_ifeval, native), ConversionContext({}, env))
    assert isinstance(task, TaskSpec)
    assert [event.content for event in task.context.events] == [native["messages"][0]["content"]]
    files = {file.path: resource_bytes(file) for file in task.resources.verifier}
    config = json.loads(files["config.json"])
    assert config["checks"] == native["llm_judge"] and config["aggregation"] == "all"
    assert (
        config["native_judge_template"] == json.loads((FIXTURES / "native-judge-template.json").read_text())["template"]
    )
    assert json.loads(files["source-row.json"]) == native
    assert not task.resources.worker and not task.resources.oracle
    changed = {**native, "instructions": [{"uid": 999, "content": "undeclared deterministic validator"}]}
    assert isinstance(
        inverse_ifeval.convert_inverse(raw_row(inverse_ifeval, changed), ConversionContext({}, env)), ImportRejection
    )


@pytest.mark.parametrize("verdicts,expected", [(["[[YES]]", "[[NO]]"], 1.0), (["[[YES]]", "[[YES]]"], 0.0)])
def test_inverse_uses_all_checks_and_expected_no(verdicts, expected):
    checks = [
        {"uid": 1, "content": "First private check", "pass_criteria": "YES"},
        {"uid": 2, "content": "Second private check", "pass_criteria": "NO"},
    ]
    outputs = iter(verdicts)
    reward, details = score_checks(
        checks, "<think>private reasoning</think>final", lambda question, candidate: next(outputs)
    )
    assert reward == expected and len(details) == 2


def test_malformed_provider_output_cannot_pass_an_expected_no_check():
    with pytest.raises(ValueError, match="no conclusive"):
        score_checks(
            [{"uid": 1, "content": "Private check", "pass_criteria": "NO"}],
            "answer",
            lambda question, candidate: "Provider returned no verdict",
        )
