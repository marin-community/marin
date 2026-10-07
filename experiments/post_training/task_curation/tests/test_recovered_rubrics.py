# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recovered sources preserve their original public tasks and private judges."""

import base64
import json
import tomllib
from pathlib import Path

import pytest
from taskcompendium.grader import grader_config
from taskcompendium.models import AnswerType, Source, TaskSpec
from taskcompendium.pipeline.models import CheckStatus, NormalizedTask, RawRow

from experiments.post_training.task_curation.pipeline import SourceRuntimeConfig
from experiments.post_training.task_curation.sources import rl_data_pipelines
from experiments.post_training.tasktrove.taskbinary import read_task_binary

SOURCES = {source.source_key: source for source in rl_data_pipelines().values()}


@pytest.mark.parametrize("name", ["multichallenge"])
@pytest.mark.parametrize("campaign_credentials", [{}, {"TOGETHER_API_KEY": ("env:UNUSED_TEST_SECRET",)}])
def test_recovered_rubrics_preserve_task_and_judge_contract_without_inventing_goldens(name, campaign_credentials):
    # Exemplars from TaskTrove revision 02923004846e4e73862c20962f823a6d05100e7a.
    fixture = Path(__file__).parent / f"fixtures/{name}.tar.gz"
    if name == "multichallenge":
        # File contents match archived row 0, multichallenge-9ade61e420d6.tar.gz;
        # the fixture's tar/gzip packaging differs from the source archive.
        fixture = Path(__file__).parents[2] / "tasktrove/fixtures/nemotron_multichallenge.tar.gz"
    files = read_task_binary(fixture.read_bytes())
    declaration = SOURCES["Task Trove:laion__nemotron-gym-multichallenge-advanced-v4"]
    pipeline = declaration.recipe(
        SourceRuntimeConfig(images={}, controller_url=None, verifier_secret_env=campaign_credentials)
    ).policy
    instruction = files.text("instruction.md")
    data = json.loads(files.text("tests/verifier_data.json"))
    judge_toml = files.text("tests/judge.toml")
    source = Source(dataset=declaration.hf_id, revision=declaration.revision, row="fixture", importer_revision="1")
    result = pipeline.normalize(
        RawRow(
            "fixture",
            source,
            {
                "instruction": instruction,
                "files": {path: base64.b64encode(blob).decode() for path, blob in files.files.items()},
                "verifier_data": data,
            },
        )
    )
    assert isinstance(result, NormalizedTask)
    task = TaskSpec.model_validate_json(result.task.model_dump_json())
    assert task.source == source
    assert task.answer_type == AnswerType.TEXT
    question = files.text("tests/conversation.txt") if name == "multichallenge" else data["instruction"]
    public_instruction = task.context.events[0].content
    assert public_instruction is not None and question in public_instruction
    assert "/app/response.txt" not in public_instruction
    assert "task_complete" not in public_instruction
    assert result.changes[0].original == instruction
    assert not task.resources.worker and not task.resources.all and not task.resources.oracle
    contract = grader_config(task)["contract"]
    assert contract["source_judge_data"] == data
    assert contract["source_judge_toml"] == judge_toml
    assert contract["aggregation"] == tomllib.loads(judge_toml)
    assert contract["question"] == question
    if name in {"multichallenge", "multichallenge_vanilla"}:
        assert contract["criteria"] == [entry["description"] for entry in tomllib.loads(judge_toml)["criterion"]]
        assert contract["aggregation"]["scoring"]["aggregation"] == "all_pass"
        paths = ["tests/test.sh", "task.toml", "environment/Dockerfile"]
        if name == "multichallenge":
            paths.append("tests/sitecustomize.py")
        assert contract["source_runtime_files"] == {path: files.text(path) for path in paths}
        assert contract["source_verifier_settings"] == tomllib.loads(files.text("task.toml"))["verifier"]
    else:
        assert data["principle"] in task.context.events[0].content
        assert contract["mode"] == "holistic_numeric"
    report = pipeline.check_suite.run(task)
    assert {check.status for check in report.checks} == {CheckStatus.UNSUPPORTED}
