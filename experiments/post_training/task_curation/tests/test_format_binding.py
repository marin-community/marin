# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from functools import partial

import pytest
from shellbox.machine import DockerImage, MachineSpec
from taskcompendium.datasets.nemotron_ultra import normalization
from taskcompendium.grader import grader_config
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.models import Source
from taskcompendium.pipeline.models import CheckStatus, ImportRejection, NormalizedTask, RawRow

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading import format_binding

from .test_calendar_binding import SourceScoreMachines, grade, grader_image


@pytest.fixture
def row():
    return RawRow(
        "format",
        Source(dataset="fixture", revision="pinned", row="1", importer_revision="1"),
        {
            "dataset": "fixture",
            "agent_ref": {"name": "citation_format_simple_agent"},
            "responses_create_params": {"input": [{"role": "user", "content": "Cite the supplied references."}]},
            "verifier": {"type": "regex", "verify_regex": [r"\[\d+\]"], "verify_min_matches": 2},
        },
    )


def task_for(row):
    result = format_binding.normalize_isolated(
        row,
        image="fixture@sha256:" + "1" * 64,
        normalize_task=partial(normalization.normalize, selector="fixture", family="instruction-following"),
    )
    assert isinstance(result, NormalizedTask)
    return result.task


@pytest.mark.parametrize(
    "candidate,reward,matching_lines",
    [
        ("First [1]\nSecond [2]", 1.0, 2),
        ("Both [1] [2] on one line", 0.0, 1),
        ("Missing citations", 0.0, 0),
        ("<think>Hidden [1]\n[2]</think>Visible [3]", 0.0, 1),
        ("<think>reasoning</think>First [1]\nSecond [2]", 1.0, 2),
    ],
)
def test_original_regex_counts_matching_lines_after_dispatcher_extraction(row, candidate, reward, matching_lines):
    result = grade(task_for(row), candidate)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)
    assert result.detail is not None and result.detail["matching_lines"] == matching_lines


@pytest.mark.parametrize("mode", ["regex", "inline_prose"])
def test_original_regex_modes_keep_zero_threshold_vacuous_success(row, mode):
    row.data["verifier"] = {"type": mode, "verify_regex": [], "verify_min_matches": 0}
    assert grade(task_for(row), "A response without any marker.").reward == 1.0


@pytest.mark.parametrize(
    "candidate,reward,missing,spurious",
    [
        ("References [1] and [2]", 1.0, [], []),
        ("Repeated [1] [1] and [2]", 1.0, [], []),
        ("Only [1]", 0.0, ["[2]"], []),
        ("References [1], [2], [3]", 0.0, [], ["[3]"]),
    ],
)
def test_original_marker_matching_detects_missing_and_spurious_without_counting_duplicates(
    row, candidate, reward, missing, spurious
):
    row.data["agent_ref"] = {"name": "freeform_formatting_simple_agent"}
    row.data["verifier"] = {"type": "string_match", "expected_markers": ["[1]", "[2]"], "patterns": [r"\[\d+\]"]}
    result = grade(task_for(row), candidate)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)
    assert result.detail is not None
    assert (result.detail["missing"], result.detail["spurious"]) == (missing, spurious)


@pytest.mark.parametrize(
    "verifier",
    [
        {"type": "regex", "verify_regex": ["["]},
        {"type": "string_match", "patterns": ["["]},
        {"type": "semantic_judge"},
    ],
)
def test_broken_source_format_contract_writes_no_reward(row, verifier):
    row.data["verifier"] = verifier
    result = grade(task_for(row), "Candidate response.")
    assert (result.status, result.reward, result.failure) == (Outcome.INFRA_ERROR, None, GradingFailure.MISSING_REWARD)


def test_missing_verifier_writes_no_reward(row):
    del row.data["verifier"]
    result = grade(task_for(row), "Candidate response.")
    assert (result.status, result.reward, result.failure) == (Outcome.INFRA_ERROR, None, GradingFailure.MISSING_REWARD)


def test_format_binding_preserves_private_contract_and_does_not_bind_semantic_judges(row):
    task = task_for(row)
    assert grader_config(task)["contract"]["verifier"] == row.data["verifier"]
    original = normalization.normalize(row, selector="fixture", family="instruction-following")
    assert isinstance(original, NormalizedTask)
    assert task.context == original.task.context
    assert task.resources.oracle == original.task.resources.oracle
    result = format_binding.normalize_isolated(
        replace(row, data={**row.data, "agent_ref": {"name": "multichallenge_simple_agent"}}),
        image="fixture",
        normalize_task=partial(normalization.normalize, selector="fixture", family="instruction-following"),
    )
    assert isinstance(result, ImportRejection)
    assert result.reason == "unsupported_native_text_agent"


@pytest.mark.parametrize(
    "verdict_status,expected",
    [("scored", CheckStatus.PASS), ("invalid_task", CheckStatus.INFRA_ERROR), ("infra_error", CheckStatus.INFRA_ERROR)],
)
@pytest.mark.asyncio
async def test_format_runtime_checks_do_not_invent_a_positive_witness(row, verdict_status, expected):
    task = task_for(row)
    machines = SourceScoreMachines(verdict_status=verdict_status)
    report = await format_binding.isolated_checks(
        task, factory=machines, machine_spec=MachineSpec(DockerImage(grader_image(task))), timeout=10
    )
    assert [(check.check, check.status) for check in report.checks] == [
        ("native_runtime", expected),
        ("positive_witness", CheckStatus.SKIPPED),
    ]
    assert len(machines.machines) == 1 and machines.machines[0].closed
