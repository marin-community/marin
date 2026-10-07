# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest
from shellbox.machine import DockerImage, MachineSpec
from taskcompendium.datasets.nemotron_ultra import normalization
from taskcompendium.grader import grader_config
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, Source
from taskcompendium.pipeline.models import CheckStatus, NormalizedTask, RawRow

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading import instruction_following_binding
from experiments.post_training.task_curation.datasets.nemotron_ultra.grading.binding import normalize_terminal_grader

from .test_calendar_binding import SourceScoreMachines, grade
from .test_structured_output_binding import DiagnosticMachines


@pytest.fixture
def native_dependencies():
    pytest.importorskip("verifiable_instructions")
    nltk = pytest.importorskip("nltk")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        pytest.skip("Original evaluator parity requires packaged offline NLTK punkt_tab data")


@pytest.fixture
def row():
    return RawRow(
        "instruction",
        Source(dataset="fixture", revision="pinned", row="1", importer_revision="1"),
        {
            "dataset": "fixture",
            "agent_ref": {"name": "instruction_following_simple_agent"},
            "responses_create_params": {"input": [{"role": "user", "content": "Mention tulip without commas."}]},
            "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
            "kwargs": [{"keywords": ["tulip"]}, {}],
        },
    )


def task_for(row):
    result = normalize_terminal_grader(
        row,
        image="fixture@sha256:" + "1" * 64,
        normalize_task=partial(normalization.normalize, selector="fixture", family="instruction-following"),
        allowed_agents=("instruction_following_simple_agent",),
        package=instruction_following_binding.grader_package,
        answer_type=AnswerType.TEXT,
    )
    assert isinstance(result, NormalizedTask)
    return result.task


@pytest.mark.parametrize(
    "mode,candidate,expected",
    [
        ("binary", "A tulip blooms.", 1.0),
        ("binary", "A tulip blooms, briefly.", 0.0),
        ("fraction", "A tulip blooms, briefly.", 0.5),
        ("fraction", "A rose blooms, briefly.", 0.0),
        ("fraction", "<think>A rose blooms, briefly.</think>A tulip blooms.", 1.0),
        ("fraction", "<think>unfinished A tulip blooms.", 0.5),
    ],
)
def test_original_nvidia_modes_and_dispatcher_extraction(row, native_dependencies, mode, candidate, expected):
    row.data["grading_mode"] = mode
    result = grade(task_for(row), candidate)
    assert (result.status, result.reward) == (Outcome.GRADED, expected)


@pytest.mark.parametrize("mode,expected", [("binary", 1.0), ("fraction", 0.0)])
def test_original_empty_constraint_modes_are_not_a_universal_negative(row, native_dependencies, mode, expected):
    row.data.update(instruction_id_list=[], kwargs=[], grading_mode=mode)
    result = grade(task_for(row), "<think>reasoning</think>")
    assert (result.status, result.reward) == (Outcome.GRADED, expected)


def test_original_predicate_errors_remain_false_with_diagnostics(row, native_dependencies):
    row.data.update(instruction_id_list=["unknown:predicate"], kwargs=[{}])
    result = grade(task_for(row), "A tulip blooms.")
    assert (result.status, result.reward) == (Outcome.GRADED, 0.0)
    assert result.detail is not None
    assert result.detail["follow_instruction_list"] == [False]
    assert result.detail["instruction_errors"][0].startswith("KeyError:")
    assert result.detail["invalid_instruction_ids"] == ["unknown:predicate"]


def test_original_bad_kwargs_stay_false_and_identify_broken_source_metadata(row, native_dependencies):
    row.data["kwargs"][0] = {"not_a_keyword": "value"}
    result = grade(task_for(row), "A tulip blooms.")
    assert (result.status, result.reward) == (Outcome.GRADED, 0.0)
    assert result.detail is not None
    assert result.detail["invalid_instruction_kwargs"] == [0]


def test_invalid_source_grading_mode_is_task_failure(row, native_dependencies):
    row.data["grading_mode"] = "unknown-mode"
    result = grade(task_for(row), "A tulip blooms.")
    assert result.status == Outcome.INVALID_TASK
    assert result.reward is None


def test_constraints_and_language_arguments_keep_private_original_contract(row):
    row.data.update(instruction_id_list=["language:response_language"], kwargs=[{"language": "fr"}])
    task = task_for(row)
    original = normalization.normalize(row, "fixture", "instruction-following")
    assert isinstance(original, NormalizedTask)
    assert task.context == original.task.context
    assert grader_config(task)["contract"]["kwargs"] == [{"language": "fr"}]
    assert task.resources.worker == original.task.resources.worker
    assert task.resources.oracle == original.task.resources.oracle


@pytest.mark.parametrize(
    "candidate,reward",
    [
        (
            "La méthode scientifique repose sur des observations précises, des hypothèses claires et des expériences "
            "reproductibles. Les chercheurs comparent les résultats et partagent leurs conclusions "
            "avec leurs collègues.",
            1.0,
        ),
        (
            "The scientific method depends on careful observation, clear hypotheses and reproducible experiments. "
            "Researchers compare the results and share their conclusions with colleagues.",
            0.0,
        ),
        ("<think>reasoning</think>", 1.0),
    ],
)
def test_original_language_detection_including_undetectable_empty_answer(row, native_dependencies, candidate, reward):
    row.data.update(instruction_id_list=["language:response_language"], kwargs=[{"language": "fr"}])
    result = grade(task_for(row), candidate)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


@pytest.mark.parametrize(
    "verdict_status,expected",
    [("scored", CheckStatus.PASS), ("invalid_task", CheckStatus.FAIL), ("infra_error", CheckStatus.INFRA_ERROR)],
)
@pytest.mark.asyncio
async def test_runtime_diagnostic_does_not_claim_predicate_success(row, verdict_status, expected):
    task = task_for(row)
    machines = SourceScoreMachines(verdict_status=verdict_status)
    image = task.verifier.environment_requirements.docker_image
    assert image is not None
    report = await instruction_following_binding.isolated_checks(
        task, factory=machines, machine_spec=MachineSpec(DockerImage(image)), timeout=10
    )
    assert [(check.check, check.status) for check in report.checks] == [
        ("native_runtime", expected),
        ("positive_witness", CheckStatus.SKIPPED),
    ]
    assert len(machines.machines) == 1 and machines.machines[0].closed


@pytest.mark.parametrize(
    "diagnostic,expected",
    [
        ({"instruction_errors": [None, None]}, CheckStatus.PASS),
        (
            {"instruction_errors": ["KeyError: unknown instruction"], "invalid_instruction_ids": ["unknown"]},
            CheckStatus.FAIL,
        ),
        ({"instruction_errors": ["TypeError: bad kwargs"], "invalid_instruction_kwargs": [0]}, CheckStatus.FAIL),
        ({"instruction_errors": ["RuntimeError: predicate failure"]}, CheckStatus.UNSUPPORTED),
    ],
)
@pytest.mark.asyncio
async def test_original_predicate_exceptions_distinguish_metadata_failure_from_unavailable_control(
    row, diagnostic, expected
):
    task = task_for(row)
    machines = DiagnosticMachines(diagnostic=diagnostic)
    image = task.verifier.environment_requirements.docker_image
    assert image is not None
    report = await instruction_following_binding.isolated_checks(
        task, factory=machines, machine_spec=MachineSpec(DockerImage(image)), timeout=10
    )
    assert report.checks[0].status == expected
    assert report.checks[1].status == CheckStatus.SKIPPED
