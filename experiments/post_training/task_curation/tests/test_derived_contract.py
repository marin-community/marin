# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The late-stage derived contract census binds population-wide shape and grader types."""

import pytest
from finestore.rl.curation import ActorCensusInput
from finestore.rl.status_taxonomy import DataRestriction
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grader import verifyit_package
from taskcompendium.grader_type import GraderType
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import RUBRIC_CHECKLIST, ExactSpec, JudgeSpec, PytestSpec

from experiments.post_training.task_curation.derived_contract_census import (
    DerivedContractCensusInput,
    census_source,
)
from experiments.post_training.task_curation.derived_contract_tags import bind_derived_contract_tags
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, SourceReference

IMAGE = "private/grader@sha256:" + "a" * 64
GRADER_ENVIRONMENT = EnvironmentRequirements(docker_image=IMAGE, command_semantics=CommandSemantics.LINUX_PROCESS)
JUDGE = verifyit_package(
    JudgeSpec(rubric=RUBRIC_CHECKLIST, criteria=("be complete",), question="Is the answer complete?")
).grader
EXACT = verifyit_package(ExactSpec(expected=("done",))).grader
TESTS = verifyit_package(PytestSpec(), environment=GRADER_ENVIRONMENT).grader


def reply_task(task_id: str, grader) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content="Answer in your reply."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
        source=Source(dataset="open-athena/fixture", revision="pin", row=task_id, importer_revision="1"),
    )


def population_fixture(tmp_path, tasks: tuple[TaskSpec, ...]) -> DerivedContractCensusInput:
    files = write_text_population(
        ((task.model_dump(mode="json"), {"deployment": 1}) for task in tasks), str(tmp_path / "population")
    )
    population = SourceTextPopulation(
        path=str(tmp_path / "population"),
        source_id="TaskTrove:fixture",
        dataset_id="open-athena/fixture",
        dataset_revision="pin",
        source_version="1",
        converter_revision="converter-pin",
        text_renderer_version="fixture",
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections={},
        admission_rejections={},
    )
    binding = ActorCensusInput(
        dataset_id=population.dataset_id,
        dataset_revision=population.dataset_revision,
        converter_version=population.source_version,
        converter_revision=population.converter_revision,
        measured_tasks=population.task_count,
        task_specs_sha256=population.task_specs_sha256,
        evidence_url=str(StoragePath(population.path) / ".artifact.json"),
    )
    write_record(
        ArtifactRecord(
            output_path=population.path,
            result_type=result_type_name(SourceTextPopulation),
            result=population.result_payload(),
        )
    )
    return DerivedContractCensusInput(population=population, input=binding)


def source_with(dataset: SourceReference, **info_updates) -> RlDataSource:
    return RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            id="TaskTrove:fixture",
            origin=SourceOrigin.TASKTROVE,
            upstream_dataset="fixture",
            upstream_name="fixture",
            build_path="fixture",
            dataset=dataset,
            **info_updates,
        )
    )


def test_reply_only_judged_population_binds_and_contradiction_fails(tmp_path):
    receipt = census_source(population_fixture(tmp_path, (reply_task("task-1", JUDGE),)), str(tmp_path / "census"))
    assert receipt.source_id == "TaskTrove:fixture"
    assert receipt.summary.grader_type_counts == {"llm-judge": 1}
    assert receipt.summary.agentic_count == 0
    assert receipt.summary.single_turn_count == 1
    assert receipt.summary.reply_only_count == 1
    assert receipt.summary.workspace_graded_count == 0
    assert receipt.summary.judge_rubric_counts == {"checklist": 1}
    assert receipt.summary.judge_no_reference_count == 1
    assert receipt.contradictions(frozenset()) == ()
    assert receipt.contradictions(frozenset({"agentic", "multi-turn"})) == ("agentic", "multi-turn")


def test_mixed_population_yields_mixed_breakdown(tmp_path):
    tasks = (reply_task("task-1", JUDGE), reply_task("task-2", EXACT))
    receipt = census_source(population_fixture(tmp_path, tasks), str(tmp_path / "census"))
    assert receipt.summary.grader_type_counts == {"exact-match": 1, "llm-judge": 1}
    assert receipt.contract_summary().grader_type is GraderType.MIXED


def test_agentic_workspace_population_counts(tmp_path):
    workspace = TaskSpec(
        id="task-1",
        context=ConversationInput(events=(TextMessage(role="user", content="Repair the project."),)),
        environment_requirements=GRADER_ENVIRONMENT,
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=TESTS,
        source=Source(dataset="open-athena/fixture", revision="pin", row="0", importer_revision="1"),
    )
    receipt = census_source(population_fixture(tmp_path, (workspace,)), str(tmp_path / "census"))
    assert receipt.summary.agentic_count == 1
    assert receipt.summary.workspace_graded_count == 1
    assert receipt.summary.reply_only_count == 0
    assert receipt.summary.grader_type_counts == {"unit-tests": 1}


def test_binder_binds_the_measured_population(tmp_path):
    receipt = census_source(population_fixture(tmp_path, (reply_task("task-1", JUDGE),)), str(tmp_path / "census"))
    bound = bind_derived_contract_tags(source_with(SourceReference("open-athena/fixture", "pin", "url")), receipt)
    assert bound.info.derived_contract is not None
    assert bound.info.derived_contract.grader_type is GraderType.LLM_JUDGE


def test_binder_stale_binding_stays_pending(tmp_path):
    receipt = census_source(population_fixture(tmp_path, (reply_task("task-1", JUDGE),)), str(tmp_path / "census"))
    bound = bind_derived_contract_tags(source_with(SourceReference("open-athena/fixture", "other-pin", "url")), receipt)
    assert bound.info.derived_contract is None


def test_unchanged_shape_still_emits_a_receipt_for_a_changed_population(tmp_path):
    old = population_fixture(tmp_path / "old", (reply_task("old", EXACT),))
    changed = reply_task("new", EXACT).model_copy(
        update={
            "context": ConversationInput(events=(TextMessage(role="user", content="A different task."),)),
        }
    )
    new = population_fixture(tmp_path / "new", (changed,))
    prior = census_source(old, str(tmp_path / "prior"))
    receipt = census_source(new, str(tmp_path / "new-contract"), prior=prior)
    assert receipt is not None
    assert receipt.summary == prior.summary
    assert receipt.input == new.input and receipt.input != prior.input


def test_binder_declared_contradiction_fails(tmp_path):
    receipt = census_source(population_fixture(tmp_path, (reply_task("task-1", JUDGE),)), str(tmp_path / "census"))
    with pytest.raises(ValueError, match="contradict its derived contract"):
        bind_derived_contract_tags(
            source_with(SourceReference("open-athena/fixture", "pin", "url"), tags=("agentic", "multi-turn")),
            receipt,
        )
