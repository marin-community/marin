# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert AceCode's inferred-test-case problems with a shipped assert-case grader."""

from pathlib import Path

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.script_grader import grade_script, script_package
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.models import TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizationChange, NormalizedTask, RawRow
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

HERE = Path(__file__).parent
REPO = "TIGER-Lab/AceCode-89K"
REVISION = "13216309a9f6cb40b60cb1a9750071efeac414ad"
SOURCE_ID = "Tiger-lab:AceCode-89K"
VERSION = "acecode-assert-grader-v1"
COUNT = 87149
GRADER_IMAGE = (
    "ghcr.io/marin-community/iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab"
)
GRADER_TIMEOUT = 330.0
COMPLETION_INSTRUCTION = "\nReturn the complete Python solution in this format:\n```python\n# solution\n```"
ACECODE_GRADE = grade_script(HERE / "acecode_grade.py")


def convert_problem(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    native = dict(row.data)
    required = {"question", "test_cases", "inferences", "context_messages", "source", "id"}
    if not required <= set(native) or not isinstance(native["question"], str) or not native["question"].strip():
        return source_defect("incomplete_acecode_contract", "The task fields with a nonempty question are required")
    cases = native["test_cases"]
    if not isinstance(cases, list) or not cases or not all(isinstance(case, str) and case.strip() for case in cases):
        return source_defect("invalid_test_cases", "test_cases must be a nonempty list of assert strings")
    for case in cases:
        if not case.lstrip().startswith("assert"):
            return source_defect("non_assert_test_case", "Every test case must be an assert statement")
    if not isinstance(native["id"], str) or not native["id"].strip():
        return source_defect("invalid_acecode_metadata", "The id must be a nonempty string")
    config = {"test_cases": cases, "task_id": native["id"]}
    package = script_package(
        ACECODE_GRADE,
        config,
        environment=required_grader_environment(context),
        timeout=GRADER_TIMEOUT,
        answer_path=DEFAULT_OUTPUT,
    )
    events = (TextMessage(role="user", content=native["question"] + COMPLETION_INSTRUCTION),)
    task = conversation_task(row, events=events, package=package).model_copy(
        update={
            "tags": ("train", "code", "test-runner", "acecode"),
            "labels": {"source_id": SOURCE_ID, "native_source": native["source"], "native_id": native["id"]},
        }
    )
    change = NormalizationChange(
        field="context.events",
        reason="Append the completion format instruction to the source question; the question stays verbatim.",
        original="The native question asks for an implementation without fixing a reply format.",
        replacement=(
            "The task adds the fenced-block completion instruction so the grader can extract the candidate's code."
        ),
    )
    return NormalizedTask(task, (change,))


def source(files: tuple[str, ...]) -> RlDataSource:
    return RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.TIGER_LAB,
            upstream_dataset="AceCode-89K",
            upstream_name="AceCode-89K",
            build_path=VERSION,
            family="code",
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=COUNT,
            notes=(
                "The source's inferred assert cases are the grader: each task's context is the verbatim question, the "
                "candidate completes it in a fenced block, and the shipped runner execs the candidate's block then "
                "every assert case in one namespace. Reward 1 when every case passes, 0 on any candidate-code "
                "failure; harness defects exit nonzero as infrastructure errors. The source's 64 sampled model "
                "inferences stay private provenance, never judge gold. The completion format instruction is "
                "recorded as a normalization. No live execution is claimed by conversion."
            ),
        ),
        config=CurationRecipe(
            name="acecode-assert-grader",
            version=VERSION,
            source=HfSource(REPO, REVISION, files, SourceFormat.PARQUET),
            grader=Environment(image=GRADER_IMAGE),
            convert=convert_problem,
            intended_use=IntendedUse.TRAIN,
        ),
        pipeline=process_rows,
    )
