# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""QASPER Verifiable questions with annotator-agreed exact answers."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.script_grader import grade_script, script_package
from taskcompendium.grader import grader_config
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import Controls, IntendedUse, RawRow, WorkspaceFiles

from experiments.post_training.task_curation.datasets.ai2_qasper import (
    ANSWER_PATH,
    REPOSITORY,
    REVISION,
    TRAIN_FILE,
    CategoricalAnnotation,
    build_task,
    uses_missing_evidence,
)
from experiments.post_training.task_curation.datasets.ai2_qasper import (
    read_questions as read_qasper_questions,
)
from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.pipeline import (
    CurationRecipe,
    HfSource,
    process_rows,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

GRADER = Path(__file__).with_name("ai2_qasper_verifiable_grade.py")

RUBRIC = """
The complete paper is public in paper.json. Check that the question is understandable from that paper and that the
agreed yes, no, or unanswerable reference is supported. Do not require outside knowledge. The JSON answer contract is
intentional and should not be treated as a quality defect.
"""


def agreed_answer(annotations: list[CategoricalAnnotation]) -> str | None:
    """This function returns the answers the annotators agreed on. If there was disagreement, it returns None.

    Empty annotations and non-categorical answers also return None.
    """
    labels = []
    for answer in annotations:
        if answer["unanswerable"]:
            labels.append("unanswerable")
        elif answer["yes_no"] is not None:
            labels.append("yes" if answer["yes_no"] else "no")
        else:
            return None
    return labels[0] if labels and len(set(labels)) == 1 else None


def read_questions(path: StoragePath, _context: ConversionContext) -> Iterator[dict[str, Any]]:
    for row in read_qasper_questions(path):
        expected = agreed_answer(row["answers"]["answer"])
        if expected is not None and not uses_missing_evidence(row["answers"]["answer"]):
            yield {**row, "expected": expected}


def convert(row: RawRow, context: ConversionContext) -> TaskSpec:
    data = row.data
    package = script_package(
        grade_script(GRADER),
        {
            "expected": data["expected"],
            "paper_id": data["paper"]["id"],
            "question_id": data["question_id"],
            "annotations": data["answers"],
            "answer_path": ANSWER_PATH,
        },
        environment=required_grader_environment(context),
        timeout=10,
        answer_path=None,
    )
    return build_task(row, package, ("qa", "paper", "exact", "qasper"))


def golden_answer(task: TaskSpec) -> WorkspaceFiles:
    answer = json.dumps({"answer": grader_config(task)["expected"]}).encode()
    return WorkspaceFiles({ANSWER_PATH: answer})


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                id="QASPER:ai2_qasper_verifiable",
                title="QASPER Verifiable",
                origin="QASPER",
                family="paper-question-answering",
                tags=("rlvr", "agentic", "single-turn", "license:cc-by-4.0"),
                count=633,
                notes="Training questions with agreed exact answers and no unavailable figure or table evidence.",
            ),
            config=CurationRecipe(
                name="ai2_qasper_verifiable",
                source=HfSource(REPOSITORY, REVISION, (TRAIN_FILE,), SourceFormat.PARQUET, read=read_questions),
                convert=convert,
                version="4",
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                controls=Controls(golden=golden_answer),
                grader=GRADER_PACKAGES,
            ),
        )
    ]
