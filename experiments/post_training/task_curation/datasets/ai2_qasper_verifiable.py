# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""QASPER Verifiable questions with annotator-agreed exact answers."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.script_grader import grade_script, script_package
from taskcompendium.convert.tasks import shell_environment
from taskcompendium.models import AnswerType, ConversationInput, PlainText, ResourceGroups, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import Controls, IntendedUse, RawRow, WorkspaceFiles
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import (
    CurationRecipe,
    HfSource,
    environment_requirements,
    process_rows,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

REPOSITORY = "allenai/qasper"
REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
TRAIN_FILE = "qasper/train/0000.parquet"
ANSWER_PATH = "/app/answer.json"
PAPER_FIELDS = ("id", "title", "abstract", "full_text", "figures_and_tables")
GRADER = Path(__file__).with_name("ai2_qasper_verifiable_grade.py")
AGENT_IMAGE = Environment(
    image="ghcr.io/marin-community/iris-task@sha256:66cba7cb3eb682f9a53e444876ef2468670336a71e03559de85b5b2b5d4cdde6"
)

PROMPT = """Access and read the research paper located at `/app/paper.json`. Answer the following question using only information from the paper.

Write your answer to `/app/answer.json` using this format:

{{"answer": "<answer>"}}

If the question can be answered succinctly with yes or no, use exactly `yes` or `no` as the answer value. If the paper does not provide enough information to answer the question, use exactly `unanswerable`. Otherwise, provide a clear, concise answer supported by the paper.

Question: {question}"""

RUBRIC = """
The complete paper is public in paper.json. Check that the question is understandable from that paper and that the
agreed yes, no, or unanswerable reference is supported. Do not require outside knowledge. The JSON answer contract is
intentional and should not be treated as a quality defect.
"""


def agreed_answer(annotations: list[dict[str, Any]]) -> str | None:
    '''This function returns the answers the annotators agreed on. If there was disagreement, it returns None.'''
    labels = []
    for answer in annotations:
        if answer["unanswerable"]:
            labels.append("unanswerable")
        elif answer["yes_no"] is not None:
            labels.append("yes" if answer["yes_no"] else "no")
        else:
            return None
    return labels[0] if labels and len(set(labels)) == 1 else None


def uses_missing_evidence(annotations: list[dict[str, Any]]) -> bool:
    '''This function checks if the answer requires evidence (like images) that the model doesn't have access to through paper.json'''
    evidence = (item for answer in annotations for item in answer["evidence"])
    return any(item.startswith("FLOAT SELECTED:") for item in evidence)


def read_questions(path: StoragePath, _context: ConversionContext) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=32, use_threads=False):
            for paper in batch.to_pylist():
                public_paper = {field: paper[field] for field in PAPER_FIELDS}
                qas = paper["qas"]
                for question, question_id, answers in zip(
                    qas["question"], qas["question_id"], qas["answers"], strict=True
                ):
                    expected = agreed_answer(answers["answer"])
                    if expected is not None and not uses_missing_evidence(answers["answer"]):
                        yield {
                            "path": f"{paper['id']}/{question_id}",
                            "paper": public_paper,
                            "question": question,
                            "question_id": question_id,
                            "answers": answers,
                            "expected": expected,
                        }


def convert(row: RawRow, context: ConversionContext) -> TaskSpec:
    data = row.data
    package = script_package(
        grade_script(GRADER),
        {
            "expected": data["expected"],
            "paper_id": data["paper"]["id"],
            "question_id": data["question_id"],
            "annotations": data["answers"],
        },
        environment=required_grader_environment(context),
        timeout=10,
        answer_path=None,
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=PROMPT.format(question=data["question"])),)),
        environment_requirements=shell_environment(environment_requirements(AGENT_IMAGE)).model_copy(
            update={"working_directory": "/app"}
        ),
        resources=ResourceGroups(
            worker=(inline_resource("app/paper.json", json.dumps(data["paper"], ensure_ascii=False).encode()),),
            verifier=package.resources,
        ),
        output_paths=(ANSWER_PATH,),
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        grader=package.grader,
        tags=("qa", "paper", "exact", "qasper"),
    )


def golden_answer(task: TaskSpec) -> WorkspaceFiles:
    config = json.loads(resource_bytes(next(file for file in task.resources.verifier if file.path == "config.json")))
    answer = json.dumps({"answer": config["expected"]}).encode()
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
