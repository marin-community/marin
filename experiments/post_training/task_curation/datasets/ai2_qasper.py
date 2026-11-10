# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared QASPER data and task construction."""

import json
from collections.abc import Iterator, Sequence
from typing import Any, TypedDict

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.tasks import shell_environment
from taskcompendium.grader import GraderPackage
from taskcompendium.models import AnswerType, ConversationInput, PlainText, ResourceGroups, TaskSpec, TextMessage
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import environment_requirements

REPOSITORY = "allenai/qasper"
REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
TRAIN_FILE = "qasper/train/0000.parquet"
PAPER_PATH = "/app/paper.json"
ANSWER_PATH = "/app/answer.json"
PAPER_FIELDS = ("id", "title", "abstract", "full_text", "figures_and_tables")
AGENT_IMAGE = Environment(
    image="ghcr.io/marin-community/iris-task@sha256:66cba7cb3eb682f9a53e444876ef2468670336a71e03559de85b5b2b5d4cdde6"
)

PROMPT = (
    "Access and read the research paper located at `{paper_path}`. "
    "Answer the following question using only information from the paper.\n\n"
    "Write your answer to `{answer_path}` using this format:\n\n"
    '{{"answer": "<answer>"}}\n\n'
    "If the question can be answered succinctly with yes or no, use exactly `yes` or `no` as the answer value. "
    "If the paper does not provide enough information to answer the question, use exactly `unanswerable`. "
    "Otherwise, provide a clear, concise answer supported by the paper.\n\n"
    "Question: {question}"
)


class EvidenceAnnotation(TypedDict):
    evidence: list[str]


class CategoricalAnnotation(TypedDict):
    unanswerable: bool
    yes_no: bool | None


def uses_missing_evidence(annotations: Sequence[EvidenceAnnotation]) -> bool:
    """Return whether an answer cites a figure or table unavailable in paper.json."""
    evidence = (item for answer in annotations for item in answer["evidence"])
    return any(item.startswith("FLOAT SELECTED:") for item in evidence)


def read_questions(path: StoragePath) -> Iterator[dict[str, Any]]:
    """Read one row per QASPER question with the public paper fields."""
    with path.open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=32, use_threads=False):
            for paper in batch.to_pylist():
                public_paper = {field: paper[field] for field in PAPER_FIELDS}
                qas = paper["qas"]
                for question, question_id, answers in zip(
                    qas["question"], qas["question_id"], qas["answers"], strict=True
                ):
                    yield {
                        "path": f"{paper['id']}/{question_id}",
                        "paper": public_paper,
                        "question": question,
                        "question_id": question_id,
                        "answers": answers,
                    }


def build_task(row: RawRow, package: GraderPackage, tags: tuple[str, ...]) -> TaskSpec:
    """Build a QASPER paper workspace task with the shared answer contract."""
    data = row.data
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(
            events=(
                TextMessage(
                    role="user",
                    content=PROMPT.format(
                        paper_path=PAPER_PATH,
                        answer_path=ANSWER_PATH,
                        question=data["question"],
                    ),
                ),
            )
        ),
        environment_requirements=shell_environment(environment_requirements(AGENT_IMAGE)).model_copy(
            update={"working_directory": "/app"}
        ),
        resources=ResourceGroups(
            worker=(
                inline_resource(PAPER_PATH.removeprefix("/"), json.dumps(data["paper"], ensure_ascii=False).encode()),
            ),
            verifier=package.resources,
        ),
        output_paths=(ANSWER_PATH,),
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        grader=package.grader,
        tags=tags,
    )
