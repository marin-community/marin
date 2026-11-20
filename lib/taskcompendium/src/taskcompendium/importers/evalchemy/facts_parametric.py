# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Parametric public questions as single-turn tasks graded by the starter notebook's judge prompt."""

import csv

from rigging.filesystem.storage_path import StoragePath
from verifyit.spec import JudgeSpec

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

FACTS_PARAMETRIC_COLUMNS = ["url", "query", "answer", "topic"]
JUDGE_REQUEST_TIMEOUT = 300
JUDGE_TRANSPORT_BUDGET = 600


def read_facts_parametric(file: StoragePath, _context: ConversionContext):
    with file.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FACTS_PARAMETRIC_COLUMNS:
            raise ValueError(f"Unexpected FACTS Parametric columns: {reader.fieldnames}")
        for index, row in enumerate(reader):
            yield {"source_index": index, **row}


def convert_facts_parametric(row: RawRow, context: ConversionContext) -> TaskSpec:
    """Send the bare question as one user message, as the starter's query template does."""
    judge = JudgeSpec(
        # The scorer reads the question and gold answer from the judge contract.
        question=row.data["query"],
        references=(row.data["answer"],),
        exact_gate=False,
        request_timeout=JUDGE_REQUEST_TIMEOUT,
        transport_attempts=3,
        transport_budget=JUDGE_TRANSPORT_BUDGET,
    )
    environment = required_grader_environment(context)
    environment = environment.model_copy(update={"capabilities": (*environment.capabilities, "network")})
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=row.data["query"]),),
        package=source_package(
            "facts_parametric",
            {},
            environment=environment,
            judge=judge,
            timeout=60 + JUDGE_TRANSPORT_BUDGET,
        ),
        evidence={key: row.data[key] for key in ("source_index", "url", "topic")},
    )
    return task.model_copy(update={"tags": ("eval", "source:facts-parametric", "benchmark:facts-parametric")})


def facts_parametric_item(task: TaskSpec) -> EvaluationItem:
    """Grade only the final answer, and report scores by question topic."""
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"topic": source_evidence(task)["topic"]}}
    )
