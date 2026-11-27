# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Parametric public questions as single-turn tasks graded by the starter notebook's judge prompt."""

import csv

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.convert.answers import answer_task, source_evidence
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.judge_prompts import facts_parametric_spec
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow

FACTS_PARAMETRIC_COLUMNS = ["url", "query", "answer", "topic"]


def read_facts_parametric(file: StoragePath, _context: ConversionContext):
    with file.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FACTS_PARAMETRIC_COLUMNS:
            raise ValueError(f"Unexpected FACTS Parametric columns: {reader.fieldnames}")
        for index, row in enumerate(reader):
            yield {"source_index": index, **row}


def convert_facts_parametric(row: RawRow, _context: ConversionContext) -> TaskSpec:
    """Send the bare question as one user message, as the starter's query template does."""
    task = answer_task(
        row,
        prompt=row.data["query"],
        spec=facts_parametric_spec(row.data["query"], row.data["answer"]),
        evidence={key: row.data[key] for key in ("source_index", "url", "topic")},
    )
    return task.model_copy(update={"tags": ("eval", "source:facts-parametric", "benchmark:facts-parametric")})


def facts_parametric_item(task: TaskSpec) -> EvaluationItem:
    """Grade only the final answer, and report scores by question topic."""
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"topic": source_evidence(task)["topic"]}}
    )
