# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical SimpleQA source rows and seeded SimpleQAMini sample.

Source locators index the emitted stream; source_index keeps the original CSV
identity for paired comparisons independently of task IDs and sampled order.
"""

import ast
import csv
import random

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.convert.answers import answer_task, source_evidence
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.judge_prompts import simpleqa_spec
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow


def read_simpleqa(file: StoragePath, _context: ConversionContext, *, mini: bool):
    with file.open("r", newline="", encoding="utf-8") as stream:
        rows = [
            {
                "source_index": index,
                "question": row["problem"],
                "answer": row["answer"],
                "metadata": ast.literal_eval(row["metadata"]),
            }
            for index, row in enumerate(csv.DictReader(stream))
        ]
    yield from random.Random(0).sample(rows, 500) if mini else rows


def convert_simpleqa(row: RawRow, _context: ConversionContext, *, benchmark: str) -> TaskSpec:
    task = answer_task(
        row,
        prompt=row.data["question"],
        spec=simpleqa_spec(row.data["question"], row.data["answer"]),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:" + benchmark)})


def simpleqa_item(task: TaskSpec) -> EvaluationItem:
    evidence = source_evidence(task)
    groups = {"source_index": str(evidence["source_index"])} if "source_index" in evidence else {}
    return chat_item(task).model_copy(update={"groups": groups})
