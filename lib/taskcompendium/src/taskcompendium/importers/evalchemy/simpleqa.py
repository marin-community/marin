# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical SimpleQA and SimpleQA Verified source rows, and the seeded SimpleQAMini sample.

Source locators index the emitted stream; source_index keeps the original CSV
identity for paired comparisons independently of task IDs and sampled order.
"""

import ast
import csv
import random
from collections.abc import Callable

from rigging.filesystem.storage_path import StoragePath
from verifyit.spec import JudgeSpec

from taskcompendium.convert.answers import answer_task, source_evidence
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.judge_prompts import simpleqa_spec, simpleqa_verified_spec
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


_BOOLEANS = {"True": True, "False": False}


def read_simpleqa_verified(file: StoragePath, _context: ConversionContext):
    with file.open("r", newline="", encoding="utf-8") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            yield {
                "source_index": index,
                "question": row["problem"],
                "answer": row["answer"],
                "metadata": {
                    # Row index in OpenAI's SimpleQA CSV, for pairing with SimpleQA results.
                    "original_index": int(row["original_index"]),
                    "topic": row["topic"],
                    "answer_type": row["answer_type"],
                    "multi_step": _BOOLEANS[row["multi_step"]],
                    "requires_reasoning": _BOOLEANS[row["requires_reasoning"]],
                    "urls": row["urls"],
                },
            }


def convert_simpleqa(row: RawRow, _context: ConversionContext, *, benchmark: str) -> TaskSpec:
    return _simpleqa_task(row, benchmark, simpleqa_spec)


def convert_simpleqa_verified(row: RawRow, _context: ConversionContext, *, benchmark: str) -> TaskSpec:
    return _simpleqa_task(row, benchmark, simpleqa_verified_spec)


def _simpleqa_task(row: RawRow, benchmark: str, judge_spec: Callable[[str, str], JudgeSpec]) -> TaskSpec:
    task = answer_task(
        row,
        prompt=row.data["question"],
        spec=judge_spec(row.data["question"], row.data["answer"]),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:" + benchmark)})


def simpleqa_item(task: TaskSpec) -> EvaluationItem:
    evidence = source_evidence(task)
    groups = {"source_index": str(evidence["source_index"])} if "source_index" in evidence else {}
    return chat_item(task).model_copy(update={"groups": groups})


def simpleqa_verified_item(task: TaskSpec) -> EvaluationItem:
    """Grade only the final answer: the reference implementation never shows the judge a reasoning trace."""
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"source_index": str(source_evidence(task)["source_index"])}}
    )
