# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned GenomeQA source conversion; upstream prompt remains an external input."""

import ast
from collections.abc import Iterator
from functools import lru_cache
from string import ascii_uppercase

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item, file_sha256, parquet_rows
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

PROMPT_FILENAME = "run.py"
FORMAT_BY_SHA256 = {
    "0ef4c3d33541980e3fded9a5b23a952296ce3c51cb0415b42c0299a61c21938a": "binary",
    "c1cbb51449eb8a5241890337c88dc1bc0a58974b8eb4cb8bc49f310b12faeb75": "mcq",
}


@lru_cache(maxsize=8)
def load_system_prompt(path: str) -> str:
    """Read the pinned literal without executing upstream code."""
    with StoragePath(path).open("r") as stream:
        tree = ast.parse(stream.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SIMPLE_SYSTEM_PROMPT" for target in node.targets
        ):
            prompt = ast.literal_eval(node.value)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("GenomeQA system prompt must be a nonempty string literal")
            return prompt
    raise ValueError("Pinned GenomeQA runner has no SIMPLE_SYSTEM_PROMPT literal")


def read_genomeqa(path: StoragePath, _context: ConversionContext) -> Iterator[dict]:
    """Identify the pinned format by file content, including renamed local inputs."""
    answer_format = FORMAT_BY_SHA256.get(file_sha256(path))
    if answer_format is None:
        raise ValueError("GenomeQA checksum mismatch")
    for row in parquet_rows(path):
        yield {**row, "format": answer_format}


def render_question(row: dict) -> str:
    options = "\n".join(f"{letter}. {text}" for letter, text in zip(ascii_uppercase, row["options"], strict=False))
    return f"Question:\n{row['question']}\n\nOptions:\n{options}\n\nPlease select the correct option."


def convert_genomeqa(row: RawRow, context: ConversionContext) -> TaskSpec:
    data = dict(row.data)
    declared_format = row.source.row.partition("/")[0]
    if data["format"] != declared_format:
        raise ValueError(f"GenomeQA {declared_format} input contains {data['format']} data")
    prompt = load_system_prompt(str(context.inputs["prompt"] / PROMPT_FILENAME))
    task = conversation_task(
        row,
        events=(TextMessage(role="system", content=prompt), TextMessage(role="user", content=render_question(data))),
        package=source_package(
            "genomeqa",
            {
                "expected": ascii_uppercase[data["answer_index"]],
                "options": ascii_uppercase[: len(data["options"])],
                # Each motif cell has 100 questions. The other cells have 500.
                "cell_weight": 5 if data["subtask"] == "6_motif" else 1,
            },
            environment=required_grader_environment(context),
        ),
        evidence=data,
    )
    return task.model_copy(update={"tags": ("eval", "benchmark:GenomeQA")})


def genomeqa_item(task: TaskSpec) -> EvaluationItem:
    data = source_evidence(task)
    return chat_item(task, response_selection="final_only").model_copy(
        update={
            "reference_text": ascii_uppercase[data["answer_index"]],
            "groups": {
                "subtask": data["subtask"],
                "format": data["format"],
                "cell": f"{data['subtask']}/{data['format']}",
            },
        }
    )
