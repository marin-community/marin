# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned GenomeQA source conversion; upstream prompt remains an external input."""

import ast
from collections import Counter
from collections.abc import Iterator
from functools import lru_cache
from string import ascii_uppercase

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item, file_sha256
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

DATASET = "HuggingFaceBio/genomeqa"
REVISION = "e655e415fa30681dddbaf4acc182bcbd8ac41943"
PROMPT_REVISION = "7ccc230e734fc11a365fd1963b593b5fd11e834d"
PROMPT_URL = f"https://raw.githubusercontent.com/ai4nucleome/GenomeQA/{PROMPT_REVISION}/run.py"
SUBTASKS = (
    "1_enhancer_vs_promoter",
    "2_splice_sites",
    "3_species_cls",
    "4_histone_marks",
    "5_tfbs",
    "6_motif",
)
FORMATS = ("binary", "mcq")
CELL_COUNTS = {
    f"{task}/{answer_format}": 100 if task == "6_motif" else 500 for task in SUBTASKS for answer_format in FORMATS
}
FILE_HASHES = {
    "binary": "0ef4c3d33541980e3fded9a5b23a952296ce3c51cb0415b42c0299a61c21938a",
    "mcq": "c1cbb51449eb8a5241890337c88dc1bc0a58974b8eb4cb8bc49f310b12faeb75",
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


def read_genomeqa(path: StoragePath, context: ConversionContext) -> Iterator[dict]:
    """Validate each complete pinned format before publishing its questions."""
    answer_format = str(path).split("/")[-2]
    if file_sha256(path) != FILE_HASHES[answer_format]:
        raise ValueError(f"GenomeQA {answer_format} checksum mismatch")
    counts = Counter()
    seen = set()
    with path.open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                cell = f"{row['subtask']}/{answer_format}"
                if cell not in CELL_COUNTS:
                    raise ValueError(f"Unknown GenomeQA cell: {cell}")
                key = (row["subtask"], row["id"])
                if key in seen:
                    raise ValueError(f"Duplicate GenomeQA identity: {key}")
                seen.add(key)
                choices, answer = row["options"], row["answer_index"]
                if len(choices) != (2 if answer_format == "binary" else 4) or not 0 <= answer < len(choices):
                    raise ValueError(f"Invalid GenomeQA options or answer: {key}")
                if choices[answer] != row["answer_text"]:
                    raise ValueError(f"Inconsistent GenomeQA answer: {key}")
                counts[cell] += 1
                yield {**row, "format": answer_format}
    if counts != {cell: count for cell, count in CELL_COUNTS.items() if cell.endswith("/" + answer_format)}:
        raise ValueError(f"GenomeQA population differs from pinned counts: {counts}")


def render_question(row: dict) -> str:
    options = "\n".join(f"{letter}. {text}" for letter, text in zip(ascii_uppercase, row["options"], strict=False))
    return f"Question:\n{row['question']}\n\nOptions:\n{options}\n\nPlease select the correct option."


def convert_genomeqa(row: RawRow, context: ConversionContext) -> TaskSpec:
    data = dict(row.data)
    prompt = load_system_prompt(str(context.inputs["prompt"] / "run.py"))
    task = conversation_task(
        row,
        events=(TextMessage(role="system", content=prompt), TextMessage(role="user", content=render_question(data))),
        package=source_package(
            "genomeqa",
            {
                "expected": ascii_uppercase[data["answer_index"]],
                "options": ascii_uppercase[: len(data["options"])],
                "cell_weight": 500 / CELL_COUNTS[f"{data['subtask']}/{data['format']}"],
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
            "groups": {
                "subtask": data["subtask"],
                "format": data["format"],
                "cell": f"{data['subtask']}/{data['format']}",
            }
        }
    )
