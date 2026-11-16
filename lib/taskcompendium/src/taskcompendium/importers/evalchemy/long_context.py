# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned MRCR and GraphWalks source decoding and evaluation inputs."""

import json
from functools import cache

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import ChatGeneration, EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.importers.evalchemy.scorers.long_context import FINAL_ANSWER_PREFIX, mrcr_bin
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

# Full population of the pinned openai/mrcr revision, before run-time eligibility.
MRCR_SOURCE_POPULATION = 2400


@cache
def official_tokenizer():
    # o200k is a curation dependency; inference uses the saved official counts.
    from tiktoken import get_encoding  # noqa: PLC0415

    return get_encoding("o200k_base")


def official_token_count(row: dict) -> int:
    encode = official_tokenizer().encode
    return sum(len(encode(message["content"])) for message in json.loads(row["prompt"])) + len(encode(row["answer"]))


def decode_mrcr(row: dict) -> dict:
    """Attach official counts without materializing a full Parquet row group."""
    tokens = official_token_count(row)
    return {**row, "official_tokens": tokens, "mrcr_bin_upper": mrcr_bin(tokens)}


def read_mrcr(path: StoragePath, _context: ConversionContext):
    """Yield source rows annotated with official token counts and benchmark bins."""
    with path.open("rb") as file:
        parquet = pq.ParquetFile(file)
        # A full MRCR row group can exceed a curation worker's memory budget.
        for batch in parquet.iter_batches(batch_size=1):
            for row in batch.to_pylist():
                yield decode_mrcr(row)


def convert_mrcr(row: RawRow, context: ConversionContext) -> TaskSpec:
    data = row.data
    messages = json.loads(data["prompt"])
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=messages[-1]["content"]),),
        package=source_package(
            "mrcr",
            {"expected": data["answer"], "nonce": data["random_string_to_prepend"]},
            environment=required_grader_environment(context),
        ),
        evidence=dict(data),
    )
    return task.model_copy(
        update={
            "context": ConversationInput(
                events=tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
            ),
            "tags": ("eval", "source:evalchemy", "benchmark:MRCR", "long-context"),
            "labels": {
                "mrcr.needles": str(data["n_needles"]),
                **({"mrcr.bin_upper": str(data["mrcr_bin_upper"])} if data["mrcr_bin_upper"] is not None else {}),
            },
        }
    )


def mrcr_item(task: TaskSpec) -> EvaluationItem:
    row = source_evidence(task)
    return chat_item(task).model_copy(
        update={
            "operation": ChatGeneration(historical_assistant_reasoning="empty"),
            "groups": {"cell": f"{row['mrcr_bin_upper']}_{row['n_needles']}needle"},
        }
    )


def convert_graphwalks(row: RawRow, context: ConversionContext) -> TaskSpec:
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=row.data["prompt"]),),
        package=source_package(
            "graphwalks",
            {"expected": row.data["answer_nodes"]},
            environment=required_grader_environment(context),
        ),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:GraphWalks", "long-context")})


def graphwalks_item(task: TaskSpec) -> EvaluationItem:
    row = source_evidence(task)
    return chat_item(task).model_copy(
        update={
            "prompt_characters": row["prompt_chars"],
            "reference_text": FINAL_ANSWER_PREFIX + " [" + ", ".join(row["answer_nodes"]) + "]",
            "groups": {"problem_type": row["problem_type"]},
        }
    )
