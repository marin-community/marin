# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty-graded math problem pool for the curriculum-RL experiment.

The pool is one partition shared by every sampling arm: bins ordered by grade
(0 easiest), each bin a pinned slice of a public math dataset. Rows use the
SkyRL parquet schema with a per-row ``env_class`` so one training run mixes
verifier environments freely; ``extra_info`` carries the bin name and grade for
curriculum samplers and per-source metrics.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from datasets import concatenate_datasets, load_dataset
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from rigging.filesystem.storage_path import prefix_join
from transformers import AutoTokenizer
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)

GSM8K_DATASET = "openai/gsm8k"
GSM8K_REVISION = "e53f048"
MATH_DATASET = "EleutherAI/hendrycks_math"
MATH_REVISION = "21a5633"
MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
AIME_DATASET = "di-zhang-fdu/AIME_1983_2024"
AIME_REVISION = "3e2cc86"
MATH500_DATASET = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a"

GSM8K_ENV = "gsm8k"
BOXED_ENV = "aime"

GSM8K_INSTRUCTION = ' Let\'s think step by step and output the final answer after "####".'
# The aime env verifies with the Minerva "Answer: ..." extraction (not \boxed),
# so the instruction must elicit a final Answer line.
BOXED_INSTRUCTION = " Please reason step by step, and end your response with a final line 'Answer: <answer>'."

GSM8K_TRAIN_ROWS = 2000
GSM8K_VALIDATION_ROWS = 256

QWEN3_MODEL = "Qwen/Qwen3-0.6B"
QWEN3_REVISION = "c1899de"
# The trainer skips generation for prompts above max_input_length
# (request_window_tokens - max_new_tokens = 1024 in both presets), and a fully
# skipped GRPO group fails group admission. Drop over-length train prompts at
# pool build with a margin for chat-template variants.
MAX_PROMPT_TOKENS = 1000

TRAIN_FILENAME = "train.parquet"
VALIDATION_FILENAME = "validation.parquet"

_GSM8K_FINAL_ANSWER = re.compile(r"####\s*(-?[0-9.,]+)")


@dataclass(frozen=True)
class PoolBin:
    """One difficulty bin: a named, graded slice of a source dataset."""

    name: str
    grade: int
    env_class: str


# Grades order bins easiest to hardest. MATH levels 1-2 sit in one bin because
# level 1 alone has only 564 train rows.
GSM8K_BIN = PoolBin("g0-gsm8k", grade=0, env_class=GSM8K_ENV)
MATH_L12_BIN = PoolBin("g1-math-l12", grade=1, env_class=BOXED_ENV)
MATH_L3_BIN = PoolBin("g2-math-l3", grade=2, env_class=BOXED_ENV)
MATH_L4_BIN = PoolBin("g3-math-l4", grade=3, env_class=BOXED_ENV)
MATH_L5_BIN = PoolBin("g4-math-l5", grade=4, env_class=BOXED_ENV)
AIME_BIN = PoolBin("g5-aime", grade=5, env_class=BOXED_ENV)
MATH_BINS_BY_LEVEL: Mapping[str, PoolBin] = {
    "Level 1": MATH_L12_BIN,
    "Level 2": MATH_L12_BIN,
    "Level 3": MATH_L3_BIN,
    "Level 4": MATH_L4_BIN,
    "Level 5": MATH_L5_BIN,
}
POOL_BINS = (GSM8K_BIN, MATH_L12_BIN, MATH_L3_BIN, MATH_L4_BIN, MATH_L5_BIN, AIME_BIN)
POOL_GRADE_COUNT = max(b.grade for b in POOL_BINS) + 1

VALIDATION_GSM8K_SOURCE = "val-gsm8k"
VALIDATION_MATH500_SOURCE = "val-math500"


@dataclass(frozen=True)
class PoolParquetConfig:
    output_path: str


def boxed_answer(solution: str) -> str | None:
    """Extract the final ``\\boxed{...}`` argument, honoring nested braces."""
    start = solution.rfind("\\boxed")
    if start == -1:
        return None
    cursor = start + len("\\boxed")
    if cursor >= len(solution) or solution[cursor] != "{":
        return None
    depth = 0
    for index in range(cursor, len(solution)):
        if solution[index] == "{":
            depth += 1
        elif solution[index] == "}":
            depth -= 1
            if depth == 0:
                answer = solution[cursor + 1 : index].strip()
                return answer or None
    return None


def _pool_record(
    *,
    question: str,
    answer: str,
    pool_bin: PoolBin,
    split: str,
    index: int,
    data_source: str | None = None,
) -> dict[str, object]:
    instruction = GSM8K_INSTRUCTION if pool_bin.env_class == GSM8K_ENV else BOXED_INSTRUCTION
    source = data_source or pool_bin.name
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": f"{question}{instruction}"}],
        "env_class": pool_bin.env_class,
        # The gsm8k env reads reward_spec.ground_truth; the aime env reads
        # reward_model.ground_truth. Every row carries both.
        "reward_spec": {"method": "rule", "ground_truth": answer},
        "reward_model": {"ground_truth": answer},
        "extra_info": {
            "data_source": source,
            "grade": pool_bin.grade,
            "split": split,
            "index": index,
        },
    }


def _gsm8k_records(split: str, rows: int, *, data_source: str | None = None) -> list[dict[str, object]]:
    dataset = load_dataset(GSM8K_DATASET, "main", split=split, revision=GSM8K_REVISION).select(range(rows))
    records = []
    for index, example in enumerate(dataset):
        match = _GSM8K_FINAL_ANSWER.search(example["answer"])
        if match is None:
            raise ValueError(f"GSM8K row {split}/{index} has no final answer marker")
        answer = match.group(1).replace(",", "")
        records.append(
            _pool_record(
                question=example["question"],
                answer=answer,
                pool_bin=GSM8K_BIN,
                split=split,
                index=index,
                data_source=data_source,
            )
        )
    return records


def _math_records() -> list[dict[str, object]]:
    subjects = [load_dataset(MATH_DATASET, subject, split="train", revision=MATH_REVISION) for subject in MATH_SUBJECTS]
    dataset = concatenate_datasets(subjects)
    records = []
    skipped = 0
    for index, example in enumerate(dataset):
        pool_bin = MATH_BINS_BY_LEVEL.get(example["level"])
        answer = boxed_answer(example["solution"])
        if pool_bin is None or answer is None:
            skipped += 1
            continue
        records.append(
            _pool_record(question=example["problem"], answer=answer, pool_bin=pool_bin, split="train", index=index)
        )
    logger.info("MATH: kept %d rows, skipped %d without level or boxed answer", len(records), skipped)
    return records


def _aime_records() -> list[dict[str, object]]:
    dataset = load_dataset(AIME_DATASET, split="train", revision=AIME_REVISION)
    records = []
    for index, example in enumerate(dataset):
        answer = str(example["Answer"]).strip()
        if not answer or answer.lower() == "none":
            continue
        records.append(
            _pool_record(question=example["Question"], answer=answer, pool_bin=AIME_BIN, split="train", index=index)
        )
    return records


def _math500_records() -> list[dict[str, object]]:
    dataset = load_dataset(MATH500_DATASET, split="test", revision=MATH500_REVISION)
    math500_bin = PoolBin(VALIDATION_MATH500_SOURCE, grade=2, env_class=BOXED_ENV)
    return [
        _pool_record(
            question=example["problem"], answer=example["answer"], pool_bin=math500_bin, split="test", index=index
        )
        for index, example in enumerate(dataset)
    ]


def _drop_over_length_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Drop rows whose templated prompt would be skipped by the trainer."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL, revision=QWEN3_REVISION)
    kept = []
    dropped: Counter[str] = Counter()
    for record in records:
        encoded = tokenizer.apply_chat_template(
            record["prompt"], add_generation_prompt=True, tokenize=True, return_dict=True, enable_thinking=False
        )
        if len(encoded["input_ids"]) > MAX_PROMPT_TOKENS:
            dropped[str(record["data_source"])] += 1
        else:
            kept.append(record)
    if dropped:
        logger.info("Dropped over-length prompts per bin: %s", dict(dropped))
    return kept


def write_pool_parquet(config: PoolParquetConfig) -> None:
    """Write the graded train pool and the fixed validation set."""
    train = _drop_over_length_records(
        [
            *_gsm8k_records("train", GSM8K_TRAIN_ROWS),
            *_math_records(),
            *_aime_records(),
        ]
    )
    validation = [
        *_gsm8k_records("test", GSM8K_VALIDATION_ROWS, data_source=VALIDATION_GSM8K_SOURCE),
        *_math500_records(),
    ]
    for records, filename in ((train, TRAIN_FILENAME), (validation, VALIDATION_FILENAME)):
        destination = prefix_join(config.output_path, filename)
        write_parquet_file(records, destination)
        logger.info("Wrote %d rows to %s", len(records), destination)


def pool_step(name: str, version: str) -> ArtifactStep[Artifact]:
    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=Artifact,
        run=remote(write_pool_parquet, resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g")),
        build_config=lambda ctx: PoolParquetConfig(output_path=ctx.output_path),
    )
