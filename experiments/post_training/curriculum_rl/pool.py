# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty-graded problem pool for the curriculum-RL experiment.

The pool is one partition shared by every sampling arm: bins ordered by grade
(0 easiest), each bin either a pinned slice of a public math dataset or a
seeded procedurally generated reasoning-gym task family. Rows use the SkyRL
parquet schema with a per-row ``env_class`` so one training run mixes verifier
environments freely; ``extra_info`` carries the bin name and grade for
curriculum samplers and per-source metrics.

Bins sit on one 0-13 ladder from single-digit sums to graduate mathematics,
anchored to school grade and contest tier. Grades come from the strongest
available signal per source: explicit per-problem school grades (ASDiv),
dataset difficulty metadata (MATH levels, Omni-MATH AoPS ratings), contest or
curriculum provenance (GSM8K, NuminaMath source tags, AIME), or generator
knobs anchored to measured pass rates (reasoning-gym
arithmetic). The top rungs are university/graduate applied math with plain
verifiable answers (TheoremQA, HARDMath).
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

import requests
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
# The HF ASDiv mirrors drop the per-problem school-grade attribute, so the pool
# reads the original XML at a pinned commit.
ASDIV_REVISION = "883f90a9a65bf00304ba8f37423910fe743abc47"
ASDIV_XML_URL = f"https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/{ASDIV_REVISION}/dataset/ASDiv.xml"
SVAMP_DATASET = "ChilleD/SVAMP"
SVAMP_REVISION = "5e0bf1e"
NUMINA_DATASET = "AI-MO/NuminaMath-CoT"
NUMINA_REVISION = "9d8d210"
OMNI_MATH_DATASET = "KbsdJames/Omni-MATH"
OMNI_MATH_REVISION = "40ba231"
THEOREMQA_DATASET = "TIGER-Lab/TheoremQA"
THEOREMQA_REVISION = "a340b17"
# Community mirror of the original HARDMath generator output; the official
# repo publishes no HF dataset.
HARDMATH_DATASET = "pafitis/HARDMath_processed_training"
HARDMATH_REVISION = "937e9f1"

GSM8K_ENV = "gsm8k"
ANSWER_LINE_ENV = "aime"
REASONING_GYM_ENV = "reasoning_gym"

# Each instruction restates the grader's exact final-line contract, including
# the anti-\boxed clause: without it the SFT answer style (\boxed) wins and
# correct math rule-grades to zero.
GSM8K_INSTRUCTION = (
    " Let's think step by step. End your response with one final line of the exact form"
    ' "#### <number>". The automated grader reads only that line; do not use \\boxed{}.'
)
# The aime env verifies with the Minerva "Answer: ..." extraction (not \boxed),
# so the instruction must elicit a final Answer line.
ANSWER_LINE_INSTRUCTION = (
    " Please reason step by step. End your response with one final line of the exact form"
    " 'Answer: <answer>'. The automated grader reads only that line; do not use \\boxed{}."
)

# The format contract also lives in a system turn: a trailing user
# instruction alone loses to the SFT answer style (\boxed) on every bin whose
# grader wants a different final line. Kept to 37 tokens; longer wordings of
# the same contract measured no better on served-model compliance or pass, so
# resist re-expanding it.
SYSTEM_PROMPT = (
    "Solve the problem step by step. End with the exact final-answer line the problem "
    "requests; an automated grader reads only that line. No \\boxed{}, no text after it."
)

GSM8K_TRAIN_ROWS = 2000
GSM8K_VALIDATION_ROWS = 256

QWEN3_MODEL = "Qwen/Qwen3-0.6B"
QWEN3_REVISION = "c1899de"
# The trainer skips generation for prompts above max_input_length
# (request_window_tokens - max_new_tokens; every launch.py preset keeps this
# at 1024 and asserts it), and a fully skipped GRPO group fails group
# admission. Drop over-length train prompts at pool build with a margin for
# chat-template variants.
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


# Grades order bins easiest to hardest on one shared 0-13 ladder. MATH levels
# 1-2 sit in one bin because level 1 alone has only 564 train rows. Two bins
# may share a grade (they are distinct arms of the same difficulty band).
ASDIV_ELEM_BIN = PoolBin("g01-asdiv-elem", grade=1, env_class=ANSWER_LINE_ENV)
ASDIV_UPPER_BIN = PoolBin("g02-asdiv-upper", grade=2, env_class=ANSWER_LINE_ENV)
SVAMP_BIN = PoolBin("g02-svamp", grade=2, env_class=ANSWER_LINE_ENV)
GSM8K_BIN = PoolBin("g03-gsm8k", grade=3, env_class=GSM8K_ENV)
MATH_L12_BIN = PoolBin("g05-math-l12", grade=5, env_class=ANSWER_LINE_ENV)
MATH_L3_BIN = PoolBin("g06-math-l3", grade=6, env_class=ANSWER_LINE_ENV)
MATH_L4_BIN = PoolBin("g07-math-l4", grade=7, env_class=ANSWER_LINE_ENV)
MATH_L5_BIN = PoolBin("g08-math-l5", grade=8, env_class=ANSWER_LINE_ENV)
AIME_BIN = PoolBin("g10-aime", grade=10, env_class=ANSWER_LINE_ENV)
# ASDiv school grades 1-3 form the elementary bin; 4-6 the upper bin.
ASDIV_ELEM_GRADES = frozenset({"1", "2", "3"})

# The top half of the ladder. NuminaMath source tags give contest tiers
# (Chinese K-12 above GSM8K, synthetic AMC below AIME, olympiads above);
# Omni-MATH carries per-problem AoPS-anchored difficulty ratings, split here
# at 7.0 (hard olympiad) and 8.5 (the hardest band, standing in for Putnam:
# every Putnam-AXIOM HF mirror is gated). TheoremQA and HARDMath carry
# university/graduate applied math with plain verifiable answers.
NUMINA_CNK12_BIN = PoolBin("g04-numina-cnk12", grade=4, env_class=ANSWER_LINE_ENV)
NUMINA_AMC_BIN = PoolBin("g09-numina-amc", grade=9, env_class=ANSWER_LINE_ENV)
NUMINA_OLY_BIN = PoolBin("g11-numina-oly", grade=11, env_class=ANSWER_LINE_ENV)
OMNI_MID_BIN = PoolBin("g11-omni", grade=11, env_class=ANSWER_LINE_ENV)
OMNI_TOP_BIN = PoolBin("g12-omni-top", grade=12, env_class=ANSWER_LINE_ENV)
THEOREMQA_BIN = PoolBin("g13-theoremqa", grade=13, env_class=ANSWER_LINE_ENV)
HARDMATH_BIN = PoolBin("g13-hardmath", grade=13, env_class=ANSWER_LINE_ENV)
NUMINA_BINS_BY_SOURCE: Mapping[str, tuple[PoolBin, int]] = {
    "cn_k12": (NUMINA_CNK12_BIN, 2000),
    "synthetic_amc": (NUMINA_AMC_BIN, 1500),
    "olympiads": (NUMINA_OLY_BIN, 1500),
}
OMNI_MID_DIFFICULTY = 7.0
OMNI_TOP_DIFFICULTY = 8.5
# Answers longer than this are proofs-in-disguise or multi-part results the
# Answer-line verifier cannot check reliably.
MAX_ANSWER_CHARS = 60
VALIDATION_OMNI_SOURCE = "val-omni"
OMNI_VALIDATION_ROWS = 128
THEOREMQA_ANSWER_TYPES = frozenset({"integer", "float"})
# Held-out slices at the AMC and graduate rungs so the grade-weighted end
# metric can see the top of the ladder.
VALIDATION_AMC_SOURCE = "val-amc"
NUMINA_AMC_VALIDATION_ROWS = 128
VALIDATION_THEOREMQA_SOURCE = "val-theoremqa"
THEOREMQA_VALIDATION_STRIDE = 5
MATH_BINS_BY_LEVEL: Mapping[str, PoolBin] = {
    "Level 1": MATH_L12_BIN,
    "Level 2": MATH_L12_BIN,
    "Level 3": MATH_L3_BIN,
    "Level 4": MATH_L4_BIN,
    "Level 5": MATH_L5_BIN,
}
VALIDATION_GSM8K_SOURCE = "val-gsm8k"
VALIDATION_MATH500_SOURCE = "val-math500"


@dataclass(frozen=True)
class ReasoningGymBin:
    """A seeded, procedurally generated reasoning-gym bin.

    ``knobs`` are the generator's difficulty parameters; the grade is assigned
    a priori from those knobs, before any model sees a sample.
    """

    pool_bin: PoolBin
    task: str
    size: int
    seed: int
    knobs: Mapping[str, object] = field(default_factory=dict)


RG_TRAIN_ROWS = 1200
RG_VALIDATION_ROWS = 128

# Procedural arithmetic anchors the bottom of the ladder ("2+2"). chain_sum
# grades are anchored to measured pass rates rather than generator knobs
# (knob-guessed grades overshot badly: 3-4-term sums passed at ~0.97 while
# graded 3), and the non-math spelling/base-conversion families are excluded
# from the math ladder.
REASONING_GYM_BINS = (
    ReasoningGymBin(
        PoolBin("g00-rg-sum-easy", grade=0, env_class=REASONING_GYM_ENV),
        task="chain_sum",
        size=RG_TRAIN_ROWS,
        seed=101,
        knobs={"min_terms": 2, "max_terms": 2, "min_digits": 1, "max_digits": 2},
    ),
    ReasoningGymBin(
        PoolBin("g01-rg-sum-med", grade=1, env_class=REASONING_GYM_ENV),
        task="chain_sum",
        size=RG_TRAIN_ROWS,
        seed=102,
        knobs={"min_terms": 3, "max_terms": 4, "min_digits": 2, "max_digits": 3},
    ),
    ReasoningGymBin(
        PoolBin("g02-rg-sum-hard", grade=2, env_class=REASONING_GYM_ENV),
        task="chain_sum",
        size=RG_TRAIN_ROWS,
        seed=105,
        knobs={"min_terms": 5, "max_terms": 6, "min_digits": 4, "max_digits": 4, "allow_negation": True},
    ),
)

# Held-out generator draws at the medium difficulty of the arithmetic family,
# for an in-run validation curve on the procedural task.
VALIDATION_REASONING_GYM_BINS = (
    ReasoningGymBin(
        PoolBin("val-rg-sum", grade=1, env_class=REASONING_GYM_ENV),
        task="chain_sum",
        size=RG_VALIDATION_ROWS,
        seed=201,
        knobs={"min_terms": 3, "max_terms": 4, "min_digits": 2, "max_digits": 3},
    ),
)


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
    instruction = GSM8K_INSTRUCTION if pool_bin.env_class == GSM8K_ENV else ANSWER_LINE_INSTRUCTION
    source = data_source or pool_bin.name
    return {
        "data_source": source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{question}{instruction}"},
        ],
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


# Word-problem answers arrive as "9 (apples)" (ASDiv) or "7.0" (SVAMP); keep
# plain numbers and simple fractions, drop the few clock-time/date leftovers.
_NUMERIC_ANSWER = re.compile(r"^-?\d+(?:\.\d+)?(?:/\d+)?$")


def _plain_number(text: str) -> str | None:
    value = text.split("(")[0].strip().replace(",", "")
    if not _NUMERIC_ANSWER.match(value):
        return None
    if value.endswith(".0"):
        value = value[: -len(".0")]
    return value


def _asdiv_records() -> list[dict[str, object]]:
    response = requests.get(ASDIV_XML_URL, timeout=60)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    records = []
    skipped = 0
    for index, problem in enumerate(root.iter("Problem")):
        answer = _plain_number(problem.findtext("Answer", ""))
        if answer is None:
            skipped += 1
            continue
        pool_bin = ASDIV_ELEM_BIN if problem.get("Grade") in ASDIV_ELEM_GRADES else ASDIV_UPPER_BIN
        question = f"{problem.findtext('Body', '').strip()} {problem.findtext('Question', '').strip()}"
        records.append(_pool_record(question=question, answer=answer, pool_bin=pool_bin, split="train", index=index))
    logger.info("ASDiv: kept %d rows, skipped %d without a plain numeric answer", len(records), skipped)
    return records


def _svamp_records() -> list[dict[str, object]]:
    dataset = load_dataset(SVAMP_DATASET, split="train", revision=SVAMP_REVISION)
    records = []
    skipped = 0
    for index, example in enumerate(dataset):
        answer = _plain_number(str(example["Answer"]))
        if answer is None:
            skipped += 1
            continue
        question = f"{example['Body'].strip()} {example['Question'].strip()}"
        records.append(_pool_record(question=question, answer=answer, pool_bin=SVAMP_BIN, split="train", index=index))
    logger.info("SVAMP: kept %d rows, skipped %d without a plain numeric answer", len(records), skipped)
    return records


def _numina_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Tiered NuminaMath slices, plus a held-out AMC validation slice.

    The AMC quota is over-collected by the validation size; the tail past the
    train quota becomes ``val-amc``, so the two splits are disjoint by
    construction.
    """
    dataset = load_dataset(NUMINA_DATASET, split="train", revision=NUMINA_REVISION)
    validation_bin = PoolBin(VALIDATION_AMC_SOURCE, grade=NUMINA_AMC_BIN.grade, env_class=ANSWER_LINE_ENV)
    quotas = {pool_bin.name: quota for pool_bin, quota in NUMINA_BINS_BY_SOURCE.values()}
    quotas[NUMINA_AMC_BIN.name] += NUMINA_AMC_VALIDATION_ROWS
    taken: Counter[str] = Counter()
    train, validation = [], []
    for index, example in enumerate(dataset):
        selected = NUMINA_BINS_BY_SOURCE.get(example["source"])
        if selected is None:
            continue
        pool_bin, train_quota = selected
        if taken[pool_bin.name] >= quotas[pool_bin.name]:
            if sum(taken.values()) >= sum(quotas.values()):
                break
            continue
        answer = boxed_answer(example["solution"])
        if answer is None or len(answer) > MAX_ANSWER_CHARS:
            continue
        taken[pool_bin.name] += 1
        if pool_bin is NUMINA_AMC_BIN and taken[pool_bin.name] > train_quota:
            validation.append(
                _pool_record(
                    question=example["problem"], answer=answer, pool_bin=validation_bin, split="test", index=index
                )
            )
        else:
            train.append(
                _pool_record(question=example["problem"], answer=answer, pool_bin=pool_bin, split="train", index=index)
            )
    logger.info("NuminaMath: kept rows per bin %s (val-amc %d)", dict(taken), len(validation))
    return train, validation


def _omni_math_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Omni-MATH difficulty bands: train bins at >=7.0 and >=8.5, plus a held-out slice."""
    dataset = load_dataset(OMNI_MATH_DATASET, split="test", revision=OMNI_MATH_REVISION)
    validation_bin = PoolBin(VALIDATION_OMNI_SOURCE, grade=11, env_class=ANSWER_LINE_ENV)
    train, validation = [], []
    for index, example in enumerate(dataset):
        difficulty = example["difficulty"]
        answer = (example["answer"] or "").strip()
        if difficulty is None or float(difficulty) < OMNI_MID_DIFFICULTY:
            continue
        if not answer or len(answer) > MAX_ANSWER_CHARS:
            continue
        if float(difficulty) >= OMNI_TOP_DIFFICULTY:
            pool_bin, split, out = OMNI_TOP_BIN, "train", train
        elif index % 8 == 0 and len(validation) < OMNI_VALIDATION_ROWS:
            pool_bin, split, out = validation_bin, "test", validation
        else:
            pool_bin, split, out = OMNI_MID_BIN, "train", train
        out.append(_pool_record(question=example["problem"], answer=answer, pool_bin=pool_bin, split=split, index=index))
    logger.info("Omni-MATH: %d train rows, %d validation rows", len(train), len(validation))
    return train, validation


def _theoremqa_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Numeric no-picture TheoremQA rows, with every Nth kept out as validation."""
    dataset = load_dataset(THEOREMQA_DATASET, split="test", revision=THEOREMQA_REVISION)
    validation_bin = PoolBin(VALIDATION_THEOREMQA_SOURCE, grade=THEOREMQA_BIN.grade, env_class=ANSWER_LINE_ENV)
    train, validation = [], []
    kept = 0
    for index, example in enumerate(dataset):
        if example["Answer_type"] not in THEOREMQA_ANSWER_TYPES or str(example["Picture"]) not in ("None", ""):
            continue
        held_out = kept % THEOREMQA_VALIDATION_STRIDE == 0
        kept += 1
        pool_bin = validation_bin if held_out else THEOREMQA_BIN
        out = validation if held_out else train
        out.append(
            _pool_record(
                question=example["Question"],
                answer=str(example["Answer"]),
                pool_bin=pool_bin,
                split="test" if held_out else "train",
                index=index,
            )
        )
    logger.info("TheoremQA: %d train rows, %d validation rows", len(train), len(validation))
    return train, validation


def _hardmath_records() -> list[dict[str, object]]:
    dataset = load_dataset(HARDMATH_DATASET, split="train", revision=HARDMATH_REVISION)
    records = []
    skipped = 0
    for index, example in enumerate(dataset):
        answer = boxed_answer(example["ground_truths"] or "")
        # List-valued answers (asymptotic regime pairs) defeat the Answer-line
        # equality check; keep single closed-form results.
        if answer is None or "[" in answer or len(answer) > MAX_ANSWER_CHARS:
            skipped += 1
            continue
        # Ground truths are statements ("\epsilon \approx 4.16"); the value the
        # question asks for (with explicit rounding) is the right-hand side.
        if "\\approx" in answer:
            answer = answer.rsplit("\\approx", 1)[1].strip()
        elif "=" in answer:
            answer = answer.rsplit("=", 1)[1].strip()
        if not answer:
            skipped += 1
            continue
        records.append(
            _pool_record(question=example["question"], answer=answer, pool_bin=HARDMATH_BIN, split="train", index=index)
        )
    logger.info("HARDMath: kept %d rows, skipped %d list/overlong answers", len(records), skipped)
    return records


def _math500_records() -> list[dict[str, object]]:
    dataset = load_dataset(MATH500_DATASET, split="test", revision=MATH500_REVISION)
    math500_bin = PoolBin(VALIDATION_MATH500_SOURCE, grade=5, env_class=ANSWER_LINE_ENV)
    return [
        _pool_record(
            question=example["problem"], answer=example["answer"], pool_bin=math500_bin, split="test", index=index
        )
        for index, example in enumerate(dataset)
    ]


def _reasoning_gym_records(rg_bin: ReasoningGymBin, split: str) -> list[dict[str, object]]:
    # Optional dependency: only the remote pool build installs the
    # reasoning-gym group, so keep the import out of module scope.
    import reasoning_gym  # noqa: PLC0415

    dataset = reasoning_gym.create_dataset(rg_bin.task, size=rg_bin.size, seed=rg_bin.seed, **rg_bin.knobs)
    records = []
    for index, entry in enumerate(dataset):
        # The reasoning_gym env re-scores against the full entry (question,
        # answer, metadata), so the ground truth carries it verbatim.
        ground_truth = json.dumps({"task": rg_bin.task, "entry": entry}, sort_keys=True)
        records.append(
            _pool_record(
                question=entry["question"],
                answer=ground_truth,
                pool_bin=rg_bin.pool_bin,
                split=split,
                index=index,
            )
        )
    return records


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
    omni_train, omni_validation = _omni_math_records()
    numina_train, numina_validation = _numina_records()
    theoremqa_train, theoremqa_validation = _theoremqa_records()
    train = _drop_over_length_records(
        [
            *_asdiv_records(),
            *_svamp_records(),
            *_gsm8k_records("train", GSM8K_TRAIN_ROWS),
            *numina_train,
            *_math_records(),
            *_aime_records(),
            *omni_train,
            *theoremqa_train,
            *_hardmath_records(),
            *(record for rg_bin in REASONING_GYM_BINS for record in _reasoning_gym_records(rg_bin, "train")),
        ]
    )
    validation = [
        *_gsm8k_records("test", GSM8K_VALIDATION_ROWS, data_source=VALIDATION_GSM8K_SOURCE),
        *_math500_records(),
        *numina_validation,
        *omni_validation,
        *theoremqa_validation,
        *(record for rg_bin in VALIDATION_REASONING_GYM_BINS for record in _reasoning_gym_records(rg_bin, "test")),
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
        run=remote(
            write_pool_parquet,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g"),
            pip_dependency_groups=["reasoning-gym"],
        ),
        build_config=lambda ctx: PoolParquetConfig(output_path=ctx.output_path),
    )
