# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPQA and MMLU-Pro source prompts, staged fewshots, and evaluation items."""

import hashlib
import random
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from zephyr.readers import load_parquet

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import (
    ChatGeneration,
    ChatPromptVariant,
    EvaluationItem,
)
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

MMLU_VALIDATION_FILE = "data/validation-00000-of-00001.parquet"
GPQA_PROMPT = (
    "Return your final response within \\boxed{{}} and only include the letter choice "
    "(A, B, C, or D) as your final response.\nProblem: {problem}\nOptions: {options}\nAnswer:"
)
# The source package file ends with three newlines; generate_cot_prompt adds another.
MMLU_INITIAL_PROMPT = (
    "The following are multiple choice questions (with answers) about {$}. Think step by step "
    'and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n\n'
)
MMLU_FEWSHOT_COUNT = 5


def gpqa_choices(data: Mapping[str, Any]) -> tuple[str, str]:
    answers = [data[key] for key in ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")]
    rng = random.Random(42 + int(hashlib.md5(data["Question"].encode()).hexdigest(), 16))
    rng.shuffle(answers)
    options = ", ".join(f"{letter}) {answer}" for letter, answer in zip("ABCD", answers, strict=True))
    expected = next(letter for letter, answer in zip("ABCD", answers, strict=True) if answer == data["Correct Answer"])
    return options, expected


def convert_gpqa(row: RawRow, context: ConversionContext) -> TaskSpec:
    options, expected = gpqa_choices(row.data)
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=GPQA_PROMPT.format(problem=row.data["Question"], options=options)),),
        package=source_package("gpqa", {"expected": expected}, environment=required_grader_environment(context)),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:GPQADiamond")})


def mmlu_example(example: Mapping[str, Any], *, including_answer: bool) -> str:
    options = [option for option in example["options"] if option != "N/A"]
    text = "Question:\n" + example["question"] + "\nOptions:\n"
    text += "".join(f"{chr(65 + index)}. {option}\n" for index, option in enumerate(options))
    if including_answer:
        return (
            text
            + example["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.")
            + "\n\n"
        )
    return text + "Answer: Let's think step by step."


def mmlu_prompt(support: tuple[dict, ...], example: Mapping[str, Any], *, num_fewshot: int) -> str:
    category = example["category"]
    selected = [shot for shot in support if shot["category"] == category][:num_fewshot]
    return (
        MMLU_INITIAL_PROMPT.replace("{$}", category)
        + "\n"
        + "".join(mmlu_example(shot, including_answer=True) for shot in selected)
        + mmlu_example(example, including_answer=False)
    )


@lru_cache
def mmlu_support(root: str) -> tuple[dict, ...]:
    """Read the small, immutable validation split once per conversion worker."""
    return tuple(load_parquet(str(StoragePath(root) / MMLU_VALIDATION_FILE)))


def convert_mmlu(row: RawRow, context: ConversionContext) -> TaskSpec:
    support = mmlu_support(str(context.inputs["fewshot"]))
    variants = tuple(mmlu_prompt(support, row.data, num_fewshot=shots) for shots in range(MMLU_FEWSHOT_COUNT + 1))
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=variants[-1]),),
        package=source_package(
            "mmlu_pro", {"expected": row.data["answer"]}, environment=required_grader_environment(context)
        ),
        evidence={**row.data, "prompt_variants": variants},
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:MMLUPro")})


def reasoning_item(task: TaskSpec, *, benchmark: str) -> EvaluationItem:
    evidence = source_evidence(task)
    groups = (
        {"category": evidence["category"]}
        if benchmark == "MMLUPro"
        else {key: evidence[key] for key in ("High-level domain", "Subdomain") if key in evidence}
    )
    operation = (
        ChatGeneration(
            prompt_selection="fit_context",
            response_selection="final_or_complete_reasoning",
            variants=tuple(
                ChatPromptVariant(
                    context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
                    fewshot=shots,
                )
                for shots, prompt in enumerate(evidence["prompt_variants"])
            ),
        )
        if benchmark == "MMLUPro"
        else ChatGeneration()
    )
    return chat_item(task).model_copy(update={"groups": groups, "operation": operation})
