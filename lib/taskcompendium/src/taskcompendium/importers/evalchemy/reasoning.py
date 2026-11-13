# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPQA and MMLU-Pro source prompts, staged fewshots, and evaluation items."""

import hashlib
import random
import re
import string
from collections.abc import Mapping
from functools import lru_cache, partial
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
from taskcompendium.importers.fewshot import (
    ChoiceDocument,
    ChoiceTemplate,
    Document,
    FewshotPolicy,
    FewshotSampling,
    GenerationTemplate,
    pinned_prompts,
)
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


def clean_gpqa_answer(text: str) -> str:
    return re.sub(r"\[.*?\]", "", text.strip().replace(" [title]", ". ")).replace("  ", " ")


def prepare_gpqa_choices(doc: Document, rng: random.Random) -> dict[str, Any]:
    answers = [
        clean_gpqa_answer(doc[key])
        for key in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Correct Answer")
    ]
    rng.shuffle(answers)
    return {**doc, "prepared_choices": answers}


def gpqa_likelihood_document(doc: Document) -> ChoiceDocument:
    choices = tuple(f"({letter})" for letter in "ABCD")
    answers = doc["prepared_choices"]
    prompt = "What is the correct answer to this question:" + doc["Question"] + "\nChoices:\n"
    prompt += "\n".join(f"{letter} {text}" for letter, text in zip(choices, answers, strict=True)) + "\nAnswer: "
    return ChoiceDocument(prompt, choices, answers.index(clean_gpqa_answer(doc["Correct Answer"])))


def gpqa_likelihood_template() -> ChoiceTemplate:
    return ChoiceTemplate(
        "leaderboard_gpqa",
        gpqa_likelihood_document,
        FewshotPolicy(0, FewshotSampling.FIRST),
        prepare=prepare_gpqa_choices,
    )


def mmlu_pro_likelihood_document(doc: Document) -> ChoiceDocument:
    choices = tuple(string.ascii_uppercase[: len(doc["options"])])
    prompt = (
        doc["question"]
        + "\n"
        + "\n".join(f"{letter}. {text}" for letter, text in zip(choices, doc["options"], strict=True))
        + "\nAnswer:"
    )
    return ChoiceDocument(prompt, choices, int(doc["answer_index"]))


def mmlu_pro_likelihood_template(pool_file: str) -> ChoiceTemplate:
    return ChoiceTemplate(
        "leaderboard_mmlu_pro",
        mmlu_pro_likelihood_document,
        FewshotPolicy(5, FewshotSampling.FIRST, pool_file=pool_file),
    )


def agieval_document(doc: Document) -> ChoiceDocument:
    return ChoiceDocument(doc["query"], tuple(doc["choices"]), int(doc["gold"][0]))


def agieval_gold(doc: Document) -> tuple[int, ...]:
    return tuple(doc["gold"])


def agieval_template(shots: int = 3) -> ChoiceTemplate:
    return ChoiceTemplate(
        "agieval_lsat_ar",
        agieval_document,
        FewshotPolicy(shots, FewshotSampling.RANDOM),
        gold_indices=agieval_gold,
    )


def bbh_likelihood_document(doc: Document, *, subject: str) -> ChoiceDocument:
    choices = tuple(pinned_prompts()["bbh/" + subject]["doc_to_choice"])
    return ChoiceDocument("Q: " + doc["input"] + "\nA:", choices, choices.index(doc["target"]))


def bbh_likelihood_template(subject: str) -> ChoiceTemplate:
    assets = pinned_prompts()["bbh/" + subject]
    return ChoiceTemplate(
        "leaderboard_bbh",
        partial(bbh_likelihood_document, subject=subject),
        FewshotPolicy(
            3,
            FewshotSampling.FIRST,
            examples=tuple(assets["fewshot_config"]["samples"]),
            description=assets["description"] + "\n\n",
        ),
    )


def bbh_cot_prompt(doc: Document) -> str:
    return "Q: " + doc["input"] + "\nA: Let's think step by step.\n"


def bbh_cot_demonstration(doc: Document) -> str:
    return bbh_cot_prompt(doc) + doc["target"] + "\n\n"


def bbh_cot_template(subject: str) -> GenerationTemplate:
    assets = pinned_prompts()["bbh_cot/" + subject]
    return GenerationTemplate(
        "bbh_cot_fewshot",
        bbh_cot_prompt,
        bbh_cot_demonstration,
        FewshotPolicy(
            3,
            FewshotSampling.FIRST,
            examples=tuple(assets["fewshot_config"]["samples"]),
            description=assets["description"] + "\n\n",
        ),
        subject=subject,
    )


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
