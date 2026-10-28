# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPQA-Diamond and MMLU-Pro public prompts from Marin's Evalchemy source protocols."""

import hashlib
import random
from functools import lru_cache, partial

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import answer_task
from taskcompendium.convert.evaluation import source_evidence
from taskcompendium.evaluation import (
    ChatGeneration,
    ChatPromptVariant,
    EvaluationItem,
    EvaluationProtocol,
    Sampling,
    TrialPolicy,
)
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.evaluation import chat_item
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Converter, IntendedUse, RawRow
from verifyit.adapters.evalchemy_math import CHAT_REQUEST_STOPS
from verifyit.spec import EmptyOutputPolicy, McqProfile, McqSpec
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.datasets.evalchemy.datasets import ACCURACY
from experiments.post_training.task_curation.datasets.evalchemy.pipeline import EvalDataSet, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

GPQA_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"
MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_TEST_FILE = "data/test-00000-of-00001.parquet"
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


def gpqa_choices(data: dict) -> tuple[str, str]:
    answers = [data[key] for key in ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")]
    rng = random.Random(42 + int(hashlib.md5(data["Question"].encode()).hexdigest(), 16))
    rng.shuffle(answers)
    options = ", ".join(f"{letter}) {answer}" for letter, answer in zip("ABCD", answers, strict=True))
    expected = next(letter for letter, answer in zip("ABCD", answers, strict=True) if answer == data["Correct Answer"])
    return options, expected


def convert_gpqa(row: RawRow, _context: ConversionContext) -> TaskSpec:
    options, expected = gpqa_choices(row.data)
    task = answer_task(
        row,
        prompt=GPQA_PROMPT.format(problem=row.data["Question"], options=options),
        spec=McqSpec(
            expected=expected,
            options=4,
            profile=McqProfile.EVALCHEMY_GPQA,
            empty_output=EmptyOutputPolicy.GRADE,
        ),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:GPQADiamond")})


def mmlu_example(example: dict, *, including_answer: bool) -> str:
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


def mmlu_prompt(support: tuple[dict, ...], example: dict, *, num_fewshot: int) -> str:
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
    task = answer_task(
        row,
        prompt=variants[-1],
        spec=McqSpec(
            expected=row.data["answer"],
            options=10,
            profile=McqProfile.EVALCHEMY_MMLU_PRO,
            empty_output=EmptyOutputPolicy.GRADE,
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


def choice_dataset(
    name: str,
    benchmark: str,
    source: HfSource,
    converter: Converter,
    rows: int,
    trials: int,
    temperature: float,
    inputs: dict[str, HfSource],
) -> RlDataSource[EvalDataSet]:
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name,
            title=benchmark,
            origin="Evalchemy",
            family="generated-choice",
            tags=("eval", "source:evalchemy", "single-turn", "benchmark", "reasoning", "benchmark:" + benchmark),
            count=rows,
        ),
        config=EvalDataSet(
            CurationRecipe(
                name="evalchemy-" + name,
                source=source,
                convert=converter,
                version="1",
                intended_use=IntendedUse.EVAL,
                inputs=inputs,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                version="maintained-generated-choice-v2",
                operation="chat_generation",
                sampling=Sampling(
                    temperature=temperature,
                    top_p=1,
                    top_k=None,
                    max_tokens=None,
                    stop=CHAT_REQUEST_STOPS,
                    chat_template_kwargs={"enable_thinking": True},
                ),
                trials=TrialPolicy(count=trials, seed=0 if temperature else None),
                metrics=(ACCURACY,),
                primary_metric="accuracy",
                reducer="mean_over_examples_and_trials_v1",
                expected_population=rows,
                context_window=65536 if benchmark == "MMLUPro" else None,
            ),
            partial(reasoning_item, benchmark=benchmark),
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataSet]]:
    return [
        choice_dataset(
            "gpqa-diamond",
            "GPQADiamond",
            HfSource("Idavidrein/gpqa", GPQA_REVISION, ("gpqa_diamond.csv",), SourceFormat.CSV),
            convert_gpqa,
            198,
            3,
            0.7,
            {},
        ),
        choice_dataset(
            "mmlu-pro",
            "MMLUPro",
            HfSource("TIGER-Lab/MMLU-Pro", MMLU_PRO_REVISION, (MMLU_TEST_FILE,), SourceFormat.PARQUET),
            convert_mmlu,
            12032,
            1,
            0,
            {
                "fewshot": HfSource(
                    "TIGER-Lab/MMLU-Pro",
                    MMLU_PRO_REVISION,
                    (MMLU_VALIDATION_FILE,),
                    SourceFormat.PARQUET,
                )
            },
        ),
    ]
