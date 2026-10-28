# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile pinned lm-eval v0.4.12 completion inputs into ordinary evaluation data.

Templates and HellaSwag preprocessing follow EleutherAI/lm-evaluation-harness
(commit 6d642546f4688648fced259eb3302efd36ece5af, MIT license). Compilation
runs before selection: the source sampler advances once for every source ordinal.
No task object or harness dependency is retained in a released corpus.
"""

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from verifyit.modes.grade_harness_generation import drop_references
from verifyit.spec import ExactSpec, HarnessGenerationProfile, HarnessGenerationSpec

from taskcompendium.convert.answers import answer_task
from taskcompendium.convert.evaluation import source_evidence
from taskcompendium.evaluation import (
    ChoiceLikelihood,
    ContinuationLikelihood,
    EvaluationItem,
    LikelihoodRequest,
    TextGeneration,
)
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.evaluation import source_order
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow

HARNESS_REVISION = "6d642546f4688648fced259eb3302efd36ece5af"
FEWSHOT_SEED = 1234
TRUTHFULQA_PREFIX = """Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: I have no comment.

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain."""


@dataclass(frozen=True)
class HarnessTemplate:
    name: str
    num_fewshot: int
    subject: str = ""
    fewshot_file: str = ""


@dataclass(frozen=True)
class ChoiceDocument:
    prompt: str
    choices: tuple[str, ...]
    gold: int


@dataclass(frozen=True)
class GenerationDocument:
    prompt: str
    demonstration_target: str
    references: tuple[tuple[str, ...], ...]


def original_task_name(template: HarnessTemplate) -> str:
    return "mmlu_" + template.subject if template.name == "mmlu" else template.name


def _hellaswag_text(text: str) -> str:
    return re.sub(r"\[.*?\]", "", text.strip().replace(" [title]", ". ")).replace("  ", " ")


def choice_document(doc: Mapping[str, Any], template: HarnessTemplate) -> ChoiceDocument:
    """Render public questions and original choices separately from the target delimiter."""
    name = template.name
    if name == "mmlu":
        prompt = (
            doc["question"].strip()
            + "\n"
            + "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", doc["choices"], strict=True))
            + "\nAnswer:"
        )
        return ChoiceDocument(prompt=prompt, choices=tuple("ABCD"), gold=int(doc["answer"]))
    if name in {"arc_challenge", "arc_easy"}:
        return ChoiceDocument(
            prompt=f"Question: {doc['question']}\nAnswer:",
            choices=tuple(doc["choices"]["text"]),
            gold=doc["choices"]["label"].index(doc["answerKey"]),
        )
    if name == "openbookqa":
        return ChoiceDocument(
            prompt=doc["question_stem"],
            choices=tuple(doc["choices"]["text"]),
            gold=doc["choices"]["label"].index(doc["answerKey"].lstrip()),
        )
    if name == "piqa":
        return ChoiceDocument(
            prompt=f"Question: {doc['goal']}\nAnswer:", choices=(doc["sol1"], doc["sol2"]), gold=int(doc["label"])
        )
    if name == "boolq":
        return ChoiceDocument(
            prompt=f"{doc['passage']}\nQuestion: {doc['question']}?\nAnswer:",
            choices=("no", "yes"),
            gold=int(doc["label"]),
        )
    if name == "hellaswag":
        context = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return ChoiceDocument(
            prompt=_hellaswag_text(doc["activity_label"] + ": " + context),
            choices=tuple(_hellaswag_text(ending) for ending in doc["endings"]),
            gold=int(doc["label"]),
        )
    if name == "truthfulqa_mc2":
        return ChoiceDocument(
            prompt=TRUTHFULQA_PREFIX + f"\n\nQ: {doc['question']}\nA:",
            choices=tuple(doc["mc2_targets"]["choices"]),
            gold=0,
        )
    if name == "winogrande":
        beginning, ending = doc["sentence"].split("_", 1)
        choices = (beginning + doc["option1"], beginning + doc["option2"])
        return ChoiceDocument(prompt=ending.strip(), choices=choices, gold=int(doc["answer"]) - 1)
    raise ValueError(f"Unsupported pinned likelihood template: {name}")


def compile_choice_documents(
    documents: Iterable[dict[str, Any]], fewshot_documents: Sequence[dict[str, Any]], template: HarnessTemplate
) -> Iterator[dict[str, Any]]:
    """Compile a complete source file in source order, preserving stateful few-shot sampling."""
    random = Random(FEWSHOT_SEED)
    for doc in documents:
        if template.name == "mmlu":
            indices = list(range(template.num_fewshot))
            subject = template.subject.replace("_", " ")
            prefix = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
        else:
            indices = _sample_indices(random, fewshot_documents, doc, template)
            prefix = ""
        for index in indices:
            demonstration = choice_document(fewshot_documents[index], template)
            target = demonstration.choices[demonstration.gold]
            prefix += (
                target + " " + demonstration.prompt
                if template.name == "winogrande"
                else demonstration.prompt + " " + target
            ) + "\n\n"
        document = choice_document(doc, template)
        requests = tuple(
            (
                LikelihoodRequest(context=prefix + choice, continuation=" " + document.prompt)
                if template.name == "winogrande"
                else LikelihoodRequest(context=prefix + document.prompt, continuation=" " + choice)
            )
            for choice in document.choices
        )
        operation = ChoiceLikelihood(
            choices=document.choices,
            requests=requests,
            gold=(
                tuple(index for index, label in enumerate(doc["mc2_targets"]["labels"]) if label)
                if template.name == "truthfulqa_mc2"
                else (document.gold,)
            ),
        )
        yield {
            "document": doc,
            "task_name": original_task_name(template),
            "operation": operation.model_dump(mode="json"),
            "fewshot_indices": indices,
            "fewshot_seed": FEWSHOT_SEED,
            "num_fewshot": template.num_fewshot,
            "harness_revision": HARNESS_REVISION,
        }


def _sample_indices(
    random: Random, pool: Sequence[dict[str, Any]], doc: dict[str, Any], template: HarnessTemplate
) -> list[int]:
    # The pinned task passes eval_doc when both fewshot_split and test_split
    # are None (validation-only tasks), so its sampler draws one extra.
    exclude = template.name in {"hellaswag", "winogrande", "triviaqa", "nq_open", "drop"} and template.num_fewshot > 0
    indices = random.sample(range(len(pool)), template.num_fewshot + int(exclude))
    return [index for index in indices if pool[index] != doc][: template.num_fewshot] if exclude else indices


def parquet_documents(path: StoragePath) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches():
            yield from batch.to_pylist()


def read_choice_file(
    path: StoragePath, context: ConversionContext, *, template: HarnessTemplate
) -> Iterator[dict[str, Any]]:
    fewshot = tuple(parquet_documents(context.inputs["fewshot"] / template.fewshot_file)) if template.num_fewshot else ()
    yield from compile_choice_documents(parquet_documents(path), fewshot, template)


def convert_choice_document(row: RawRow, _context: ConversionContext, *, benchmark: str) -> TaskSpec:
    operation = ChoiceLikelihood.model_validate(row.data["operation"])
    task = answer_task(
        row,
        prompt=operation.requests[0].context,
        spec=ExactSpec(
            expected=tuple(operation.choices[index] for index in operation.gold),
            ignore_case=False,
            ignore_whitespace=False,
        ),
        evidence=row.data,
    )
    return task.model_copy(
        update={
            "tags": (
                "eval",
                "source:evalchemy",
                "family:likelihood",
                f"benchmark:{benchmark}",
                f"task:{row.data['task_name']}",
            )
        }
    )


def choice_item(task: TaskSpec, *, subject: str = "") -> EvaluationItem:
    return EvaluationItem(
        task_id=task.id,
        source_row=task.source.row,
        source_ordinal=source_order(task.source.row)[1],
        operation=ChoiceLikelihood.model_validate(source_evidence(task)["operation"]),
        groups={"subject": subject} if subject else {},
    )


def compile_continuation_document(doc: Mapping[str, Any]) -> ContinuationLikelihood:
    words = doc["text"].split(" ")
    return ContinuationLikelihood(request=LikelihoodRequest(context=" ".join(words[:-1]), continuation=" " + words[-1]))


def read_continuation_file(path: StoragePath, _context: ConversionContext) -> Iterator[dict[str, Any]]:
    for doc in parquet_documents(path):
        yield {
            "document": doc,
            "operation": compile_continuation_document(doc).model_dump(mode="json"),
            "harness_revision": HARNESS_REVISION,
        }


def convert_continuation_document(row: RawRow, _context: ConversionContext) -> TaskSpec:
    operation = ContinuationLikelihood.model_validate(row.data["operation"])
    task = answer_task(
        row,
        prompt=operation.request.context,
        spec=ExactSpec(
            expected=(operation.request.continuation,),
            ignore_case=False,
            ignore_whitespace=False,
        ),
        evidence=row.data,
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:lambada_openai")})


def continuation_item(task: TaskSpec) -> EvaluationItem:
    return EvaluationItem(
        task_id=task.id,
        source_row=task.source.row,
        source_ordinal=source_order(task.source.row)[1],
        operation=ContinuationLikelihood.model_validate(source_evidence(task)["operation"]),
    )


def generation_document(doc: dict[str, Any], name: str) -> GenerationDocument:
    if name == "gsm8k":
        return GenerationDocument(
            prompt=f"Question: {doc['question']}\nAnswer:",
            demonstration_target=doc["answer"],
            references=((doc["answer"].rsplit("#### ", 1)[-1],),),
        )
    if name == "triviaqa":
        references = tuple((alias,) for alias in doc["answer"]["aliases"])
        return GenerationDocument(
            prompt=f"Question: {doc['question']}?\nRespond with exactly \"Answer: <short answer>\".",
            demonstration_target="Answer: " + references[0][0],
            references=references,
        )
    if name == "nq_open":
        references = tuple((alias,) for alias in doc["answer"])
        return GenerationDocument(
            prompt=f"Q: {doc['question']}?\nRespond with exactly \"Answer: <short answer>\".",
            demonstration_target=f"Answer: {doc['answer']}",
            references=references,
        )
    if name == "drop":
        references = drop_references(doc)
        return GenerationDocument(
            prompt=f"{doc['passage']} {doc['question']}\nRespond with exactly \"Answer: <short answer>\".",
            demonstration_target="Answer: " + ",".join(references[0]),
            references=references,
        )
    raise ValueError(f"Unsupported pinned generation template: {name}")


def compile_generation_documents(
    documents: Iterable[dict[str, Any]], fewshot_documents: Sequence[dict[str, Any]], template: HarnessTemplate
) -> Iterator[dict[str, Any]]:
    random = Random(FEWSHOT_SEED)
    for doc in documents:
        indices = _sample_indices(random, fewshot_documents, doc, template)
        prefix = "Answer these questions:\n\n" if template.name == "nq_open" else ""
        delimiter = "\n" if template.name == "drop" else " "
        separator = "\n" if template.name == "nq_open" else "\n\n"
        for index in indices:
            demonstration = generation_document(fewshot_documents[index], template.name)
            prefix += demonstration.prompt + delimiter + demonstration.demonstration_target + separator
        document = generation_document(doc, template.name)
        yield {
            "document": doc,
            "operation": TextGeneration(context=prefix + document.prompt).model_dump(mode="json"),
            "references": document.references,
            "profile": template.name,
            "fewshot_indices": indices,
            "fewshot_seed": FEWSHOT_SEED,
            "num_fewshot": template.num_fewshot,
            "harness_revision": HARNESS_REVISION,
        }


def read_generation_file(
    path: StoragePath, context: ConversionContext, *, template: HarnessTemplate
) -> Iterator[dict[str, Any]]:
    fewshot = tuple(parquet_documents(context.inputs["fewshot"] / template.fewshot_file)) if template.num_fewshot else ()
    yield from compile_generation_documents(parquet_documents(path), fewshot, template)


def convert_generation_document(row: RawRow, _context: ConversionContext, *, benchmark: str) -> TaskSpec:
    operation = TextGeneration.model_validate(row.data["operation"])
    task = answer_task(
        row,
        prompt=operation.context,
        spec=HarnessGenerationSpec(
            profile=HarnessGenerationProfile(row.data["profile"]),
            references=tuple(tuple(spans) for spans in row.data["references"]),
        ),
        evidence=row.data,
    )
    return task.model_copy(
        update={
            "tags": (
                "eval",
                "source:evalchemy",
                "family:short-answer",
                f"benchmark:{benchmark}",
                f"task:{row.data['profile']}",
            )
        }
    )


def generation_item(task: TaskSpec) -> EvaluationItem:
    return EvaluationItem(
        task_id=task.id,
        source_row=task.source.row,
        source_ordinal=source_order(task.source.row)[1],
        operation=TextGeneration.model_validate(source_evidence(task)["operation"]),
    )
