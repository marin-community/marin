# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile source-ordered few-shot examples without dataset-specific policy."""

import csv
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path
from random import Random
from typing import Any

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.spec import ChoiceLikelihood, LikelihoodRequest
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

Document = Mapping[str, Any]
FEWSHOT_SEED = 1234


@dataclass(frozen=True)
class ChoiceDocument:
    prompt: str
    choices: tuple[str, ...]
    gold: int


def parquet_documents(path: StoragePath) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches():
            yield from batch.to_pylist()


def sample_indices(
    random: Random,
    pool: Sequence[Document],
    shots: int,
    excluded_document: Document | None,
) -> list[int]:
    """Preserve the extra source-sampler draw when excluding the current document."""
    exclude = excluded_document is not None and shots > 0
    indices = random.sample(range(len(pool)), shots + int(exclude))
    return [index for index in indices if pool[index] != excluded_document][:shots] if exclude else indices


class FewshotSampling(StrEnum):
    FIRST = "first"
    RANDOM = "random"
    EXCLUDE_CURRENT = "exclude_current"


class ChoiceLayout(StrEnum):
    CONTINUATION_CHOICES = "continuation_choices"
    CONTEXT_CHOICES = "context_choices"


def copy_document(document: Document, _rng: Random) -> dict[str, Any]:
    return dict(document)


@dataclass(frozen=True)
class FewshotPolicy:
    shots: int
    sampling: FewshotSampling
    pool_file: str = ""
    examples: tuple[dict[str, Any], ...] | None = None
    description: str = ""


@dataclass(frozen=True)
class ChoiceTemplate:
    name: str
    render: Callable[[Document], ChoiceDocument]
    fewshot: FewshotPolicy
    layout: ChoiceLayout = ChoiceLayout.CONTINUATION_CHOICES
    prepare: Callable[[Document, Random], dict[str, Any]] = copy_document
    gold_indices: Callable[[Document], tuple[int, ...]] | None = None


@dataclass(frozen=True)
class GenerationTemplate:
    name: str
    prompt: Callable[[Document], str]
    demonstration: Callable[[Document], str]
    fewshot: FewshotPolicy
    subject: str = ""


@cache
def pinned_prompts() -> dict[str, Any]:
    """Read demonstrations retained from lm-eval v0.4.12 (see assets/NOTICE.txt)."""
    return json.loads((Path(__file__).parent / "evalchemy/assets/checkpoint_prompts.json").read_text())


def fewshot_indices(policy: FewshotPolicy, pool: Sequence[Document], document: Document, random: Random) -> list[int]:
    if policy.sampling is FewshotSampling.FIRST:
        return list(range(policy.shots))
    excluded_document = document if policy.sampling is FewshotSampling.EXCLUDE_CURRENT else None
    return sample_indices(random, pool, policy.shots, excluded_document)


def source_documents(path: StoragePath) -> list[dict[str, Any]]:
    if path.name.endswith(".csv"):
        with path.open("rt") as stream:
            return list(csv.DictReader(stream))
    return list(parquet_documents(path))


def fewshot_pool(policy: FewshotPolicy, documents: Sequence[Document], context: ConversionContext) -> Sequence[Document]:
    if policy.examples is not None:
        return policy.examples
    if policy.pool_file:
        return list(parquet_documents(context.inputs["fewshot"] / policy.pool_file))
    return documents


def read_choice_file(
    path: StoragePath, context: ConversionContext, *, template: ChoiceTemplate
) -> Iterator[dict[str, Any]]:
    """Compile source rows into choice-likelihood requests and their few-shot evidence."""
    documents = source_documents(path)
    policy = template.fewshot
    pool = fewshot_pool(policy, documents, context)
    yield from compile_choice_documents(documents, pool, template)


def compile_choice_documents(
    documents: Iterable[Document],
    pool: Sequence[Document],
    template: ChoiceTemplate,
) -> Iterator[dict[str, Any]]:
    """Compile complete source order with one sampler and explicit request layout."""
    policy = template.fewshot
    sampler, preparation = Random(FEWSHOT_SEED), Random(0)
    for original in documents:
        document = template.prepare(original, preparation)
        indices = fewshot_indices(policy, pool, original, sampler)
        prefix = policy.description
        for index in indices:
            example = template.render(pool[index])
            target = example.choices[example.gold]
            prefix += (
                target + " " + example.prompt
                if template.layout is ChoiceLayout.CONTEXT_CHOICES
                else example.prompt + " " + target
            ) + "\n\n"
        rendered = template.render(document)
        requests = tuple(
            (
                LikelihoodRequest(context=prefix + choice, continuation=" " + rendered.prompt)
                if template.layout is ChoiceLayout.CONTEXT_CHOICES
                else LikelihoodRequest(context=prefix + rendered.prompt, continuation=" " + choice)
            )
            for choice in rendered.choices
        )
        yield {
            "document": document,
            "task_name": template.name,
            "operation": (
                ChoiceLikelihood(
                    choices=rendered.choices,
                    requests=requests,
                    gold=template.gold_indices(document) if template.gold_indices is not None else (rendered.gold,),
                ).model_dump(mode="json")
            ),
            "fewshot_indices": indices,
            "num_fewshot": policy.shots,
        }


def read_generation_file(
    path: StoragePath, context: ConversionContext, *, template: GenerationTemplate
) -> Iterator[dict[str, Any]]:
    documents = source_documents(path)
    policy = template.fewshot
    pool = fewshot_pool(policy, documents, context)
    sampler = Random(FEWSHOT_SEED)
    for document in documents:
        indices = fewshot_indices(policy, pool, document, sampler)
        yield {
            "document": document,
            "prompt": (
                policy.description
                + "".join(template.demonstration(pool[index]) for index in indices)
                + template.prompt(document)
            ),
            "profile": template.name,
            "subject": template.subject,
            "num_fewshot": policy.shots,
        }


def convert_generation(row: RawRow, context: ConversionContext, *, scorer: str) -> TaskSpec:
    return conversation_task(
        row,
        events=(TextMessage(role="user", content=row.data["prompt"]),),
        package=source_package(
            scorer,
            {"document": row.data["document"]},
            environment=required_grader_environment(context),
        ),
        evidence=row.data,
    ).model_copy(update={"tags": ("eval", "source:lm-eval", "checkpoint")})
