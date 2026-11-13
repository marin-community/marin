# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact bindings for compiled likelihood and generation protocols."""

from dataclasses import dataclass, replace
from functools import partial
from typing import Literal

from marin.evaluation.records import PromptFormat
from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    LikelihoodMetric,
    LikelihoodNormalization,
    Metric,
    MetricKind,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.evaluation_harness import (
    HarnessTemplate,
    choice_item,
    convert_choice_document,
    convert_generation_document,
    generation_item,
    original_task_name,
    read_choice_file,
)
from taskcompendium.importers.evaluation_harness import (
    read_generation_file as read_lm_eval_generation,
)
from taskcompendium.importers.fewshot import ChoiceTemplate, GenerationTemplate, convert_generation, read_generation_file
from taskcompendium.importers.fewshot import read_choice_file as read_fewshot_choices
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, ProtocolVariant
from experiments.evaluation.preparation import EvaluationCorpusRecipe, chat_source, prepare_evaluation
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

EXPLICIT_TEMPLATE_VERSION = "2026.11.05.3"


def multiple_choice_evaluation(
    *,
    name: str,
    benchmark: str,
    source: HfSource,
    population: int,
    template: ChoiceTemplate,
    metrics: tuple[LikelihoodMetric, ...],
    subject: str = "",
) -> RlDataSource[EvaluationCorpusRecipe]:
    """Bind a family's explicit prompt and normalization policies to a curated corpus."""
    policy = template.fewshot
    return RlDataSource(
        info=SourceInfo(
            id="Checkpoint:" + name,
            title=name,
            origin="lm-eval",
            family="checkpoint",
            tags=("eval", "benchmark"),
            count=population,
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name="checkpoint-" + name,
                source=replace(source, read=partial(read_fewshot_choices, template=template)),
                inputs=(
                    {"fewshot": HfSource(source.repo, source.revision, (policy.pool_file,), SourceFormat.PARQUET)}
                    if policy.pool_file
                    else {}
                ),
                convert=partial(convert_choice_document, benchmark=benchmark),
                version=EXPLICIT_TEMPLATE_VERSION,
                intended_use=IntendedUse.EVAL,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                version=EXPLICIT_TEMPLATE_VERSION,
                operation=OperationKind.CHOICE_LIKELIHOOD,
                sampling=None,
                trials=TrialPolicy(count=1, seed=None),
                metrics=tuple(
                    Metric(name=metric.name, kind=MetricKind.BINARY, higher_is_better=True) for metric in metrics
                ),
                primary_metric=metrics[-1].name,
                num_fewshot=policy.shots,
                summary=SummaryKind.MEAN,
                expected_population=population,
                likelihood_metrics=metrics,
            ),
            partial(choice_item, subject=subject),
        ),
        pipeline=prepare_evaluation,
    )


def lm_eval_multiple_choice(
    *,
    name: str,
    benchmark: str,
    source: HfSource,
    population: int,
    template: HarnessTemplate,
    normalizations: tuple[LikelihoodNormalization, ...],
    version: str,
    metric_kind: MetricKind = MetricKind.BINARY,
    scoring: Literal["most_likely", "probability_mass"] = "most_likely",
    partition: str | None = None,
) -> RlDataSource[EvaluationCorpusRecipe]:
    """Bind pinned rows and few-shots, with the first normalization as the headline."""
    names = {
        LikelihoodNormalization.RAW: "acc,none",
        LikelihoodNormalization.CHARACTERS: "acc_norm,none",
        LikelihoodNormalization.BYTES: "acc_bytes,none",
    }
    inputs = (
        {"fewshot": HfSource(source.repo, source.revision, (template.fewshot_file,), SourceFormat.PARQUET)}
        if template.num_fewshot
        else {}
    )
    recipe = CurationRecipe(
        name="evalchemy-" + name,
        source=replace(source, read=partial(read_choice_file, template=template)),
        inputs=inputs,
        convert=partial(convert_choice_document, benchmark=name),
        version=version,
        intended_use=IntendedUse.EVAL,
    )
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name,
            title=name,
            origin="Evalchemy",
            family="likelihood-choice",
            tags=(
                "eval",
                "source:evalchemy",
                "single-turn",
                "benchmark",
                "family:likelihood",
                "task:" + original_task_name(template),
            ),
            count=population,
        ),
        config=EvaluationCorpusRecipe(
            recipe=recipe,
            protocol=EvaluationProtocol(
                benchmark=benchmark,
                version=version,
                operation=OperationKind.CHOICE_LIKELIHOOD,
                sampling=None,
                trials=TrialPolicy(count=1, seed=None),
                metrics=tuple(
                    Metric(
                        name=names[value],
                        kind=metric_kind,
                        higher_is_better=True,
                    )
                    for value in normalizations
                ),
                primary_metric=names[normalizations[0]],
                num_fewshot=template.num_fewshot,
                summary=SummaryKind.MEAN,
                expected_population=population,
                likelihood_metrics=tuple(
                    LikelihoodMetric(
                        name=names[value],
                        normalization=value,
                        scoring=scoring,
                    )
                    for value in normalizations
                ),
            ),
            item_builder=partial(choice_item, subject=template.subject),
            protocol_name="harness",
            partition=partition,
        ),
        pipeline=prepare_evaluation,
    )


def text_generation_evaluation(
    *,
    name: str,
    benchmark: str,
    source: HfSource,
    population: int,
    template: GenerationTemplate,
    metrics: tuple[str, ...],
    sampling: Sampling,
    scorer: str,
    grader: Environment,
) -> RlDataSource[EvaluationCorpusRecipe]:
    """Bind a family's compiled demonstrations, scorer and sampling protocol."""
    return RlDataSource(
        info=SourceInfo(
            id="Checkpoint:" + name,
            title=name,
            origin="lm-eval",
            family="checkpoint",
            tags=("eval", "benchmark"),
            count=population,
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name="checkpoint-" + name,
                source=replace(source, read=partial(read_generation_file, template=template)),
                convert=partial(convert_generation, scorer=scorer),
                version=EXPLICIT_TEMPLATE_VERSION + "-" + scorer_revision(scorer),
                intended_use=IntendedUse.EVAL,
                grader=grader,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                version=EXPLICIT_TEMPLATE_VERSION,
                operation=OperationKind.TEXT_GENERATION,
                sampling=sampling,
                trials=TrialPolicy(count=1, seed=0),
                metrics=tuple(Metric(name=metric, kind=MetricKind.BINARY, higher_is_better=True) for metric in metrics),
                primary_metric=metrics[0],
                num_fewshot=template.fewshot.shots,
                summary=SummaryKind.MEAN,
                expected_population=population,
            ),
            generation_item,
        ),
        pipeline=prepare_evaluation,
    )


@dataclass(frozen=True)
class QuestionAnsweringDataset:
    name: str
    source: HfSource
    population: int
    pool_file: str


def _compile_lm_eval_text_generation(
    dataset: QuestionAnsweringDataset,
    variant: ProtocolVariant,
    *,
    grader: Environment,
    metrics: tuple[str, ...],
    primary_metric: str,
    stop: tuple[str, ...],
    chat_options: dict[int, dict[str, bool | str]],
) -> tuple[RlDataSource[EvaluationCorpusRecipe], ...]:
    shots = variant.num_fewshot
    assert shots is not None
    template = HarnessTemplate(name=dataset.name, num_fewshot=shots, fewshot_file=dataset.pool_file)
    source = RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + dataset.name,
            title=dataset.name,
            origin="Evalchemy",
            family="short-answer",
            tags=("eval", "source:evalchemy", "benchmark", "family:short-answer"),
            count=dataset.population,
        ),
        config=EvaluationCorpusRecipe(
            recipe=CurationRecipe(
                name="evalchemy-" + dataset.name,
                source=replace(dataset.source, read=partial(read_lm_eval_generation, template=template)),
                inputs={"fewshot": replace(dataset.source, files=(dataset.pool_file,), read=None)} if shots else {},
                convert=partial(convert_generation_document, benchmark=dataset.name),
                grader=grader,
                version=scorer_revision("harness_generation"),
                intended_use=IntendedUse.EVAL,
            ),
            protocol=EvaluationProtocol(
                benchmark=dataset.name.replace("_", "-"),
                version="2026.11.13",
                operation=OperationKind.TEXT_GENERATION,
                sampling=Sampling(
                    temperature=None, top_p=None, top_k=None, max_tokens=None, stop=stop, chat_template_kwargs={}
                ),
                trials=TrialPolicy(count=1, seed=0),
                num_fewshot=shots,
                metrics=tuple(
                    Metric(
                        name=metric,
                        kind=MetricKind.CONTINUOUS if metric.startswith("f1,") else MetricKind.BINARY,
                        higher_is_better=True,
                    )
                    for metric in metrics
                ),
                primary_metric=primary_metric,
                summary=SummaryKind.MEAN,
                expected_population=dataset.population,
            ),
            item_builder=generation_item,
            protocol_name="harness",
        ),
        pipeline=prepare_evaluation,
    )
    if variant.prompt_format is PromptFormat.CHAT:
        source = chat_source(source, chat_options[shots])
    return (source,)


def lm_eval_text_generation(
    *,
    dataset: QuestionAnsweringDataset,
    shots: tuple[int, ...],
    grader: Environment,
    metrics: tuple[str, ...],
    primary_metric: str,
    stop: tuple[str, ...],
    chat_options: dict[int, dict[str, bool | str]],
) -> EvaluationBenchmark:
    return EvaluationBenchmark(
        name=dataset.name.replace("_", "-"),
        variants=tuple(
            ProtocolVariant("harness", count, form)
            for count in shots
            for form in (
                (PromptFormat.COMPLETION, PromptFormat.CHAT) if count in chat_options else (PromptFormat.COMPLETION,)
            )
        ),
        build=partial(
            _compile_lm_eval_text_generation,
            dataset,
            grader=grader,
            metrics=metrics,
            primary_metric=primary_metric,
            stop=stop,
            chat_options=chat_options,
        ),
    )
