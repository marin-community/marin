# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact bindings for compiled likelihood and generation protocols."""

from dataclasses import replace
from functools import partial
from typing import Literal

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
    generation_item,
    original_task_name,
    read_choice_file,
)
from taskcompendium.importers.fewshot import ChoiceTemplate, GenerationTemplate, convert_generation, read_generation_file
from taskcompendium.importers.fewshot import read_choice_file as read_fewshot_choices
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.evaluation import EvalDataset, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo


def checkpoint_choice_source(
    name: str,
    benchmark: str,
    source: HfSource,
    population: int,
    template: ChoiceTemplate,
    metrics: tuple[LikelihoodMetric, ...],
    *,
    subject: str = "",
) -> RlDataSource[EvalDataset]:
    """Bind a family's explicit prompt and normalization policies to a curated corpus."""
    version = "2026.11.05.3"
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
        config=EvalDataset(
            CurationRecipe(
                name="checkpoint-" + name,
                source=replace(source, read=partial(read_fewshot_choices, template=template)),
                inputs=(
                    {"fewshot": HfSource(source.repo, source.revision, (policy.pool_file,), SourceFormat.PARQUET)}
                    if policy.pool_file
                    else {}
                ),
                convert=partial(convert_choice_document, benchmark=benchmark),
                version=version,
                intended_use=IntendedUse.EVAL,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                version=version,
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
        pipeline=curate,
    )


def choice_source(
    name: str,
    repo: str,
    revision: str,
    file: str,
    population: int,
    template: HarnessTemplate,
    normalizations: tuple[LikelihoodNormalization, ...],
    *,
    metric_kind: MetricKind = MetricKind.BINARY,
    scoring: Literal["most_likely", "probability_mass"] = "most_likely",
    version: str = "2026.11.02",
) -> RlDataSource[EvalDataset]:
    """Bind pinned rows and few-shots, with the first normalization as the headline."""
    names = {
        LikelihoodNormalization.RAW: "acc,none",
        LikelihoodNormalization.CHARACTERS: "acc_norm,none",
        LikelihoodNormalization.BYTES: "acc_bytes,none",
    }
    inputs = (
        {"fewshot": HfSource(repo, revision, (template.fewshot_file,), SourceFormat.PARQUET)}
        if template.num_fewshot
        else {}
    )
    recipe = CurationRecipe(
        name="evalchemy-" + name,
        source=HfSource(
            repo, revision, (file,), SourceFormat.PARQUET, read=partial(read_choice_file, template=template)
        ),
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
        config=EvalDataset(
            recipe=recipe,
            protocol=EvaluationProtocol(
                benchmark=name,
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
        ),
        pipeline=curate,
    )


def fewshot_generation_source(
    name: str,
    benchmark: str,
    source: HfSource,
    population: int,
    template: GenerationTemplate,
    metrics: tuple[str, ...],
    sampling: Sampling,
    scorer: str,
    grader: Environment,
) -> RlDataSource[EvalDataset]:
    """Bind a family's compiled demonstrations, scorer and sampling protocol."""
    version = "2026.11.05.3"
    return RlDataSource(
        info=SourceInfo(
            id="Checkpoint:" + name,
            title=name,
            origin="lm-eval",
            family="checkpoint",
            tags=("eval", "benchmark"),
            count=population,
        ),
        config=EvalDataset(
            CurationRecipe(
                name="checkpoint-" + name,
                source=replace(source, read=partial(read_generation_file, template=template)),
                convert=partial(convert_generation, scorer=scorer),
                version=version + "-" + scorer_revision(scorer),
                intended_use=IntendedUse.EVAL,
                grader=grader,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                version=version,
                operation=OperationKind.TEXT_GENERATION,
                sampling=sampling,
                trials=TrialPolicy(count=1, seed=None),
                metrics=tuple(Metric(name=metric, kind=MetricKind.BINARY, higher_is_better=True) for metric in metrics),
                primary_metric=metrics[0],
                num_fewshot=template.fewshot.shots,
                summary=SummaryKind.MEAN,
                expected_population=population,
            ),
            generation_item,
        ),
        pipeline=curate,
    )
