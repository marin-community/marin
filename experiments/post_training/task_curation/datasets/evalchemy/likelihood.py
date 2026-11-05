# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared artifact binding for native likelihood-choice families."""

from functools import partial

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    LikelihoodMetric,
    LikelihoodNormalization,
    Metric,
    MetricKind,
    OperationKind,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evaluation_harness import (
    HarnessTemplate,
    choice_item,
    convert_choice_document,
    original_task_name,
    read_choice_file,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.evaluation import EvalDataset, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo


def choice_source(
    name: str,
    repo: str,
    revision: str,
    file: str,
    population: int,
    template: HarnessTemplate,
    normalizations: tuple[LikelihoodNormalization, ...],
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
        version="2026.11.02",
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
                version="2026.11.02",
                operation=OperationKind.CHOICE_LIKELIHOOD,
                sampling=None,
                trials=TrialPolicy(count=1, seed=None),
                metrics=tuple(
                    Metric(
                        name=names[value],
                        kind=MetricKind.CONTINUOUS if template.name == "truthfulqa_mc2" else MetricKind.BINARY,
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
                        scoring="probability_mass" if template.name == "truthfulqa_mc2" else "most_likely",
                    )
                    for value in normalizations
                ),
            ),
            item_builder=partial(choice_item, subject=template.subject),
        ),
        pipeline=curate,
    )
