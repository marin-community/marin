# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LAMBADA continuation-likelihood evaluation."""

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
    continuation_item,
    convert_continuation_document,
    read_continuation_file,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo


def benchmarks() -> list[EvaluationBenchmark]:
    source = RlDataSource(
        info=SourceInfo(
            id="Evalchemy:lambada_openai",
            title="lambada_openai",
            origin="Evalchemy",
            family="likelihood-continuation",
            tags=("eval", "source:evalchemy", "benchmark"),
            count=5153,
        ),
        config=EvaluationCorpusRecipe(
            recipe=CurationRecipe(
                name="evalchemy-lambada_openai",
                source=HfSource(
                    "EleutherAI/lambada_openai",
                    "900124bf3b8235c6daf21033af9948b3f07346c4",
                    ("default/test/default.parquet",),
                    SourceFormat.PARQUET,
                    read=read_continuation_file,
                ),
                convert=convert_continuation_document,
                version="2026.11.02",
                intended_use=IntendedUse.EVAL,
            ),
            protocol=EvaluationProtocol(
                benchmark="lambada_openai",
                version="2026.11.02",
                operation=OperationKind.CONTINUATION_LIKELIHOOD,
                sampling=None,
                trials=TrialPolicy(count=1, seed=None),
                metrics=(
                    Metric(name="acc,none", kind=MetricKind.BINARY, higher_is_better=True),
                    Metric(name="perplexity,none", kind=MetricKind.CONTINUOUS, higher_is_better=False),
                ),
                primary_metric="acc,none",
                summary=SummaryKind.PERPLEXITY,
                expected_population=5153,
                likelihood_metrics=(
                    LikelihoodMetric(
                        name="acc,none", normalization=LikelihoodNormalization.RAW, scoring="greedy_accuracy"
                    ),
                    LikelihoodMetric(
                        name="perplexity,none", normalization=LikelihoodNormalization.RAW, scoring="log_likelihood"
                    ),
                ),
            ),
            item_builder=continuation_item,
            protocol_name="harness",
        ),
        pipeline=prepare_evaluation,
    )
    return fixed_benchmarks([source], "harness", {"lambada_openai": "lambada"})
