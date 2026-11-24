# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned public FrontierScience evaluations."""

from taskcompendium.evaluation.spec import (
    ContextPolicy,
    EvaluationProtocol,
    Metric,
    MetricKind,
    Metrics,
    OperationKind,
    OutputFit,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.frontierscience import convert_olympiad, convert_research, frontierscience_item
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

SOURCE_REVISION = "25ed67db7da8f4591484e764008ff585544f5a30"
RESEARCH_METRICS = (
    Metric(name="pass_rate", kind=MetricKind.BINARY, higher_is_better=True),
    Metric(name="rubric_points", kind=MetricKind.CONTINUOUS, higher_is_better=True),
)


def sources(output_fit: OutputFit = OutputFit.REQUIRE) -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                id=f"frontierscience-{track}",
                title=f"FrontierScience {track.title()}",
                origin="OpenAI",
                family="judged-answer",
                tags=("eval", "single-turn", "benchmark", "llm-judge"),
                count=count,
            ),
            config=EvaluationCorpusRecipe(
                recipe=CurationRecipe(
                    name=f"frontierscience-{track}",
                    source=HfSource(
                        "openai/frontierscience", SOURCE_REVISION, (f"{track}/test.jsonl",), SourceFormat.JSONL
                    ),
                    convert=convert,
                    version=version if output_fit is OutputFit.REQUIRE else f"{version}-remaining-context",
                    intended_use=IntendedUse.EVAL,
                ),
                protocol_name="evalchemy" if output_fit is OutputFit.REQUIRE else "remaining-context",
                protocol=EvaluationProtocol(
                    benchmark=f"frontierscience-{track}",
                    version=version if output_fit is OutputFit.REQUIRE else f"{version}-remaining-context",
                    operation=OperationKind.CHAT_GENERATION,
                    requires_judge=True,
                    context_policy=ContextPolicy(output_fit=output_fit),
                    sampling=Sampling(
                        temperature=None, top_p=None, top_k=None, max_tokens=None, stop=(), chat_template_kwargs={}
                    ),
                    trials=TrialPolicy(count=1, seed=None),
                    metrics=metrics,
                    primary_metric=metrics[0].name,
                    summary=SummaryKind.MEAN,
                    expected_population=count,
                ),
                item_builder=frontierscience_item,
            ),
            pipeline=prepare_evaluation,
        )
        for track, count, convert, metrics, version in (
            ("olympiad", 100, convert_olympiad, (Metrics.ACCURACY,), "2026.11.19"),
            ("research", 60, convert_research, RESEARCH_METRICS, "2026.11.19.1"),
        )
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks([*sources(), *sources(OutputFit.SHRINK)])
