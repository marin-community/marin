# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full pinned long-context populations and run-time eligibility policy."""

from taskcompendium.evaluation.spec import (
    CohortOrder,
    CohortPolicy,
    ContextPolicy,
    EvaluationProtocol,
    LimitStage,
    Metric,
    MetricKind,
    OperationKind,
    OutputBudget,
    OutputFit,
    PrefixProbe,
    Sampling,
    SummaryKind,
    TaskGroup,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.evalchemy.long_context import (
    MRCR_SOURCE_POPULATION,
    convert_graphwalks,
    convert_mrcr,
    graphwalks_item,
    mrcr_item,
    read_mrcr,
)
from taskcompendium.importers.evalchemy.scorers.long_context import (
    GRAPHWALKS_CONTEXT_MARGIN,
    GRAPHWALKS_MIN_OUTPUT_TOKENS,
    GRAPHWALKS_OUTPUT_BUDGET_MULTIPLIER,
    GRAPHWALKS_PREFIX_CHARS_PER_TOKEN,
    GRAPHWALKS_PREFIX_TOKEN_MARGIN,
    GRAPHWALKS_REASONING_RESERVE,
    MRCR_BIN_UPPER_BOUNDS,
    MRCR_NEEDLE_COUNTS,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

MRCR_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
LONG_CONTEXT_RESOURCE_BYTES = 64 * 1024 * 1024
MRCR_FILES = (
    "2needle/2needle_0.parquet",
    "2needle/2needle_1.parquet",
    "4needle/4needle_0.parquet",
    "4needle/4needle_1.parquet",
    "8needle/8needle_0.parquet",
    "8needle/8needle_1.parquet",
)
GRAPHWALKS_REVISION = "be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7"
GRAPHWALKS_FILES = ("graphwalks_128k_and_shorter.parquet", "graphwalks_256k_to_1mil.parquet")
MRCR_COHORT = CohortPolicy(
    groups=tuple(
        TaskGroup(
            labels={"mrcr.bin_upper": str(upper), "mrcr.needles": str(needles)},
            minimum_context=upper,
        )
        for upper in MRCR_BIN_UPPER_BOUNDS
        for needles in MRCR_NEEDLE_COUNTS
    ),
    order=CohortOrder.ROUND_ROBIN,
    limit_stage=LimitStage.ELIGIBLE,
)
GRAPHWALKS_COHORT = CohortPolicy(limit_stage=LimitStage.ELIGIBLE)
GRAPHWALKS_CONTEXT_POLICY = ContextPolicy(
    output_fit=OutputFit.REQUIRE,
    context_reserve=GRAPHWALKS_CONTEXT_MARGIN,
    output_budget=OutputBudget(
        minimum=GRAPHWALKS_MIN_OUTPUT_TOKENS,
        reserve=GRAPHWALKS_REASONING_RESERVE,
        multiplier=GRAPHWALKS_OUTPUT_BUDGET_MULTIPLIER,
    ),
    prefix_probe=PrefixProbe(
        message_index=0,
        characters_per_token=GRAPHWALKS_PREFIX_CHARS_PER_TOKEN,
        token_margin=GRAPHWALKS_PREFIX_TOKEN_MARGIN,
    ),
)


def mrcr_source(*, name: str, max_context: int) -> RlDataSource[EvaluationCorpusRecipe]:
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name.lower(),
            title=name,
            origin="Evalchemy",
            family="long-context",
            tags=("eval", "source:evalchemy", "benchmark", "long-context", "retrieval"),
            count=MRCR_SOURCE_POPULATION,
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name="evalchemy-" + name.lower(),
                source=HfSource("openai/mrcr", MRCR_REVISION, MRCR_FILES, SourceFormat.PARQUET, read=read_mrcr),
                convert=convert_mrcr,
                version=scorer_revision("mrcr"),
                intended_use=IntendedUse.EVAL,
                resource_budget_bytes=LONG_CONTEXT_RESOURCE_BYTES,
                grader=BASIC_GRADER,
            ),
            EvaluationProtocol(
                benchmark=name,
                version="2026.11.02",
                operation=OperationKind.CHAT_GENERATION,
                sampling=Sampling(
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_tokens=4096,
                    stop=(),
                    chat_template_kwargs={"enable_thinking": False},
                ),
                trials=TrialPolicy(count=1, seed=None),
                metrics=(
                    Metric(name="accuracy", kind=MetricKind.CONTINUOUS, higher_is_better=True),
                    Metric(name="prefix_hit_rate", kind=MetricKind.BINARY, higher_is_better=True),
                ),
                primary_metric="accuracy",
                summary=SummaryKind.MEAN,
                expected_population=MRCR_SOURCE_POPULATION,
                context_window=max_context,
                cohort=MRCR_COHORT,
            ),
            mrcr_item,
        ),
        pipeline=prepare_evaluation,
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        mrcr_source(name="MRCR", max_context=73728),
        mrcr_source(name="MRCR-32k", max_context=32704),
        mrcr_source(name="MRCR-65k", max_context=65472),
        RlDataSource(
            info=SourceInfo(
                id="Evalchemy:graphwalks",
                title="GraphWalks",
                origin="Evalchemy",
                family="long-context",
                tags=("eval", "source:evalchemy", "benchmark", "long-context", "graph"),
                count=1150,
            ),
            config=EvaluationCorpusRecipe(
                CurationRecipe(
                    name="evalchemy-graphwalks",
                    source=HfSource("openai/graphwalks", GRAPHWALKS_REVISION, GRAPHWALKS_FILES, SourceFormat.PARQUET),
                    convert=convert_graphwalks,
                    version=scorer_revision("graphwalks"),
                    intended_use=IntendedUse.EVAL,
                    resource_budget_bytes=LONG_CONTEXT_RESOURCE_BYTES,
                    grader=BASIC_GRADER,
                ),
                EvaluationProtocol(
                    benchmark="GraphWalks",
                    version="2026.11.02",
                    operation=OperationKind.CHAT_GENERATION,
                    sampling=Sampling(
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        max_tokens=131072,
                        stop=(),
                        chat_template_kwargs={"enable_thinking": True},
                    ),
                    trials=TrialPolicy(count=1, seed=None),
                    metrics=tuple(
                        Metric(
                            name=name,
                            kind=MetricKind.BINARY if name == "exact_match" else MetricKind.CONTINUOUS,
                            higher_is_better=True,
                        )
                        for name in ("f1", "precision", "recall", "exact_match")
                    ),
                    primary_metric="f1",
                    summary=SummaryKind.MEAN,
                    expected_population=1150,
                    cohort=GRAPHWALKS_COHORT,
                    context_policy=GRAPHWALKS_CONTEXT_POLICY,
                ),
                graphwalks_item,
            ),
            pipeline=prepare_evaluation,
        ),
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    populations = sources()
    return [
        *fixed_benchmarks(
            [
                source
                for source in populations
                if source.config is not None and source.config.protocol.benchmark == "GraphWalks"
            ]
        ),
        *fixed_benchmarks([mrcr_source(name="MRCR", max_context=73728)], "evalchemy", {"MRCR": "mrcr"}),
        *fixed_benchmarks([mrcr_source(name="MRCR-32k", max_context=32704)], "32k", {"MRCR-32k": "mrcr"}),
        *fixed_benchmarks([mrcr_source(name="MRCR-65k", max_context=65472)], "65k", {"MRCR-65k": "mrcr"}),
    ]
