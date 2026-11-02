# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full pinned long-context populations and run-time eligibility policy."""

from taskcompendium.evaluation.spec import (
    ContextPolicy,
    EvaluationProtocol,
    Metric,
    MetricKind,
    OperationKind,
    Sampling,
    SummaryKind,
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
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.evalchemy.environments import BASIC_GRADER
from experiments.post_training.task_curation.evaluation import EvalDataset, curate
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


def mrcr_source(name: str, max_context: int) -> RlDataSource[EvalDataset]:
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name.lower(),
            title=name,
            origin="Evalchemy",
            family="long-context",
            tags=("eval", "source:evalchemy", "benchmark", "long-context", "retrieval"),
            count=MRCR_SOURCE_POPULATION,
        ),
        config=EvalDataset(
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
                    temperature=0,
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
                context_policy=ContextPolicy(kind="mrcr", max_context=max_context),
            ),
            mrcr_item,
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataset]]:
    return [
        mrcr_source("MRCR", 73728),
        mrcr_source("MRCR-32k", 32704),
        mrcr_source("MRCR-65k", 65472),
        RlDataSource(
            info=SourceInfo(
                id="Evalchemy:graphwalks",
                title="GraphWalks",
                origin="Evalchemy",
                family="long-context",
                tags=("eval", "source:evalchemy", "benchmark", "long-context", "graph"),
                count=1150,
            ),
            config=EvalDataset(
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
                        temperature=0,
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
                    context_policy=ContextPolicy(kind="graphwalks", max_context=None),
                ),
                graphwalks_item,
            ),
            pipeline=curate,
        ),
    ]
