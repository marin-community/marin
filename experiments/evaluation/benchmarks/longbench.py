# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LongBench v2 with precompiled middle-truncated prompts for every model context."""

from functools import partial

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.evaluation.spec import (
    CohortOrder,
    CohortPolicy,
    EvaluationProtocol,
    Metric,
    MetricKind,
    OperationKind,
    Sampling,
    SummaryKind,
    TaskGroup,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.longbench_v2 import (
    LONGBENCH_V2_LENGTH_LABEL,
    LONGBENCH_V2_LENGTHS,
    LONGBENCH_V2_TOKEN_BUDGETS,
    convert_longbench_v2,
    longbench_v2_item,
    read_longbench_v2,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin

LONGBENCH_V2_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
# The longest item's full prompt and truncated copies.
LONGBENCH_V2_RESOURCE_BYTES = 128 * 1024 * 1024
# A limited run takes short, medium and long items in turn.
LONGBENCH_V2_COHORT = CohortPolicy(
    groups=tuple(TaskGroup(labels={LONGBENCH_V2_LENGTH_LABEL: length}) for length in LONGBENCH_V2_LENGTHS),
    order=CohortOrder.ROUND_ROBIN,
)
# Upstream samples at temperature 0.1 with 128 output tokens. Greedy decoding needs no request
# seed, which TPU vLLM does not support, and 4,096 tokens let a model explain before answering.
NO_THINKING = Sampling(
    temperature=0.0,
    top_p=None,
    top_k=None,
    max_tokens=4096,
    stop=(),
    chat_template_kwargs={"enable_thinking": False},
)
# The Qwen3.5 model card's thinking-mode settings, without its presence penalty, which Sampling
# does not express.
THINKING = Sampling(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    max_tokens=32768,
    stop=(),
    chat_template_kwargs={"enable_thinking": True},
)


def longbench_source(protocol_name: str, sampling: Sampling, seed: int | None) -> RlDataSource[EvaluationCorpusRecipe]:
    return RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            origin=SourceOrigin.LM_EVAL,
            upstream_dataset="LongBench-v2",
            upstream_name="LongBench-v2",
            variant=protocol_name,
            build_path="Checkpoint",
            family="long-context",
            tags=("eval", "source:longbench-v2", "benchmark", "long-context", "multiple-choice"),
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name="longbench-v2-" + protocol_name,
                source=HfSource(
                    "zai-org/LongBench-v2",
                    LONGBENCH_V2_REVISION,
                    ("data.json",),
                    SourceFormat.JSON,
                    read=read_longbench_v2,
                ),
                convert=convert_longbench_v2,
                version=scorer_revision("longbench_v2"),
                intended_use=IntendedUse.EVAL,
                resource_budget_bytes=LONGBENCH_V2_RESOURCE_BYTES,
                grader=BASIC_GRADER,
            ),
            EvaluationProtocol(
                benchmark="LongBenchV2",
                version="2026.11.13",
                operation=OperationKind.CHAT_GENERATION,
                sampling=sampling,
                trials=TrialPolicy(count=1, seed=seed),
                metrics=(Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True),),
                primary_metric="accuracy",
                summary=SummaryKind.MEAN,
                cohort=LONGBENCH_V2_COHORT,
            ),
            partial(longbench_v2_item, budgets=LONGBENCH_V2_TOKEN_BUDGETS),
            protocol_name=protocol_name,
        ),
        pipeline=prepare_evaluation,
    )


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(
        [longbench_source("no-thinking", NO_THINKING, seed=None), longbench_source("thinking", THINKING, seed=42)],
        names={"LongBenchV2": "longbench-v2"},
    )
