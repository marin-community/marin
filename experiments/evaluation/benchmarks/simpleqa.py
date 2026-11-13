# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SimpleQA evaluation declarations."""

from functools import partial

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.simpleqa import convert_simpleqa, read_simpleqa, simpleqa_item
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

SIMPLEQA_REVISION = "e319282ab125c3dbd0c7fd00be2e4dd54e7e8f94"


def simpleqa_source(benchmark: str, *, mini: bool) -> RlDataSource[EvaluationCorpusRecipe]:
    name = "simpleqa-mini" if mini else "simpleqa"
    rows = 500 if mini else 4326
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name,
            title=benchmark,
            origin="Evalchemy",
            family="judged-answer",
            tags=("eval", "source:evalchemy", "single-turn", "benchmark", "llm-judge"),
            count=rows,
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name="evalchemy-" + name,
                source=HfSource(
                    "basicv8vc/SimpleQA",
                    SIMPLEQA_REVISION,
                    ("simple_qa_test_set.csv",),
                    SourceFormat.CSV,
                    read=partial(read_simpleqa, mini=mini),
                ),
                convert=partial(convert_simpleqa, benchmark=benchmark),
                version="2026.11.02",
                intended_use=IntendedUse.EVAL,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                requires_judge=True,
                version="2026.11.02",
                operation=OperationKind.CHAT_GENERATION,
                sampling=Sampling(
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_tokens=512,
                    stop=(),
                    chat_template_kwargs={"enable_thinking": True},
                ),
                trials=TrialPolicy(count=1, seed=None),
                metrics=Metrics.CLASSIFICATION,
                primary_metric="accuracy",
                summary=SummaryKind.CLASSIFICATION,
                expected_population=rows,
            ),
            simpleqa_item,
        ),
        pipeline=prepare_evaluation,
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [simpleqa_source(benchmark, mini=mini) for benchmark, mini in (("SimpleQA", False), ("SimpleQAMini", True))]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources())
