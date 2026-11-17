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
from taskcompendium.importers.evalchemy.simpleqa import (
    convert_simpleqa,
    convert_simpleqa_verified,
    read_simpleqa,
    read_simpleqa_verified,
    simpleqa_item,
    simpleqa_verified_item,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

SIMPLEQA_REVISION = "e319282ab125c3dbd0c7fd00be2e4dd54e7e8f94"
# Includes the publisher's 2026-03-09 correction of the Lilongwe question ("latitude" to "longitude").
SIMPLEQA_VERIFIED_REVISION = "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc"
SIMPLEQA_VERIFIED_ROWS = 1000


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


def simpleqa_verified_source(*, thinking: bool) -> RlDataSource[EvaluationCorpusRecipe]:
    """Google DeepMind's SimpleQA Verified (arXiv:2509.07968), the leaderboard's 1,000 cleaned SimpleQA questions.

    The judge sees only the final answer, as in the reference notebook. The leaderboard judge is
    gpt-4.1-2025-04-14; runs name their own judge, so scores are comparable only under the same judge.
    """
    benchmark = "SimpleQAVerifiedThinking" if thinking else "SimpleQAVerified"
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + ("simpleqa-verified-thinking" if thinking else "simpleqa-verified"),
            title=benchmark,
            origin="Evalchemy",
            family="judged-answer",
            tags=("eval", "source:evalchemy", "single-turn", "benchmark", "llm-judge"),
            count=SIMPLEQA_VERIFIED_ROWS,
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name="evalchemy-" + ("simpleqa-verified-thinking" if thinking else "simpleqa-verified"),
                source=HfSource(
                    "google/simpleqa-verified",
                    SIMPLEQA_VERIFIED_REVISION,
                    ("simpleqa_verified.csv",),
                    SourceFormat.CSV,
                    read=read_simpleqa_verified,
                ),
                # Both variants grade the same task corpus.
                convert=partial(convert_simpleqa_verified, benchmark="SimpleQAVerified"),
                version="2026.11.13",
                intended_use=IntendedUse.EVAL,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                requires_judge=True,
                version="2026.11.13",
                operation=OperationKind.CHAT_GENERATION,
                # Like the reference notebook, keep each model's own sampling defaults; greedy decoding
                # sends reasoning models such as gpt-oss into repetition loops that exhaust the cap.
                sampling=Sampling(
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_tokens=16384 if thinking else 2048,
                    stop=(),
                    chat_template_kwargs={"enable_thinking": thinking},
                ),
                trials=TrialPolicy(count=1, seed=None),
                # The classification summary also reports f1, the leaderboard's headline metric.
                metrics=Metrics.CLASSIFICATION,
                primary_metric="accuracy",
                summary=SummaryKind.CLASSIFICATION,
                expected_population=SIMPLEQA_VERIFIED_ROWS,
            ),
            simpleqa_verified_item,
        ),
        pipeline=prepare_evaluation,
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        *(simpleqa_source(benchmark, mini=mini) for benchmark, mini in (("SimpleQA", False), ("SimpleQAMini", True))),
        simpleqa_verified_source(thinking=False),
        simpleqa_verified_source(thinking=True),
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(
        sources(),
        names={"SimpleQAVerified": "simpleqa-verified", "SimpleQAVerifiedThinking": "simpleqa-verified-thinking"},
    )
