# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IMO-AnswerBench (https://imobench.github.io): olympiad problems with short answers.

The run's judge grades with the paper's AnswerAutoGrader prompt; the paper used Gemini 2.5 Pro, so
scores are comparable only under the same judge.
"""

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.imobench import (
    answerbench_item,
    answerbench_row_selected,
    convert_answerbench,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

SUPERHUMAN_REVISION = "9b620a92886b3a7655b30ddcfe36e725e9596509"
# answerbench_v2 (2026-02-12) fixes ambiguous statements and wrong answers in v1.
ANSWERBENCH_SHA256 = "275877a9d988d85278fad3a5f8a41d7f83393a60bf259531ec0a5161e6b21cf9"
VERSION = "2026.11.19"
BENCHMARK = "IMO-AnswerBench"
POPULATION = 399


def answerbench_source() -> RlDataSource[EvaluationCorpusRecipe]:
    """399 short-answer problems (the malformed algebra-036 is dropped); the paper averages 8 runs, this protocol 4."""
    return RlDataSource(
        info=SourceInfo(
            id="IMOBench:answerbench",
            title=BENCHMARK,
            origin="Google DeepMind",
            family="judged-math",
            tags=("eval", "source:imobench", "single-turn", "benchmark", "math", "llm-judge"),
            count=POPULATION,
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name="imobench-answerbench",
                source=UrlSource(
                    "https://raw.githubusercontent.com/google-deepmind/superhuman/"
                    f"{SUPERHUMAN_REVISION}/imobench/answerbench_v2.csv",
                    ANSWERBENCH_SHA256,
                    "answerbench_v2.csv",
                    SourceFormat.CSV,
                    select=answerbench_row_selected,
                ),
                convert=convert_answerbench,
                version=VERSION,
                intended_use=IntendedUse.EVAL,
            ),
            EvaluationProtocol(
                benchmark=BENCHMARK,
                requires_judge=True,
                version=VERSION,
                operation=OperationKind.CHAT_GENERATION,
                # The paper does not publish solver sampling; keep each model's served defaults and
                # let the run's token budget bound generation.
                sampling=Sampling(
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_tokens=None,
                    stop=(),
                    chat_template_kwargs={"enable_thinking": True},
                ),
                trials=TrialPolicy(count=4, seed=None),
                metrics=(Metrics.ACCURACY,),
                primary_metric=Metrics.ACCURACY.name,
                summary=SummaryKind.MEAN,
                expected_population=POPULATION,
            ),
            answerbench_item,
        ),
        pipeline=prepare_evaluation,
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [answerbench_source()]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources(), names={BENCHMARK: "imo-answerbench"})
