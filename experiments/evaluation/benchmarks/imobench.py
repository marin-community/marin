# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IMO-Bench (https://imobench.github.io): olympiad short answers and judged proofs.

Both benchmarks are graded by the run's judge with the paper's autograder prompts; the paper used
Gemini 2.5 Pro, so scores are comparable only under the same judge. IMO-GradingBench measures
graders rather than solvers and chose ProofBench's judge; IMO-LeanProofBench needs a Lean toolchain.
Neither is declared.
"""

from collections.abc import Callable

from taskcompendium.evaluation.spec import (
    EvaluationItem,
    EvaluationProtocol,
    Metric,
    MetricKind,
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
    convert_proofbench,
    proofbench_item,
)
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import Converter, IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, RowSelector, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

SUPERHUMAN_REVISION = "9b620a92886b3a7655b30ddcfe36e725e9596509"
# answerbench_v2 (2026-02-12) fixes ambiguous statements and wrong answers in v1.
ANSWERBENCH_SHA256 = "275877a9d988d85278fad3a5f8a41d7f83393a60bf259531ec0a5161e6b21cf9"
# proofbench_v2 (2025-05-29) fixes a typo in PB-Advanced-022.
PROOFBENCH_SHA256 = "aa8b813dbd4068137e3d165e5da228f6e0e1cc85a91c37883e1791b954e43af0"
VERSION = "2026.11.24"
PROOF_SCORE = Metric(name="score", kind=MetricKind.CONTINUOUS, higher_is_better=True)


def _source(
    *,
    name: str,
    benchmark: str,
    filename: str,
    sha256: str,
    population: int,
    select: RowSelector | None,
    convert: Converter,
    item_builder: Callable[[TaskSpec], EvaluationItem],
    metric: Metric,
    trials: int,
) -> RlDataSource[EvaluationCorpusRecipe]:
    return RlDataSource(
        info=SourceInfo(
            id="IMOBench:" + name,
            title=benchmark,
            origin="Google DeepMind",
            family="judged-math",
            tags=("eval", "source:imobench", "single-turn", "benchmark", "math", "llm-judge"),
            count=population,
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name="imobench-" + name,
                source=UrlSource(
                    "https://raw.githubusercontent.com/google-deepmind/superhuman/"
                    f"{SUPERHUMAN_REVISION}/imobench/{filename}",
                    sha256,
                    filename,
                    SourceFormat.CSV,
                    select=select,
                ),
                convert=convert,
                version=VERSION,
                intended_use=IntendedUse.EVAL,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
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
                trials=TrialPolicy(count=trials, seed=None),
                metrics=(metric,),
                primary_metric=metric.name,
                summary=SummaryKind.MEAN,
                expected_population=population,
            ),
            item_builder,
        ),
        pipeline=prepare_evaluation,
    )


def answerbench_source() -> RlDataSource[EvaluationCorpusRecipe]:
    """399 short-answer problems (the malformed algebra-036 is rejected); the paper averages 8 runs, this protocol 4."""
    return _source(
        name="answerbench",
        benchmark="IMO-AnswerBench",
        filename="answerbench_v2.csv",
        sha256=ANSWERBENCH_SHA256,
        population=399,
        select=answerbench_row_selected,
        convert=convert_answerbench,
        item_builder=answerbench_item,
        metric=Metrics.ACCURACY,
        trials=4,
    )


def proofbench_source() -> RlDataSource[EvaluationCorpusRecipe]:
    """60 proof problems (30 Basic, 30 Advanced); score is awarded points over 7."""
    return _source(
        name="proofbench",
        benchmark="IMO-ProofBench",
        filename="proofbench_v2.csv",
        sha256=PROOFBENCH_SHA256,
        population=60,
        select=None,
        convert=convert_proofbench,
        item_builder=proofbench_item,
        metric=PROOF_SCORE,
        trials=4,
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [answerbench_source(), proofbench_source()]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources(), names={"IMO-AnswerBench": "imo-answerbench", "IMO-ProofBench": "imo-proofbench"})
