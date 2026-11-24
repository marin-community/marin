# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EconEvals synthetic DWA and occupation benchmarks, judged pairwise against a baseline answer.

Questions and baseline answers are the Arena-Hard inputs EconEvals judged for its paper, pinned in
the EconEvalsIntermediate release. The paper's baseline is o3-mini, which strong models beat on most
items; the ``-o3`` variants judge against the release's o3 answers to leave headroom. The ``-Sampled``
variants keep a fixed sample of each DWA or occupation category for development runs. The real-query
DWA benchmark is not included: its prompts are pointers into LMSYS-Chat-1M and WildChat that must be
rebuilt from those corpora.
"""

from dataclasses import dataclass
from functools import partial

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metric,
    MetricKind,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.econevals import (
    BASELINE_FILENAME,
    BASELINE_INPUT,
    CATEGORIES_FILENAME,
    CATEGORIES_INPUT,
    convert_econevals,
    econevals_item,
    read_econevals,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse
from verifyit.modes.grade_judge import PAIRWISE_METRIC

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

INTERMEDIATE_REVISION = "bcd63e15ede5085f94fb14f5152f7daf1dc956a5"
INTERMEDIATE_URL = f"https://huggingface.co/datasets/EconEvals/EconEvalsIntermediate/resolve/{INTERMEDIATE_REVISION}"
VERSION = "2026.11.13"

PAPER_BASELINE = "openai:o3-mini-2025-01-31"
O3_BASELINE = "openai:o3-2025-04-16"
# Benchmark-name suffix for each baseline; the paper's baseline keeps the bare name.
BASELINE_SUFFIXES = {PAPER_BASELINE: "", O3_BASELINE: "-o3"}
# Six questions per category give about 250 per benchmark: roughly 0.02 standard error near parity.
SAMPLED_PER_CATEGORY = 6


@dataclass(frozen=True)
class EconEvalsRun:
    benchmark: str
    run: str
    rows: int
    categories: int
    questions_sha256: str
    categories_sha256: str
    baseline_sha256: dict[str, str]


RUNS = (
    EconEvalsRun(
        benchmark="EconEvals-Synthetic-DWA",
        run="synth_sampled_dwas",
        rows=2351,
        categories=40,
        questions_sha256="d4de7271642857a71d6db597f6139d9d6bfdeaebc6721d70c36459ae25a3c137",
        categories_sha256="4e33e6fa4458e5a41a08588fff4678e610db20c6239f2b11ef55667558e1510d",
        baseline_sha256={
            PAPER_BASELINE: "604c4c4338fedaa776f1b772e3a5210f989915553eb977b3a76bca484976e875",
            O3_BASELINE: "dbb5186e744e29b97c1884d659459167bcf31a3a62e2f5ec056cc3f42c473ad5",
        },
    ),
    EconEvalsRun(
        benchmark="EconEvals-Synthetic-Occupation",
        run="synth_sampled_occs_gdpval",
        rows=2426,
        categories=43,
        questions_sha256="5f94198cd50b519b86aedf23369c7a36e895169130a86f87c395e17dd27a645f",
        categories_sha256="c5819bfeb0cf75bdbfdb761a67a95f34d2b6eb380aaaee70a1a9b78ea6c088ca",
        baseline_sha256={
            PAPER_BASELINE: "7aecd675fd594b94706bc68f3de005bd5058d5be6ba17e6e8b41773b4ee829d4",
            O3_BASELINE: "ba3b3871f8f4ce203d9eb75c86eac35bf7fb4a4c0de99227ab051fa49b4abbf3",
        },
    ),
)


def econevals_source(
    run: EconEvalsRun, baseline_model: str, *, per_category: int | None
) -> RlDataSource[EvaluationCorpusRecipe]:
    root = f"{INTERMEDIATE_URL}/.runs/{run.run}/predictions-arena-hard"
    suffix = ("" if per_category is None else "-Sampled") + BASELINE_SUFFIXES[baseline_model]
    benchmark = run.benchmark + suffix
    rows = run.rows if per_category is None else per_category * run.categories
    return RlDataSource(
        info=SourceInfo(
            id="EconEvals:" + run.run + suffix,
            title=benchmark,
            origin="EconEvals",
            family="judged-preference",
            tags=("eval", "source:econevals", "single-turn", "benchmark", "llm-judge"),
            count=rows,
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name="econevals-" + run.run.replace("_", "-") + suffix,
                source=UrlSource(
                    f"{root}/question.jsonl",
                    run.questions_sha256,
                    "question.jsonl",
                    SourceFormat.JSONL,
                    read=partial(read_econevals, baseline_model=baseline_model, per_category=per_category),
                ),
                inputs={
                    BASELINE_INPUT: UrlSource(
                        f"{root}/model_answer/{baseline_model}.jsonl",
                        run.baseline_sha256[baseline_model],
                        BASELINE_FILENAME,
                        SourceFormat.JSONL,
                    ),
                    CATEGORIES_INPUT: UrlSource(
                        f"{root}/category_data.json", run.categories_sha256, CATEGORIES_FILENAME, SourceFormat.JSON
                    ),
                },
                convert=partial(convert_econevals, benchmark=benchmark),
                version=VERSION,
                intended_use=IntendedUse.EVAL,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                requires_judge=True,
                version=VERSION,
                operation=OperationKind.CHAT_GENERATION,
                # EconEvals sampled candidate models greedily with large output budgets.
                sampling=Sampling(
                    temperature=0.0,
                    top_p=None,
                    top_k=None,
                    max_tokens=None,
                    stop=(),
                    chat_template_kwargs={},
                ),
                trials=TrialPolicy(count=1, seed=None),
                metrics=(Metric(name=PAIRWISE_METRIC, kind=MetricKind.CONTINUOUS, higher_is_better=True),),
                primary_metric=PAIRWISE_METRIC,
                summary=SummaryKind.WEIGHTED_MEAN,
                expected_population=rows,
            ),
            econevals_item,
        ),
        pipeline=prepare_evaluation,
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        econevals_source(run, baseline, per_category=per_category)
        for per_category in (None, SAMPLED_PER_CATEGORY)
        for baseline in BASELINE_SUFFIXES
        for run in RUNS
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources())
