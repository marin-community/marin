# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LEXam four-choice multiple-choice questions from Swiss law exams, with the authors' prompt and extraction.

LEXam's ``mcq_4_choices`` test split (Fan et al., 2025, arXiv:2505.12864) is published as two benchmarks:
its 619 English questions and its 1,036 German questions. The dataset is CC-BY-4.0 and the reference
code (LEXam-Benchmark/LEXam) is Apache-2.0.

Sampling keeps each served model's defaults and token budget. The reference decodes non-reasoning
models greedily with 4,096 tokens and reasoning models at vendor settings with 8,192 tokens, grading
only the text after the reasoning trace; Marin's reasoning models need a larger budget.
"""

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.evalchemy.lexam import convert_lexam_mcq, is_english, is_german, lexam_mcq_item
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.pipeline import HfSource, RowSelector
from experiments.post_training.task_curation.source import RlDataSource

LEXAM_REVISION = "31e57ae395f92ed7284a4c28d278c55da892b898"
LEXAM_MCQ_4_FILE = "mcq_4_choices/test-00000-of-00001.parquet"
LEXAM_MCQ_4_EN_POPULATION = 619
LEXAM_MCQ_4_DE_POPULATION = 1036
VERSION = "2026.11.20"


def _lexam_mcq_4(name: str, select: RowSelector, population: int) -> RlDataSource[EvaluationCorpusRecipe]:
    protocol = EvaluationProtocol(
        benchmark=name,
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=(),
            chat_template_kwargs={},
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metrics.ACCURACY,),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
        expected_population=population,
    )
    return evaluation_source(
        name=name,
        source=HfSource(
            "LEXam-Benchmark/LEXam",
            LEXAM_REVISION,
            (LEXAM_MCQ_4_FILE,),
            SourceFormat.PARQUET,
            select=select,
        ),
        convert=convert_lexam_mcq,
        version=VERSION + "-" + scorer_revision("lexam_mcq"),
        protocol=protocol,
        item_builder=lexam_mcq_item,
        grader=BASIC_GRADER,
    )


def benchmarks() -> list[EvaluationBenchmark]:
    sources = [
        _lexam_mcq_4("lexam-mcq-4-en", is_english, LEXAM_MCQ_4_EN_POPULATION),
        _lexam_mcq_4("lexam-mcq-4-de", is_german, LEXAM_MCQ_4_DE_POPULATION),
    ]
    return fixed_benchmarks(sources, "chat")
