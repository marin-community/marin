# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TruthfulQA binary choice, the authors' January 2025 multiple-choice version (Apache-2.0).

The population is all 790 questions in ``TruthfulQA.csv`` at sylinrl/TruthfulQA
d71c110897f5d31c5d7f309e7bc316c152f6f031, each with its Best Answer and Best Incorrect Answer as
options A and B. The ``likelihood`` protocol compares the letters' log-probabilities; the ``chat``
protocol grades the final answer's ``Answer: X`` line. The original MC2 protocol stays in
``multiple_choice.py`` as ``truthfulqa``.
"""

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    LikelihoodMetric,
    LikelihoodNormalization,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.truthfulqa import (
    convert_truthfulqa_binary,
    truthfulqa_binary_item,
    truthfulqa_binary_template,
)
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.evaluation.prompt_compilation import evaluation_source, multiple_choice_evaluation
from experiments.post_training.task_curation.pipeline import UrlSource

TRUTHFULQA_COMMIT = "d71c110897f5d31c5d7f309e7bc316c152f6f031"
TRUTHFULQA_SOURCE = UrlSource(
    f"https://raw.githubusercontent.com/sylinrl/TruthfulQA/{TRUTHFULQA_COMMIT}/TruthfulQA.csv",
    "b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c",
    "TruthfulQA.csv",
    SourceFormat.CSV,
)
TRUTHFULQA_BINARY_POPULATION = 790
VERSION = "2026.11.24.1"


def benchmarks() -> list[EvaluationBenchmark]:
    likelihood = multiple_choice_evaluation(
        name="truthfulqa_binary",
        benchmark="truthfulqa_binary",
        source=TRUTHFULQA_SOURCE,
        population=TRUTHFULQA_BINARY_POPULATION,
        template=truthfulqa_binary_template(),
        metrics=(LikelihoodMetric(name="acc,none", normalization=LikelihoodNormalization.RAW),),
    )
    chat = evaluation_source(
        name="truthfulqa-binary",
        source=TRUTHFULQA_SOURCE,
        convert=convert_truthfulqa_binary,
        version=VERSION,
        protocol=EvaluationProtocol(
            benchmark="truthfulqa-binary",
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
            expected_population=TRUTHFULQA_BINARY_POPULATION,
        ),
        item_builder=truthfulqa_binary_item,
        grader=BASIC_GRADER,
    )
    return [*fixed_benchmarks([likelihood], "likelihood"), *fixed_benchmarks([chat], "chat")]
