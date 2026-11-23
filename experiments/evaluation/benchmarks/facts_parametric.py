# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Parametric on the 1,052 public questions, judged once by the run's configured judge.

FACTS Parametric (Google DeepMind; section 4 of arXiv:2512.10791) asks closed-book factoid questions
whose answers are on Wikipedia. The rows are Kaggle dataset ``kaggle/facts-parametric-public-examples``
version 2 (Apache 2.0); the 1,052-row private split is held by Kaggle. The judge prompt is the starter
notebook's (kaggle.com/code/yulongt/facts-parametric-benchmark-starter-code, version 10). The paper
averages three Gemini 2.5 Pro grades and the leaderboard pools both splits, so scores are not
comparable to the Kaggle leaderboard.
"""

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metric,
    MetricKind,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.facts_parametric import (
    convert_facts_parametric,
    facts_parametric_item,
    read_facts_parametric,
)
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.pipeline import UrlSource

FACTS_PARAMETRIC_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/kaggle/facts-parametric-public-examples/"
    "FACTS-Parametric-public.csv?datasetVersionNumber=2"
)
FACTS_PARAMETRIC_SHA256 = "23fdc39d681656c87c6790b158f3d3a54f335903691a3285599c67a1578439f8"
FACTS_PARAMETRIC_POPULATION = 1052
VERSION = "2026.11.23"


def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="facts-parametric",
        requires_judge=True,
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        # The notebook sets no sampling; the cap leaves room for reasoning before a short answer.
        sampling=Sampling(
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=16384,
            stop=(),
            chat_template_kwargs={"enable_thinking": True},
        ),
        trials=TrialPolicy(count=1, seed=None),
        # UNKNOWN verdicts also count as incorrect, so the summary's attempted accuracy matches the paper's.
        metrics=(*Metrics.CLASSIFICATION, Metric(name="unknown", kind=MetricKind.BINARY, higher_is_better=False)),
        primary_metric="accuracy",
        summary=SummaryKind.CLASSIFICATION,
        expected_population=FACTS_PARAMETRIC_POPULATION,
    )
    source = evaluation_source(
        name="facts-parametric",
        source=UrlSource(
            FACTS_PARAMETRIC_URL,
            FACTS_PARAMETRIC_SHA256,
            "FACTS-Parametric-public.csv",
            SourceFormat.CSV,
            read=read_facts_parametric,
        ),
        convert=convert_facts_parametric,
        version=VERSION,
        protocol=protocol,
        item_builder=facts_parametric_item,
        grader=None,
    )
    return fixed_benchmarks([source], "chat")
