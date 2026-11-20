# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GenomeQA's complete pinned population and one-response evaluation protocol."""

from taskcompendium.evaluation.spec import (
    ContextPolicy,
    EvaluationProtocol,
    Metric,
    MetricKind,
    Metrics,
    OperationKind,
    OutputFit,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.evalchemy.scorers.genomeqa import MACRO_METRIC
from taskcompendium.importers.genomeqa import (
    DATASET,
    FORMATS,
    POPULATION,
    PROMPT_FILENAME,
    PROMPT_URL,
    REVISION,
    convert_genomeqa,
    genomeqa_item,
    read_genomeqa,
)
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.pipeline import HfSource, UrlSource

VERSION = "2026.11.20"


def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="genomeqa",
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            # Use the served model's decoding settings for routine evaluation.
            temperature=None,
            top_p=None,
            top_k=None,
            stop=(),
            max_tokens=None,
            chat_template_kwargs={},
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(
            Metric(name=MACRO_METRIC, kind=MetricKind.CONTINUOUS, higher_is_better=True),
            Metrics.ACCURACY,
            Metric(name="unanswered", kind=MetricKind.BINARY, higher_is_better=False),
        ),
        primary_metric=MACRO_METRIC,
        summary=SummaryKind.WEIGHTED_MEAN,
        expected_population=POPULATION,
        context_policy=ContextPolicy(
            output_fit=OutputFit.REQUIRE,
            context_reserve=0,
        ),
    )
    source = evaluation_source(
        name="genomeqa",
        source=HfSource(
            DATASET,
            REVISION,
            tuple(f"{answer_format}/train-00000-of-00001.parquet" for answer_format in FORMATS),
            SourceFormat.PARQUET,
            read=read_genomeqa,
        ),
        convert=convert_genomeqa,
        version=VERSION + "-" + scorer_revision("genomeqa"),
        protocol=protocol,
        item_builder=genomeqa_item,
        grader=BASIC_GRADER,
        inputs={
            "prompt": UrlSource(
                PROMPT_URL,
                "bc932144c20e02ccb18e4d8407f90814ca4f8fdc389d9309cd0cbc1525db58a3",
                PROMPT_FILENAME,
                SourceFormat.GENERATED,
            )
        },
    )
    return fixed_benchmarks([source], "chat")
