# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned CommonsenseQA, COPA and WSC checkpoint protocols."""

from taskcompendium.evaluation.spec import LikelihoodMetric, LikelihoodNormalization
from taskcompendium.importers.evalchemy.commonsense import commonsense_qa_template, copa_template, wsc_template
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.evaluation.prompt_compilation import multiple_choice_evaluation
from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.source import RlDataSource


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    raw_accuracy = (LikelihoodMetric(name="acc,none", normalization=LikelihoodNormalization.RAW),)
    return [
        multiple_choice_evaluation(
            name="commonsense_qa",
            benchmark="commonsense_qa",
            source=HfSource(
                "tau/commonsense_qa",
                "94630fe30dad47192a8546eb75f094926d47e155",
                ("data/validation-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            population=1221,
            template=commonsense_qa_template(shots=10, pool_file="data/train-00000-of-00001.parquet"),
            metrics=raw_accuracy,
        ),
        multiple_choice_evaluation(
            name="copa",
            benchmark="copa",
            source=HfSource(
                "aps/super_glue",
                "3de24cf8022e94f4ee4b9d55a6f539891524d646",
                ("copa/validation-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            population=100,
            template=copa_template(),
            metrics=raw_accuracy,
        ),
        multiple_choice_evaluation(
            name="wsc273",
            benchmark="wsc273",
            source=HfSource(
                "lighteval/winograd_wsc",
                "5df8523abfe0c256d133b9887328e3e5697cb67a",
                ("wsc273/test-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            population=273,
            template=wsc_template(),
            metrics=raw_accuracy,
        ),
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources(), "likelihood")
