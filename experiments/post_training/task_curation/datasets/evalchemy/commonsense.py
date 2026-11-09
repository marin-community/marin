# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned CommonsenseQA, COPA and WSC checkpoint protocols."""

from taskcompendium.evaluation.spec import LikelihoodMetric, LikelihoodNormalization
from taskcompendium.importers.evalchemy.commonsense import commonsense_qa_template, copa_template, wsc_template
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.post_training.task_curation.datasets.evalchemy.protocols import fewshot_choice_source
from experiments.post_training.task_curation.evaluation import EvalDataset, EvaluationBenchmark, fixed_benchmarks
from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.source import RlDataSource


def sources() -> list[RlDataSource[EvalDataset]]:
    raw_accuracy = (LikelihoodMetric(name="acc,none", normalization=LikelihoodNormalization.RAW),)
    return [
        fewshot_choice_source(
            "commonsense_qa",
            "commonsense_qa",
            HfSource(
                "tau/commonsense_qa",
                "94630fe30dad47192a8546eb75f094926d47e155",
                ("data/validation-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            1221,
            commonsense_qa_template(10, pool_file="data/train-00000-of-00001.parquet"),
            raw_accuracy,
        ),
        fewshot_choice_source(
            "copa",
            "copa",
            HfSource(
                "aps/super_glue",
                "3de24cf8022e94f4ee4b9d55a6f539891524d646",
                ("copa/validation-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            100,
            copa_template(),
            raw_accuracy,
        ),
        fewshot_choice_source(
            "wsc273",
            "wsc273",
            HfSource(
                "lighteval/winograd_wsc",
                "5df8523abfe0c256d133b9887328e3e5697cb67a",
                ("wsc273/test-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            273,
            wsc_template(),
            raw_accuracy,
        ),
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources(), "likelihood")
