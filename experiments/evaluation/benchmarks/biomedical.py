# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned PubMedQA checkpoint protocol.

PubMedQA (Jin et al., 2019) is MIT-licensed. The source is the Hub's Parquet
export of `bigbio/pubmed_qa`, configuration `pubmed_qa_labeled_fold0_source`,
test split: the 500 expert-labeled questions that lm-eval's `pubmedqa` task
scores zero-shot.
"""

from taskcompendium.evaluation.spec import LikelihoodMetric, LikelihoodNormalization
from taskcompendium.importers.evalchemy.biomedical import pubmedqa_template
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.evaluation.prompt_compilation import multiple_choice_evaluation
from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.source import RlDataSource

# Commit of the `refs/convert/parquet` revision of bigbio/pubmed_qa.
PUBMEDQA_PARQUET_REVISION = "13a7d15476092370cbabb6475390e7e69b74d2f2"


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        multiple_choice_evaluation(
            name="pubmedqa",
            benchmark="pubmedqa",
            source=HfSource(
                "bigbio/pubmed_qa",
                PUBMEDQA_PARQUET_REVISION,
                ("pubmed_qa_labeled_fold0_source/test/0000.parquet",),
                SourceFormat.PARQUET,
            ),
            population=500,
            template=pubmedqa_template(),
            metrics=(LikelihoodMetric(name="acc,none", normalization=LikelihoodNormalization.RAW),),
        )
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources(), "likelihood")
