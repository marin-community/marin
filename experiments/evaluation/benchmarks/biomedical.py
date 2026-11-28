# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned PubMedQA and MedMCQA protocols.

PubMedQA (Jin et al., 2019) is MIT-licensed. The source is the Hub's Parquet
export of `bigbio/pubmed_qa`, configuration `pubmed_qa_labeled_fold0_source`,
test split: the 500 expert-labeled questions that lm-eval's `pubmedqa` task
scores zero-shot.

MedMCQA (Pal et al., 2022) is Apache-2.0 on the Hub and MIT on GitHub. Both
protocols score the 4,183-question validation split, as lm-eval's `medmcqa`
tasks do; the test split's answers are hidden. `medmcqa` is the zero-shot
letter likelihood task. `medmcqa-cot` is lm-eval's `medmcqa_cot` chat prompt
graded on the final answer, with each served model's sampling defaults and
token budget instead of the reference's greedy 512-token decoding, which would
truncate reasoning models. Its answer letter must stand alone, so option text
such as `Final Answer: Bone marrow` does not count as B.
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
from taskcompendium.importers.evalchemy.biomedical import (
    convert_medmcqa_cot,
    medmcqa_cot_item,
    medmcqa_template,
    pubmedqa_template,
)
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.evaluation.prompt_compilation import evaluation_source, multiple_choice_evaluation
from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.source import RlDataSource

# Commit of the `refs/convert/parquet` revision of bigbio/pubmed_qa.
PUBMEDQA_PARQUET_REVISION = "13a7d15476092370cbabb6475390e7e69b74d2f2"
MEDMCQA_REVISION = "91c6572c454088bf71b679ad90aa8dffcd0d5868"
MEDMCQA_SOURCE = HfSource(
    "openlifescienceai/medmcqa",
    MEDMCQA_REVISION,
    ("data/validation-00000-of-00001.parquet",),
    SourceFormat.PARQUET,
)
MEDMCQA_COT_VERSION = "2026.11.24"


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    raw = LikelihoodMetric(name="acc,none", normalization=LikelihoodNormalization.RAW)
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
            template=pubmedqa_template(),
            metrics=(raw,),
        ),
        multiple_choice_evaluation(
            name="medmcqa",
            benchmark="medmcqa",
            source=MEDMCQA_SOURCE,
            template=medmcqa_template(),
            metrics=(raw,),
        ),
    ]


def chat_sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    protocol = EvaluationProtocol(
        benchmark="medmcqa-cot",
        version=MEDMCQA_COT_VERSION,
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=(),
            chat_template_kwargs={"enable_thinking": True},
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metrics.ACCURACY,),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
    )
    return [
        evaluation_source(
            name="medmcqa-cot",
            source=MEDMCQA_SOURCE,
            convert=convert_medmcqa_cot,
            version=MEDMCQA_COT_VERSION,
            protocol=protocol,
            item_builder=medmcqa_cot_item,
            grader=BASIC_GRADER,
        )
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return [*fixed_benchmarks(sources(), "likelihood"), *fixed_benchmarks(chat_sources(), "chat")]
