# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SuperGPQA zero-shot chat generation with the authors' prompt and answer extraction.

The population is all 26,529 graduate-level questions in ``m-a-p/SuperGPQA`` (Du et al., 2025), with
four to ten options each, across 13 disciplines. The prompt, greedy decoding and letter extraction
follow the authors' zero-shot runner (SuperGPQA/SuperGPQA at e31e8bf3e7089aef808be5ddecafd4538dee6e41).
Output length is the model budget; the runner caps it at 4,096 tokens and scores truncated answers as
misses. Rows are selected in a fixed hashed order, so ``--limit N`` evaluates a uniform random sample.
``supergpqa-2500`` is a fixed uniform sample: the 2,500 questions whose ``uuid`` has the smallest SHA-256.
SuperGPQA is ODC-By.
"""

import hashlib
from functools import partial

from taskcompendium.evaluation.spec import (
    CohortOrder,
    CohortPolicy,
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.supergpqa import convert_supergpqa, supergpqa_item
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.pipeline import HfSource, RowSelector
from experiments.post_training.task_curation.source import RlDataSource

SUPERGPQA_REVISION = "4430d4458112c7d4497fdcf94d7cc223313d6acf"
SUPERGPQA_FILE = "SuperGPQA-all.jsonl"
SUPERGPQA_POPULATION = 26529
VERSION = "2026.11.19"
SAMPLE_POPULATION = 2500
SAMPLE_MAX_UUID_SHA256 = "1873b9a34a36540a6b3e05faafb9a426c32923e157db3023a7ca2f5bb476e853"
"""The 2,500th smallest ``uuid`` SHA-256 at ``SUPERGPQA_REVISION``."""


def in_sample(data: dict, context: ConversionContext, *, max_uuid_sha256: str) -> bool:
    return hashlib.sha256(data["uuid"].encode()).hexdigest() <= max_uuid_sha256


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(
        [
            _source("supergpqa", SUPERGPQA_POPULATION, select=None),
            # Bound by keyword so the corpus identity records the threshold.
            _source(
                "supergpqa-2500", SAMPLE_POPULATION, select=partial(in_sample, max_uuid_sha256=SAMPLE_MAX_UUID_SHA256)
            ),
        ],
        "chat",
    )


def _source(name: str, population: int, select: RowSelector | None) -> RlDataSource[EvaluationCorpusRecipe]:
    protocol = EvaluationProtocol(
        benchmark=name,
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            temperature=0.0,
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
        cohort=CohortPolicy(order=CohortOrder.SHUFFLED),
    )
    return evaluation_source(
        name=name,
        source=HfSource("m-a-p/SuperGPQA", SUPERGPQA_REVISION, (SUPERGPQA_FILE,), SourceFormat.JSONL, select=select),
        convert=convert_supergpqa,
        version=VERSION,
        protocol=protocol,
        item_builder=supergpqa_item,
        grader=BASIC_GRADER,
    )
