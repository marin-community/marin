# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SuperGPQA zero-shot chat generation with the authors' prompt and answer extraction.

The population is all 26,529 graduate-level questions in ``m-a-p/SuperGPQA`` (Du et al., 2025), with
four to ten options each, across 13 disciplines. The prompt, greedy decoding and letter extraction
follow the authors' zero-shot runner (SuperGPQA/SuperGPQA at e31e8bf3e7089aef808be5ddecafd4538dee6e41).
Output length is the model budget; the runner caps it at 4,096 tokens and scores truncated answers as
misses. Rows are selected in a fixed hashed order, so ``--limit N`` (and
``supergpqa-sample``) evaluates a uniform random sample. SuperGPQA is ODC-By.
"""

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
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.pipeline import HfSource

SUPERGPQA_REVISION = "4430d4458112c7d4497fdcf94d7cc223313d6acf"
SUPERGPQA_FILE = "SuperGPQA-all.jsonl"
VERSION = "2026.11.19"


def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="supergpqa",
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
        cohort=CohortPolicy(order=CohortOrder.SHUFFLED),
    )
    source = evaluation_source(
        name="supergpqa",
        source=HfSource("m-a-p/SuperGPQA", SUPERGPQA_REVISION, (SUPERGPQA_FILE,), SourceFormat.JSONL),
        convert=convert_supergpqa,
        version=VERSION,
        protocol=protocol,
        item_builder=supergpqa_item,
        grader=BASIC_GRADER,
    )
    return fixed_benchmarks([source], "chat")
