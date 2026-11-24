# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MMLU-Redux zero-shot chat generation with ZeroEval's prompt, rows and answer extraction.

The population is ZeroEval's frozen ``mmlu-redux`` test split: 2,778 four-choice questions from 30
subjects of MMLU-Redux 1.0 (``edinburgh-dawg/mmlu-redux``, Gema et al., 2024), with questions the
annotators marked as erroneous removed. The Qwen2.5-Instruct MMLU-Redux scores in the Qwen3 technical
report track ZeroEval's published greedy scores for those models within 0.3 points at 3B to 32B.
MMLU-Redux is CC-BY-4.0; ZeroEval is Apache-2.0.

Decoding is greedy, as in ZeroEval, so every model is scored under the same deterministic setting.
The Qwen3 report instead samples at temperature 0.7, top-p 0.8, top-k 20 and presence penalty 1.5.
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
from taskcompendium.importers.evalchemy.mmlu_redux import convert_mmlu_redux, mmlu_redux_item
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.pipeline import HfSource

ZEROEVAL_DATA_REVISION = "a100ff81ecec0ed983dd3f0038d6d6bb2ab3dc1c"
MMLU_REDUX_FILE = "mmlu-redux/test-00000-of-00001.parquet"
MMLU_REDUX_POPULATION = 2778
VERSION = "2026.11.24"


def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="mmlu-redux",
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            temperature=0.0,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=(),
            chat_template_kwargs={"enable_thinking": False},
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metrics.ACCURACY,),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
        expected_population=MMLU_REDUX_POPULATION,
    )
    source = evaluation_source(
        name="mmlu-redux",
        source=HfSource("yuchenlin/zero-eval", ZEROEVAL_DATA_REVISION, (MMLU_REDUX_FILE,), SourceFormat.PARQUET),
        convert=convert_mmlu_redux,
        version=VERSION + "-" + scorer_revision("zeroeval_mcqa"),
        protocol=protocol,
        item_builder=mmlu_redux_item,
        grader=BASIC_GRADER,
    )
    return fixed_benchmarks([source], "chat")
