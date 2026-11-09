# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GSM8K source data and standard/chain-of-thought evaluation configurations."""

from taskcompendium.evaluation.spec import Sampling
from taskcompendium.importers.evalchemy.math import gsm8k_cot_template
from taskcompendium.importers.evalchemy.scorers.harness_generation import GSM8K_REQUEST_STOPS
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import MATH_GRADER
from experiments.evaluation.benchmarks.question_answering import QuestionAnsweringDataset, question_answering_benchmark
from experiments.evaluation.prompt_compilation import text_generation_evaluation
from experiments.post_training.task_curation.pipeline import HfSource

GSM8K_SOURCE = HfSource(
    "openai/gsm8k",
    "740312add88f781978c0658806c59bc2815b9866",
    ("main/test-00000-of-00001.parquet",),
    SourceFormat.PARQUET,
)


def benchmarks() -> list[EvaluationBenchmark]:
    return [
        question_answering_benchmark(
            dataset=QuestionAnsweringDataset(
                name="gsm8k", source=GSM8K_SOURCE, population=1319, pool_file="main/train-00000-of-00001.parquet"
            ),
            shots=(5, 0),
            grader=MATH_GRADER,
            metrics=("exact_match,strict-match", "exact_match,flexible-extract"),
            primary_metric="exact_match,strict-match",
            stop=GSM8K_REQUEST_STOPS,
            chat_options={0: {"enable_thinking": False}, 5: {}},
        ),
        *fixed_benchmarks(
            [
                text_generation_evaluation(
                    name="gsm8k_cot",
                    benchmark="gsm8k_cot",
                    source=GSM8K_SOURCE,
                    population=1319,
                    template=gsm8k_cot_template(),
                    metrics=("exact_match,strict-match", "exact_match,flexible-extract"),
                    sampling=Sampling(
                        temperature=0,
                        top_p=None,
                        top_k=None,
                        max_tokens=None,
                        stop=("Q:", "</s>", "<|im_end|>"),
                        chat_template_kwargs={},
                    ),
                    scorer="gsm8k_cot",
                    grader=MATH_GRADER,
                ),
            ],
            "cot",
            {"gsm8k_cot": "gsm8k"},
        ),
    ]
