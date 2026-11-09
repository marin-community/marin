# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TriviaQA, Natural Questions and DROP evaluation configurations."""

from taskcompendium.importers.evalchemy.scorers.harness_generation import COMPLETION_REQUEST_STOPS
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark
from experiments.evaluation.benchmarks.environments import ALIAS_GRADER, BASIC_GRADER
from experiments.evaluation.prompt_compilation import QuestionAnsweringDataset, lm_eval_text_generation
from experiments.post_training.task_curation.pipeline import HfSource


def benchmarks() -> list[EvaluationBenchmark]:
    return [
        lm_eval_text_generation(
            dataset=QuestionAnsweringDataset(
                name="triviaqa",
                source=HfSource(
                    "mandarjoshi/trivia_qa",
                    "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
                    ("rc.nocontext/validation-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=17944,
                pool_file="rc.nocontext/train-00000-of-00001.parquet",
            ),
            shots=(5, 0),
            grader=ALIAS_GRADER,
            metrics=("exact_match,strict_answer", "exact_match,extract_answer"),
            primary_metric="exact_match,strict_answer",
            stop=COMPLETION_REQUEST_STOPS,
            chat_options={5: {"enable_thinking": False}},
        ),
        lm_eval_text_generation(
            dataset=QuestionAnsweringDataset(
                name="nq_open",
                source=HfSource(
                    "google-research-datasets/nq_open",
                    "5dd9790a83002ad084ddeb7c420dc716852c6f28",
                    ("nq_open/validation-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=3610,
                pool_file="nq_open/train-00000-of-00001.parquet",
            ),
            shots=(5, 0),
            grader=ALIAS_GRADER,
            metrics=("exact_match,strict_answer", "exact_match,extract_answer"),
            primary_metric="exact_match,strict_answer",
            stop=COMPLETION_REQUEST_STOPS,
            chat_options={5: {}},
        ),
        lm_eval_text_generation(
            dataset=QuestionAnsweringDataset(
                name="drop",
                source=HfSource(
                    "EleutherAI/drop",
                    "619386fc7ffe6dace3cc6d3483c69b164ebf6bf3",
                    ("drop_validation.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=9536,
                pool_file="drop_train.parquet",
            ),
            shots=(3, 0),
            grader=BASIC_GRADER,
            metrics=("em,extract_answer", "f1,extract_answer"),
            primary_metric="f1,extract_answer",
            stop=COMPLETION_REQUEST_STOPS,
            chat_options={3: {}},
        ),
    ]
