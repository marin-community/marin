# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned evaluation datasets; simple benchmarks bind a prompt, grader and protocol."""

from functools import partial

from taskcompendium.convert.evaluation import boxed_math_task
from taskcompendium.evaluation import (
    EvaluationProtocol,
    Metric,
    MetricKind,
    Sampling,
    TrialPolicy,
)
from taskcompendium.pipeline.evaluation import chat_item
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse
from verifyit.adapters.evalchemy_math import CHAT_REQUEST_STOPS

from experiments.post_training.task_curation.datasets.evalchemy.pipeline import EvalDataSet, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

MATH_PROMPT = "Problem: {problem}\nMark your solution with \\boxed\nAnswer:"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MARIN_SOURCE_REVISION = "623134de90723e7f189036da005067f36817b6e7"
AIME24_SHA256 = "2dfe4f134aee7989cd9978218ca914b3c3db4685ea06d841560ec1391ba0a409"
ACCURACY = Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True)


def math_dataset(
    name: str,
    benchmark: str,
    source: HfSource | UrlSource,
    answer_field: str,
    rows: int,
    trials: int,
    temperature: float,
) -> RlDataSource[EvalDataSet]:
    recipe = CurationRecipe(
        name="evalchemy-" + name,
        source=source,
        convert=partial(boxed_math_task, prompt_template=MATH_PROMPT, answer_field=answer_field, benchmark=benchmark),
        version="1",
        intended_use=IntendedUse.EVAL,
    )
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name,
            title=benchmark,
            origin="Evalchemy",
            family="math-answer",
            tags=("eval", "source:evalchemy", "single-turn", "benchmark", "math"),
            count=rows,
        ),
        config=EvalDataSet(
            recipe,
            EvaluationProtocol(
                benchmark=benchmark,
                version="source-sampling-v2",
                operation="chat_generation",
                sampling=Sampling(
                    temperature=temperature,
                    top_p=None,
                    top_k=None,
                    max_tokens=None,
                    stop=CHAT_REQUEST_STOPS,
                    chat_template_kwargs={"enable_thinking": True},
                ),
                trials=TrialPolicy(count=trials, seed=0 if temperature else None),
                metrics=(ACCURACY,),
                primary_metric="accuracy",
                reducer="mean_over_examples_and_trials_v1",
                expected_population=rows,
            ),
            partial(chat_item, response_selection="final_or_complete_boxed_reasoning"),
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataSet]]:
    return [
        math_dataset(
            "math500",
            "MATH500",
            HfSource("HuggingFaceH4/MATH-500", MATH500_REVISION, ("test.jsonl",), SourceFormat.JSONL),
            "answer",
            500,
            1,
            0,
        ),
        math_dataset(
            "aime24",
            "AIME24",
            UrlSource(
                f"https://raw.githubusercontent.com/marin-community/marin/{MARIN_SOURCE_REVISION}/"
                "vendor/evalchemy/eval/chat_benchmarks/AIME24/data/aime24.json",
                AIME24_SHA256,
                "aime24.jsonl",
                SourceFormat.JSONL,
            ),
            "expected_answer",
            30,
            10,
            0.7,
        ),
    ]
