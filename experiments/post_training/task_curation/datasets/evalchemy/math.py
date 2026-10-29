# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Legacy Evalchemy math populations with their original extraction policies."""

import json
from dataclasses import replace
from functools import partial

from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation import EvaluationProtocol, Sampling, TrialPolicy
from taskcompendium.importers.evalchemy.grader import scorer_revision, source_package
from taskcompendium.importers.evalchemy.scorers.math import CHAT_REQUEST_STOPS
from taskcompendium.importers.evalchemy.scorers.olympiad import reference_answers
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.evaluation import chat_item
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import IntendedUse, RawRow
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation.datasets.evalchemy.datasets import (
    ACCURACY,
    MARIN_SOURCE_REVISION,
    MATH_PROMPT,
    math_dataset,
)
from experiments.post_training.task_curation.datasets.evalchemy.grader_environments import MATH_GRADER, MATH_JUDGE_GRADER
from experiments.post_training.task_curation.datasets.evalchemy.judge_prompts import EQUIVALENCE_JUDGE_TEMPLATE
from experiments.post_training.task_curation.datasets.evalchemy.pipeline import EvalDataSet, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

OLYMPIAD_SHA256 = "bcafaabc0128adfeef69c830d1682254428887d4e1b96b5b1b03fb5d3818f958"
AIME25_SHA256 = "9affac318c4dd399482a67226936e229f954628567c7d7b7ee8f1360c15d6f7c"
AMC23_SHA256 = "02ae54425741562778916b5fa8f6ff07749689cfcd7a53e65e1d88a89e5065f9"


def bundled_source(directory: str, filename: str, sha256: str) -> UrlSource:
    return UrlSource(
        f"https://raw.githubusercontent.com/marin-community/marin/{MARIN_SOURCE_REVISION}/"
        f"vendor/evalchemy/eval/chat_benchmarks/{directory}/data/{filename}",
        sha256,
        filename + ".jsonl",
        SourceFormat.JSONL,
    )


def convert_amc(row: RawRow, context: ConversionContext) -> TaskSpec:
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=MATH_PROMPT.format(problem=row.data["question"])),),
        package=source_package(
            "amc", {"expected": str(row.data["answer"])}, environment=required_grader_environment(context)
        ),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:AMC23")})


def convert_olympiad(row: RawRow, context: ConversionContext, *, judged: bool) -> TaskSpec:
    answers = reference_answers(row.data["answer"])
    judge = (
        JudgeSpec(
            references=(json.dumps(list(answers), ensure_ascii=False),),
            question=row.data["problem"],
            rubric="labels",
            exact_gate=False,
            prompt_template=EQUIVALENCE_JUDGE_TEMPLATE.replace("{reference_answers}", "{reference}").replace(
                "{candidate_answer}", "{candidate}"
            ),
            label_scores={"correct": 1.0, "incorrect": 0.0, "not_attempted": 0.0},
            label_metrics={
                label: {
                    "accuracy": float(label == "correct"),
                    "incorrect": float(label == "incorrect"),
                    "not_attempted": float(label == "not_attempted"),
                    "judge_failed": 0.0,
                }
                for label in ("correct", "incorrect", "not_attempted")
            },
            label_scan="whole",
            label_case="upper",
            max_completion_tokens=2048,
        )
        if judged
        else None
    )
    environment = required_grader_environment(context)
    if judged:
        environment = environment.model_copy(update={"capabilities": (*environment.capabilities, "network")})
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=MATH_PROMPT.format(problem=row.data["problem"])),),
        package=source_package(
            "olympiad",
            {"expected": answers, "judged": judged},
            environment=environment,
            judge=judge,
            timeout=60 + judge.transport_budget if judge is not None else 60,
        ),
        evidence=dict(row.data),
    )
    benchmark = "OlympiadBench" if judged else "OlympiadBenchDeterministic"
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", f"benchmark:{benchmark}")})


def olympiad_source(*, judged: bool) -> RlDataSource[EvalDataSet]:
    benchmark = "OlympiadBench" if judged else "OlympiadBenchDeterministic"
    count = 10 if judged else 1
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + benchmark.lower(),
            title=benchmark,
            origin="Evalchemy",
            family="math-answer",
            tags=("eval", "source:evalchemy", "single-turn", "benchmark", "math", "legacy-subset"),
            count=30,
        ),
        config=EvalDataSet(
            CurationRecipe(
                name="evalchemy-" + benchmark.lower(),
                source=bundled_source("OlympiadBench", "olympiadbench.jsonl", OLYMPIAD_SHA256),
                convert=partial(convert_olympiad, judged=judged),
                version=scorer_revision("olympiad"),
                intended_use=IntendedUse.EVAL,
                grader=MATH_JUDGE_GRADER if judged else MATH_GRADER,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                version="olympiad-observations-v2" if judged else "1",
                operation="chat_generation",
                sampling=Sampling(
                    temperature=0.7 if judged else 0,
                    top_p=1,
                    top_k=None,
                    max_tokens=32768,
                    stop=CHAT_REQUEST_STOPS,
                    chat_template_kwargs={"enable_thinking": True},
                ),
                trials=TrialPolicy(count=count, seed=0 if judged else None),
                metrics=(ACCURACY,),
                primary_metric="accuracy",
                reducer="mean_over_examples_and_trials_v1",
                expected_population=30,
            ),
            chat_item,
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataSet]]:
    # AIME25 and AMC23 source generation sets do_sample=False despite temperature=.7.
    aime = math_dataset(
        "aime25",
        "AIME25",
        bundled_source("AIME25", "aime25.json", AIME25_SHA256),
        "answer",
        30,
        10,
        0,
    )
    amc = math_dataset(
        "amc23",
        "AMC23",
        bundled_source("AMC23", "amc23.json", AMC23_SHA256),
        "answer",
        40,
        10,
        0,
    )
    assert aime.config is not None and amc.config is not None
    assert aime.config.protocol.sampling is not None and amc.config.protocol.sampling is not None
    aime = replace(
        aime,
        config=replace(
            aime.config,
            protocol=aime.config.protocol.model_copy(
                update={
                    "sampling": aime.config.protocol.sampling.model_copy(update={"max_tokens": 32768, "stop": ()}),
                }
            ),
        ),
    )
    amc = replace(
        amc,
        config=replace(
            amc.config,
            item_builder=chat_item,
            recipe=replace(amc.config.recipe, convert=convert_amc, version=scorer_revision("amc")),
            protocol=amc.config.protocol.model_copy(
                update={
                    "sampling": amc.config.protocol.sampling.model_copy(update={"max_tokens": 32768, "stop": ()}),
                }
            ),
        ),
    )
    return [aime, amc, olympiad_source(judged=False), olympiad_source(judged=True)]
