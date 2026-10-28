# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical full-population harness IFEval and native IFBench instruction evaluations."""

from functools import partial

from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation import (
    EvaluationItem,
    EvaluationProtocol,
    Metric,
    MetricKind,
    Sampling,
    TextGeneration,
    TrialPolicy,
)
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.evaluation import chat_item
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import IntendedUse, RawRow
from verifyit.adapters.evalchemy_math import CHAT_REQUEST_STOPS
from verifyit.spec import Constraint, EmptyOutputPolicy, IfevalProfile, IfevalSpec

from experiments.post_training.task_curation.datasets.evalchemy.instruction_environments import (
    IFBENCH_GRADER,
    IFEVAL_GRADER,
)
from experiments.post_training.task_curation.datasets.evalchemy.pipeline import EvalDataSet, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
IFBENCH_REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"
IFEVAL_METRICS = (
    "prompt_level_strict_acc,none",
    "inst_level_strict_acc,none",
    "prompt_level_loose_acc,none",
    "inst_level_loose_acc,none",
)
IFBENCH_METRICS = (
    "strict_prompt_accuracy",
    "strict_instruction_accuracy",
    "loose_prompt_accuracy",
    "loose_instruction_accuracy",
)


def convert_instructions(row: RawRow, context: ConversionContext, *, profile: IfevalProfile) -> TaskSpec:
    # Parquet pads kwargs with nulls for other checkers; both source protocols discard them.
    constraints = tuple(
        Constraint(name=name, params={key: value for key, value in params.items() if value is not None})
        for name, params in zip(row.data["instruction_id_list"], row.data["kwargs"], strict=True)
    )
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=row.data["prompt"]),),
        package=verifyit_package(
            IfevalSpec(
                constraints=constraints,
                profile=profile,
                prompt=row.data["prompt"],
                empty_output=EmptyOutputPolicy.GRADE,
            ),
            environment=required_grader_environment(context),
        ),
        evidence=dict(row.data),
    )
    benchmark = "ifeval" if profile is IfevalProfile.HARNESS else "IFBench"
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:" + benchmark)})


def instruction_item(task: TaskSpec, *, profile: IfevalProfile) -> EvaluationItem:
    item = chat_item(task)
    if profile is IfevalProfile.HARNESS:
        (message,) = task.context.events
        return item.model_copy(update={"operation": TextGeneration(context=message.content)})
    return item


def instruction_source(
    name: str,
    benchmark: str,
    source: HfSource,
    population: int,
    profile: IfevalProfile,
    max_tokens: int,
) -> RlDataSource[EvalDataSet]:
    names = IFEVAL_METRICS if profile is IfevalProfile.HARNESS else IFBENCH_METRICS
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name,
            title=benchmark,
            origin="Evalchemy",
            family="instruction-following",
            tags=("eval", "source:evalchemy", "benchmark", "single-turn", "instruction-following"),
            count=population,
        ),
        config=EvalDataSet(
            CurationRecipe(
                name="evalchemy-" + name,
                source=source,
                convert=partial(convert_instructions, profile=profile),
                version="2",
                intended_use=IntendedUse.EVAL,
                grader=IFEVAL_GRADER if profile is IfevalProfile.HARNESS else IFBENCH_GRADER,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                version="maintained-instructions-v3",
                operation="text_generation" if profile is IfevalProfile.HARNESS else "chat_generation",
                sampling=Sampling(
                    temperature=0,
                    top_p=1,
                    top_k=None,
                    max_tokens=max_tokens,
                    stop=() if profile is IfevalProfile.HARNESS else CHAT_REQUEST_STOPS,
                    chat_template_kwargs={} if profile is IfevalProfile.HARNESS else {"enable_thinking": False},
                ),
                trials=TrialPolicy(count=1, seed=None),
                metrics=tuple(
                    Metric(
                        name=name,
                        kind=MetricKind.BINARY if "prompt" in name else MetricKind.CONTINUOUS,
                        higher_is_better=True,
                    )
                    for name in names
                ),
                primary_metric=names[0],
                reducer="instruction_weighted_v1",
                expected_population=population,
            ),
            partial(instruction_item, profile=profile),
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataSet]]:
    return [
        instruction_source(
            "ifeval",
            "ifeval",
            HfSource("google/IFEval", IFEVAL_REVISION, ("ifeval_input_data.jsonl",), SourceFormat.JSONL),
            541,
            IfevalProfile.HARNESS,
            1280,
        ),
        instruction_source(
            "ifbench",
            "IFBench",
            HfSource(
                "allenai/IFBench_test", IFBENCH_REVISION, ("data/train-00000-of-00001.parquet",), SourceFormat.PARQUET
            ),
            300,
            IfevalProfile.IFBENCH,
            1024,
        ),
    ]
