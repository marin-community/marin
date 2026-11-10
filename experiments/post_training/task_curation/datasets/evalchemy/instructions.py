# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical full-population harness IFEval and native IFBench instruction evaluations."""

from functools import partial

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metric,
    MetricKind,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.evalchemy.instructions import convert_instructions, instruction_item
from taskcompendium.importers.evalchemy.scorers.instruction_spec import InstructionProfile
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.evalchemy.environments import (
    IFBENCH_GRADER,
    IFEVAL_GRADER,
)
from experiments.post_training.task_curation.evaluation import EvalDataset, curate
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


def instruction_source(
    name: str,
    benchmark: str,
    source: HfSource,
    population: int,
    profile: InstructionProfile,
    max_tokens: int,
) -> RlDataSource[EvalDataset]:
    names = IFEVAL_METRICS if profile is InstructionProfile.HARNESS else IFBENCH_METRICS
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name,
            title=benchmark,
            origin="Evalchemy",
            family="instruction-following",
            tags=("eval", "source:evalchemy", "benchmark", "single-turn", "instruction-following"),
            count=population,
        ),
        config=EvalDataset(
            CurationRecipe(
                name="evalchemy-" + name,
                source=source,
                convert=partial(convert_instructions, profile=profile),
                version=scorer_revision("ifeval" if profile is InstructionProfile.HARNESS else "ifbench"),
                intended_use=IntendedUse.EVAL,
                grader=IFEVAL_GRADER if profile is InstructionProfile.HARNESS else IFBENCH_GRADER,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                version="2026.11.02",
                operation=(
                    OperationKind.TEXT_GENERATION
                    if profile is InstructionProfile.HARNESS
                    else OperationKind.CHAT_GENERATION
                ),
                sampling=Sampling(
                    temperature=0,
                    top_p=None,
                    top_k=None,
                    max_tokens=max_tokens,
                    stop=(),
                    chat_template_kwargs={} if profile is InstructionProfile.HARNESS else {"enable_thinking": False},
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
                summary=SummaryKind.WEIGHTED_MEAN,
                expected_population=population,
            ),
            partial(instruction_item, profile=profile),
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataset]]:
    return [
        instruction_source(
            "ifeval",
            "ifeval",
            HfSource("google/IFEval", IFEVAL_REVISION, ("ifeval_input_data.jsonl",), SourceFormat.JSONL),
            541,
            InstructionProfile.HARNESS,
            1280,
        ),
        instruction_source(
            "ifbench",
            "IFBench",
            HfSource(
                "allenai/IFBench_test", IFBENCH_REVISION, ("data/train-00000-of-00001.parquet",), SourceFormat.PARQUET
            ),
            300,
            InstructionProfile.IFBENCH,
            1024,
        ),
    ]
