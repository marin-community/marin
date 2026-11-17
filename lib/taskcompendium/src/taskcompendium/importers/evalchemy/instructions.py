# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert pinned instruction-following rows and preserve their request channels."""

from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem, TextGeneration
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.importers.evalchemy.scorers.instruction_spec import InstructionProfile
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow


def convert_instructions(row: RawRow, context: ConversionContext, *, profile: InstructionProfile) -> TaskSpec:
    # Parquet pads kwargs with nulls for other checkers; both source protocols discard them.
    constraints = [
        {"name": name, "params": {key: value for key, value in params.items() if value is not None}}
        for name, params in zip(row.data["instruction_id_list"], row.data["kwargs"], strict=True)
    ]
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=row.data["prompt"]),),
        package=source_package(
            "ifeval" if profile is InstructionProfile.HARNESS else "ifbench",
            {"constraints": constraints, "profile": profile, "prompt": row.data["prompt"]},
            environment=required_grader_environment(context),
        ),
        evidence=dict(row.data),
    )
    benchmark = "ifeval" if profile is InstructionProfile.HARNESS else "IFBench"
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:" + benchmark)})


def instruction_item(task: TaskSpec, *, profile: InstructionProfile) -> EvaluationItem:
    item = chat_item(task)
    if profile is InstructionProfile.HARNESS:
        return item.model_copy(update={"operation": TextGeneration()})
    return item
