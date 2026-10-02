# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned source contracts for TaskTrove and direct dataset families."""

from dataclasses import dataclass

from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat, hub_inputs
from taskcompendium.pipeline.models import ReviewRubric
from taskcompendium.pipeline.sources import unpack_task_binary

TASKTROVE_DATASET = "open-thoughts/TaskTrove"


@dataclass(frozen=True)
class SourceDefinition:
    dataset: str
    revision: str
    config: str
    split: str
    rubric: ReviewRubric
    files: SourceFiles


def tasktrove_source(config: str, revision: str, rubric: ReviewRubric) -> SourceDefinition:
    """Describe one pinned TaskTrove component and its archived task records."""
    return SourceDefinition(
        dataset=TASKTROVE_DATASET,
        revision=revision,
        config=config,
        split="train",
        rubric=rubric,
        files=SourceFiles(
            patterns=(f"{config}/tasks.parquet",), format=SourceFormat.PARQUET, decoder=unpack_task_binary
        ),
    )


def tasktrove_inputs(config: str, revision: str) -> RecipeInputs:
    """Declare the pinned archive consumed by a TaskTrove recipe."""
    return hub_inputs(
        TASKTROVE_DATASET,
        revision,
        SourceFiles((f"{config}/tasks.parquet",), SourceFormat.PARQUET, decoder=unpack_task_binary),
    )
