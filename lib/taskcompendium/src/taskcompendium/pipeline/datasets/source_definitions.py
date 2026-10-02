# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned source contracts for TaskTrove and direct dataset families."""

from dataclasses import dataclass

from taskcompendium.pipeline.models import ReviewRubric
from taskcompendium.pipeline.sources import SourceFiles, SourceFormat, unpack_task_binary


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
        dataset="open-thoughts/TaskTrove",
        revision=revision,
        config=config,
        split="train",
        rubric=rubric,
        files=SourceFiles(
            patterns=(f"{config}/tasks.parquet",), format=SourceFormat.PARQUET, decoder=unpack_task_binary
        ),
    )
