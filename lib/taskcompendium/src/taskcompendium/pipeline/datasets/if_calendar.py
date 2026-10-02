# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Instruction-following calendar source with its final-schedule contract."""

from pathlib import Path

from taskcompendium.pipeline.datasets.calendar_tasks import normalize as normalize_calendar
from taskcompendium.pipeline.datasets.calendar_tasks import verification_report
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    ImportRejection,
    IntendedUse,
    NormalizationChange,
    NormalizedTask,
    RawRow,
    ReviewRubric,
    SnapshotSource,
)

CONFIG = "laion__nemotron-gym-instruction-following-calendar-v3"
REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
RUBRIC = ReviewRubric(
    id="if-calendar-answerability",
    version="1",
    criteria=(
        "Read the complete conversation and derive the final state after additions, updates, removals, and refusals.",
        "Compare event IDs, names, durations, windows, permanent constraints, and working hours with private data.",
        "Check that every required event fits its allowed window without overlaps; do not invent exceptions.",
        "The contract accepts any feasible final schedule. Different valid times are not reference conflicts.",
        "Before constrains event end and after constrains event start; compare these meanings with the wording.",
        "A source witness proves only grader compatibility; missing witness controls imply verification uncertainty.",
        "This requests a final JSON calendar, not an interactive tool episode; flag contradictory delivery promises.",
    ),
)


def normalize(row: RawRow) -> NormalizedTask | ImportRejection:
    task = normalize_calendar(row)
    if isinstance(task, ImportRejection):
        return task
    original = row.data["instruction"]
    replacement = task.context.events[0].content
    changes = (
        ()
        if original == replacement
        else (
            NormalizationChange(
                field="instruction",
                reason="Adapt final calendar file delivery to the assistant response",
                original=original,
                replacement=replacement,
            ),
        )
    )
    return NormalizedTask(task, changes)


def recipe(snapshot: Path) -> DatasetRecipe:
    return DatasetRecipe(
        name="tasktrove-if_calendar",
        version="tasktrove-if_calendar-v1",
        source=SnapshotSource("open-thoughts/TaskTrove", REVISION, CONFIG, "train", str(snapshot)),
        normalize=normalize,
        rubric=RUBRIC,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(
            id="calendar-source-witness-controls", revision="1", parameters={}, run=verification_report
        ),
    )
