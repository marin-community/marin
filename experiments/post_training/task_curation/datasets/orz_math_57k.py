# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the ORZ math collection with the shipped boxed-answer verifier.

The Open-Reasoner-Zero training collection's rows are chat transcripts: the human turn
carries the verbatim problem and the assistant turn carries the exact reference answer
(``ground_truth.value``). The context is the verbatim problem; the grader is the shipped
boxed-answer extractor plus ``math_verify`` against the reference (reward 1 when the
candidate's final answer verifies). The reference answers are the dataset's own gold — the
same closed-form contract as the MEDIUM reference-judge sources — and never ship in the
candidate's context.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from finestore.rl.status_taxonomy import DataRestriction
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.script_grader import grade_script, script_package
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.models import TextMessage
from taskcompendium.pipeline.inputs import (
    ConversionContext,
    SourceFormat,
    required_grader_environment,
)
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizationChange, NormalizedTask, RawRow
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

HERE = Path(__file__).parent
REPO = "Open-Reasoner-Zero/orz_math_57k_collection"
REVISION = "09e8ddf7ff58fe0f2ea5e8e74ed7d20d0f8ea24e"
SOURCE_ID = "Open-Reasoner-Zero:orz_math_57k_collection"
VERSION = "orz-boxed-verify-v1"
COUNT = 56878
GRADER = grade_script(HERE / "orz_math_grade.py")


def read_rows(path: StoragePath, _context: ConversionContext) -> Iterator[dict[str, Any]]:
    """One dict per transcript: the verbatim problem and the exact reference answer."""
    with path.open("rb") as stream:
        array = json.load(stream)
    for transcript in array:
        human = [m for m in transcript if m.get("from") == "human"]
        assistant = [m for m in transcript if m.get("from") == "assistant"]
        problem = human[-1]["value"] if human else None
        truth = assistant[-1].get("ground_truth", {}).get("value") if assistant else None
        yield {"problem": problem, "ground_truth": truth}


def convert_problem(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    native = dict(row.data)
    problem, truth = native.get("problem"), native.get("ground_truth")
    if not isinstance(problem, str) or not problem.strip() or not isinstance(truth, str) or not truth.strip():
        return source_defect(
            "incomplete_orz_contract", "The verbatim problem and a nonempty reference answer are required"
        )
    package = script_package(
        GRADER,
        {"ground_truth": truth},
        environment=required_grader_environment(context),
        timeout=330.0,
        answer_path=DEFAULT_OUTPUT,
    )
    events = (TextMessage(role="user", content=problem),)
    task = conversation_task(row, events=events, package=package).model_copy(
        update={
            "tags": ("train", "math", "exact-answer", "orz"),
            "labels": {"source_id": SOURCE_ID},
        }
    )
    change = NormalizationChange(
        field="context.events",
        reason="Drop the assistant turn from the transcript; the context is the verbatim problem only.",
        original="The native row is a two-turn transcript ending in the reference answer.",
        replacement=(
            "The task presents the verbatim problem alone; the reference answer is the grader's gold, never "
            "in the candidate's context."
        ),
    )
    return NormalizedTask(task, (change,))


def source(files: tuple[str, ...]) -> RlDataSource:
    return RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.OPEN_REASONER_ZERO,
            upstream_dataset="orz_math_57k_collection",
            upstream_name="ORZ-Math-57K",
            build_path=VERSION,
            family="math",
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=COUNT,
            notes=(
                "TRAIN math tasks from the Open-Reasoner-Zero training collection: the context is the "
                "verbatim problem (the transcript's human turn) and the grader is the shipped boxed-answer "
                "extractor plus math_verify against the dataset's own exact reference answer (reward 1 when "
                "the candidate's final answer verifies; the reference answers are the dataset's gold, never "
                "in the candidate's context). Contract defects fail closed. No live execution is claimed by "
                "conversion."
            ),
        ),
        config=CurationRecipe(
            name="orz-boxed-verify",
            version=VERSION,
            source=HfSource(REPO, REVISION, files, SourceFormat.JSON, read=read_rows),
            grader=GRADER_PACKAGES,
            convert=convert_problem,
            intended_use=IntendedUse.TRAIN,
        ),
        pipeline=process_rows,
    )
