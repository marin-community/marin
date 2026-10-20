# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native Snowball structural tasks with ordinary assistant prefills."""

import json

from taskcompendium.convert.structure_closure import structure_closure_task
from taskcompendium.models import TaskSpec, TextMessage, VerifyitGrader, verifyit_spec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, NormalizedTask, RawRow, Reply
from verifyit.json_objects import unique_object
from verifyit.spec import StructureClosureSpec

from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

REPOSITORY = "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data"
REVISION = "cb728a7636caad3fc6fa6b6ba85a49194d9d69a3"
TERMINAL_MARKERS = ("<|eot_id|>", "<|end_of_text|>")

RUBRIC = """
These tasks complete an unfinished assistant message. Present its text as an assistant prefill and
grade the generated continuation. Historical prompt tokens are provenance, not a model requirement.

The native config.json extras.prefix is the supplied assistant prefix, and extras.expected is the required
completed reference. Check that the reference extends the prefix, preserves required content and closes one
outer structure. The reference is deliberately part of these reproduction tasks; do not flag its presence
as accidental answer leakage. The grader scores the first completed outer region, permitting later output.

Keep the native development validation split for evaluation. It was repeatedly used during development;
it is not an independent final benchmark. Static closure checks and controls do not establish RL efficacy.
"""


def structural_row(row: dict, _context: ConversionContext) -> bool:
    native = json.loads(row["lowered_task_spec"], object_pairs_hook=unique_object)
    return native["task"]["source"]["dataset"] == "structure-closure"


def convert(row: RawRow, _context: ConversionContext) -> NormalizedTask | ImportRejection:
    return structure_closure_task(
        row,
        terminal_markers=TERMINAL_MARKERS,
    )


def reference_continuation(task: TaskSpec) -> Reply:
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, StructureClosureSpec)
    return Reply(TextMessage(role="assistant", content=spec.expected[len(spec.prefix) :]))


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                id="SnowballV01:structure-closure" + ("/validation" if split == "validation" else ""),
                title=f"Snowball v0.1 structural closure · {split}",
                origin="SnowballV01",
                family="structural",
                tags=("rlvr", "single-turn", "structural-closure", "assistant-prefill"),
                count=count,
                notes="Text assistant prefills; validation is a reused development panel.",
            ),
            config=CurationRecipe(
                name=f"snowball-v01-structure-closure-{split}",
                version="2",
                source=HfSource(
                    REPOSITORY, REVISION, (f"{split}.parquet",), SourceFormat.PARQUET, select=structural_row
                ),
                convert=convert,
                intended_use=intended_use,
                rubric=RUBRIC,
                controls=Controls(golden=reference_continuation),
            ),
            pipeline=process_rows,
        )
        for split, intended_use, count in (
            ("train", IntendedUse.TRAIN, 57_364),
            ("validation", IntendedUse.EVAL, 243),
        )
    ]
