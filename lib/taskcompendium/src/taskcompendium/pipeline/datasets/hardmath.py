# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned HARDMath training questions with their symbolic and list references."""

from pathlib import Path

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets.hf_math import math_controls, normalize_math
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    SnapshotSource,
)

REVISION = "937e9f10356e31e854f6efb9a2507f1e200c8b25"


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    return normalize_math(row, "question", "ground_truths")


def recipe(snapshot: Path) -> DatasetRecipe:
    return DatasetRecipe(
        name="hardmath",
        version="hardmath-v1",
        source=SnapshotSource("pafitis/HARDMath_processed_training", REVISION, "default", "train", str(snapshot)),
        normalize=normalize,
        intended_use=IntendedUse.TRAIN,
        rubric=ReviewRubric(
            id="hardmath-asymptotics",
            version="1",
            criteria=(
                "Check the question, private solution, and ground_truths together. Symbolic "
                "regimes, approximations, boundary conditions, and equations are part of the "
                "task.",
                "Compare every requested regime or deliverable with the reference list. A key "
                "omitting a requested asymptotic regime is a concrete mismatch.",
                "Check limiting powers and coefficients before certifying asymptotic formulas; "
                "do not confuse necessary and sufficient regimes or invent precision "
                "requirements.",
                "The training source is not an evaluation benchmark binding. The cleanup typed "
                "math comparator is used without claiming source scorer parity; hard mathematics "
                "alone is not a defect.",
            ),
        ),
        check_suite=CheckSuite(
            id="hardmath-math-controls",
            revision="1",
            parameters={"comparator": "cleanup-math-verify"},
            run=math_controls,
        ),
    )
