# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepScaleR's pinned training blend, with solutions kept as private evidence."""

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

REVISION = "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29"


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    return normalize_math(row, "problem", "answer")


def recipe(snapshot: Path) -> DatasetRecipe:
    return DatasetRecipe(
        name="deepscaler",
        version="deepscaler-v1",
        source=SnapshotSource("agentica-org/DeepScaleR-Preview-Dataset", REVISION, "default", "train", str(snapshot)),
        normalize=normalize,
        intended_use=IntendedUse.TRAIN,
        rubric=ReviewRubric(
            id="deepscaler-math",
            version="1",
            criteria=(
                "Check the problem and private answer for a unique mathematical result; preserve "
                "domains, LaTeX, and units.",
                "The answer field is the key. A solution ending with an option letter does not "
                "override a numeric answer; compare the actual arithmetic before alleging a "
                "contradiction.",
                "This training blend includes historical AIME/AMC, Omni-MATH, and Still "
                "problems. Retain provenance and do not treat it as uncontaminated evaluation "
                "data.",
                "The cleanup typed math comparator is used; source reward-scorer parity is not "
                "claimed. Difficulty and inability to solve immediately are not defects.",
            ),
        ),
        check_suite=CheckSuite(
            id="deepscaler-math-controls",
            revision="1",
            parameters={"comparator": "cleanup-math-verify"},
            run=math_controls,
        ),
    )
