# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pinned algebra training split of MATH, separate from evaluation splits."""

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets.hf_math import math_controls, normalize_math
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
)
from taskcompendium.pipeline.sources import SourceFiles, SourceFormat

REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
SOURCE_FILES = SourceFiles(patterns=("algebra/train-00000-of-00001.parquet",), format=SourceFormat.PARQUET)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    solution = row.data.get("solution")
    if isinstance(solution, str) and r"\boxed" not in solution:
        return ImportRejection(reason="missing_final_answer", detail="A boxed final answer is required in solution")
    return normalize_math(row, "problem", "solution")


def recipe() -> DatasetRecipe:
    return DatasetRecipe(
        name="hendrycks_math",
        version="hendrycks-math-algebra-train-v1",
        source=HFSource("EleutherAI/hendrycks_math", REVISION, "algebra", "train"),
        normalize=normalize,
        intended_use=IntendedUse.TRAIN,
        rubric=ReviewRubric(
            id="hendrycks-math-algebra-train",
            version="1",
            criteria=(
                "Preserve the complete algebra problem, domains, quantifiers, units, and requested answer form.",
                "The last boxed solution answer is private reference evidence. Check short "
                "calculations and contradictions; ordered pairs and half-open intervals are "
                "different contracts.",
                "Only the algebra train split is bound. MATH test and MATH-500 remain evaluation "
                "sources and must not be merged through this binding.",
                "Assess well-posedness independently of difficulty. The cleanup typed math "
                "comparator is used without claiming original scorer parity.",
            ),
        ),
        check_suite=CheckSuite(
            id="hendrycks-math-controls",
            revision="1",
            parameters={"comparator": "cleanup-math-verify"},
            run=math_controls,
        ),
    )
