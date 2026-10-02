# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned swe rebench repository repair source and review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets import repository_tasks
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "DCAgent__swe_rebench_v2_patched_oracle-v2"
RUBRIC = ReviewRubric(
    id="swe_rebench-answerability",
    version="1",
    criteria=(
        "Compare the issue request and source checkout with the hidden test patch, restored trusted "
        "paths, and test IDs.",
        "The public repository and checkout identify necessary context; unavailable local checkout is a runtime "
        "limitation rather than proof that the issue is underspecified.",
        "Flag hidden requirements unrelated to the public issue, wrong base references, and inconsistent test IDs.",
        "Repository source, multi-file changes, dependencies, and trusted-test restoration require an isolated "
        "runtime; do not certify a repair using a generic solution.py sandbox.",
        "Source oracle scripts are private review controls; their existence does not prove the issue or grader correct.",
        "Distinguish installation/network failures from task defects and retain concrete unresolved evidence.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    return repository_tasks.recipe("swe_rebench", snapshot, config=CONFIG, revision=REVISION, rubric=RUBRIC)
