# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned swesmith repository repair source and review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets import repository_tasks
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "laion__swesmith-oracle-filtered-v2"
REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
RUBRIC = ReviewRubric(
    id="swesmith-answerability",
    version="1",
    criteria=(
        "Compare the stated repository bug and behavioral requirements with FAIL_TO_PASS and PASS_TO_PASS tests.",
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
    return repository_tasks.recipe("swesmith", snapshot, config=CONFIG, revision=REVISION, rubric=RUBRIC)
