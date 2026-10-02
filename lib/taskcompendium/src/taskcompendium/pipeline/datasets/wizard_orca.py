# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned wizard orca rubric-judged source and review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets import rubric_tasks
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "laion__wizardlm-orca-v4"
REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
RUBRIC = ReviewRubric(
    id="wizard_orca-answerability",
    version="2",
    criteria=(
        "Trace supplied code with each stated example, including actual printed strings, divisions, "
        "return values, and arithmetic. A purported correct example that disagrees with the code is a "
        "defect unless the public task explicitly asks to debug or correct that discrepancy.",
        "Check the complete instruction, facts, and requested reasoning against the original private rubric.",
        "Reject missing inputs, contradictions, and rubric requirements absent from the public request.",
        "The source grades one holistic numeric reward; preserve its rubric, judge policy, and threshold.",
        "No reference answer is supplied; do not invent one.",
        "An unavailable semantic judge is a verification limitation, not evidence that the content is bad.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    return rubric_tasks.recipe("wizard_orca", snapshot, config=CONFIG, revision=REVISION, rubric=RUBRIC)
