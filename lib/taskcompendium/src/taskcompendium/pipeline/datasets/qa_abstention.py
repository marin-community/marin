# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned qa abstention source binding and quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.atlas_math_qa import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

RUBRIC = ReviewRubric(
    id="qa_abstention-answerability",
    version="1",
    criteria=(
        "Judge answerability from the actual question. The wrapper's claim that every question is "
        "knowable does not supply omitted passages, diagrams, personal facts, or needed "
        "experimental conditions.",
        "An optional [IDK] response does not repair an unanswerable task. The source rejects "
        "abstention for ordinary answerable rows. Check reference accuracy and whether it fully "
        "answers the question.",
        "The semantic paraphrase judge is unbound. That is a grading integration annotation, not a "
        "reason to reject an otherwise coherent static task. Do not confuse output delivery with "
        "subject matter.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned snapshot to this source's normalization and rubric."""
    return family_recipe("qa_abstention", snapshot, rubric=RUBRIC)
