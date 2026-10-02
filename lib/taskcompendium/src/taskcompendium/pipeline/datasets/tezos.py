# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned tezos rubric-judged source and review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets import rubric_tasks
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "laion__stackexchange-tezos-sandboxes-verified-v2"
RUBRIC = ReviewRubric(
    id="tezos-answerability",
    version="1",
    criteria=(
        "Check that Tezos questions supply necessary code, transaction details, versions, and error context.",
        "Reject missing inputs, contradictions, and rubric requirements absent from the public request.",
        "The source grades one holistic numeric reward; preserve its rubric, judge policy, and threshold.",
        "No reference answer is supplied; do not invent one.",
        "An unavailable semantic judge is a verification limitation, not evidence that the content is bad.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    return rubric_tasks.recipe("tezos", snapshot, config=CONFIG, revision=REVISION, rubric=RUBRIC)
