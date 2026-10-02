# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned aggregate KTO mixture with its unpaired binary preference labels."""

from pathlib import Path

from taskcompendium.pipeline.datasets import preference_tasks
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

DATASET = "trl-lib/kto-mix-14k"
REVISION = "4470f033f33364e7d064c9f920c3df54d0cce767"
CONFIG = "default"
RUBRIC = ReviewRubric(
    id="kto-mix-answerability",
    version="1",
    criteria=(
        "Read the complete public prompt messages; the labeled candidate completion remains private.",
        "The boolean label is an unpaired preference observation; do not invent a chosen/rejected counterpart.",
        "Assess public task coherence separately from candidate quality or the source preference label.",
        "The pinned mixture has no contributor column; do not claim a sampled row belongs to a named contributor.",
        "Missing inputs and contradictions are task defects; an unbound reward model alone is not.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    return preference_tasks.recipe(
        "kto_mix",
        snapshot,
        dataset=DATASET,
        revision=REVISION,
        config=CONFIG,
        rubric=RUBRIC,
        normalize=preference_tasks.normalize_binary,
    )
