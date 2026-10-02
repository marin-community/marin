# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned dapo_math source recipe."""

from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.datasets.direct_math import math_recipe, math_task
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
)

DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
CONFIG = "default"
SPLIT = "train"
SOURCE_FILE = "data/dapo-math-17k.parquet"
SOURCE_FORMAT = "parquet"

RUBRIC = ReviewRubric(
    id="dapo_math-quality",
    version="1",
    criteria=(
        "Preserve every prompt message and reward_model ground truth; source scorer parity is not implied by "
        "matching a reference.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    messages, reward = row.data.get("prompt"), row.data.get("reward_model")
    if not isinstance(messages, list) or not messages or not isinstance(reward, dict) or "ground_truth" not in reward:
        return ImportRejection(
            reason="missing_prompt_or_reference", detail="prompt messages and reward_model ground_truth are required"
        )
    events = tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
    evidence = {key: row.data[key] for key in ("reward_model", "data_source", "ability", "extra_info")}
    return math_task(row, events, str(reward["ground_truth"]), evidence)


def recipe() -> DatasetRecipe:
    return math_recipe("dapo_math", HFSource(DATASET, REVISION, CONFIG, SPLIT), normalize, IntendedUse.TRAIN, RUBRIC)
