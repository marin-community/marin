# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned eurus2_code source and its private evaluator contract."""

from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.datasets.direct_contracts import contract_task
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
)

DATASET = "PRIME-RL/Eurus-2-RL-Data"
REVISION = "9776b13264b5aaa0b16495fcf086a0a8d86fd655"
CONFIG = "default"
SPLIT = "train"
SOURCE_FILE = "train.parquet"
SOURCE_FORMAT = "parquet"

RUBRIC = ReviewRubric(
    id="eurus2_code-quality",
    version="1",
    criteria=(
        "Require ability=code. Preserve source reward_model ground truth and every prompt message; private "
        "function tests and source evaluator requirements remain private.",
        "Missing runtime binding is a readiness limitation, not a task quality defect. Identify missing "
        "public context separately from implementation difficulty.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    if row.data.get("ability") != "code":
        return ImportRejection(reason="source_selector_mismatch", detail="Eurus code requires ability=code")
    messages, reward = row.data.get("prompt"), row.data.get("reward_model")
    if not isinstance(messages, list) or not messages or not isinstance(reward, dict) or not reward.get("ground_truth"):
        return ImportRejection(
            reason="missing_prompt_or_tests", detail="prompt and reward_model ground_truth are required"
        )
    events = tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
    contract = {key: row.data[key] for key in ("reward_model", "extra_info", "data_source", "ability")}
    return contract_task(
        row,
        events,
        "eurus2_code",
        contract,
        ("PRIME code evaluator, function-call/stdin harness and source comparator",),
    )


def recipe() -> DatasetRecipe:
    return DatasetRecipe(
        name="eurus2_code",
        version="eurus2_code-v1",
        source=HFSource(DATASET, REVISION, CONFIG, SPLIT),
        normalize=normalize,
        intended_use=IntendedUse.TRAIN,
        rubric=RUBRIC,
    )
