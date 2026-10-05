# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-turn checklist tasks retaining the original scoring and criterion polarity."""

import base64
import tomllib

from taskcompendium.pipeline.datasets import rubric_tasks
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    ImportRejection,
    NormalizedTask,
    RawRow,
    ReviewRubric,
)

CONFIG = "laion__nemotron-gym-multichallenge-advanced-v4"
RUBRIC = ReviewRubric(
    id="multichallenge-answerability",
    version="1",
    criteria=(
        "Read the full persona and conversation; grade only the requested next response, not a historical turn.",
        "Compare every private criterion with the public conversation and final user request, including earlier rules.",
        "Preserve negated criterion polarity and the source aggregation rule; all-pass is not a mean score.",
        "Reject conflicting required formats, absent context, and invented checklist conditions.",
        "Judge availability is a verification limitation; no canonical response should be fabricated.",
    ),
)


def normalize(row: RawRow) -> NormalizedTask | ImportRejection:
    files = row.data.get("files", {})
    if "tests/judge.toml" not in files or "tests/conversation.txt" not in files:
        return ImportRejection(reason="missing_judge_context", detail="judge.toml and conversation.txt are required")
    judge_toml = base64.b64decode(files["tests/judge.toml"], validate=True).decode()
    conversation = base64.b64decode(files["tests/conversation.txt"], validate=True).decode()
    configuration = tomllib.loads(judge_toml)
    criteria = tuple(criterion["description"] for criterion in configuration.get("criterion", []))
    if not criteria or any(not criterion.strip() for criterion in criteria):
        return ImportRejection(reason="empty_criteria", detail="The source provides no complete checklist criteria")
    return rubric_tasks.normalized_task(
        row,
        {
            "mode": "checklist",
            "question": conversation,
            "criteria": criteria,
            "aggregation": configuration,
            "source_judge_data": row.data["verifier_data"],
            "source_judge_toml": judge_toml,
        },
    )


def recipe() -> DatasetRecipe:
    return rubric_tasks.recipe(
        "multichallenge",
        config=CONFIG,
        revision=REVISION,
        rubric=RUBRIC,
        normalize_row=normalize,
    )
