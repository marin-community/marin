# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-turn checklist tasks retaining the original scoring and criterion polarity."""

import base64
import tomllib

from taskcompendium.datasets import rubric_tasks
from taskcompendium.pipeline.models import (
    ImportFailureKind,
    ImportRejection,
    NormalizedTask,
    RawRow,
    ReviewRubric,
    TaskPolicy,
)

RUBRIC = ReviewRubric(
    id="multichallenge-answerability",
    version="2",
    criteria=(
        "Read the full persona and conversation; grade only the requested next response, not a historical turn.",
        "Compare every private criterion with the public conversation and final user request, including earlier rules.",
        "Preserve negated criterion polarity and the source aggregation rule; all-pass is not a mean score.",
        "RewardKit 0.1.4 all_pass requires every normalized criterion score to be greater than zero. "
        "For numeric criteria, even a small positive score passes; flag this permissive threshold without changing it.",
        "Reject conflicting required formats, absent context, and invented checklist conditions.",
        "Judge availability is a verification limitation; no canonical response should be fabricated.",
    ),
)


def normalize(row: RawRow) -> NormalizedTask | ImportRejection:
    files = row.data.get("files", {})
    if "tests/judge.toml" not in files or "tests/conversation.txt" not in files:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="missing_judge_context",
            detail="judge.toml and conversation.txt are required",
        )
    conversation = base64.b64decode(files["tests/conversation.txt"], validate=True).decode()
    return normalized_checklist(row, conversation)


def normalize_vanilla(row: RawRow) -> NormalizedTask | ImportRejection:
    """Read the vanilla source's complete conversation from its verifier data."""
    data = row.data.get("verifier_data")
    if not isinstance(data, dict) or not isinstance(data.get("instruction"), str) or not data["instruction"].strip():
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="missing_judge_context",
            detail="verifier_data.instruction is required",
        )
    if "tests/judge.toml" not in row.data.get("files", {}):
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED, reason="missing_judge_context", detail="judge.toml is required"
        )
    return normalized_checklist(row, data["instruction"])


def normalized_checklist(row: RawRow, conversation: str) -> NormalizedTask | ImportRejection:
    """Preserve the judge's criteria, scoring and original private runtime files."""
    files = row.data["files"]
    runtime_files = {}
    for path in ("tests/test.sh", "task.toml", "environment/Dockerfile"):
        if path not in files:
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED,
                reason="missing_judge_runtime",
                detail=f"Original source {path} is required",
            )
        runtime_files[path] = base64.b64decode(files[path], validate=True).decode()
    if "tests/sitecustomize.py" in files:
        runtime_files["tests/sitecustomize.py"] = base64.b64decode(
            files["tests/sitecustomize.py"], validate=True
        ).decode()
    judge_toml = base64.b64decode(files["tests/judge.toml"], validate=True).decode()
    configuration = tomllib.loads(judge_toml)
    criteria = tuple(criterion["description"] for criterion in configuration.get("criterion", []))
    if not criteria or any(not criterion.strip() for criterion in criteria):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="empty_criteria",
            detail="The source provides no complete checklist criteria",
        )
    return rubric_tasks.normalized_task(
        row,
        {
            "mode": "checklist",
            "question": conversation,
            "criteria": criteria,
            "aggregation": configuration,
            "source_judge_data": row.data["verifier_data"],
            "source_judge_toml": judge_toml,
            "source_runtime_files": runtime_files,
            "source_verifier_settings": tomllib.loads(runtime_files["task.toml"])["verifier"],
        },
    )


def policy() -> TaskPolicy:
    """Build the multichallenge normalization and review policy."""
    return rubric_tasks.policy(RUBRIC, normalize_row=normalize)


def vanilla_policy() -> TaskPolicy:
    """Review vanilla checklists without requiring the advanced source's file layout."""
    return rubric_tasks.policy(
        ReviewRubric(
            id="multichallenge-vanilla-answerability",
            version="2",
            criteria=(
                *RUBRIC.criteria,
                "A single criterion and no canonical response are valid source choices. Check that the full "
                "criterion, including its expected YES/NO verdict, measures the requested final response.",
            ),
        ),
        normalize_row=normalize_vanilla,
    )
