# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen five-family development panels, independent of execution backends."""

import hashlib
import json
from collections import Counter
from enum import StrEnum

FAMILY_COUNTS = {
    "calendar": 167,
    "competitive_coding": 167,
    "reasoning_gym": 167,
    "stem": 166,
    "structural": 243,
}


PROMPT_TOKEN_BUDGET = 16_384
FAST_TASKS_PER_FAMILY = 20
PANEL_SELECTION_VERSION = "snowball-five-family-v1"


class DevelopmentPanel(StrEnum):
    FAST = "fast"
    FULL = "full"

    @property
    def family_counts(self) -> dict[str, int]:
        if self == DevelopmentPanel.FULL:
            return dict(FAMILY_COUNTS)
        return dict.fromkeys(FAMILY_COUNTS, FAST_TASKS_PER_FAMILY)


def select_development_rows(rows: list[dict], panel: DevelopmentPanel) -> list[dict]:
    """Select fixed paired rows independently of file order, scores and initializer.

    Selection ranks each family's source contract and task ID with a versioned
    hash. Selected rows remain byte-for-byte equivalent as records: no task,
    grader, deadline, prompt or pairing field is rewritten.
    """
    if Counter(row["evaluation_family"] for row in rows) != FAMILY_COUNTS:
        raise ValueError("Development selection requires the complete original 910-row panel")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Development panel task IDs must be unique")
    if panel == DevelopmentPanel.FULL:
        return list(rows)

    def rank(row: dict) -> tuple[str, str]:
        identity = (
            f"{PANEL_SELECTION_VERSION}\0{row['evaluation_family']}\0{row['source_contract_id']}\0{row['task_id']}"
        )
        return hashlib.sha256(identity.encode()).hexdigest(), row["task_id"]

    selected = []
    for family, count in panel.family_counts.items():
        candidates = [row for row in rows if row["evaluation_family"] == family]
        selected.extend(sorted(candidates, key=rank)[:count])
    return selected


def pairing_identity(pairs: list[tuple[str, str, str]]) -> str:
    """Identify ordered task IDs, families and original source contracts."""
    encoded = json.dumps(pairs, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def evaluation_dataset_uid(uid: str, step: int | None) -> str:
    """Read the dataset UID from the step evaluator's length-prefixed instance ID."""
    if step is None:
        return uid
    prefix = f"eval-{step}-"
    if not uid.startswith(prefix):
        raise ValueError("Periodic evaluation UID does not match the requested update")
    name_length, separator, tail = uid[len(prefix) :].partition("-")
    if not separator or not name_length.isdecimal():
        raise ValueError("Periodic evaluation UID lacks a validation-name length")
    length = int(name_length)
    if len(tail) <= length + 1 or tail[length] != "-":
        raise ValueError("Periodic evaluation UID has an invalid validation-name boundary")
    return tail[length + 1 :]
