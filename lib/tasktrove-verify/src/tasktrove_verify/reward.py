# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The reward record every mode produces and the files it is written to."""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

REWARD_JSON = "reward.json"
REWARD_TXT = "reward.txt"


class Status(StrEnum):
    SCORED = "scored"
    INVALID_TASK = "invalid_task"
    INFRA_ERROR = "infra_error"


class InvalidTask(Exception):
    """The task is malformed: a missing reference, an unknown constraint, too few cases."""


@dataclass(frozen=True)
class Reward:
    reward: float
    status: Status
    detail: dict = field(default_factory=dict)


def scored(reward: float, **detail: object) -> Reward:
    return Reward(float(reward), Status.SCORED, dict(detail))


def invalid_task(message: str) -> Reward:
    return Reward(0.0, Status.INVALID_TASK, {"error": message})


def infra_error(message: str) -> Reward:
    return Reward(0.0, Status.INFRA_ERROR, {"error": message})


def write_reward(logs_dir: Path, reward: Reward) -> None:
    """Write reward.json and Harbor's reward.txt. A non-scored status writes reward.txt as 0."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    payload = {"reward": reward.reward, "status": reward.status.value, "detail": reward.detail}
    (logs_dir / REWARD_JSON).write_text(json.dumps(payload) + "\n")
    (logs_dir / REWARD_TXT).write_text(f"{reward.reward if reward.status == Status.SCORED else 0.0}\n")
