# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""W&B run normalization for roofline reports."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ENTITY = "marin-community"
DEFAULT_PROJECT = "marin_moe"
WANDB_RUN_PREFIX = "https://wandb.ai/"


@dataclass(frozen=True)
class WandbRunRef:
    entity: str
    project: str
    run_id: str

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"

    @property
    def url(self) -> str:
        return f"{WANDB_RUN_PREFIX}{self.entity}/{self.project}/runs/{self.run_id}"


def normalize_wandb_run(value: str | None) -> WandbRunRef | None:
    if value is None or not value.strip():
        return None
    stripped = value.strip()
    if stripped.startswith(WANDB_RUN_PREFIX):
        parts = stripped.removeprefix(WANDB_RUN_PREFIX).strip("/").split("/")
        if len(parts) >= 4 and parts[2] == "runs":
            return WandbRunRef(entity=parts[0], project=parts[1], run_id=parts[3])
        raise ValueError(f"Unsupported W&B run URL: {value}")

    parts = stripped.strip("/").split("/")
    if len(parts) == 1:
        return WandbRunRef(entity=DEFAULT_ENTITY, project=DEFAULT_PROJECT, run_id=parts[0])
    if len(parts) == 3:
        return WandbRunRef(entity=parts[0], project=parts[1], run_id=parts[2])
    raise ValueError(
        "W&B run must be a run id, entity/project/run id, or https://wandb.ai/<entity>/<project>/runs/<run-id>"
    )
