# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Teacher-model configuration for RL distillation losses."""

from dataclasses import dataclass
from enum import StrEnum

INITIAL_POLICY_TEACHER_CHECKPOINT = "initial_policy_checkpoint"
"""Marker resolved by RL experiment launchers to the runtime student initial checkpoint."""


class TeacherScoringMode(StrEnum):
    """Supported teacher scoring modes."""

    LOCAL_SAMPLED_TOKEN = "local_sampled_token"


@dataclass(frozen=True)
class TeacherConfig:
    """Configuration for a local teacher model used by train-time losses."""

    checkpoint: str
    """Checkpoint or Hugging Face repo used to load the teacher model."""

    scoring_mode: TeacherScoringMode = TeacherScoringMode.LOCAL_SAMPLED_TOKEN
    """Teacher scoring implementation. The MVP supports sampled-token scoring only."""

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("TeacherConfig.checkpoint must be non-empty")

        if not isinstance(self.scoring_mode, TeacherScoringMode):
            object.__setattr__(self, "scoring_mode", TeacherScoringMode(self.scoring_mode))
