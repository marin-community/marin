# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from .model import GrugModelConfig


@dataclass(frozen=True)
class GrugTrainingConfig:
    """Full training recipe, nested around a model."""

    model: GrugModelConfig
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    seed: int = 0
    steps: int = 10
    global_batch_size: int = 8


__all__ = ["GrugModelConfig", "GrugTrainingConfig"]
