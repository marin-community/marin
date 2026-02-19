# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from levanter.data.text import LmDataConfig
from levanter.grug.model import GrugModelConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import TrainerConfig


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug-native training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None
    z_loss_weight: float = 0.0


@dataclass(frozen=True)
class GrugEvalConfig:
    """Perplexity eval settings for grug-native training."""

    steps_per_eval: int | None = None
    max_eval_batches: int | None = None
    prefix: str = "eval"
    eval_current: bool = True
    eval_ema: bool = True
    compute_bpb: bool = True


@dataclass(frozen=True)
class GrugNativeRunConfig:
    """Top-level config for grug-native training."""

    model: GrugModelConfig
    data: LmDataConfig
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig = field(default_factory=GrugEvalConfig)
