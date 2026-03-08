# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JPEG-tokenizer training surface built on the grug trainer path.

The JPEG experiments need one extra indirection versus the plain grug template:
the token store should be materialized on the training worker, not serialized
through the Ray job payload. This module keeps the public launch surface small
while deferring token-store loading until the TPU-local entrypoint runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig, OptimizerConfig

from experiments.grug.base.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    _run_grug_local_impl,
    build_tagged_evaluator,
    build_train_dataset,
    build_train_loader,
    initial_state,
)
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.jpeg_tokenizer.base.data import (
    build_passthrough_lm_data_config_from_store,
)
from experiments.jpeg_tokenizer.base.model import JpegLmConfig

JpegEvalConfig = GrugEvalConfig
JpegTrainerConfig = GrugTrainerConfig


@dataclass(frozen=True)
class JpegRunConfig:
    """Top-level config for JPEG-tokenizer training."""

    model: JpegLmConfig
    token_store_path: str
    resources: ResourceConfig
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: JpegTrainerConfig = field(default_factory=JpegTrainerConfig)
    eval: JpegEvalConfig | None = field(default_factory=JpegEvalConfig)


def _run_jpeg_local(config: JpegRunConfig) -> None:
    trainer = config.trainer.trainer
    trainer.initialize()
    grug_config = GrugRunConfig(
        model=config.model,
        data=build_passthrough_lm_data_config_from_store(store_dir=config.token_store_path),
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=config.trainer,
        eval=config.eval,
    )
    _run_grug_local_impl(grug_config)


def run_jpeg_tokenizer(config: JpegRunConfig) -> None:
    """Dispatch JPEG-tokenizer training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching JPEG-tokenizer training.")

    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=_run_jpeg_local,
        resources=config.resources,
    )


__all__ = [
    "JpegEvalConfig",
    "JpegRunConfig",
    "JpegTrainerConfig",
    "build_tagged_evaluator",
    "build_train_dataset",
    "build_train_loader",
    "initial_state",
    "run_jpeg_tokenizer",
]
