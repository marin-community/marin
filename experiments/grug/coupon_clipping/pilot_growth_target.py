# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main
from marin.training.training import LevanterCheckpoint

from experiments.grug.coupon_clipping.depth_launch import build_growth_target_only_checkpoint

_SOURCE_CHECKPOINT_ENV = "CC16_GROWTH_SOURCE_CHECKPOINT_ROOT"


def build(*, source_checkpoint_root: str, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    return build_growth_target_only_checkpoint(source_checkpoint_root=source_checkpoint_root, version=version)


def _build_from_environment(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    source_checkpoint_root = os.environ.get(_SOURCE_CHECKPOINT_ENV)
    if source_checkpoint_root is None:
        raise ValueError(f"{_SOURCE_CHECKPOINT_ENV} is required for target-only growth recovery")
    return build(source_checkpoint_root=source_checkpoint_root, version=version)


if __name__ == "__main__":
    experiment_main(_build_from_environment)()
