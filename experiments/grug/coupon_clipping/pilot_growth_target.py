# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main
from marin.training.training import LevanterCheckpoint

from experiments.grug.coupon_clipping.depth_launch import build_growth_target_only_checkpoint


def build(*, source_checkpoint_root: str, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    return build_growth_target_only_checkpoint(source_checkpoint_root=source_checkpoint_root, version=version)


if __name__ == "__main__":
    experiment_main(build)()
