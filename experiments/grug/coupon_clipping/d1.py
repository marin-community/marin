# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main
from marin.training.training import LevanterCheckpoint

from experiments.grug.coupon_clipping.depth_launch import build_d1_checkpoint


def build(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    return build_d1_checkpoint(version=version)


if __name__ == "__main__":
    experiment_main(build)()
