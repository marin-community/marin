# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the d1536/L1 throughput gate for aggressive coupon clipping."""

from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main
from marin.training.training import LevanterCheckpoint

from experiments.grug.coupon_clipping.depth_launch import build_aggressive_source_pilot_checkpoint


def build(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    return build_aggressive_source_pilot_checkpoint(version=version)


if __name__ == "__main__":
    experiment_main(build)()
