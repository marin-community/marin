# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the terminal WD1 checkpoint on a bounded Paloma sample."""

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main

from experiments.grug.coupon_clipping.depth_launch import build_aggressive_checkpoint
from experiments.grug.coupon_clipping.eval_launch import build_paloma_eval


def build(*, version: str | None = None) -> ArtifactStep[Artifact]:
    checkpoint = build_aggressive_checkpoint(version=version)
    return build_paloma_eval(checkpoint, label="wd1", version=version)


if __name__ == "__main__":
    experiment_main(build)()
