# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the clean C0 checkpoint on a bounded Paloma sample."""

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main
from marin.training.training import LevanterCheckpoint

from experiments.grug.coupon_clipping.eval_launch import build_paloma_eval

_CHECKPOINT = ArtifactStep.adopt(
    "grug/coupon-clipping/adopted/cc16-c0-terminal",
    "2026.08.01",
    source="grug/coupon-clipping/cc16-c0-p0/dev",
    kind=LevanterCheckpoint,
)


def build(*, version: str | None = None) -> ArtifactStep[Artifact]:
    return build_paloma_eval(_CHECKPOINT, label="c0", version=version)


if __name__ == "__main__":
    experiment_main(build)()
