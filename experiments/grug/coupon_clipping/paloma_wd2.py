# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the terminal WD2 checkpoint on a bounded Paloma sample."""

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main
from marin.training.training import LevanterCheckpoint

from experiments.grug.coupon_clipping.eval_launch import build_paloma_eval

_CHECKPOINT = ArtifactStep.adopt(
    "grug/coupon-clipping/adopted/ccx-wd2-terminal",
    "2026.08.01",
    source="users/power/grug/coupon-clipping/ccx-wd2-d768-l1-to-d3072-l48-tail640/dev",
    kind=LevanterCheckpoint,
)


def build(*, version: str | None = None) -> ArtifactStep[Artifact]:
    return build_paloma_eval(_CHECKPOINT, label="wd2", version=version)


if __name__ == "__main__":
    experiment_main(build)()
