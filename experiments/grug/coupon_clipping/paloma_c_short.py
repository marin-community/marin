# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the compressed full-model control on a bounded Paloma sample."""

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main
from marin.training.training import LevanterCheckpoint

from experiments.grug.coupon_clipping.eval_launch import build_paloma_eval

_CHECKPOINT = ArtifactStep.adopt(
    "grug/coupon-clipping/adopted/ccx-c-short-terminal",
    "2026.08.01",
    source="users/power/grug/coupon-clipping/ccx-c-short-l48-step3200/dev",
    kind=LevanterCheckpoint,
)


def build(*, version: str | None = None) -> ArtifactStep[Artifact]:
    return build_paloma_eval(_CHECKPOINT, label="c-short", version=version)


if __name__ == "__main__":
    experiment_main(build)()
