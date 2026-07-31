# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the compressed full-model control on a bounded Paloma sample."""

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main

from experiments.grug.coupon_clipping.eval_launch import build_paloma_eval
from experiments.grug.coupon_clipping.launch import build_short_control_checkpoint


def build(*, version: str | None = None) -> ArtifactStep[Artifact]:
    checkpoint = build_short_control_checkpoint(version=version)
    return build_paloma_eval(checkpoint, label="c-short", version=version)


if __name__ == "__main__":
    experiment_main(build)()
