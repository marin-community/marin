# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the clean C0 checkpoint on a bounded Paloma sample."""

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main

from experiments.grug.coupon_clipping.config import CouponClippingArm
from experiments.grug.coupon_clipping.eval_launch import build_paloma_eval
from experiments.grug.coupon_clipping.launch import build_coupon_clipping_checkpoint


def build(*, version: str | None = None) -> ArtifactStep[Artifact]:
    checkpoint = build_coupon_clipping_checkpoint(CouponClippingArm.C0_P0, version=version)
    return build_paloma_eval(checkpoint, label="c0", version=version)


if __name__ == "__main__":
    experiment_main(build)()
