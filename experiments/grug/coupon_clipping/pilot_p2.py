# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import experiment_main
from marin.training.training import LevanterCheckpoint

from experiments.grug.coupon_clipping.config import CouponClippingArm, CouponClippingLearningRate
from experiments.grug.coupon_clipping.launch import build_coupon_clipping_pilot_checkpoint


def build(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    return build_coupon_clipping_pilot_checkpoint(
        CouponClippingArm.P2,
        version=version,
        learning_rate=CouponClippingLearningRate.CENTER,
    )


if __name__ == "__main__":
    experiment_main(build)()
