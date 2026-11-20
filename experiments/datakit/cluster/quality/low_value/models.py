# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published structural scorer pins, resolved against the active MARIN_PREFIX."""

from experiments.datakit.cluster.quality.low_value.pipeline import LowValuePin

REPLAY_LOW_VALUE_V1 = LowValuePin(
    model_path="datakit/models/quality/replay_low_value_v1/2026.11.15.2",
    model_sha256="1396ad6cdaa9b7d1d9b5c0b2cabde124c4e3fe0e8c869da55e2e2b72bc7f545d",
    calibration_sha256="e650d4e9b71cc55a7feb8d54272d50a195f35a10e8894d9fea4fed119ee41c83",
)
