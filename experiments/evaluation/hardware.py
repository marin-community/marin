# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin fleet policy for the shared evaluation hardware selector."""

from marin.evaluation.hardware import (
    AcceleratorChoice,
    HardwarePolicy,
    Platform,
)
from marin.evaluation.hardware import (
    select_accelerator as select_accelerator_with_policy,
)
from marin.evaluation.model_config import ModelConfig

MARIN_EVAL_HARDWARE = HardwarePolicy(
    utilization=0.85,
    tpu_slices=("v5litepod-4", "v5litepod-8", "v6e-4", "v6e-8", "v5p-8"),
    tpu_family_regions={
        "v5e": "us-west4",
        "v6e": "europe-west4",
        "v5p": "us-central1",
    },
    gpu_preference=("H100", "GB200"),
    gpu_hbm_gb={"H100": 80, "GB200": 186},
    gpu_max_count={"H100": 8, "GB200": 4},
    gpu_clusters={"H100": "cw-us-east-02a", "GB200": "cw-us-east-08a"},
)


def select_accelerator(
    model: ModelConfig,
    platform: Platform,
    override: str | None,
) -> AcceleratorChoice:
    """Apply Marin's current fleet policy."""
    return select_accelerator_with_policy(
        model,
        platform,
        override,
        MARIN_EVAL_HARDWARE,
    )
