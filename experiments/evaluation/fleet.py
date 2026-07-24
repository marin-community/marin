# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fleet policy for Marin evaluation jobs."""

from types import MappingProxyType

from marin.evaluation.hardware import GpuProfile, HardwarePolicy

MARIN_EVAL_HARDWARE = HardwarePolicy(
    utilization=0.85,
    tpu_slices=("v5litepod-4", "v5litepod-8", "v6e-4", "v6e-8", "v5p-8"),
    tpu_family_regions=MappingProxyType(
        {
            "v5e": "us-west4",
            "v6e": "europe-west4",
            "v5p": "us-central1",
        }
    ),
    gpu_preference=("H100", "GB200"),
    gpu_profiles=MappingProxyType(
        {
            "H100": GpuProfile(hbm_gb=80, max_count=8, cluster="cw-us-east-02a"),
            "GB200": GpuProfile(hbm_gb=186, max_count=4, cluster="cw-us-east-08a"),
        }
    ),
)
