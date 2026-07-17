# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for device FLOPS lookup, focused on JAX device-kind mapping."""

import pytest
from fray.device_flops import device_flops_for_jax_device, jax_device_kind_to_fray_device_type


@pytest.mark.parametrize("kind", ["NVIDIA B200", "NVIDIA GB200"])
def test_blackwell_maps_to_b200(kind):
    """Regression: JAX reports Blackwell GPUs as "NVIDIA B200"/"NVIDIA GB200".

    Both must map to the "b200" spec rather than falling through to the raw,
    unmapped kind string (which produced a silent None and dropped MFU logging).
    """
    assert jax_device_kind_to_fray_device_type(kind) == "b200"


@pytest.mark.parametrize("kind", ["NVIDIA B200", "NVIDIA GB200"])
def test_blackwell_peak_flops_present(kind):
    """B200/GB200 must resolve to concrete peak FLOP/s, not None."""
    # dense bf16 = 4.5 PFLOP/s sparse / 2, per fray's sparse/2 convention.
    assert device_flops_for_jax_device(kind, "bf16") == 4.5e15 / 2
    assert device_flops_for_jax_device(kind, "fp8") == 9e15 / 2
