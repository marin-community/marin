# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from fray.device_flops import device_flops, device_flops_for_jax_device, jax_device_kind_to_fray_device_type


@pytest.mark.parametrize(
    ("device_kind", "expected_type", "expected_bf16"),
    [
        ("NVIDIA GH200 480GB", "gh200", 1.979e15 / 2),
        ("NVIDIA B200", "b200", 36e15 / 2 / 8),
    ],
)
def test_new_coreweave_gpu_device_flops(device_kind: str, expected_type: str, expected_bf16: float):
    assert jax_device_kind_to_fray_device_type(device_kind) == expected_type
    assert device_flops_for_jax_device(device_kind, "bf16") == expected_bf16
    assert device_flops(expected_type, "bf16") == expected_bf16
