# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from fray.cluster.device_flops import (
    device_flops,
    device_flops_for_jax_device,
    jax_device_kind_to_fray_device_type,
)


@pytest.mark.parametrize(
    "device_type,dtype,flops",
    [
        ("a100-80g", "bf16", 312e12),
        ("v4", "bf16", 275e12),
        ("v5litepod", "bf16", 197e12),
    ],
)
def test_device_flops(device_type, dtype, flops):
    assert device_flops(device_type, dtype) == flops


@pytest.mark.parametrize(
    "jax_device_kind,expected_fray_type",
    [
        ("TPU v4", "v4"),
        ("TPU v5 lite", "v5litepod"),
        ("TPU v5p", "v5p"),
        ("TPU v6 lite", "v6e"),
        ("NVIDIA H100 80GB HBM3", "h100"),
        ("NVIDIA H100-PCIE", "h100-pcie"),
        ("NVIDIA A100-SXM4-80GB", "a100-80g"),
        ("NVIDIA A10G", "a10g"),
        ("NVIDIA T4", "t4"),
    ],
)
def test_jax_device_kind_to_fray_device_type(jax_device_kind, expected_fray_type):
    assert jax_device_kind_to_fray_device_type(jax_device_kind) == expected_fray_type


@pytest.mark.parametrize(
    "jax_device_kind,expected_flops",
    [
        ("TPU v4", 275e12),
        ("TPU v5p", 459e12),
        ("TPU v5 lite", 197e12),
        ("TPU v6 lite", 918e12),
        ("NVIDIA H100 80GB HBM3", 1.979e15 / 2),
        ("NVIDIA A100-SXM4-80GB", 312e12),
    ],
)
def test_device_flops_for_jax_device(jax_device_kind, expected_flops):
    assert device_flops_for_jax_device(jax_device_kind) == expected_flops


def test_device_flops_for_invalid():
    assert device_flops_for_jax_device("Unknown Device XYZ") is None
    assert device_flops("invalid", "bf16") is None
    assert device_flops("v4", "invalid_dtype") is None
