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


def test_get_per_device_tpu_flops():
    """Test the get_per_device_tpu_flops function."""
    from marin.resources_utils import get_per_device_tpu_flops

    # Test with a valid TPU type and dtype
    flops = get_per_device_tpu_flops("v4", "bf16")
    assert flops is not None, f"Expected FLOPS for TPU type v4 and dtype bf16 to be not None, but got {flops}"

    # Test with an invalid TPU type
    with pytest.raises(ValueError, match="Unknown TPU type: invalid"):
        get_per_device_tpu_flops("invalid", "bf16")

    # Test with an invalid dtype
    with pytest.raises(ValueError, match="Unknown dtype: invalid_dtype"):
        get_per_device_tpu_flops("v4", "invalid_dtype")  # type: ignore[call-arg]


def test_flop_count_per_device_from_accel_type():
    """Test the flop_count_per_device_from_accel_type function."""
    from marin.resources_utils import flop_count_per_device_from_accel_type

    # Test with a valid accelerator type and dtype
    flops = flop_count_per_device_from_accel_type("A100-40G", "bf16")
    assert flops is not None, f"Expected FLOPS for A100-40G and bf16 to be not None, but got {flops}"

    # Test with an invalid accelerator type
    flops = flop_count_per_device_from_accel_type("invalid", "bf16")
    assert flops is None, f"Expected FLOPS for invalid accelerator type to be None, but got {flops}"
