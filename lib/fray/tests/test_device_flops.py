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
from fray.cluster.device_flops import device_flops


def test_device_flops_tpu():
    assert device_flops("v4", "bf16") == 275e12
    assert device_flops("v5litepod", "bf16") == 197e12
    with pytest.raises(ValueError):
        device_flops("v5litepod", "fp8")


def test_device_flops_gpu():
    assert device_flops("a100", "bf16") == 312e12


def test_device_flops_invalid_device():
    with pytest.raises(ValueError, match="Unknown device type"):
        device_flops("invalid", "bf16")


def test_device_flops_invalid_dtype():
    with pytest.raises(ValueError, match="No FLOPS data"):
        device_flops("v4", "invalid_dtype")
