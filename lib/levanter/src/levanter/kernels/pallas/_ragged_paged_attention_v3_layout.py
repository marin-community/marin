# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from tpu-inference at 2f327d385a90347681a7de86472b4bace6e940b2.

"""Utility functions for ragged paged attention."""
import jax
from jax._src import dtypes


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def next_power_of_2(x: int):
    """Finds the smallest power of 2 >= x using bit manipulation.

    Args:
      x: The input number (should be an integer).

    Returns:
      The smallest integer power of 2 that is >= x.
    """
    assert x > 0
    if x == 1:
        return 1
    return 1 << (x - 1).bit_length()


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        return -1
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
    if kind.endswith("p") or kind.endswith("e") or kind.endswith("i"):
        kind = kind[:-1]
    if kind.startswith("TPU7"):
        return 7
    if kind.startswith("TPU8"):
        return 8
    assert kind[:-1] == "TPU v", kind
    return int(kind[-1])
