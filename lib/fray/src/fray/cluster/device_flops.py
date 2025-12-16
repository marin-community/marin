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

"""Device FLOPS data and utilities.

Data originally from Mosaic SPDX-License-Identifier: Apache-2.0
https://github.com/mosaicml/composer/blob/56ccc2ebc59a8c68a6d075c4b61735ebf089b5a2/composer/callbacks/speed_monitor.py#L23
"""
import logging
from typing import Literal

logger = logging.getLogger(__name__)

FlopDtype = Literal["bf16", "fp16", "fp32", "fp64", "tf32", "int8", "int4", "fp8"]

# Peak FLOPS per device type. Keys are lowercase device identifiers.
DEVICE_FLOPS: dict[str, dict[str, float]] = {
    # NVIDIA GPUs
    # source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    # nvidia publishes spec sheet with a 2x sparsity factor
    "h100": {
        "fp64": 67e12,
        "fp32": 67e12,
        "tf32": 989e12 / 2,
        "fp16": 1.979e15 / 2,
        "bf16": 1.979e15 / 2,
        "fp8": 3.958e15 / 2,
        "int8": 3.958e15 / 2,
    },
    "h100-pcie": {
        "fp64": 51e12,
        "fp32": 51e12,
        "tf32": 756e12 / 2,
        "fp16": 1.513e15 / 2,
        "bf16": 1.513e15 / 2,
        "fp8": 3.026e15 / 2,
        "int8": 3.026e15 / 2,
    },
    # Source: https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23
    "h200": {
        "fp64": 67e12,
        "fp32": 67e12,
        "tf32": 989e12 / 2,
        "fp16": 1.979e15 / 2,
        "bf16": 1.979e15 / 2,
        "fp8": 3.958e15 / 2,
        "int8": 3.958e15 / 2,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    "a100": {
        "fp64": 19.5e12,
        "fp32": 19.5e12,
        "tf32": 156e12,
        "fp16": 312e12,
        "bf16": 312e12,
    },
    "a100-80g": {
        "fp64": 19.5e12,
        "fp32": 19.5e12,
        "tf32": 156e12,
        "fp16": 312e12,
        "bf16": 312e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
    "a10": {
        "fp32": 31.2e12,
        "tf32": 62.5e12,
        "fp16": 125e12,
        "bf16": 125e12,
    },
    # A10G uses same specs as A10
    "a10g": {
        "fp32": 31.2e12,
        "tf32": 62.5e12,
        "fp16": 125e12,
        "bf16": 125e12,
    },
    # source: https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf
    "a40": {
        "fp32": 37.4e12,
        "tf32": 74.8e12,
        "fp16": 149.7e12,
        "bf16": 149.7e12,
    },
    # Source: https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf
    "a6000": {
        "fp32": 38.7e12 / 2,
        "tf32": 309.7e12 / 2,
        "fp16": 309.7e12 / 2,
        "bf16": 309.7e12 / 2,
    },
    # B100 - approximate with H100 until specs available
    "b100": {
        "fp64": 67e12,
        "fp32": 67e12,
        "tf32": 989e12 / 2,
        "fp16": 1.979e15 / 2,
        "bf16": 1.979e15 / 2,
        "fp8": 3.958e15 / 2,
        "int8": 3.958e15 / 2,
    },
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    "v100": {
        "fp64": 7e12,
        "fp32": 14e12,
        "fp16": 112e12,
    },
    "v100-sxm": {
        "fp64": 7.8e12,
        "fp32": 15.7e12,
        "fp16": 125e12,
    },
    "v100s": {
        "fp64": 8.2e12,
        "fp32": 16.4e12,
        "fp16": 130e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    "t4": {
        "fp32": 8.1e12,
        "fp16": 65e12,
        "int8": 130e12,
        "int4": 260e12,
    },
    # source: https://images.nvidia.com/content/Solutions/data-center/vgpu-L4-background-image-background-image/l4-datasheet.pdf
    "l4": {
        "fp32": 30.3e12,
        "tf32": 120e12,
        "fp16": 242e12,
        "bf16": 242e12,
        "fp8": 485e12,
        "int8": 485e12,
    },
    # NVIDIA GB10
    "gb10": {
        "bf16": 100e12,
    },
    # "auto" uses H100 flops for when user doesn't care about specific GPU type
    "auto": {
        "fp64": 67e12,
        "fp32": 67e12,
        "tf32": 989e12 / 2,
        "fp16": 1.979e15 / 2,
        "bf16": 1.979e15 / 2,
        "fp8": 3.958e15 / 2,
        "int8": 3.958e15 / 2,
    },
    # TPUs
    # Source: https://cloud.google.com/tpu/docs/v3
    # v3 gives "per chip" flops, so we divide by 2 since jax device is a core
    "v3": {
        "bf16": 123e12 / 2,
    },
    # Source: https://cloud.google.com/tpu/docs/v4
    "v4": {
        "bf16": 275e12,
        "int8": 275e12,
    },
    # Source: https://cloud.google.com/tpu/docs/v5e
    "v5litepod": {
        "bf16": 197e12,
        "int8": 393e12,
    },
    # Source: https://cloud.google.com/tpu/docs/v5p
    "v5": {
        "bf16": 459e12,
    },
    "v5p": {
        "bf16": 459e12,
    },
    # Source: https://cloud.google.com/tpu/docs/v6e
    "v6e": {
        "bf16": 918e12,
        "int8": 1836e12,
    },
    # Other accelerators
    # source: https://aws.amazon.com/blogs/machine-learning/aws-inferentia2-builds-on-aws-inferentia1-by-delivering-4x-higher-throughput-and-10x-lower-latency/
    # Numbers are halved as the above flops is per chip and each chip appears as 2 devices.
    "trn1": {
        "fp32": 47.5e12 / 2,
        "tf32": 47.5e12 / 2,
        "fp16": 190e12 / 2,
        "bf16": 190e12 / 2,
        "int8": 380e12 / 2,
    },
}


def normalize_device_type(device_type: str) -> str:
    """Normalize device type string to Fray naming convention (lowercase).

    Handles fray-specific formats:
    - TPU type strings: "v4-128" -> "v4", "v5litepod-16" -> "v5litepod"
    - Direct names are lowercased: "H100" -> "h100"

    JAX device kind strings should be normalized by the caller (e.g., Levanter)
    before passing to this function.

    Returns:
        Normalized lowercase device type key for DEVICE_FLOPS lookup.
    """
    kind = device_type.lower()

    # Handle TPU type strings like "v4-128" -> "v4", "v5litepod-16" -> "v5litepod"
    if kind.startswith("v") and "-" in kind:
        return kind.split("-")[0]

    return kind


def jax_device_kind_to_fray_device_type(kind: str) -> str:
    """Normalize JAX device kind to Fray device type key.

    Converts JAX device_kind strings (e.g., "NVIDIA H100 80GB HBM3", "TPU v4")
    to Fray device type keys (e.g., "h100", "v4").

    Args:
        kind: JAX device kind string from device.device_kind

    Returns:
        Fray device type key suitable for DEVICE_FLOPS lookup
    """
    kind = kind.lower()

    # TPU looks like 'tpu v4', 'tpu v5 lite', etc.
    if kind.startswith("tpu "):
        tpu_gen = kind[4:].strip()
        # "v5 lite" -> "v5litepod"
        if "5" in tpu_gen:
            if "lite" in tpu_gen:
                return "v5litepod"
            else:
                return "v5p"
        # "v6 lite" -> "v6e"
        if "6" in tpu_gen and "lite" in tpu_gen:
            return "v6e"
        # "v4" -> "v4"
        return tpu_gen.replace(" ", "")

    # NVIDIA GPUs - check more specific patterns first
    if "h100" in kind and ("sxm" in kind or "hbm3" in kind):
        return "h100"
    if "h100" in kind:
        return "h100-pcie"
    if "h200" in kind:
        return "h200"
    if "a100" in kind:
        return "a100"
    if "a10g" in kind:
        return "a10g"
    if "a10" in kind:
        return "a10"
    if "a40" in kind:
        return "a40"
    if "v100" in kind and "sxm" in kind:
        return "v100-sxm"
    if "v100s" in kind:
        return "v100s"
    if "v100" in kind:
        return "v100"
    if "t4" in kind:
        return "t4"
    if "a6000" in kind:
        return "a6000"
    if "l40s" in kind:
        return "l40s"
    if "l4" in kind:
        return "l4"
    if "gb10" in kind:
        return "gb10"

    return kind


def normalize_dtype(dtype: str) -> str:
    """Normalize dtype string to base form.

    Strips 'amp_' prefix if present, converts to lowercase.
    """
    dtype = dtype.lower()
    if dtype.startswith("amp_"):
        return dtype[4:]
    return dtype


def device_flops(device_type: str, dtype: str = "bf16") -> float | None:
    """Get peak FLOP/s for a device.

    Args:
        device_type: Fray device type (e.g., "v4", "h100", "a100")
        dtype: Data type (e.g., "bf16", "fp16", "int8")

    Returns:
        Peak FLOP/s for a single device/chip, or None if unknown
    """
    normalized = normalize_device_type(device_type)
    flops_dict = DEVICE_FLOPS.get(normalized)
    if flops_dict is None:
        logger.warning(f"Unknown device type: {device_type} - {normalized}")
        return None

    normalized_dtype = normalize_dtype(dtype)
    if normalized_dtype not in flops_dict:
        logger.warning(f"Unknown dtype: {dtype} - {normalized_dtype} for device {device_type}")
        return None
    return flops_dict.get(normalized_dtype)


def device_flops_for_jax_device(jax_device_kind: str, dtype: str = "bf16") -> float | None:
    """Get peak FLOP/s given a JAX device kind string.

    Args:
        jax_device_kind: JAX device kind string (e.g., "TPU v4", "NVIDIA H100 80GB HBM3")
        dtype: Data type (e.g., "bf16", "fp16")

    Returns:
        Peak FLOP/s, or None if device/dtype unknown
    """
    fray_device_type = jax_device_kind_to_fray_device_type(jax_device_kind)
    return device_flops(fray_device_type, dtype)
