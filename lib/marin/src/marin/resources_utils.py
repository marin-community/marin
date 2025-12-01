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

# https://github.com/stanford-crfm/levanter/blob/162ef5f321cfc2e14b1b9c1652e2cffd42b395e2/src/levanter/utils/flop_utils.py#L40

import logging
from typing import Literal, TypeAlias

from levanter.utils.flop_utils import DEVICE_AVAILABLE_FLOPS

logger = logging.getLogger("ray")

# Our ancient version of Ray omits too many accelerator types from this module to be useful (e.g. H100)
# # ray just declares a bunch of constants, so we read them out via reflection
# _ACCEL_TYPES: list[str] = [
#     getattr(ray_accel_types, name) for name in dir(ray_accel_types) if name.isupper() and name != "TPU"
# ]
# assert all(isinstance(x, str) for x in _ACCEL_TYPES), "Expected all accelerator types to be strings"
#
# AcceleratorType: TypeAlias = Literal[tuple(_ACCEL_TYPES)]
# """
# https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html
# """
AcceleratorType: TypeAlias = str

ray_device_name_to_jax_name_map: dict[AcceleratorType, str] = {
    "H100": "h100-sxm",
    "H200": "h200-sxm",
    "H100-PCIE": "h100-pcie",
    "A100": "a100",
    "A100-40G": "a100",
    "A100-80G": "a100",
    "A10G": "a10",
    "A40": "a40",
    "V100": "v100-pcie",
    "V100-SXM": "v100-sxm",
    "V100S-PCIE": "v100s-pcie",
    "T4": "t4",
    "A6000": "a6000",
    "TRN1": "trn1",
    "TPU-V3": "tpu v3",
    "TPU-V4": "tpu v4",
    "TPU-V5LITEPOD": "tpu v5 lite",
    "TPU-V5P": "tpu v5",
    "TPU-V6E": "tpu v6 lite",
    "P100": "p100",
    "P4": "p4",
    "K80": "k80",
    "L4": "l4",
    "L40S": "l40s",
    "H20": "h20",
    "GB10": "gb10",
    # Entries with no DEVICE_AVAILABLE_FLOPS mapping
    # "Intel-GPU-Max-1550": "none",
    # "Intel-GPU-Max-1100": "none",
    # "Intel-GAUDI": "none",
    # "AMD-Instinct-MI100": "none",
    # "AMD-Instinct-MI250X": "none",
    # "AMD-Instinct-MI250X-MI250": "none",
    # "AMD-Instinct-MI210": "none",
    # "AMD-Instinct-MI300X-OAM": "none",
    # "AMD-Radeon-R9-200-HD-7900": "none",
    # "AMD-Radeon-HD-7900": "none",
    # "aws-neuron-core": "none",
    # "Ascend910B": "none",
    # "Ascend910B4": "none",
}

FlopDtype = Literal["bf16", "fp16", "fp32", "int8", "int4", "fp8"]


def flop_count_per_device_from_accel_type(accel_type: AcceleratorType, dtype: FlopDtype = "bf16") -> float | None:
    """
    Get the FLOPS for a given accelerator type and dtype.
    Args:
        accel_type: Accelerator type (e.g. "A100-40G") in Ray's format
        dtype: Data type (e.g. "bfloat16")
    Returns:
        FLOPS for the given accelerator type and dtype
    """
    jax_name = ray_device_name_to_jax_name_map.get(accel_type)
    if jax_name is None:
        logger.warning(f"Unknown accelerator type: {accel_type}. No FLOPS data available.")
        return None

    # Map dtype to FLOPS
    flops_per_device = DEVICE_AVAILABLE_FLOPS.get(jax_name, {}).get(dtype)

    if flops_per_device is None:
        logger.warning(f"Unknown dtype: {dtype} for accelerator type: {accel_type}. No FLOPS data available.")
        return None

    return flops_per_device


def get_tpu_type_and_chips(tpu_slice: str) -> tuple[str, int]:
    """Extract TPU type and number of chips from TPU name.

    Args:
        tpu_slice: TPU name like 'v4-128' or 'v5litepod-64'

    Returns:
        Tuple of (tpu_type, num_chips)

    Raises:
        ValueError: If tpu_name is not in expected format
    """
    try:
        # split by first - since count is always at the end
        tpu_type, suffix = tpu_slice.lower().split("-", maxsplit=1)

        logger.info(f"TPU type: {tpu_type}, suffix: {suffix}")

        # Map size to actual chip count
        # For v4/v5p pods the suffix encodes TensorCore count (two per chip), so halve it.
        if tpu_type in ("v4", "v5p"):
            num_chips = int(suffix) // 2
        else:
            # v5e/v6e suffixes already report chip counts directly.
            num_chips = int(suffix)

        return tpu_type, num_chips

    except ValueError as e:
        raise ValueError(f"Invalid TPU name format: {tpu_slice}. Expected format: <type>-<size> (e.g. v4-128)") from e


# We use gcloud's naming convention for TPU types (e.g. v4-128, v5litepod-64). JAX uses a different naming convention
gcloud_tpu_descriptor_to_jax_name: dict[str, str] = {
    "v3": "tpu v3",
    "v4": "tpu v4",
    "v5litepod": "tpu v5 lite",
    "v5": "tpu v5",
    "v5p": "tpu v5",
    "v6lite": "tpu v6 lite",
    "v6e": "tpu v6 lite",
}


def get_per_device_tpu_flops(tpu_generation: str, dtype: FlopDtype = "bf16") -> float:
    """Get per-device FLOPS for a TPU configuration.

    Args:
        tpu_generation: TPU name like 'v4" or "v5litepod'.  For convenience, we also accept 'v4-128' or 'v5litepod-64'.
        dtype: Data type (e.g. "bfloat16")

    Returns:
        Total FLOPS for the TPU configuration
    """
    # be nice and accept both v4-128 and v4
    tpu_generation = tpu_generation.split("-")[0]  # Extract the TPU type (e.g. v4, v5litepod)
    tpu_generation = tpu_generation.lower()

    jax_tpu_name = gcloud_tpu_descriptor_to_jax_name.get(tpu_generation)

    if jax_tpu_name is None:
        raise ValueError(f"Unknown TPU type: {tpu_generation}. No FLOPS data available.")

    # Map dtype to FLOPS
    flops_per_device = DEVICE_AVAILABLE_FLOPS.get(jax_tpu_name, {}).get(dtype)

    if flops_per_device is None:
        raise ValueError(f"Unknown dtype: {dtype} for TPU type: {tpu_generation}. No FLOPS data available.")

    return flops_per_device
