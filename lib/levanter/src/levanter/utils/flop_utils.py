# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import jax
from jax import numpy as jnp


def lm_flops_per_token(
    hidden_dim: int,
    intermediate_dim: int,
    num_layers: int,
    num_kv_heads: int,
    num_heads: int,
    seq_len: int,
    vocab_size: int,
    glu: bool,
    num_experts: int = 1,
    num_shared_experts: int = 0,
    num_experts_per_tok: int = 1,
):
    head_dim = hidden_dim / num_heads
    mlp = 2 * (3 if glu else 2) * hidden_dim * intermediate_dim * (num_experts_per_tok + num_shared_experts)
    if num_experts > 1:
        mlp += 2 * hidden_dim * num_experts  # router layer
    qkv_proj = 2 * hidden_dim * (num_heads * head_dim + 2 * num_kv_heads * head_dim)
    dense_proj = 2 * hidden_dim * hidden_dim
    # The following are across the whole sequence
    # assume full attention map like megatron-lm
    key_query_logits = 2 * seq_len**2 * num_heads * head_dim
    mask = 3 * seq_len * seq_len * num_heads
    mask_value = 2 * seq_len * seq_len * head_dim * num_heads
    seq_flops = key_query_logits + mask + mask_value
    # so we divide by the sequence length to get the per-token flops
    attn = seq_flops / seq_len
    lm_head = 2 * hidden_dim * vocab_size
    return num_layers * (mlp + qkv_proj + dense_proj + attn) + lm_head


# Copied/extended from Mosaic SPDX-License-Identifier: Apache-2.0
# https://github.com/mosaicml/composer/blob/56ccc2ebc59a8c68a6d075c4b61735ebf089b5a2/composer/callbacks/speed_monitor.py#L23
DEVICE_AVAILABLE_FLOPS = {
    # source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    # nvidia publishes spec sheet with a 2x sparsity factor
    "h100-sxm": {
        "fp64": 67e12,
        "fp32": 67e12,
        "tf32": 989e12 / 2,
        "fp16": 1.979e15 / 2,
        "amp_fp16": 1.979e15 / 2,
        "bf16": 1.979e15 / 2,
        "amp_bf16": 1.979e15 / 2,
        "fp8": 3.958e15 / 2,
        "amp_fp8": 3.958e15 / 2,
        "int8": 3.958e15 / 2,
    },
    # Source: https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23
    # nvidia publishes spec sheet with a 2x sparsity factor
    "h200-sxm": {
        "fp64": 67e12,
        "fp32": 67e12,
        "tf32": 989e12 / 2,
        "fp16": 1.979e15 / 2,
        "amp_fp16": 1.979e15 / 2,
        "bf16": 1.979e15 / 2,
        "amp_bf16": 1.979e15 / 2,
        "fp8": 3.958e15 / 2,
        "amp_fp8": 3.958e15 / 2,
        "int8": 3.958e15 / 2,
    },
    "h100-pcie": {
        "fp64": 51e12,
        "fp32": 51e12,
        "tf32": 756e12 / 2,
        "fp16": 1.513e15 / 2,
        "amp_fp16": 1.513e15 / 2,
        "bf16": 1.513e15 / 2,
        "amp_bf16": 1.513e15 / 2,
        "fp8": 3.026e15 / 2,
        "amp_fp8": 3.026e15 / 2,
        "int8": 3.026e15 / 2,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    "a100": {
        "fp64": 19.5e12,
        "fp32": 19.5e12,
        "tf32": 156e12,
        "fp16": 312e12,
        "amp_fp16": 312e12,
        "bf16": 312e12,
        "amp_bf16": 312e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
    "a10": {
        "fp32": 31.2e12,
        "tf32": 62.5e12,
        "fp16": 125e12,
        "amp_fp16": 125e12,
        "bf16": 125e12,
        "amp_bf16": 125e12,
    },
    # source: https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf
    "a40": {
        "fp32": 37.4e12,
        "tf32": 74.8e12,
        "fp16": 149.7e12,
        "amp_fp16": 149.7e12,
        "bf16": 149.7e12,
        "amp_bf16": 149.7e12,
    },
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    "v100-sxm": {
        "fp64": 7.8e12,
        "fp32": 15.7e12,
        "fp16": 125e12,
        "amp_fp16": 125e12,
    },
    "v100-pcie": {
        "fp64": 7e12,
        "fp32": 14e12,
        "fp16": 112e12,
        "amp_fp16": 112e12,
    },
    "v100s-pcie": {
        "fp64": 8.2e12,
        "fp32": 16.4e12,
        "fp16": 130e12,
        "amp_fp16": 130e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    # sxm and pcie have same flop counts
    "t4": {
        "fp32": 8.1e12,
        "fp16": 65e12,
        "amp_fp16": 65e12,
        "int8": 130e12,
        "int4": 260e12,
    },
    # source: https://aws.amazon.com/blogs/machine-learning/aws-inferentia2-builds-on-aws-inferentia1-by-delivering-4x-higher-throughput-and-10x-lower-latency/
    # Numbers are halved as the above flops is per chip and each chip appears as 2 devices.
    "trn1": {
        "fp32": 47.5e12 / 2,
        "tf32": 47.5e12 / 2,
        "fp16": 190e12 / 2,
        "amp_fp16": 190e12 / 2,
        "bf16": 190e12 / 2,
        "amp_bf16": 190e12 / 2,
        "int8": 380e12 / 2,
    },
    # Source: https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf
    "a6000": {
        "fp32": 38.7e12 / 2,
        "tf32": 309.7e12 / 2,
        "fp16": 309.7e12 / 2,
        "bf16": 309.7e12 / 2,
    },
    # TPU
    # Source: https://cloud.google.com/tpu/docs/v3
    # v3 gives "per chip" flops, so we divide by 2 since jax device is a core
    "tpu v3": {
        "bf16": 123e12 / 2,
    },
    # Source: https://cloud.google.com/tpu/docs/v4
    # don't divide by 2 since jax v4 device is a single chip
    "tpu v4": {
        "bf16": 275e12,
        "int8": 275e12,
    },
    # Source: https://cloud.google.com/tpu/docs/v5e
    "tpu v5 lite": {
        "bf16": 197e12,
        "int8": 393e12,
    },
    # Source: https://cloud.google.com/tpu/docs/v5p
    "tpu v5": {
        "bf16": 459e12,
    },
    # Source: https://cloud.google.com/tpu/docs/v6e
    "tpu v6 lite": {
        "bf16": 918e12,
        "int8": 1836e12,
    },
}


def device_hardware_flops(device: jax.Device, dtype: jnp.dtype = jnp.bfloat16) -> Optional[float]:
    """
    Returns the hardware flops of a device. If the device doesn't support memory stats, returns None.
    Args:
        device:

    Returns:

    """
    kind = _simplify_device_kind(device.device_kind)

    dtype_str = _canonical_dtype(dtype)

    flop_dict = DEVICE_AVAILABLE_FLOPS.get(kind, None)

    if flop_dict is None:
        return None

    return flop_dict.get(dtype_str, None)


def _simplify_device_kind(kind: str) -> str:
    kind = kind.lower()

    # TPU looks like 'TPU v4'
    if kind.startswith("tpu"):
        return kind

    if "h100" in kind and ("sxm" in kind or "hbm3" in kind):
        return "h100-sxm"
    if "h100" in kind:
        return "h100-pcie"
    if "a100" in kind:
        return "a100"
    if "a10" in kind:
        return "a10"
    if "v100" in kind and "sxm" in kind:
        return "v100-sxm"
    if "v100" in kind:
        return "v100-pcie"
    if "t4" in kind:
        return "t4"
    if "a6000" in kind:
        return "a6000"

    return kind


def _canonical_dtype(dtype: jnp.dtype) -> str:
    if dtype == jnp.float64:
        return "fp64"
    if dtype == jnp.float32:
        return "fp32"
    if dtype == jnp.float16:
        return "fp16"
    if dtype == jnp.bfloat16:
        return "bf16"
    if dtype == jnp.int8:
        return "int8"
    if dtype == jnp.int4:
        return "int4"
    if dtype in [
        jnp.float8_e4m3b11fnuz,
        jnp.float8_e5m2,
        jnp.float8_e4m3fn,
        jnp.float8_e4m3fn,
        jnp.float8_e4m3fnuz,
        jnp.float8_e5m2fnuz,
    ]:
        return "fp8"

    raise ValueError(f"Unsupported dtype: {dtype}")
