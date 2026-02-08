# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from .config import BlockSizes


DEFAULT_DEVICE_KEY = "default"


@dataclass(frozen=True, slots=True)
class ShapeBucket:
    """Shape bucket for selecting tuned block sizes."""

    name: str
    b_min: int
    b_max: int
    h_min: int
    h_max: int
    v_min: int
    v_max: int

    def matches(self, b: int, h: int, v: int) -> bool:
        return self.b_min <= b <= self.b_max and self.h_min <= h <= self.h_max and self.v_min <= v <= self.v_max


# key:
#   - dtype_name
#   - shape_bucket
# value:
#   - BlockSizes
TUNED_BLOCK_SIZES: dict[str, dict[tuple[str, str], BlockSizes]] = {
    DEFAULT_DEVICE_KEY: {
        ("bfloat16", "large-batch-small-h"): BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=2048),
        ("float32", "large-batch-small-h"): BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=2048),
    },
    "TPU v5e": {
        ("bfloat16", "small-vocab"): BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=512),
        ("bfloat16", "llama3-ish"): BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=1024),
        ("bfloat16", "large-batch-small-h"): BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=2048),
        ("float32", "small-vocab"): BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=512),
        ("float32", "llama3-ish"): BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=1024),
        ("float32", "large-batch-small-h"): BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=2048),
    },
    "TPU v5p": {
        ("bfloat16", "small-vocab"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=512,
        ),
        ("bfloat16", "llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("bfloat16", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("bfloat16", "medium-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=2048,
        ),
        ("float32", "small-vocab"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=512,
        ),
        ("float32", "llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("float32", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("float32", "medium-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=2048,
        ),
    },
    "TPU v5": {
        ("bfloat16", "small-vocab"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=512,
        ),
        ("bfloat16", "llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("bfloat16", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("bfloat16", "medium-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=2048,
        ),
        ("float32", "small-vocab"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=512,
        ),
        ("float32", "llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("float32", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("float32", "medium-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=2048,
        ),
    },
    "TPU v4": {
        ("bfloat16", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("bfloat16", "medium-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("float32", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("float32", "medium-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
    },
}


SHAPE_BUCKETS: list[ShapeBucket] = [
    ShapeBucket(
        name="small-vocab",
        b_min=512,
        b_max=2048,
        h_min=256,
        h_max=1024,
        v_min=4096,
        v_max=16384,
    ),
    ShapeBucket(
        name="llama3-ish",
        b_min=4096,
        b_max=16384,
        h_min=4096,
        h_max=4096,
        v_min=120000,
        v_max=131072,
    ),
    ShapeBucket(
        name="large-batch-small-h",
        b_min=32768,
        b_max=131072,
        h_min=256,
        h_max=1024,
        v_min=120000,
        v_max=131072,
    ),
    ShapeBucket(
        name="medium-batch-medium-h",
        b_min=8192,
        b_max=32768,
        h_min=1536,
        h_max=3072,
        v_min=120000,
        v_max=131072,
    ),
]


def _device_key(device_kind: Optional[str]) -> Optional[str]:
    if device_kind is None and jax.devices():
        device_kind = jax.devices()[0].device_kind.lower()
    if not device_kind:
        return None
    device_kind = device_kind.lower()
    if "v6" in device_kind:
        return "TPU v6"
    if "v4" in device_kind:
        return "TPU v4"
    if "v5p" in device_kind:
        return "TPU v5p"
    if "v5e" in device_kind:
        return "TPU v5e"
    if "v5" in device_kind:
        return "TPU v5"
    return None


def _dtype_name(dtype: Optional[jnp.dtype]) -> Optional[str]:
    if dtype is None:
        return None
    return jnp.dtype(dtype).name


def _shape_bucket(b: int, h: int, v: int) -> Optional[str]:
    for bucket in SHAPE_BUCKETS:
        if bucket.matches(b, h, v):
            return bucket.name
    return None


def infer_block_sizes(
    b: int,
    h: int,
    v: int,
    *,
    dtype: Optional[jnp.dtype],
    device_kind: Optional[str] = None,
) -> BlockSizes:
    """Infer block sizes from a small tuned table.

    Args:
        b: Batch dimension.
        h: Hidden dimension.
        v: Vocabulary dimension.
        dtype: Computation dtype for logits/softmax.
        device_kind: Optional `jax.Device.device_kind` string.

    Returns:
        BlockSizes chosen from the tuned table, or the default if no match.
    """
    dtype_name = _dtype_name(dtype)
    device_key = _device_key(device_kind)
    bucket = _shape_bucket(b, h, v)

    if dtype_name and bucket:
        for key in (device_key, DEFAULT_DEVICE_KEY):
            if not key:
                continue
            entry = TUNED_BLOCK_SIZES.get(key, {}).get((dtype_name, bucket))
            if entry is not None:
                return entry

    return BlockSizes.get_default()


def infer_xla_v_block_size(
    b: int,
    h: int,
    v: int,
    *,
    dtype: Optional[jnp.dtype],
    device_kind: Optional[str] = None,
) -> int:
    """Heuristic v-block size for the XLA streaming path."""
    del b, h, dtype, device_kind  # currently unused
    target = min(v, 32768)
    if target <= 0:
        return 1
    # Keep the block size <= v to avoid excess padding work.
    if target == v:
        return target
    return max(128, 128 * (target // 128))


__all__ = [
    "DEFAULT_DEVICE_KEY",
    "ShapeBucket",
    "TUNED_BLOCK_SIZES",
    "SHAPE_BUCKETS",
    "infer_block_sizes",
    "infer_xla_v_block_size",
]
