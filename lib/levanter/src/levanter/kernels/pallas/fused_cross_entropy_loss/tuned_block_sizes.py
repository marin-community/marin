# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import os
from typing import Optional
import warnings

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
    "NVIDIA": {
        ("bfloat16", "tiny"): BlockSizes(b_block_size=64, h_block_size=64, v_block_size=128),
        ("float32", "tiny"): BlockSizes(b_block_size=64, h_block_size=64, v_block_size=128),
        ("bfloat16", "small-vocab"): BlockSizes(b_block_size=128, h_block_size=256, v_block_size=1024),
        ("float32", "small-vocab"): BlockSizes(b_block_size=128, h_block_size=256, v_block_size=1024),
        ("bfloat16", "llama3-ish"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=1024),
        ("float32", "llama3-ish"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=1024),
        ("bfloat16", "large-batch-small-h"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=2048),
        ("float32", "large-batch-small-h"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=2048),
        ("bfloat16", "medium-batch-medium-h"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=2048),
        ("float32", "medium-batch-medium-h"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=2048),
    },
    "NVIDIA GB10": {
        ("bfloat16", "tiny"): BlockSizes(b_block_size=32, h_block_size=64, v_block_size=128),
        ("float32", "tiny"): BlockSizes(b_block_size=32, h_block_size=64, v_block_size=128),
        ("bfloat16", "small-vocab"): BlockSizes(b_block_size=32, h_block_size=64, v_block_size=256),
        ("float32", "small-vocab"): BlockSizes(b_block_size=32, h_block_size=64, v_block_size=256),
        ("bfloat16", "small-h-small-vocab"): BlockSizes(b_block_size=32, h_block_size=64, v_block_size=128),
        ("float32", "small-h-small-vocab"): BlockSizes(b_block_size=32, h_block_size=64, v_block_size=128),
        ("bfloat16", "gb10-large-vocab-mid-batch"): BlockSizes(b_block_size=1024, h_block_size=32, v_block_size=1024),
        ("float32", "gb10-large-vocab-mid-batch"): BlockSizes(b_block_size=1024, h_block_size=32, v_block_size=1024),
        ("bfloat16", "gb10-fallback"): BlockSizes(b_block_size=32, h_block_size=128, v_block_size=64),
        ("float32", "gb10-fallback"): BlockSizes(b_block_size=32, h_block_size=128, v_block_size=64),
    },
    "NVIDIA H100": {
        ("bfloat16", "tiny"): BlockSizes(b_block_size=64, h_block_size=64, v_block_size=128),
        ("float32", "tiny"): BlockSizes(b_block_size=64, h_block_size=64, v_block_size=128),
        ("bfloat16", "small-vocab"): BlockSizes(b_block_size=128, h_block_size=256, v_block_size=1024),
        ("float32", "small-vocab"): BlockSizes(b_block_size=128, h_block_size=256, v_block_size=1024),
        ("bfloat16", "llama3-ish"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=1024),
        ("float32", "llama3-ish"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=1024),
        ("bfloat16", "large-batch-small-h"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=2048),
        ("float32", "large-batch-small-h"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=2048),
        ("bfloat16", "medium-batch-medium-h"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=2048),
        ("float32", "medium-batch-medium-h"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=2048),
    },
    "NVIDIA A100": {
        ("bfloat16", "tiny"): BlockSizes(b_block_size=64, h_block_size=64, v_block_size=128),
        ("float32", "tiny"): BlockSizes(b_block_size=64, h_block_size=64, v_block_size=128),
        ("bfloat16", "small-vocab"): BlockSizes(b_block_size=128, h_block_size=256, v_block_size=1024),
        ("float32", "small-vocab"): BlockSizes(b_block_size=128, h_block_size=256, v_block_size=1024),
        ("bfloat16", "llama3-ish"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=1024),
        ("float32", "llama3-ish"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=1024),
        ("bfloat16", "large-batch-small-h"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=2048),
        ("float32", "large-batch-small-h"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=2048),
        ("bfloat16", "medium-batch-medium-h"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=2048),
        ("float32", "medium-batch-medium-h"): BlockSizes(b_block_size=256, h_block_size=256, v_block_size=2048),
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
        ("bfloat16", "huge-batch-llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=256,
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
        ("float32", "huge-batch-llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=256,
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
        ("bfloat16", "huge-batch-llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=256,
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
        ("float32", "huge-batch-llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=256,
        ),
    },
    "TPU v4": {
        ("bfloat16", "small-vocab"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=1024,
        ),
        ("bfloat16", "llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("bfloat16", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=256,
        ),
        ("bfloat16", "medium-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("float32", "small-vocab"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=1024,
        ),
        ("float32", "llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("float32", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=256,
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
        name="tiny",
        b_min=1,
        b_max=512,
        h_min=1,
        h_max=1024,
        v_min=1,
        v_max=8192,
    ),
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
        name="small-h-small-vocab",
        b_min=2048,
        b_max=8192,
        h_min=1,
        h_max=256,
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
        name="gb10-large-vocab-mid-batch",
        b_min=2048,
        b_max=32768,
        h_min=512,
        h_max=1536,
        v_min=65536,
        v_max=262144,
    ),
    ShapeBucket(
        name="huge-batch-llama3-ish",
        b_min=65536,
        b_max=1048576,
        h_min=4096,
        h_max=4096,
        v_min=120000,
        v_max=131072,
    ),
    ShapeBucket(
        name="large-batch-small-h",
        b_min=32768,
        b_max=1048576,
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
    ShapeBucket(
        name="gb10-fallback",
        b_min=1,
        b_max=1_048_576,
        h_min=1,
        h_max=1_048_576,
        v_min=1,
        v_max=1_048_576,
    ),
]

_HUGE_BATCH_BUCKET = "huge-batch-llama3-ish"
_FAST_HUGE_BATCH_SOURCE_BUCKET = "llama3-ish"
_SCOPED_VMEM_LIMIT_ARG = "xla_tpu_scoped_vmem_limit_kib="
_WARNED_HUGE_BATCH_SAFE_FALLBACK = False
_TPU_LABEL_LAYOUT_DEVICE_KEYS = {"TPU v4", "TPU v5", "TPU v5p"}


def _is_tpu_device(device_key: Optional[str]) -> bool:
    return bool(device_key and device_key.startswith("TPU"))


def _device_key(device_kind: Optional[str]) -> Optional[str]:
    if device_kind is None and jax.devices():
        device_kind = jax.devices()[0].device_kind.lower()
    if not device_kind:
        return None
    device_kind = device_kind.lower()
    if "nvidia" in device_kind:
        if "gb10" in device_kind:
            return "NVIDIA GB10"
        if "h100" in device_kind:
            return "NVIDIA H100"
        if "a100" in device_kind:
            return "NVIDIA A100"
        return "NVIDIA"
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


def _has_scoped_vmem_limit_override() -> bool:
    init_args = os.environ.get("LIBTPU_INIT_ARGS", "")
    return _SCOPED_VMEM_LIMIT_ARG in init_args


def _warn_huge_batch_safe_fallback() -> None:
    global _WARNED_HUGE_BATCH_SAFE_FALLBACK
    if _WARNED_HUGE_BATCH_SAFE_FALLBACK:
        return
    _WARNED_HUGE_BATCH_SAFE_FALLBACK = True
    warnings.warn(
        "Using safer fused CE huge-batch block sizes (v_block_size=256) because "
        "LIBTPU_INIT_ARGS does not set xla_tpu_scoped_vmem_limit_kib. "
        "On TPU v5p, set --xla_tpu_scoped_vmem_limit_kib=50000 (or higher) in "
        "LIBTPU_INIT_ARGS to use the faster tuning.",
        RuntimeWarning,
        stacklevel=3,
    )


def _maybe_override_huge_batch_block_sizes(
    *,
    entry: BlockSizes,
    dtype_name: str,
    bucket: str,
    device_key: Optional[str],
) -> BlockSizes:
    if bucket != _HUGE_BATCH_BUCKET:
        return entry
    if not _has_scoped_vmem_limit_override():
        _warn_huge_batch_safe_fallback()
        return entry

    for key in (device_key, DEFAULT_DEVICE_KEY):
        if not key:
            continue
        fast_entry = TUNED_BLOCK_SIZES.get(key, {}).get((dtype_name, _FAST_HUGE_BATCH_SOURCE_BUCKET))
        if fast_entry is not None:
            return fast_entry
    return entry


def _largest_divisor_multiple_of_128(dim: int, preferred: int) -> int:
    """Return the largest multiple-of-128 divisor of `dim` up to `preferred`.

    If `dim` has no multiple-of-128 divisor (for example dim not divisible by 128),
    return `preferred` and let runtime validation/fallback handle unsupported cases.
    """
    upper = min(dim, preferred)
    upper -= upper % 128

    for block in range(upper, 127, -128):
        if dim % block == 0:
            return block

    return preferred


def _is_valid_for_pallas_shape(
    block_sizes: BlockSizes,
    *,
    b: int,
    h: int,
    device_key: Optional[str],
) -> bool:
    if _is_tpu_device(device_key):
        if block_sizes.b_block_size % 128 != 0 or block_sizes.h_block_size % 128 != 0:
            return False
        if b % block_sizes.b_block_size != 0 or h % block_sizes.h_block_size != 0:
            return False
        if device_key in _TPU_LABEL_LAYOUT_DEVICE_KEYS and b >= 1024 and block_sizes.b_block_size % 1024 != 0:
            return False
        return True

    if block_sizes.b_block_size <= 0 or block_sizes.h_block_size <= 0 or block_sizes.v_block_size <= 0:
        return False
    if device_key and device_key.startswith("NVIDIA"):
        if block_sizes.b_block_size < 16 or block_sizes.h_block_size < 16 or block_sizes.v_block_size < 16:
            return False
        if block_sizes.b_block_size % 16 != 0 or block_sizes.h_block_size % 16 != 0:
            return False
    return True


def _sanitize_for_pallas(
    block_sizes: BlockSizes,
    *,
    b: int,
    h: int,
    device_key: Optional[str],
) -> BlockSizes:
    """Adjust inferred block sizes so B/H blocks divide local shapes when possible."""
    if not _is_tpu_device(device_key):
        return block_sizes
    b_block_size = _largest_divisor_multiple_of_128(b, block_sizes.b_block_size)
    h_block_size = _largest_divisor_multiple_of_128(h, block_sizes.h_block_size)
    return BlockSizes(
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=block_sizes.v_block_size,
    )


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
                entry = _maybe_override_huge_batch_block_sizes(
                    entry=entry,
                    dtype_name=dtype_name,
                    bucket=bucket,
                    device_key=device_key,
                )
                if _is_valid_for_pallas_shape(entry, b=b, h=h, device_key=device_key):
                    return entry

    default_entry = BlockSizes.get_default()
    if _is_valid_for_pallas_shape(default_entry, b=b, h=h, device_key=device_key):
        return default_entry
    return _sanitize_for_pallas(default_entry, b=b, h=h, device_key=device_key)


def infer_xla_v_block_size(
    b: int,
    h: int,
    v: int,
    *,
    dtype: Optional[jnp.dtype],
    device_kind: Optional[str] = None,
) -> int:
    """Heuristic v-block size for the XLA streaming path."""
    del b, h, dtype
    # Larger v-tiles improve throughput, but very large blocks can trigger OOM
    # in full training runs because backward materializes [B, v_block] temporaries.
    # Keep TPU v5p a bit larger (16384) and use 8192 elsewhere for safer memory.
    device_key = _device_key(device_kind)
    max_v_block_size = 16384 if device_key == "TPU v5p" else 8192
    target = min(v, max_v_block_size)
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
