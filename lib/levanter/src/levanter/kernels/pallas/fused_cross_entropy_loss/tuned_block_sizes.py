# Copyright The Levanter Authors
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
        ("bfloat16", "small-vocab"): BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=1024),
        ("bfloat16", "llama3-ish"): BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=512),
        ("bfloat16", "mid-h-large-vocab"): BlockSizes(b_block_size=8192, h_block_size=256, v_block_size=1024),
        ("bfloat16", "large-batch-small-h"): BlockSizes(b_block_size=4096, h_block_size=512, v_block_size=2048),
        ("bfloat16", "huge-batch-small-h"): BlockSizes(b_block_size=8192, h_block_size=256, v_block_size=1024),
        ("bfloat16", "medium-batch-medium-h"): BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=512),
        ("float32", "small-vocab"): BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=1024),
        ("float32", "llama3-ish"): BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=512),
        ("float32", "mid-h-large-vocab"): BlockSizes(b_block_size=8192, h_block_size=256, v_block_size=1024),
        ("float32", "large-batch-small-h"): BlockSizes(b_block_size=4096, h_block_size=512, v_block_size=2048),
        ("float32", "huge-batch-small-h"): BlockSizes(b_block_size=8192, h_block_size=256, v_block_size=1024),
        ("float32", "medium-batch-medium-h"): BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=512),
    },
    "TPU v6": {
        ("bfloat16", "small-vocab"): BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=2048),
        ("bfloat16", "llama3-ish"): BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=512),
        ("bfloat16", "mid-h-large-vocab"): BlockSizes(b_block_size=8192, h_block_size=1024, v_block_size=1024),
        ("bfloat16", "large-batch-small-h"): BlockSizes(b_block_size=4096, h_block_size=512, v_block_size=2048),
        ("bfloat16", "huge-batch-small-h"): BlockSizes(b_block_size=8192, h_block_size=256, v_block_size=1024),
        ("bfloat16", "medium-batch-medium-h"): BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=512),
        ("float32", "small-vocab"): BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=2048),
        ("float32", "llama3-ish"): BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=512),
        ("float32", "mid-h-large-vocab"): BlockSizes(b_block_size=8192, h_block_size=1024, v_block_size=1024),
        ("float32", "large-batch-small-h"): BlockSizes(b_block_size=4096, h_block_size=512, v_block_size=2048),
        ("float32", "huge-batch-small-h"): BlockSizes(b_block_size=8192, h_block_size=256, v_block_size=1024),
        ("float32", "medium-batch-medium-h"): BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=512),
    },
    "TPU v5p": {
        ("bfloat16", "small-vocab"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=1024,
        ),
        ("bfloat16", "llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=512,
        ),
        ("bfloat16", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=2048,
        ),
        ("bfloat16", "medium-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=512,
        ),
        ("bfloat16", "large-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=1024,
            v_block_size=768,
        ),
        ("bfloat16", "huge-batch-llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=256,
        ),
        ("float32", "small-vocab"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=1024,
        ),
        ("float32", "llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=512,
        ),
        ("float32", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=2048,
        ),
        ("float32", "medium-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=512,
        ),
        ("float32", "large-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=1024,
            v_block_size=768,
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
            v_block_size=1024,
        ),
        ("bfloat16", "llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=512,
        ),
        ("bfloat16", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=2048,
        ),
        ("bfloat16", "medium-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=512,
        ),
        ("bfloat16", "large-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=1024,
            v_block_size=768,
        ),
        ("bfloat16", "huge-batch-llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=256,
        ),
        ("float32", "small-vocab"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=1024,
        ),
        ("float32", "llama3-ish"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=512,
        ),
        ("float32", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=2048,
        ),
        ("float32", "medium-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=256,
            v_block_size=512,
        ),
        ("float32", "large-batch-medium-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=1024,
            v_block_size=768,
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
            v_block_size=512,
        ),
        ("bfloat16", "llama3-ish"): BlockSizes(
            b_block_size=8192,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("bfloat16", "mid-h-large-vocab"): BlockSizes(
            b_block_size=1024,
            h_block_size=1024,
            v_block_size=256,
        ),
        ("bfloat16", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=512,
        ),
        ("bfloat16", "huge-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=1024,
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
            v_block_size=512,
        ),
        ("float32", "llama3-ish"): BlockSizes(
            b_block_size=8192,
            h_block_size=512,
            v_block_size=1024,
        ),
        ("float32", "mid-h-large-vocab"): BlockSizes(
            b_block_size=1024,
            h_block_size=1024,
            v_block_size=256,
        ),
        ("float32", "large-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=512,
            v_block_size=512,
        ),
        ("float32", "huge-batch-small-h"): BlockSizes(
            b_block_size=1024,
            h_block_size=1024,
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
        name="mid-h-large-vocab",
        b_min=4096,
        b_max=32768,
        h_min=768,
        h_max=1536,
        v_min=120000,
        v_max=131072,
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
        name="huge-batch-small-h",
        b_min=131073,
        b_max=1048576,
        h_min=256,
        h_max=1024,
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
        name="large-batch-medium-h",
        b_min=32768,
        b_max=131072,
        h_min=1536,
        h_max=3072,
        v_min=120000,
        v_max=131072,
    ),
    ShapeBucket(
        name="medium-batch-medium-h",
        b_min=8192,
        b_max=32767,
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


def _is_tpu_device(device_key: Optional[str]) -> bool:
    return bool(device_key and device_key.startswith("TPU"))


def _normalized_device_kind(device_kind: Optional[str]) -> Optional[str]:
    if device_kind is None and jax.devices():
        device_kind = jax.devices()[0].device_kind
    if not device_kind:
        return None
    return device_kind.lower()


def _device_key(device_kind: Optional[str]) -> Optional[str]:
    device_kind = _normalized_device_kind(device_kind)
    if not device_kind:
        return None
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
    if "v5e" in device_kind or "v5lite" in device_kind or "v5 lite" in device_kind:
        return "TPU v5e"
    if "v5" in device_kind:
        return "TPU v5"
    if "tpu" in device_kind:
        return "TPU"
    return None


def _dtype_name(dtype: Optional[jnp.dtype]) -> Optional[str]:
    if dtype is None:
        return None
    return jnp.dtype(dtype).name


def widest_dtype_name(
    *,
    dtype: Optional[jnp.dtype],
    x_dtype: Optional[jnp.dtype] = None,
    w_dtype: Optional[jnp.dtype] = None,
) -> Optional[str]:
    """Return the widest dtype name among compute and operand dtypes."""
    candidates = [candidate for candidate in (dtype, x_dtype, w_dtype) if candidate is not None]
    if not candidates:
        return None
    widest = max((jnp.dtype(candidate) for candidate in candidates), key=lambda candidate: candidate.itemsize)
    return widest.name


def _shape_bucket(b: int, h: int, v: int, *, device_key: Optional[str]) -> Optional[str]:
    for bucket in SHAPE_BUCKETS:
        if bucket.name.startswith("gb10-") and device_key != "NVIDIA GB10":
            continue
        if bucket.matches(b, h, v):
            return bucket.name
    return None


def shape_bucket_name(
    b: int,
    h: int,
    v: int,
    *,
    device_kind: Optional[str] = None,
) -> Optional[str]:
    """Return the named shape bucket for a local B/H/V shape."""
    normalized_device_kind = _normalized_device_kind(device_kind)
    device_key = _device_key(normalized_device_kind)
    bucket = _shape_bucket(b, h, v, device_key=device_key)
    return _extend_tpu_v4_bucket_for_mid_vocab(bucket, b=b, h=h, v=v, device_key=device_key)


def _extend_tpu_v4_bucket_for_mid_vocab(
    bucket: Optional[str], *, b: int, h: int, v: int, device_key: Optional[str]
) -> Optional[str]:
    """Map mid-vocab TPU v4 shapes into existing tuned buckets.

    TPU v4 tuned data historically focused on small vocab and llama3-sized vocab.
    For intermediate vocab sizes (for example GPT-2 and Llama2), route shapes to
    the same regime buckets so infer chooses robust v4 block sizes instead of
    falling back to defaults.
    """
    if bucket is not None or device_key != "TPU v4":
        return bucket
    if not (16_385 <= v <= 119_999):
        return bucket

    candidate_buckets: list[str] = []
    if 512 <= b <= 2_048 and 256 <= h <= 1_024:
        candidate_buckets.append("small-vocab")
    if 4_096 <= b <= 16_384 and h == 4_096:
        candidate_buckets.append("llama3-ish")
    if 4_096 <= b <= 32_768 and 768 <= h <= 1_536:
        candidate_buckets.append("mid-h-large-vocab")
    if 131_073 <= b <= 1_048_576 and 256 <= h <= 1_024:
        candidate_buckets.append("huge-batch-small-h")
    if 32_768 <= b <= 131_072 and 256 <= h <= 1_024:
        candidate_buckets.append("large-batch-small-h")
    if 8_192 <= b <= 32_768 and 1_536 <= h <= 3_072:
        candidate_buckets.append("medium-batch-medium-h")

    # Only advertise a remapped bucket when the corresponding TPU v4 tuned entry
    # is actually valid for this local B/H shape.
    for candidate in candidate_buckets:
        for dtype_name in ("bfloat16", "float32"):
            entry = TUNED_BLOCK_SIZES["TPU v4"].get((dtype_name, candidate))
            if entry is None:
                continue
            if _is_valid_for_pallas_shape(
                entry,
                b=b,
                h=h,
                device_key=device_key,
                device_kind=device_key,
            ):
                return candidate
    return bucket


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


def _largest_divisor_multiple_of_1024(dim: int, preferred: int) -> int:
    """Return the largest multiple-of-1024 divisor of `dim` up to `preferred`."""
    upper = min(dim, preferred)
    upper -= upper % 1024

    for block in range(upper, 1023, -1024):
        if dim % block == 0:
            return block

    return preferred


def _largest_divisor_at_most(dim: int, preferred: int) -> int:
    """Return the largest divisor of `dim` up to `preferred`."""
    upper = min(dim, preferred)

    for block in range(upper, 0, -1):
        if dim % block == 0:
            return block

    return 1


def _is_valid_for_pallas_shape(
    block_sizes: BlockSizes,
    *,
    b: int,
    h: int,
    device_key: Optional[str],
    device_kind: Optional[str],
) -> bool:
    del device_kind
    if _is_tpu_device(device_key):
        if block_sizes.b_block_size % 128 != 0 or block_sizes.h_block_size % 128 != 0:
            return False
        if b % block_sizes.b_block_size != 0 or h % block_sizes.h_block_size != 0:
            return False
        if b >= 1024 and block_sizes.b_block_size % 1024 != 0:
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
    device_kind: Optional[str],
) -> BlockSizes:
    """Adjust inferred block sizes so B/H blocks divide local shapes when possible."""
    del device_kind
    if not _is_tpu_device(device_key):
        return block_sizes
    if b >= 1024:
        b_block_size = _largest_divisor_multiple_of_1024(b, block_sizes.b_block_size)
    else:
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
    x_dtype: Optional[jnp.dtype] = None,
    w_dtype: Optional[jnp.dtype] = None,
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
    block_sizes, _ = infer_block_sizes_with_tuned_match(
        b,
        h,
        v,
        dtype=dtype,
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        device_kind=device_kind,
    )
    return block_sizes


def infer_block_sizes_with_tuned_match(
    b: int,
    h: int,
    v: int,
    *,
    dtype: Optional[jnp.dtype],
    x_dtype: Optional[jnp.dtype] = None,
    w_dtype: Optional[jnp.dtype] = None,
    device_kind: Optional[str] = None,
) -> tuple[BlockSizes, bool]:
    """Infer block sizes and report whether they came from tuned lookup data."""
    normalized_device_kind = _normalized_device_kind(device_kind)
    dtype_name = widest_dtype_name(dtype=dtype, x_dtype=x_dtype, w_dtype=w_dtype)
    device_key = _device_key(normalized_device_kind)
    bucket = shape_bucket_name(b, h, v, device_kind=normalized_device_kind)

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
                if _is_valid_for_pallas_shape(
                    entry,
                    b=b,
                    h=h,
                    device_key=device_key,
                    device_kind=normalized_device_kind,
                ):
                    return entry, True

    default_entry = BlockSizes.get_default()
    if _is_valid_for_pallas_shape(
        default_entry,
        b=b,
        h=h,
        device_key=device_key,
        device_kind=normalized_device_kind,
    ):
        return default_entry, False
    return (
        _sanitize_for_pallas(
            default_entry,
            b=b,
            h=h,
            device_key=device_key,
            device_kind=normalized_device_kind,
        ),
        False,
    )


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


def infer_xla_b_block_size(b: int, v_block_size: int) -> int:
    """Heuristic batch block size for the XLA streaming path."""
    if b <= 0 or v_block_size <= 0:
        return 1

    max_b_block_size = min(b, (2**31 - 1) // v_block_size)
    if max_b_block_size <= 0:
        return 1

    preferred = max_b_block_size - (max_b_block_size % 128)
    if preferred >= 128:
        block_size = _largest_divisor_multiple_of_128(b, preferred)
        if block_size <= max_b_block_size and b % block_size == 0:
            return block_size

    return _largest_divisor_at_most(b, max_b_block_size)


__all__ = [
    "DEFAULT_DEVICE_KEY",
    "ShapeBucket",
    "TUNED_BLOCK_SIZES",
    "SHAPE_BUCKETS",
    "infer_block_sizes",
    "infer_block_sizes_with_tuned_match",
    "infer_xla_b_block_size",
    "infer_xla_v_block_size",
    "shape_bucket_name",
    "widest_dtype_name",
]
