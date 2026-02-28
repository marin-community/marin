# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import jax.numpy as jnp


DEFAULT_DEVICE_KEY = "default"


@dataclass(frozen=True, slots=True)
class ShapeBucket:
    """Shape bucket for attention block-size lookup."""

    name: str
    batch_min: int
    batch_max: int
    seq_min: int
    seq_max: int
    heads_min: int
    heads_max: int
    head_dim_min: int
    head_dim_max: int

    def matches(self, batch: int, seq_len: int, num_heads: int, head_dim: int) -> bool:
        return (
            self.batch_min <= batch <= self.batch_max
            and self.seq_min <= seq_len <= self.seq_max
            and self.heads_min <= num_heads <= self.heads_max
            and self.head_dim_min <= head_dim <= self.head_dim_max
        )


@dataclass(frozen=True, slots=True)
class AttentionBlockSizes:
    block_q: int
    block_k: int
    block_q_dkv: int
    block_kv_dkv: int
    block_q_dq: int
    block_kv_dq: int
    num_warps: int | None = None
    num_stages: int | None = None


# key:
#   - dtype_name
#   - shape_bucket
_TOKAMAX_LIKE_BLOCKS: dict[tuple[str, str], AttentionBlockSizes] = {
    ("bfloat16", "llama3-ish"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float16", "llama3-ish"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float32", "llama3-ish"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("bfloat16", "llama8b-ish"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float16", "llama8b-ish"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float32", "llama8b-ish"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("bfloat16", "qwen3-ish"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float16", "qwen3-ish"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float32", "qwen3-ish"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("bfloat16", "llama125m-high-batch"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float16", "llama125m-high-batch"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float32", "llama125m-high-batch"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("bfloat16", "llama125m-small-batch"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float16", "llama125m-small-batch"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float32", "llama125m-small-batch"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("bfloat16", "llama2-hd256"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float16", "llama2-hd256"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float32", "llama2-hd256"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("bfloat16", "default"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float16", "default"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
    ("float32", "default"): AttentionBlockSizes(128, 128, 32, 32, 32, 32),
}

_GB10_BLOCKS: dict[tuple[str, str], AttentionBlockSizes] = {
    ("bfloat16", "llama3-ish"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("float16", "llama3-ish"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("float32", "llama3-ish"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("bfloat16", "llama8b-ish"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("float16", "llama8b-ish"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("float32", "llama8b-ish"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("bfloat16", "qwen3-ish"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("float16", "qwen3-ish"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("float32", "qwen3-ish"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("bfloat16", "llama125m-high-batch"): AttentionBlockSizes(32, 32, 16, 64, 64, 16),
    ("float16", "llama125m-high-batch"): AttentionBlockSizes(32, 32, 16, 64, 64, 16),
    ("float32", "llama125m-high-batch"): AttentionBlockSizes(32, 32, 16, 64, 64, 16),
    ("bfloat16", "llama125m-small-batch"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("float16", "llama125m-small-batch"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("float32", "llama125m-small-batch"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("bfloat16", "llama2-hd256"): AttentionBlockSizes(32, 16, 16, 32, 32, 16, num_warps=8, num_stages=1),
    ("float16", "llama2-hd256"): AttentionBlockSizes(32, 16, 16, 32, 32, 16, num_warps=8, num_stages=1),
    ("float32", "llama2-hd256"): AttentionBlockSizes(32, 16, 16, 32, 32, 16, num_warps=8, num_stages=1),
    ("bfloat16", "default"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("float16", "default"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
    ("float32", "default"): AttentionBlockSizes(32, 32, 16, 32, 32, 16),
}


TUNED_BLOCK_SIZES: dict[str, dict[tuple[str, str], AttentionBlockSizes]] = {
    DEFAULT_DEVICE_KEY: _TOKAMAX_LIKE_BLOCKS,
    "NVIDIA GB10": _GB10_BLOCKS,
}

# Bucket order is significant: first matching bucket wins.
SHAPE_BUCKETS: list[ShapeBucket] = [
    ShapeBucket(
        name="llama3-ish",
        batch_min=1,
        batch_max=4,
        seq_min=2048,
        seq_max=8192,
        heads_min=24,
        heads_max=48,
        head_dim_min=128,
        head_dim_max=128,
    ),
    ShapeBucket(
        name="llama8b-ish",
        batch_min=1,
        batch_max=4,
        seq_min=2048,
        seq_max=8192,
        heads_min=24,
        heads_max=40,
        head_dim_min=128,
        head_dim_max=128,
    ),
    ShapeBucket(
        name="llama2-hd256",
        batch_min=1,
        batch_max=16,
        seq_min=1024,
        seq_max=4096,
        heads_min=1,
        heads_max=16,
        head_dim_min=256,
        head_dim_max=256,
    ),
    ShapeBucket(
        name="qwen3-ish",
        batch_min=1,
        batch_max=8,
        seq_min=1024,
        seq_max=8192,
        heads_min=4,
        heads_max=64,
        head_dim_min=128,
        head_dim_max=256,
    ),
    ShapeBucket(
        name="llama125m-high-batch",
        batch_min=8,
        batch_max=64,
        seq_min=1024,
        seq_max=4096,
        heads_min=4,
        heads_max=16,
        head_dim_min=64,
        head_dim_max=256,
    ),
    ShapeBucket(
        name="llama125m-small-batch",
        batch_min=1,
        batch_max=7,
        seq_min=1024,
        seq_max=4096,
        heads_min=4,
        heads_max=16,
        head_dim_min=64,
        head_dim_max=256,
    ),
    ShapeBucket(
        name="default",
        batch_min=1,
        batch_max=128,
        seq_min=1,
        seq_max=16384,
        heads_min=1,
        heads_max=128,
        head_dim_min=64,
        head_dim_max=256,
    ),
]


def _device_key(device_kind: str | None) -> str | None:
    if device_kind is None and jax.devices():
        device_kind = jax.devices()[0].device_kind
    if not device_kind:
        return None
    norm = str(device_kind).strip()
    if "gb10" in norm.lower():
        return "NVIDIA GB10"
    return norm


def _dtype_name(dtype: jnp.dtype | None) -> str | None:
    if dtype is None:
        return None
    return jnp.dtype(dtype).name


def _shape_bucket(batch: int, seq_len: int, num_heads: int, head_dim: int) -> str | None:
    for bucket in SHAPE_BUCKETS:
        if bucket.matches(batch, seq_len, num_heads, head_dim):
            return bucket.name
    return None


def infer_block_sizes(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    *,
    dtype: jnp.dtype | None,
    device_kind: str | None = None,
) -> AttentionBlockSizes:
    """Infer attention block sizes from a tuned table.

    Args:
        batch: Attention batch size.
        seq_len: Query/key sequence length.
        num_heads: Query heads after optional GQA expansion.
        head_dim: Attention head dimension.
        dtype: Logit/activation dtype.
        device_kind: Optional `jax.Device.device_kind` string.

    Returns:
        Tuned block sizes for the requested shape.
    """
    dtype_name = _dtype_name(dtype)
    bucket = _shape_bucket(batch, seq_len, num_heads, head_dim)
    device_key = _device_key(device_kind)
    dtype_names = [dtype_name] if dtype_name is not None else []
    if dtype_name in {"float16", "half"}:
        dtype_names.append("bfloat16")
    if dtype_name is not None and dtype_name not in {"float16", "half", "bfloat16", "float32"}:
        dtype_names.append("bfloat16")
    if "bfloat16" not in dtype_names:
        dtype_names.append("bfloat16")
    if "float32" not in dtype_names:
        dtype_names.append("float32")

    if bucket is not None:
        for key in (device_key, DEFAULT_DEVICE_KEY):
            for candidate_dtype in dtype_names:
                entry = TUNED_BLOCK_SIZES.get(key, {}).get((candidate_dtype, bucket))
                if entry is not None:
                    return entry

    for key in (device_key, DEFAULT_DEVICE_KEY):
        for candidate_dtype in dtype_names:
            entry = TUNED_BLOCK_SIZES.get(key, {}).get((candidate_dtype, "default"))
            if entry is not None:
                return entry

    return TUNED_BLOCK_SIZES[DEFAULT_DEVICE_KEY][("bfloat16", "default")]


__all__ = [
    "AttentionBlockSizes",
    "DEFAULT_DEVICE_KEY",
    "SHAPE_BUCKETS",
    "ShapeBucket",
    "TUNED_BLOCK_SIZES",
    "infer_block_sizes",
]
