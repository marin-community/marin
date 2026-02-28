# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from levanter.grug import attention as attention_mod
from levanter.kernels.pallas.attention import tuned_block_sizes as tuned_mod


class _FakeDevice:
    def __init__(self, device_kind: str):
        self.device_kind = device_kind


def test_infer_block_sizes_uses_gb10_hd256_profile():
    blocks = tuned_mod.infer_block_sizes(
        batch=8,
        seq_len=2048,
        num_heads=4,
        head_dim=256,
        dtype=jnp.bfloat16,
        device_kind="NVIDIA GB10",
    )

    assert blocks.block_q == 32
    assert blocks.block_k == 16
    assert blocks.block_q_dkv == 16
    assert blocks.block_kv_dkv == 32
    assert blocks.block_q_dq == 32
    assert blocks.block_kv_dq == 16
    assert blocks.num_warps == 8
    assert blocks.num_stages == 1


def test_shape_bucket_order_is_first_match_wins():
    # This shape overlaps llama3-ish/llama8b-ish/qwen3-ish; the first bucket in
    # SHAPE_BUCKETS must win so tuning remains deterministic.
    assert tuned_mod._shape_bucket(batch=2, seq_len=4096, num_heads=32, head_dim=128) == "llama3-ish"


def test_backend_block_size_keys_include_device_and_auto(monkeypatch):
    monkeypatch.setattr(attention_mod.jax, "devices", lambda: [_FakeDevice("NVIDIA GB10")])

    keys = attention_mod._backend_block_size_keys("pallas_gpu")

    assert keys[0] == "nvidia gb10"
    assert "gpu" in keys
    assert keys[-1] == "auto"


def test_backend_block_sizes_prefers_device_specific_entry(monkeypatch):
    monkeypatch.setattr(attention_mod.jax, "devices", lambda: [_FakeDevice("NVIDIA GB10")])

    gb10_cfg = attention_mod.BlockSizes(block_q=64, block_k=32)
    auto_cfg = attention_mod.BlockSizes(block_q=16, block_k=16)
    resolved = attention_mod._backend_block_sizes(
        {"auto": auto_cfg, "nvidia gb10": gb10_cfg},
        "pallas_gpu",
    )

    assert resolved == gb10_cfg
