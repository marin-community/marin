# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from safetensors.numpy import save_file

from experiments.post_training.weight_analysis.checkpoint_io import SafetensorsCheckpoint
from experiments.post_training.weight_analysis.expert_metrics import seed_ceiling
from experiments.post_training.weight_analysis.master_deltas import fp32_master_delta_blocks


def test_fused_attention_and_experts_recover_hf_layout(tmp_path):
    """Sentinel rows distinguish grouped QKV, layers, experts, and gate/up halves."""
    tensors = {}
    masters = {
        "decoder.layers.self_attention.linear_qkv.weight": [],
        "decoder.layers.self_attention.attn_gate.weight": [],
        "decoder.layers.mlp.experts.experts.linear_fc1.weight": [],
        "decoder.layers.mlp.experts.experts.linear_fc2.weight": [],
    }
    expected = {}
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        q = (np.arange(32).reshape(8, 4) + layer * 200 + 1).astype(np.float32) / 100003
        k = (np.arange(16).reshape(4, 4) + layer * 200 + 51).astype(np.float32) / 100003
        v = (np.arange(16).reshape(4, 4) + layer * 200 + 81).astype(np.float32) / 100003
        packed = np.concatenate([q[:4], k[:2], v[:2], q[4:], k[2:], v[2:]])
        masters["decoder.layers.self_attention.linear_qkv.weight"].append(packed)
        gate = np.full((4, 4), (layer + 1) / 100003, dtype=np.float32)
        masters["decoder.layers.self_attention.attn_gate.weight"].append(gate)
        for part, value in [("q_proj", q), ("k_proj", k), ("v_proj", v), ("attn_gate", gate)]:
            name = f"{prefix}.self_attn.{part}.weight"
            tensors[name] = np.zeros_like(value)
            expected[(name, None)] = value
        gates = (np.arange(16).reshape(2, 2, 4) + layer * 200 + 101).astype(np.float32) / 100003
        ups = gates + np.float32(0.000321)
        downs = gates.transpose(0, 2, 1).copy() + np.float32(0.000123)
        masters["decoder.layers.mlp.experts.experts.linear_fc1.weight"].append(np.concatenate([gates, ups], axis=1))
        masters["decoder.layers.mlp.experts.experts.linear_fc2.weight"].append(downs)
        for part, value in [("gate_proj", gates), ("up_proj", ups), ("down_proj", downs)]:
            name = f"{prefix}.mlp.experts.{part}.weight"
            tensors[name] = np.zeros_like(value)
            for expert in range(2):
                expected[(name, (layer, expert))] = value[expert]
    masters = {name: np.stack(value) for name, value in masters.items()}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    blocks = list(fp32_master_delta_blocks(masters, masters.__getitem__, SafetensorsCheckpoint(str(tmp_path))))
    actual = {(b.name, b.expert): b.array for b in blocks}
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)


def test_seed_cosines_keep_updates_erased_by_bf16_rounding(tmp_path):
    base = np.ones((2, 2), dtype=np.float32)
    save_file({"model.embed_tokens.weight": base}, str(tmp_path / "model.safetensors"))
    checkpoint = SafetensorsCheckpoint(str(tmp_path))
    arrays = [
        base + np.array([[0.0001, 0], [0, 0]], dtype=np.float32),
        base + np.array([[0, 0.0001], [0, 0]], dtype=np.float32),
        base + np.array([[-0.0001, 0], [0, 0]], dtype=np.float32),
    ]
    records = [{"embedding.word_embeddings.weight": value} for value in arrays]
    result = seed_ceiling([fp32_master_delta_blocks(row, row.__getitem__, checkpoint) for row in records], near_zero=0.0)
    np.testing.assert_allclose(result.overall.pair_cosines, [0, -1, 0], rtol=0, atol=1e-12)
    assert result.overall.defined


def test_bf16_values_upcast_to_fp32_cannot_establish_master_updates(tmp_path):
    base = np.ones((2, 2), dtype=np.float32)
    save_file({"model.embed_tokens.weight": base}, str(tmp_path / "model.safetensors"))
    rounded = {"embedding.word_embeddings.weight": base.copy()}
    with pytest.raises(ValueError, match="BF16-representable"):
        list(fp32_master_delta_blocks(rounded, rounded.__getitem__, SafetensorsCheckpoint(str(tmp_path))))
