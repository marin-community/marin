# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import safetensors
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.export_hf_bf16 import (
    apply_pending_qb_betas,
    inference_model,
    save_hf_bf16,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig as ArrayStackedGrugModelConfig
from experiments.june_tpu_67b_a2b.moe.model import Transformer as ArrayStackedTransformer


def test_save_hf_bf16_applies_pending_qb_and_inference_overrides(tmp_path: Path) -> None:
    checkpoint_config = ArrayStackedGrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=8,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=32,
        sliding_window=8,
        disable_pko=True,
        disable_long_rope=True,
        use_array_stacked_blocks=True,
    )
    inference_config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=8,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=262_144,
        sliding_window=8,
        qk_mult=1.57,
        disable_pko=True,
        disable_long_rope=True,
    )
    pending_qb_betas = jnp.asarray([[1.0, 3.0], [6.0, 2.0]], dtype=jnp.float32)

    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        checkpoint_model = ArrayStackedTransformer.init(checkpoint_config, key=jax.random.PRNGKey(0))
        checkpoint_model = apply_pending_qb_betas(checkpoint_model, pending_qb_betas)
        checkpoint_model = jax.tree.map(
            lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value,
            checkpoint_model,
        )
        save_hf_bf16(
            inference_model(checkpoint_model, inference_config),
            inference_config,
            str(tmp_path),
            tokenizer=None,
        )

    exported_config = json.loads((tmp_path / "config.json").read_text())
    assert exported_config["max_position_embeddings"] == 262_144
    assert exported_config["qk_mult"] == 1.57
    assert exported_config["dtype"] == "bfloat16"

    tensors: dict[str, np.ndarray] = {}
    tensor_dtypes: set[str] = set()
    for shard_path in tmp_path.glob("*.safetensors"):
        with safetensors.safe_open(shard_path, framework="numpy") as shard:
            tensor_dtypes.update(shard.get_slice(name).get_dtype() for name in shard.keys())
            if "model.layers.0.mlp.router.bias" in shard.keys():
                tensors["router_bias"] = shard.get_tensor("model.layers.0.mlp.router.bias")

    assert tensor_dtypes == {"BF16"}
    np.testing.assert_array_equal(tensors["router_bias"].astype(np.float32), np.asarray([1.0, -1.0]))
