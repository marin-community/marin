# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from safetensors.numpy import save_file

from experiments.evaluation.grug_parity.reference import restore_hero_bf16
from experiments.evaluation.grug_parity.snowball_routers import extract
from experiments.evaluation.grug_parity.snowball_routers import restore as restore_snowball_routers
from experiments.grug.moe_hero_ep.ops.test_export_vllm import native_fixture


def test_reference_restores_master_weights_and_pending_router_update(tmp_path):
    request, original = native_fixture(str(tmp_path / "checkpoint"), master=True)
    manifest = {
        "checkpoint": {"uri": request.checkpoint, "metadata_digest": request.metadata_digest},
        "source_revision": request.source_revision,
    }
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        restored = restore_hero_bf16(manifest, request.model, mesh)
        expected = jax.tree.map(lambda value: np.asarray(value).astype(jnp.bfloat16), original)
        # The fixture stores bias 9 in both saved models. Pending QB replaces it.
        bias = np.asarray([[-0.5, 2.5, -3.5, 1.5], [5.5, 0.5, -0.5, -5.5]], dtype=jnp.bfloat16)
        for actual, wanted in zip(jax.tree.leaves(restored), jax.tree.leaves(expected), strict=True):
            if actual is restored.stacked_blocks.stacked.mlp.router_bias:
                wanted = bias
            np.testing.assert_array_equal(np.asarray(actual), wanted)
            assert actual.dtype == jnp.bfloat16


def test_snowball_reference_preserves_fp32_expert_selection(tmp_path):
    """Two distinct exported logits become a tie if the loader rounds the router."""
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    config = SnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        moe_implementation="scatter",
    )
    with jax.set_mesh(mesh):
        model = SnowballLMHeadModel.init(Axis("vocab", 32), config, key=jax.random.PRNGKey(0))
        rounded = jax.tree.map(lambda value: value.astype(jnp.bfloat16), model)
        weight = np.zeros((2, 16), dtype=np.float32)
        weight[:, 0] = [1.0001, 1.0002]
        path = tmp_path / "routers.npz"
        tensors = {"model.layers.0.mlp.router.weight": weight, "model.layers.0.mlp.router.bias": np.zeros(2, np.float32)}
        save_file(tensors, tmp_path / "model.safetensors")
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}})
        )
        source = tmp_path.as_uri() + "/"
        extract(source, "pinned", path)
        rounded = eqx.tree_at(
            lambda item: item.transformer.blocks[0].mlp.router,
            rounded,
            jnp.asarray(weight.T, dtype=jnp.bfloat16),
        )
        restored = restore_snowball_routers(rounded, path, source, "pinned")
        inputs = jnp.zeros((1, 1, 16), dtype=jnp.bfloat16).at[0, 0, 0].set(1)
        expected = restored.transformer.blocks[0].mlp.expert_mlp(
            inputs.reshape(1, 16),
            jnp.ones((1, 1), dtype=jnp.int32),
            jnp.full((1, 1), 2.5, dtype=jnp.bfloat16),
            mesh=mesh.abstract_mesh,
        )
        actual = restored.transformer.blocks[0].mlp(inputs)
        np.testing.assert_array_equal(np.asarray(actual).reshape(1, 16), np.asarray(expected))
        assert not np.array_equal(np.asarray(rounded.transformer.blocks[0].mlp(inputs)), np.asarray(actual))
