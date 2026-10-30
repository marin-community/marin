# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.cpu_devices import run_on_cpu_devices
from safetensors.numpy import save_file

from experiments.grug.snowball_sft.import_hf import ImportConfig, convert, import_weights, verify_weights
from experiments.grug.snowball_sft.model import GrugModelConfig, Transformer, grugmoe_inference_state_dict


def test_hf_import_preserves_experts_through_native_restore(tmp_path):
    config = GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=24,
        shared_expert_intermediate_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=32,
        sliding_window=16,
        global_every=2,
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
        original = Transformer.init(config, key=jax.random.key(91))
        original = jax.tree.map(lambda a: a.astype(jnp.bfloat16), original)
        source = {
            name.replace(".shared_experts.0.", ".shared_expert."): value
            for name, value in grugmoe_inference_state_dict(original).items()
        }
        loaded = import_weights(config, source)
        for expected, actual in zip(jax.tree.leaves(original), jax.tree.leaves(loaded), strict=True):
            np.testing.assert_array_equal(expected, actual)
        broken = eqx.tree_at(
            lambda m: m.stacked_blocks.stacked.mlp.expert_mlp.w_gate,
            loaded,
            jnp.zeros_like(loaded.stacked_blocks.stacked.mlp.expert_mlp.w_gate),
        )
        with pytest.raises(ValueError, match=r"Tensor mismatch: model\.layers\.0\.mlp\.experts\.gate_proj\.weight"):
            verify_weights(broken, source)
        hf = tmp_path / "hf"
        hf.mkdir()
        save_file({name: np.asarray(value) for name, value in source.items()}, str(hf / "model.safetensors"))
    destination = tmp_path / "native"
    convert(ImportConfig(str(hf), None, config, str(destination)))
    verification = json.loads((destination / "verification.json").read_text())
    assert verification["exact_parity"] is True
    assert {row["name"] for row in verification["tensors"]} == set(source)


@pytest.mark.timeout(180)
def test_hf_import_verifies_different_staging_and_model_shardings():
    run_on_cpu_devices(
        """
        import jax
        from levanter.grug.sharding import compact_grug_mesh
        from levanter.utils.jax_utils import best_effort_sharding
        from experiments.grug.snowball_sft.import_hf import import_weights, verify_weights
        from experiments.grug.snowball_sft.model import GrugModelConfig, Transformer, grugmoe_inference_state_dict

        config = GrugModelConfig(
            vocab_size=32, hidden_dim=16, intermediate_dim=24, shared_expert_intermediate_dim=32,
            num_experts=4, num_experts_per_token=2, num_layers=2, num_heads=2, num_kv_heads=1,
            head_dim=8, max_seq_len=32, sliding_window=16, global_every=2,
        )
        mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
        with jax.set_mesh(mesh):
            original = Transformer.init(config, key=jax.random.key(29))
            source = {
                name.replace('.shared_experts.0.', '.shared_expert.'):
                    jax.sharding.reshard(value, best_effort_sharding(value.shape, mesh=mesh))
                for name, value in grugmoe_inference_state_dict(original).items()
            }
            imported = import_weights(config, source)
            records = verify_weights(imported, source)
            assert {record['name'] for record in records} == set(source)
        """,
        device_count=4,
    )
