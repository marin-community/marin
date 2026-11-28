# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU integration of strict restore, pinned optimizer, rebind and one stock update."""

from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import numpy as np
from levanter.checkpoint import save_checkpoint
from levanter.data.text.examples import GrugLmExample

from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.post_training.long_loop.runs.e12 import carried_restore
from experiments.post_training.long_loop.sft_runtime.model import GrugModelConfig
from experiments.post_training.long_loop.sft_runtime.state_rebind import rebind_first_state
from experiments.post_training.long_loop.sft_runtime.train import (
    _make_train_step,
    initial_state,
    reset_carried_trainer_step,
    restore_template_from,
)


def test_pinned_checkpoint_roundtrip_then_stock_update(tmp_path, monkeypatch):
    # Tiny CPU fixture tests the real SFT optimizer/step, not GPU SYRK or 64-device behavior.
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 5,
    )
    model = GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=4.0,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
    )
    with jax.set_mesh(mesh):
        optimizer = GrugMoeMuonHConfig(
            learning_rate=5e-6,
            adam_lr=5e-6,
            lr_schedule="constant",
            min_lr_ratio=1.0,
            warmup=0,
            decay=0,
            max_grad_norm=None,
            global_grad_norm=1.0,
            beta1=0.9062,
            beta2=0.95,
            epsilon=3.8339433005718795e-15,
            expert_normalization="all_experts",
            use_syrk=False,
            backend_steps=5,  # Real Newton-Schulz iterations; GPU-only SYRK is unavailable on CPU.
        ).build(15)
        mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
        state = jax.jit(
            lambda key: initial_state(model, optimizer=optimizer, mp=mp, key=key, ema_beta=None, offload_opt_state=True)
        )(jax.random.key(1))
        # A single-device replicated layout avoids size-one explicit-axis ambiguity on CPU.
        state = jax.tree.map(
            lambda x: jax.device_put(
                x,
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()).with_memory_kind(x.sharding.memory_kind),
            )
            if isinstance(x, jax.Array)
            else x,
            state,
        )
        assert all(
            x.sharding.memory_kind == "pinned_host" for x in jax.tree.leaves(state.opt_state) if isinstance(x, jax.Array)
        )
        legacy = carried_restore._shared(state, legacy=True)

        def populate(path, value):
            if not isinstance(value, jax.Array):
                return value
            name = jax.tree_util.keystr(path)
            if name.endswith(".step") or name.endswith(".count"):
                return jnp.full(value.shape, 161750, dtype=value.dtype)
            if "hyperparams[" in name:
                return jnp.full(value.shape, 5.0947383e-6, dtype=jnp.float32)
            if ".opt_state" in name and any(token in name for token in (".mu.", ".nu.", ".momentum_buffer.")):
                return jnp.full(value.shape, 0.01, dtype=jnp.float32)
            if value.dtype == jnp.bfloat16:
                return value.astype(jnp.float32) + jnp.float32(0.01)
            return value

        saved = jax.tree_util.tree_map_with_path(populate, legacy)
        expected_leaves = [np.asarray(jax.device_get(x)).copy() for x in jax.tree.leaves(saved)]
        checkpoint = tmp_path / "checkpoint"
        save_checkpoint(saved, step=161750, checkpoint_path=checkpoint, is_temporary=False)
        specs = carried_restore._array_specs(carried_restore._shared(state, legacy=True))
        monkeypatch.setattr(
            carried_restore, "_EXPECTED_MOMENT_LEAVES", sum(carried_restore._is_moment_path(p) for p in specs)
        )
        restored = carried_restore.restore_1005_carried_state(
            restore_template_from(state), str(checkpoint), mesh=mesh, expected_step=161750
        )
        actual_leaves = jax.tree.leaves(carried_restore._shared(restored, legacy=True))
        assert len(actual_leaves) == len(expected_leaves)
        for expected, actual in zip(expected_leaves, actual_leaves, strict=True):
            np.testing.assert_array_equal(np.asarray(actual), expected.astype(np.dtype(actual.dtype)))
        assert int(restored.opt_state.count) == 161750
        assert all(
            x.sharding.memory_kind == "pinned_host"
            for x in jax.tree.leaves(restored.opt_state)
            if isinstance(x, jax.Array)
        )
        restored = reset_carried_trainer_step(restored, load_checkpoint=True, allow_partial=False, expected_step=161750)
        tokens = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
        batch = GrugLmExample(tokens, jnp.ones_like(tokens, dtype=jnp.float32).at[:, -1].set(0))
        step = _make_train_step(optimizer, mp, z_loss_weight=0, ema_beta=None, offload_opt_state=True)
        compiled = step.lower(restored, batch).compile()
        samples = []
        rebound = rebind_first_state(
            restored,
            batch,
            cgroup_path=Path("/nonexistent"),
            observe=lambda phase, sample: samples.append((phase, sample)),
        )
        assert [phase for phase, _ in samples] == ["before_snapshot", "after_snapshot"]
        del compiled
        result, metrics, _ = step(rebound, batch)
        assert int(result.step) == 1
        assert int(result.opt_state.count) == 161751
        assert np.isfinite(float(metrics["train/loss"]))
        assert all(
            x.sharding.memory_kind == "pinned_host"
            for x in jax.tree.leaves(result.opt_state)
            if isinstance(x, jax.Array)
        )
