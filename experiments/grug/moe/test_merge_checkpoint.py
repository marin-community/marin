# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.expert_merge import AssignmentMode
from experiments.grug.moe.merge_checkpoint import (
    MergeCheckpointManifest,
    OnePairMergeCheckpointSpec,
    convert_grug_state_for_one_pair_merge,
    read_merge_checkpoint_manifest,
    write_merge_checkpoint_manifest,
)
from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.grug.moe.train import GrugTrainState


def _tiny_config(mapping: tuple[int, ...]) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=3,
        shared_expert_intermediate_dim=4,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=len(mapping),
        expert_bank_for_layer=mapping,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
        moe_implementation="scatter",
    )


def _source_state(model: Transformer) -> GrugTrainState:
    pending_qb_betas = jnp.arange(len(model.blocks) * model.config.num_experts, dtype=jnp.float32).reshape(
        len(model.blocks), model.config.num_experts
    )
    ema_params = jax.tree.map(lambda value: value + 0.25, model)
    return GrugTrainState(
        step=jnp.array(37, dtype=jnp.int32),
        params=model,
        opt_state={"stale_optimizer_state": jnp.array([-1.0])},
        ema_params=ema_params,
        pending_qb_betas=pending_qb_betas,
    )


def _spec() -> OnePairMergeCheckpointSpec:
    return OnePairMergeCheckpointSpec(
        representative_layer=2,
        source_layer=3,
        source_to_shared=(2, 0, 3, 1),
        assignment_mode=AssignmentMode.SPECTRAL,
        source_checkpoint="source/checkpoints/step-37",
        source_commit="0123456789abcdef",
        calibration_path="calibration/layers-2-3",
        cost_matrix_path="matching/spectral-cost.npy",
        probe_path="matching/probes",
        prefit_applied=False,
    )


def _fresh_optimizer_state(model: Transformer):
    return {"recovery_bank_count": jnp.array(len(model.expert_banks), dtype=jnp.int32)}


def test_state_conversion_resets_recovery_state_and_preserves_qb_semantics():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        source = _source_state(Transformer.init(_tiny_config(tuple(range(6))), key=jax.random.key(0)))
        converted = convert_grug_state_for_one_pair_merge(
            source,
            spec=_spec(),
            init_optimizer_state=_fresh_optimizer_state,
        )

    assert converted.state.params.config.resolved_expert_bank_for_layer == (0, 1, 2, 2, 3, 4)
    assert converted.state.ema_params is not None
    assert converted.state.ema_params.config.resolved_expert_bank_for_layer == (0, 1, 2, 2, 3, 4)
    assert int(converted.state.step) == 0
    assert set(converted.state.opt_state) == {"recovery_bank_count"}
    assert int(converted.state.opt_state["recovery_bank_count"]) == 5

    permutation = np.asarray(_spec().source_to_shared)
    source_layer = _spec().source_layer
    np.testing.assert_array_equal(
        converted.state.pending_qb_betas[source_layer][permutation],
        source.pending_qb_betas[source_layer],
    )
    np.testing.assert_array_equal(
        np.delete(converted.state.pending_qb_betas, source_layer, axis=0),
        np.delete(source.pending_qb_betas, source_layer, axis=0),
    )

    assert converted.manifest.source_step == 37
    assert converted.manifest.recovery_step == 0
    assert converted.manifest.source_topology == tuple(range(6))
    assert converted.manifest.target_topology == (0, 1, 2, 2, 3, 4)
    assert converted.manifest.ema_converted
    assert converted.manifest.optimizer_state_reset


def test_state_conversion_installs_prefitted_bank_and_resets_ema_from_params():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        source = _source_state(Transformer.init(_tiny_config(tuple(range(6))), key=jax.random.key(4)))
        representative_bank = source.params.expert_banks[2]
        prefitted_bank = dataclasses.replace(
            representative_bank,
            w_gate=representative_bank.w_gate + 0.125,
            w_up=representative_bank.w_up - 0.25,
        )
        converted = convert_grug_state_for_one_pair_merge(
            source,
            spec=dataclasses.replace(_spec(), prefit_applied=True),
            init_optimizer_state=_fresh_optimizer_state,
            shared_bank=prefitted_bank,
        )

    for actual, expected in zip(
        jax.tree.leaves(converted.state.params.expert_banks[2]),
        jax.tree.leaves(prefitted_bank),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert converted.state.ema_params is not None
    for actual, expected in zip(
        jax.tree.leaves(converted.state.ema_params),
        jax.tree.leaves(converted.state.params),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_state_conversion_requires_prefitted_bank_when_manifest_claims_prefit():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        source = _source_state(Transformer.init(_tiny_config(tuple(range(6))), key=jax.random.key(5)))

        with np.testing.assert_raises(ValueError):
            convert_grug_state_for_one_pair_merge(
                source,
                spec=dataclasses.replace(_spec(), prefit_applied=True),
                init_optimizer_state=_fresh_optimizer_state,
            )


def test_merge_checkpoint_manifest_and_state_round_trip(tmp_path):
    mesh = compact_grug_mesh(expert_axis_size=1)
    source_config = _tiny_config(tuple(range(6)))
    with jax.set_mesh(mesh):
        source = _source_state(Transformer.init(source_config, key=jax.random.key(1)))
        converted = convert_grug_state_for_one_pair_merge(
            source,
            spec=_spec(),
            init_optimizer_state=_fresh_optimizer_state,
            recovery_step=4,
        )
        save_checkpoint(converted.state, step=4, checkpoint_path=tmp_path)
        write_merge_checkpoint_manifest(str(tmp_path), converted.manifest)

        restored_manifest = read_merge_checkpoint_manifest(str(tmp_path))
        restored_config = dataclasses.replace(
            source_config,
            expert_bank_for_layer=restored_manifest.target_topology,
        )
        template_model = Transformer.init(restored_config, key=jax.random.key(2))
        template = GrugTrainState(
            step=jnp.array(0, dtype=jnp.int32),
            params=template_model,
            opt_state=_fresh_optimizer_state(template_model),
            ema_params=template_model,
            pending_qb_betas=jnp.zeros((6, 4), dtype=jnp.float32),
        )
        restored_state = load_checkpoint(template, checkpoint_path=tmp_path, mesh=mesh)

    assert restored_manifest == converted.manifest
    assert restored_manifest == MergeCheckpointManifest.from_dict(converted.manifest.to_dict())
    assert restored_state.params.config.resolved_expert_bank_for_layer == (0, 1, 2, 2, 3, 4)
    assert tuple(block.expert_bank_index for block in restored_state.params.blocks) == (0, 1, 2, 2, 3, 4)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(restored_state),
        jax.tree_util.tree_leaves(converted.state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
