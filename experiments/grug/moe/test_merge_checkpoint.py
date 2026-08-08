# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.expert_merge import AssignmentMode
from experiments.grug.moe.merge_checkpoint import (
    CapacityOracleKind,
    JointRefactorKind,
    JointRefactorProvenance,
    LayerAdapterKind,
    MergeCheckpointManifest,
    OnePairMergeCheckpointSpec,
    convert_grug_state_for_capacity_oracle_split,
    convert_grug_state_for_joint_refactor,
    convert_grug_state_for_layer_adapter,
    convert_grug_state_for_one_pair_merge,
    read_merge_checkpoint_manifest,
    write_merge_checkpoint_manifest,
)
from experiments.grug.moe.merge_recovery import RecoveryInitialization
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


def _joint_refactor_provenance() -> JointRefactorProvenance:
    return JointRefactorProvenance(
        kind=JointRefactorKind.CACHED_TRACE_HARD_TOP4,
        source_checkpoint="source/checkpoints/step-37",
        source_commit="0123456789abcdef",
        calibration_path="calibration/layers-2-3",
        calibration_artifact_version="2026.08.06",
        calibration_artifact_fingerprint="0dafe93d",
        trace_paths=("calibration/traces/layer_02.npz", "calibration/traces/layer_03.npz"),
        representative_layer=2,
        source_layer=3,
        source_topology=(0, 1, 2, 3, 4, 5),
        target_topology=(0, 1, 2, 2, 3, 4),
        correspondence_used=False,
        matching_path=None,
        router_initialization="unpermuted_source_columns",
        router_bias_trained=False,
        pending_qb_trained=False,
        optimizer="adamw",
        learning_rate=1e-4,
        weight_decay=0.0,
        optimizer_b1=0.9,
        optimizer_b2=0.999,
        optimizer_epsilon=1e-8,
        normalization_epsilon=1e-8,
        heldout_fraction=0.2,
        split_seeds=(2, 3),
        train_examples_per_layer=256,
        heldout_examples_per_layer=512,
        max_steps=2_000,
        eval_every=100,
        early_stopping_patience=5,
        seed=0,
        train_tokens_by_layer=(6_554, 6_554),
        heldout_tokens_by_layer=(1_638, 1_638),
        selected_refactor_step=700,
        best_heldout_loss=0.31,
        best_heldout_nrmse_by_layer=(0.5, 0.61),
        routing_entropy_by_layer=(5.4, 5.35),
        active_experts_by_layer=(4, 4),
        capacity_overflow_by_layer=(0.0, 0.0),
        output_step=0,
    )


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
    optimizer_state = cast("dict[str, Any]", converted.state.opt_state)
    assert int(optimizer_state["recovery_bank_count"]) == 5

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


def test_joint_refactor_conversion_installs_one_bank_and_unpermuted_routers():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        source = _source_state(Transformer.init(_tiny_config(tuple(range(6))), key=jax.random.key(15)))
        shared_bank = dataclasses.replace(
            source.params.expert_banks[2],
            w_gate=source.params.expert_banks[2].w_gate + 0.125,
        )
        routers = (
            dataclasses.replace(
                source.params.blocks[2].mlp,
                router=source.params.blocks[2].mlp.router + 0.25,
            ),
            dataclasses.replace(
                source.params.blocks[3].mlp,
                router=source.params.blocks[3].mlp.router - 0.5,
            ),
        )
        converted = convert_grug_state_for_joint_refactor(
            source,
            shared_bank=shared_bank,
            routers=routers,
            provenance=_joint_refactor_provenance(),
            init_optimizer_state=_fresh_optimizer_state,
        )

    assert converted.state.params.config.resolved_expert_bank_for_layer == (0, 1, 2, 2, 3, 4)
    assert len(converted.state.params.expert_banks) == 5
    for actual, expected in zip(
        jax.tree.leaves(converted.state.params.expert_banks[2]),
        jax.tree.leaves(shared_bank),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    for layer, expected in zip((2, 3), routers, strict=True):
        np.testing.assert_array_equal(converted.state.params.blocks[layer].mlp.router, expected.router)
        np.testing.assert_array_equal(converted.state.params.blocks[layer].mlp.router_bias, expected.router_bias)
    np.testing.assert_array_equal(converted.state.pending_qb_betas, source.pending_qb_betas)
    assert converted.manifest.spec.assignment_mode is AssignmentMode.IDENTITY
    assert converted.manifest.spec.source_to_shared == tuple(range(4))
    assert converted.manifest.spec.cost_matrix_path is None
    assert converted.manifest.spec.probe_path is None
    assert converted.manifest.joint_refactor == _joint_refactor_provenance()
    assert MergeCheckpointManifest.from_dict(converted.manifest.to_dict()) == converted.manifest


def _recovered_one_pair_state_and_manifest() -> tuple[GrugTrainState, MergeCheckpointManifest]:
    source = _source_state(Transformer.init(_tiny_config(tuple(range(6))), key=jax.random.key(6)))
    converted = convert_grug_state_for_one_pair_merge(
        source,
        spec=_spec(),
        init_optimizer_state=_fresh_optimizer_state,
        recovery_step=19,
    )
    recovered_state = dataclasses.replace(
        converted.state,
        step=jnp.array(1526, dtype=jnp.int32),
        opt_state={"recovered_optimizer_state": jnp.array([42.0])},
        pending_qb_betas=converted.state.pending_qb_betas + 0.5,
    )
    recovered_manifest = dataclasses.replace(
        converted.manifest,
        recovery_step=1526,
        recovery_initialization=RecoveryInitialization.LOCAL_RECOVERY,
        recovery_initial_checkpoint="stage-a/checkpoints/step-382",
    )
    return recovered_state, recovered_manifest


def test_capacity_oracle_split_is_functionally_identical_and_unties_source_bank():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        recovered, manifest = _recovered_one_pair_state_and_manifest()
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
        tied_logits = recovered.params.logits(tokens)
        oracle = convert_grug_state_for_capacity_oracle_split(
            recovered,
            source_manifest=manifest,
            source_checkpoint="recovery/checkpoints/step-1526",
            init_optimizer_state=_fresh_optimizer_state,
        )
        untied_logits = oracle.state.params.logits(tokens)

    assert oracle.state.params.config.resolved_expert_bank_for_layer == (0, 1, 2, 3, 4, 5)
    assert tuple(block.expert_bank_index for block in oracle.state.params.blocks) == (0, 1, 2, 3, 4, 5)
    assert len(oracle.state.params.expert_banks) == 6
    np.testing.assert_array_equal(untied_logits, tied_logits)

    representative_bank = oracle.state.params.expert_banks[2]
    duplicated_bank = oracle.state.params.expert_banks[3]
    for representative_leaf, duplicated_leaf in zip(
        jax.tree.leaves(representative_bank),
        jax.tree.leaves(duplicated_bank),
        strict=True,
    ):
        assert representative_leaf is not duplicated_leaf
        np.testing.assert_array_equal(duplicated_leaf, representative_leaf)

    np.testing.assert_array_equal(
        oracle.state.params.blocks[3].mlp.router,
        recovered.params.blocks[3].mlp.router,
    )
    np.testing.assert_array_equal(
        oracle.state.params.blocks[3].mlp.router_bias,
        recovered.params.blocks[3].mlp.router_bias,
    )
    assert oracle.state.pending_qb_betas is recovered.pending_qb_betas
    np.testing.assert_array_equal(oracle.state.pending_qb_betas, recovered.pending_qb_betas)
    assert int(oracle.state.step) == 0
    assert set(oracle.state.opt_state) == {"recovery_bank_count"}
    optimizer_state = cast("dict[str, Any]", oracle.state.opt_state)
    assert int(optimizer_state["recovery_bank_count"]) == 6


def test_capacity_oracle_split_records_diagnostic_provenance_and_converts_ema():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        recovered, manifest = _recovered_one_pair_state_and_manifest()
        oracle = convert_grug_state_for_capacity_oracle_split(
            recovered,
            source_manifest=manifest,
            source_checkpoint="recovery/checkpoints/step-1526",
            init_optimizer_state=_fresh_optimizer_state,
            oracle_step=7,
        )

    provenance = oracle.manifest.capacity_oracle
    assert provenance is not None
    assert provenance.kind is CapacityOracleKind.UNTIED_IDENTICAL_START_DIAGNOSTIC
    assert provenance.source_checkpoint == "recovery/checkpoints/step-1526"
    assert provenance.representative_layer == 2
    assert provenance.source_layer == 3
    assert provenance.input_topology == (0, 1, 2, 2, 3, 4)
    assert provenance.source_shared_bank_index == 2
    assert provenance.duplicated_bank_index == 3
    assert provenance.source_recovery_step == 1526
    assert provenance.output_step == 7
    assert oracle.manifest.target_topology == (0, 1, 2, 3, 4, 5)
    assert oracle.manifest.recovery_step == 7
    assert oracle.manifest.optimizer_state_reset
    assert oracle.manifest.recovery_initialization is None
    assert oracle.manifest.recovery_initial_checkpoint is None
    assert MergeCheckpointManifest.from_dict(oracle.manifest.to_dict()) == oracle.manifest

    assert oracle.state.ema_params is not None
    assert oracle.state.ema_params.config.resolved_expert_bank_for_layer == (0, 1, 2, 3, 4, 5)
    for representative_leaf, duplicated_leaf in zip(
        jax.tree.leaves(oracle.state.ema_params.expert_banks[2]),
        jax.tree.leaves(oracle.state.ema_params.expert_banks[3]),
        strict=True,
    ):
        assert representative_leaf is not duplicated_leaf
        np.testing.assert_array_equal(duplicated_leaf, representative_leaf)


def test_layer_adapter_conversion_preserves_function_routes_and_qb_state():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        recovered, manifest = _recovered_one_pair_state_and_manifest()
        recovered = dataclasses.replace(recovered, step=jnp.array(382, dtype=jnp.int32))
        manifest = dataclasses.replace(manifest, recovery_step=382)
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
        mlp_input = jnp.linspace(-1.0, 1.0, 32, dtype=jnp.float32).reshape(1, 4, 8)
        source_block = recovered.params.blocks[3]
        source_bank = recovered.params.expert_banks[source_block.expert_bank_index]
        source_trace = source_block.mlp.forward_with_trace(mlp_input, source_bank)
        source_logits = recovered.params.logits(tokens)
        converted = convert_grug_state_for_layer_adapter(
            recovered,
            source_manifest=manifest,
            source_checkpoint="stage-a/checkpoints/step-382",
            layer_index=3,
            rank=8,
            key=jax.random.key(13),
            init_optimizer_state=_fresh_optimizer_state,
        )
        converted_block = converted.state.params.blocks[3]
        converted_bank = converted.state.params.expert_banks[converted_block.expert_bank_index]
        converted_trace = converted_block.mlp.forward_with_trace(
            mlp_input,
            converted_bank,
            converted_block.routed_expert_adapter,
        )
        converted_logits = converted.state.params.logits(tokens)

    assert converted.state.params.config.resolved_expert_bank_for_layer == (0, 1, 2, 2, 3, 4)
    assert converted.state.params.config.resolved_expert_adapter_rank_for_layer == (0, 0, 0, 8, 0, 0)
    assert [
        index for index, block in enumerate(converted.state.params.blocks) if block.routed_expert_adapter is not None
    ] == [3]
    assert tuple(block.expert_bank_index for block in converted.state.params.blocks) == (0, 1, 2, 2, 3, 4)
    np.testing.assert_array_equal(converted_logits, source_logits)
    np.testing.assert_array_equal(converted_trace.routed_output, source_trace.routed_output)
    np.testing.assert_array_equal(
        converted_trace.routing.selected_experts,
        source_trace.routing.selected_experts,
    )
    np.testing.assert_array_equal(converted_trace.routing.combine_weights, source_trace.routing.combine_weights)
    np.testing.assert_array_equal(converted_trace.router_stats["qb_beta"], source_trace.router_stats["qb_beta"])
    np.testing.assert_array_equal(converted.state.pending_qb_betas, recovered.pending_qb_betas)
    np.testing.assert_array_equal(
        converted.state.params.blocks[3].mlp.router,
        recovered.params.blocks[3].mlp.router,
    )
    np.testing.assert_array_equal(
        converted.state.params.blocks[3].mlp.router_bias,
        recovered.params.blocks[3].mlp.router_bias,
    )
    assert int(converted.state.step) == 0
    optimizer_state = cast("dict[str, Any]", converted.state.opt_state)
    assert int(optimizer_state["recovery_bank_count"]) == 5

    provenance = converted.manifest.layer_adapter
    assert provenance is not None
    assert provenance.kind is LayerAdapterKind.ZERO_INITIALIZED_INPUT_OUTPUT
    assert provenance.source_checkpoint == "stage-a/checkpoints/step-382"
    assert provenance.layer_index == 3
    assert provenance.rank == 8
    assert provenance.input_topology == (0, 1, 2, 2, 3, 4)
    assert provenance.source_recovery_step == 382
    assert provenance.output_step == 0

    assert converted.state.ema_params is not None
    assert converted.state.ema_params.config.resolved_expert_adapter_rank_for_layer == (0, 0, 0, 8, 0, 0)
    ema_adapter = converted.state.ema_params.blocks[3].routed_expert_adapter
    assert ema_adapter is not None
    params_adapter = converted.state.params.blocks[3].routed_expert_adapter
    assert params_adapter is not None
    for ema_leaf, params_leaf in zip(jax.tree.leaves(ema_adapter), jax.tree.leaves(params_adapter), strict=True):
        np.testing.assert_array_equal(ema_leaf, params_leaf)


def test_layer_adapter_manifest_round_trip_and_legacy_manifest_compatibility():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        recovered, manifest = _recovered_one_pair_state_and_manifest()
        converted = convert_grug_state_for_layer_adapter(
            recovered,
            source_manifest=manifest,
            source_checkpoint="stage-a/checkpoints/step-382",
            layer_index=3,
            rank=8,
            key=jax.random.key(14),
            init_optimizer_state=_fresh_optimizer_state,
        )

    assert MergeCheckpointManifest.from_dict(converted.manifest.to_dict()) == converted.manifest

    legacy_payload = manifest.to_dict()
    legacy_payload["format_version"] = 2
    legacy_payload.pop("layer_adapter")
    restored_legacy = MergeCheckpointManifest.from_dict(legacy_payload)
    assert restored_legacy.layer_adapter is None
    assert restored_legacy.format_version == 2

    invalid_legacy_payload = converted.manifest.to_dict()
    invalid_legacy_payload["format_version"] = 2
    with pytest.raises(ValueError, match="requires merge checkpoint format version 3"):
        MergeCheckpointManifest.from_dict(invalid_legacy_payload)
