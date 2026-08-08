# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from fray.cluster import ResourceConfig
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint, save_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.grug.sharding import compact_grug_mesh
from levanter.optim.config import AdamConfig
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.fingerprint import canonical_json

from experiments.grug.moe.expert_merge import (
    AssignmentMode,
    ExpertCostMatrix,
    ExpertProbeSet,
    ExpertReservoirCollection,
    MoeLayerTrace,
    SpectralProbeConfig,
)
from experiments.grug.moe.expert_prefit import PrefitConfig, PrefitObjective
from experiments.grug.moe.merge_artifacts import (
    CalibrationArtifactManifest,
    MatchingArtifactManifest,
    write_calibration_artifact,
    write_matching_artifact,
)
from experiments.grug.moe.merge_checkpoint import read_merge_checkpoint_manifest, write_merge_checkpoint_manifest
from experiments.grug.moe.merge_jobs import (
    CapacityOracleSplitJobConfig,
    ConversionJobConfig,
    GradientConflictArtifactReference,
    GradientConflictCheckpointConfig,
    GradientConflictJobConfig,
    LayerAdapterAugmentJobConfig,
    PrefitJobConfig,
    RecoveryJobConfig,
    SourceCheckpointConfig,
)
from experiments.grug.moe.merge_recovery import (
    RecoveryCheckpointSelection,
    RecoveryInitialization,
    RecoveryStage,
    RecoveryTrainableScope,
)
from experiments.grug.moe.merge_recovery_runtime import (
    PrefitRuntimeState,
    _validate_selected_local_recovery,
    evaluate_prefit_checkpoint_local,
    run_capacity_oracle_split_local,
    run_conversion_local,
    run_gradient_conflict_local,
    run_layer_adapter_augment_local,
    run_prefit_local,
    run_recovery_local,
)
from experiments.grug.moe.model import GrugModelConfig, Transformer


@dataclass(frozen=True)
class _RuntimeInputs:
    source: SourceCheckpointConfig
    calibration_path: str
    matching_path: str
    teacher: Transformer


def _tiny_config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=3,
        shared_expert_intermediate_dim=4,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
        moe_implementation="scatter",
    )


def _probe(offset: float) -> ExpertProbeSet:
    return ExpertProbeSet(
        ordinary_inputs=np.full((2, 8), offset, dtype=np.float32),
        ordinary_weights=np.ones((2,), dtype=np.float32),
        centers=np.full((1, 8), offset + 0.1, dtype=np.float32),
        spectral_pairs=np.stack(
            [
                np.full((2, 8), offset + 0.2, dtype=np.float32),
                np.full((2, 8), offset + 0.3, dtype=np.float32),
            ]
        ),
        input_directions=np.ones((8, 1), dtype=np.float32),
        sensitivity_eigenvalues=np.ones((1,), dtype=np.float32),
    )


def _runtime_inputs(tmp_path: Path) -> _RuntimeInputs:
    model_config = _tiny_config()
    checkpoint_root = tmp_path / "teacher" / "checkpoints"
    concrete_checkpoint = checkpoint_root / "step-7"
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        teacher = Transformer.init(model_config, key=jax.random.key(0))
        save_checkpoint(
            {
                "step": jnp.array(7, dtype=jnp.int32),
                "params": teacher,
                "pending_qb_betas": jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
                "opt_state": {"source_only": jnp.array(123.0)},
            },
            step=7,
            checkpoint_path=concrete_checkpoint,
            is_temporary=False,
        )

    calibration_path = tmp_path / "calibration"
    reservoirs = {}
    traces = {}
    for layer in (1, 2):
        reservoir = ExpertReservoirCollection(
            num_experts=2,
            state_dim=8,
            capacity_per_expert=16,
            heldout_fraction=0.5,
            seed=layer,
        )
        rng = np.random.default_rng(layer)
        states = rng.normal(size=(256, 8)).astype(np.float32)
        selected = np.arange(256, dtype=np.int32).reshape(-1, 1) % 2
        reservoir.add_routes(states, selected, np.ones_like(selected, dtype=np.float32))
        reservoirs[layer] = reservoir
        trace_inputs = states[:8]
        trace_selected = selected[:8]
        trace_combine = np.linspace(0.25, 1.0, 8, dtype=np.float32).reshape(-1, 1)
        expert_bank = teacher.expert_banks[teacher.blocks[layer].expert_bank_index]
        with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
            routed_output = expert_bank(trace_inputs, trace_selected, trace_combine)
        if isinstance(routed_output, tuple):
            routed_output = routed_output[0]
        traces[layer] = MoeLayerTrace(
            mlp_input=trace_inputs,
            selected_experts=trace_selected,
            combine_weights=trace_combine,
            routed_output=routed_output,
        )
    write_calibration_artifact(
        str(calibration_path),
        reservoirs,
        CalibrationArtifactManifest(
            source_checkpoint=str(concrete_checkpoint),
            source_commit=None,
            layers=(1, 2),
            num_experts=2,
            state_dim=8,
            capacity_per_expert=16,
            heldout_fraction=0.5,
            calibration_tokens=256,
            trace_capacity=8,
        ),
        traces_by_layer=traces,
    )

    matching_path = tmp_path / "matching"
    assignments = {
        AssignmentMode.IDENTITY: (0, 1),
        AssignmentMode.NATIVE: (1, 0),
        AssignmentMode.SPECTRAL: (0, 1),
    }
    zero_costs = np.zeros((2, 2), dtype=np.float32)
    write_matching_artifact(
        str(matching_path),
        (_probe(0.0), _probe(1.0)),
        ExpertCostMatrix(native=zero_costs, tangent=zero_costs, total=zero_costs),
        MatchingArtifactManifest(
            calibration_path=str(calibration_path),
            representative_layer=1,
            source_layer=2,
            num_experts=2,
            eta=0.5,
            assignments=assignments,
        ),
    )
    return _RuntimeInputs(
        source=SourceCheckpointConfig(
            model=model_config,
            optimizer=AdamConfig(learning_rate=1e-3, weight_decay=0.0),
            training_steps=10,
            checkpoint_dir=str(checkpoint_root),
        ),
        calibration_path=str(calibration_path),
        matching_path=str(matching_path),
        teacher=teacher,
    )


def test_merge_mesh_supports_closed_expert_weights_and_aggregate_nrmse_on_multi_device() -> None:
    script = """
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.grug_moe import MoEExpertMlp
from experiments.grug.moe.expert_merge import (
    MoeLayerTrace,
    ReservoirSample,
    SpectralProbeConfig,
    build_spectral_probe_set,
    eval_expert,
)
from experiments.grug.moe.expert_prefit import (
    AggregatePrefitBatch,
    aggregate_prefit_loss,
    aggregate_routed_moe_nrmse,
)
from experiments.grug.moe.merge_recovery_runtime import compact_merge_mesh

mesh = compact_merge_mesh()
assert mesh.shape[\"replica_dcn\"] == 4
assert mesh.shape[\"data\"] == 1
with set_mesh(mesh):
    bank = jax.jit(
        lambda key: MoEExpertMlp.init(
            num_experts=4,
            hidden_dim=16,
            intermediate_dim=8,
            initializer_std=0.02,
            key=key,
            implementation=\"scatter\",
        )
    )(jax.random.key(0))
    inputs = jnp.ones((16, 16), dtype=jnp.float32)
    selected = jnp.arange(16, dtype=jnp.int32).reshape(-1, 1) % 4
    weights = jnp.ones((16, 1), dtype=jnp.float32)
    closed = jax.jit(lambda x, expert_ids, combine: bank(x, expert_ids, combine))(
        inputs, selected, weights
    )
    explicit = jax.jit(lambda current, x, expert_ids, combine: current(x, expert_ids, combine))(
        bank, inputs, selected, weights
    )
    states = np.random.default_rng(0).normal(size=(32, 16)).astype(np.float32)
    sample = ReservoirSample(states=states, weights=np.ones(32, dtype=np.float32))
    probes = build_spectral_probe_set(
        bank,
        0,
        sample,
        sample,
        config=SpectralProbeConfig(
            covariance_rank=4,
            num_centers=4,
            num_sensitive_directions=2,
            directions_per_center=1,
            ordinary_samples=4,
        ),
    )
    expert_output = eval_expert(bank, 0, states[:4])

    teacher = jax.jit(
        lambda key: MoEExpertMlp.init(
            num_experts=4,
            hidden_dim=16,
            intermediate_dim=8,
            initializer_std=0.02,
            key=key,
            implementation="scatter",
        )
    )(jax.random.key(1))
    routed_inputs = np.random.default_rng(2).normal(size=(8, 16)).astype(np.float32)
    routed_experts = np.asarray(
        [[0, 1], [1, 3], [2, 0], [3, 2], [0, 3], [2, 1], [1, 0], [3, 1]],
        dtype=np.int32,
    )
    routed_weights = np.asarray(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.3, 0.7],
            [0.55, 0.45],
            [0.9, 0.1],
            [0.25, 0.75],
            [0.4, 0.6],
            [0.65, 0.35],
        ],
        dtype=np.float32,
    )
    assignment = (2, 0, 3, 1)
    teacher_weights = tuple(
        np.asarray(jax.device_get(weight)) for weight in (teacher.w_gate, teacher.w_up, teacher.w_down)
    )
    shared_weights = tuple(
        np.asarray(jax.device_get(weight)) for weight in (bank.w_gate, bank.w_up, bank.w_down)
    )

    def explicit_expert(current_weights, expert, x):
        w_gate, w_up, w_down = current_weights
        gate = x @ w_gate[expert]
        up = x @ w_up[expert]
        return (np.asarray(jax.nn.silu(gate)) * up) @ w_down[expert]

    teacher_output = np.zeros_like(routed_inputs)
    shared_output = np.zeros_like(routed_inputs)
    for token in range(routed_inputs.shape[0]):
        for route in range(routed_experts.shape[1]):
            source_expert = int(routed_experts[token, route])
            weight = routed_weights[token, route]
            teacher_output[token] += weight * explicit_expert(teacher_weights, source_expert, routed_inputs[token])
            shared_output[token] += weight * explicit_expert(
                shared_weights, assignment[source_expert], routed_inputs[token]
            )
    trace = MoeLayerTrace(
        mlp_input=routed_inputs,
        selected_experts=routed_experts,
        combine_weights=routed_weights,
        routed_output=teacher_output,
    )
    aggregate_nrmse = aggregate_routed_moe_nrmse(bank, trace, assignment)
    target_power_by_layer = np.asarray(
        [
            np.mean(np.sum(np.square(teacher_output[:4]), axis=-1)),
            np.mean(np.sum(np.square(teacher_output[4:]), axis=-1)),
        ],
        dtype=np.float32,
    )
    aggregate_batch = AggregatePrefitBatch(
        inputs=jnp.asarray(routed_inputs),
        targets=jnp.asarray(teacher_output),
        shared_experts=jnp.asarray(np.asarray(assignment)[routed_experts]),
        combine_weights=jnp.asarray(routed_weights),
        layer_indices=jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32),
        target_power_by_layer=jnp.asarray(target_power_by_layer),
    )
    routed_shared_output = bank(
        aggregate_batch.inputs,
        aggregate_batch.shared_experts,
        aggregate_batch.combine_weights,
    )
    assert not isinstance(routed_shared_output, tuple)
    aggregate_loss, aggregate_nrmse_by_layer = aggregate_prefit_loss(bank, aggregate_batch)
    gradient = jax.grad(lambda current: aggregate_prefit_loss(current, aggregate_batch)[0])(bank)
    expected_nrmse = np.sqrt(
        np.sum(np.square(shared_output - teacher_output)) / np.sum(np.square(teacher_output))
    )
    expected_nrmse_by_layer = []
    for layer_slice in (slice(0, 4), slice(4, 8)):
        error = np.mean(
            np.sum(
                np.square(np.asarray(routed_shared_output)[layer_slice] - teacher_output[layer_slice]),
                axis=-1,
            )
        )
        power = np.mean(np.sum(np.square(teacher_output[layer_slice]), axis=-1))
        expected_nrmse_by_layer.append(np.sqrt(error / (power + 1e-8)))
np.testing.assert_allclose(closed, explicit, rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(aggregate_nrmse, expected_nrmse, rtol=5e-3, atol=1e-6)
np.testing.assert_allclose(aggregate_nrmse_by_layer, expected_nrmse_by_layer, rtol=5e-3, atol=1e-6)
np.testing.assert_allclose(aggregate_loss, np.mean(np.square(expected_nrmse_by_layer)), rtol=5e-3, atol=1e-6)
assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(gradient))
assert probes.spectral_pairs.shape == (8, 2, 16)
assert expert_output.shape == (4, 16)
"""
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_conversion_worker_persists_merged_checkpoint_and_ignores_source_optimizer(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    output_path = tmp_path / "converted"
    config = ConversionJobConfig(
        source=inputs.source,
        calibration_path=inputs.calibration_path,
        matching_path=inputs.matching_path,
        prefit_path=None,
        output_path=str(output_path),
        resources=ResourceConfig.with_cpu(),
        run_id="test-conversion",
        assignment_mode=AssignmentMode.SPECTRAL,
        representative_layer=1,
        source_layer=2,
    )

    run_conversion_local(config)
    run_conversion_local(config)

    checkpoint = latest_checkpoint_path(str(output_path / "checkpoints"))
    manifest = read_merge_checkpoint_manifest(checkpoint)
    assert manifest.target_topology == (0, 1, 1, 2)
    assert manifest.optimizer_state_reset
    assert manifest.source_step == 7
    assert manifest.recovery_step == 0
    assert len(tuple((output_path / "checkpoints").glob("step-*"))) == 1

    second_root = tmp_path / "second-teacher" / "checkpoints"
    second_checkpoint = second_root / "step-7"
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        save_checkpoint(
            {
                "step": jnp.array(7, dtype=jnp.int32),
                "params": inputs.teacher,
                "pending_qb_betas": jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
                "opt_state": {"source_only": jnp.array(123.0)},
            },
            step=7,
            checkpoint_path=second_checkpoint,
            is_temporary=False,
        )
    with pytest.raises(ValueError, match="stale provenance"):
        run_conversion_local(
            dataclasses.replace(
                config,
                source=dataclasses.replace(inputs.source, checkpoint_dir=str(second_root)),
            )
        )


def test_prefit_worker_persists_best_bank_and_balanced_layer_metrics(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    output_path = tmp_path / "prefit"
    config = PrefitJobConfig(
        source=inputs.source,
        calibration_path=inputs.calibration_path,
        matching_path=inputs.matching_path,
        output_path=str(output_path),
        resources=ResourceConfig.with_cpu(),
        run_id="test-prefit",
        assignment_mode=AssignmentMode.SPECTRAL,
        objective=PrefitObjective.PER_EXPERT,
        representative_layer=1,
        source_layer=2,
        probe=SpectralProbeConfig(
            covariance_rank=2,
            num_centers=2,
            num_sensitive_directions=2,
            directions_per_center=1,
            radii=(0.15,),
            ordinary_samples=2,
        ),
        config=PrefitConfig(
            learning_rate=1e-3,
            steps=1,
            examples_per_source=1,
            heldout_examples_per_source=1,
            eval_every=1,
            early_stopping_patience=1,
        ),
    )
    run_prefit_local(config)
    posthoc_routed_nrmse = evaluate_prefit_checkpoint_local(config)

    checkpoint = latest_checkpoint_path(str(output_path / "checkpoints"))
    manifest = json.loads((Path(checkpoint) / "prefit_manifest.json").read_text())
    assert manifest["step"] == 1
    assert manifest["assignment_mode"] == AssignmentMode.SPECTRAL.value
    assert manifest["objective"] == PrefitObjective.PER_EXPERT.value
    assert not any(key.startswith("aggregate_") for key in manifest["prefit_config"])
    assert len(manifest["nrmse_by_source"]) == 4
    assert set(manifest["merge/expert_holdout_nrmse_by_cluster"]) == {"shared_0000", "shared_0001"}
    routed_nrmse = manifest["merge/prefit_routed_moe_nrmse_by_layer"]
    assert set(routed_nrmse) == {"layer_1", "layer_2"}
    assert all(np.isfinite(value) for value in routed_nrmse.values())
    assert posthoc_routed_nrmse == {1: routed_nrmse["layer_1"], 2: routed_nrmse["layer_2"]}
    assert len(tuple((output_path / "checkpoints").glob("step-*"))) == 1

    representative_bank = inputs.teacher.expert_banks[inputs.teacher.blocks[1].expert_bank_index]
    template_optimizer = optax.adamw(1e-3, weight_decay=0.0)
    template = PrefitRuntimeState(
        step=jnp.array(0, dtype=jnp.int32),
        bank=representative_bank,
        opt_state=template_optimizer.init(representative_bank),
        best_bank=representative_bank,
        best_loss=jnp.array(jnp.inf),
        stale_evaluations=jnp.array(0, dtype=jnp.int32),
    )
    restored = load_checkpoint(template, checkpoint, mesh=compact_grug_mesh(expert_axis_size=1))
    assert int(restored.step) == 1
    assert np.isfinite(float(restored.best_loss))

    legacy_manifest = dict(manifest)
    legacy_manifest.pop("assignment_mode")
    legacy_manifest.pop("objective")
    (Path(checkpoint) / "prefit_manifest.json").write_text(json.dumps(legacy_manifest))
    assert evaluate_prefit_checkpoint_local(config) == posthoc_routed_nrmse

    with pytest.raises(ValueError, match="stale provenance"):
        run_prefit_local(dataclasses.replace(config, config=dataclasses.replace(config.config, steps=2)))


def test_native_aggregate_prefit_checkpoint_preserves_objective_assignment_and_converts(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    output_path = tmp_path / "native-aggregate-prefit"
    config = PrefitJobConfig(
        source=inputs.source,
        calibration_path=inputs.calibration_path,
        matching_path=inputs.matching_path,
        output_path=str(output_path),
        resources=ResourceConfig.with_cpu(),
        run_id="test-native-aggregate-prefit",
        assignment_mode=AssignmentMode.NATIVE,
        objective=PrefitObjective.AGGREGATE_ROUTED,
        representative_layer=1,
        source_layer=2,
        config=PrefitConfig(
            learning_rate=1e-3,
            steps=1,
            eval_every=1,
            early_stopping_patience=1,
            aggregate_examples_per_layer=2,
            aggregate_heldout_examples_per_layer=2,
            aggregate_trace_heldout_fraction=0.25,
        ),
    )

    with pytest.raises(ValueError, match="requires the native Hungarian assignment"):
        run_prefit_local(dataclasses.replace(config, assignment_mode=AssignmentMode.SPECTRAL))

    run_prefit_local(config)
    posthoc_nrmse = evaluate_prefit_checkpoint_local(config)

    checkpoint = latest_checkpoint_path(str(output_path / "checkpoints"))
    manifest = json.loads((Path(checkpoint) / "prefit_manifest.json").read_text())
    assert manifest["assignment_mode"] == AssignmentMode.NATIVE.value
    assert manifest["objective"] == PrefitObjective.AGGREGATE_ROUTED.value
    assert "probe_config" not in manifest
    assert "examples_per_source" not in manifest["prefit_config"]
    assert "heldout_examples_per_source" not in manifest["prefit_config"]
    assert "merge/expert_holdout_nrmse_by_cluster" not in manifest
    heldout_nrmse = manifest["merge/prefit_heldout_routed_moe_nrmse_by_layer"]
    assert posthoc_nrmse == {1: heldout_nrmse["layer_1"], 2: heldout_nrmse["layer_2"]}
    assert manifest["aggregate_trace_split"] == {
        "layer_1": {"heldout_tokens": 2, "train_tokens": 6},
        "layer_2": {"heldout_tokens": 2, "train_tokens": 6},
    }
    with pytest.raises(ValueError, match="stale provenance"):
        evaluate_prefit_checkpoint_local(dataclasses.replace(config, objective=PrefitObjective.PER_EXPERT))

    converted_path = tmp_path / "native-aggregate-converted"
    conversion = ConversionJobConfig(
        source=inputs.source,
        calibration_path=inputs.calibration_path,
        matching_path=inputs.matching_path,
        prefit_path=str(output_path / "checkpoints"),
        output_path=str(converted_path),
        resources=ResourceConfig.with_cpu(),
        run_id="test-native-aggregate-conversion",
        assignment_mode=AssignmentMode.NATIVE,
        representative_layer=1,
        source_layer=2,
    )
    run_conversion_local(conversion)
    converted_manifest = read_merge_checkpoint_manifest(latest_checkpoint_path(str(converted_path / "checkpoints")))
    assert converted_manifest.spec.assignment_mode is AssignmentMode.NATIVE
    assert converted_manifest.spec.prefit_applied
    assert converted_manifest.spec.prefit_objective is PrefitObjective.AGGREGATE_ROUTED

    with pytest.raises(ValueError, match="prefit checkpoint uses native assignment"):
        run_conversion_local(
            dataclasses.replace(
                conversion,
                assignment_mode=AssignmentMode.SPECTRAL,
                output_path=str(tmp_path / "wrong-assignment-converted"),
            )
        )


def _data_config() -> LmDataConfig:
    examples = [GrugLmExample.causal((jnp.arange(8, dtype=jnp.int32) + offset) % 32) for offset in range(4)]
    return LmDataConfig(
        components={
            "paloma/c4_en": DirectDatasetComponent(
                datasets={
                    "train": ListAsyncDataset(examples),
                    "validation": ListAsyncDataset(examples[:2]),
                }
            )
        },
        vocab_size=32,
        tokenizer="passthrough",
    )


def test_recovery_workers_reset_each_phase_and_persist_bounded_evaluations(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    converted_path = tmp_path / "converted"
    run_conversion_local(
        ConversionJobConfig(
            source=inputs.source,
            calibration_path=inputs.calibration_path,
            matching_path=inputs.matching_path,
            prefit_path=None,
            output_path=str(converted_path),
            resources=ResourceConfig.with_cpu(),
            run_id="test-conversion-for-recovery",
            assignment_mode=AssignmentMode.SPECTRAL,
            representative_layer=1,
            source_layer=2,
        )
    )

    stage_a_path = tmp_path / "stage-a"
    common = {
        "source": inputs.source,
        "data": _data_config(),
        "matching_path": inputs.matching_path,
        "resources": ResourceConfig.with_cpu(),
        "assignment_mode": AssignmentMode.SPECTRAL,
        "prefit_applied": False,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "affected_layers": (1, 2),
        "checkpoint_every": 1,
    }
    stage_a_config = RecoveryJobConfig(
        **common,
        init_checkpoint_dir=str(converted_path / "checkpoints"),
        output_path=str(stage_a_path),
        run_id="test-stage-a",
        stage=RecoveryStage.LOCAL,
        trainable_scope=RecoveryTrainableScope.SHARED_BANK,
        initialization=RecoveryInitialization.CONVERTED_STEP_ZERO,
        training_tokens=15,
        cross_entropy_weight=0.0,
        checkpoint_token_milestones=(8,),
        select_best_validation_checkpoint=True,
    )
    run_recovery_local(stage_a_config)
    stage_a_evaluations = {
        1: json.loads((stage_a_path / "evaluations" / "tokens-8-step-1.json").read_text()),
        2: json.loads((stage_a_path / "evaluations" / "tokens-16-step-2.json").read_text()),
    }
    expected_step = min(
        stage_a_evaluations,
        key=lambda step: stage_a_evaluations[step]["metrics"]["eval/paloma/macro_loss"],
    )
    selection = json.loads((stage_a_path / "checkpoints" / "selected_checkpoint.json").read_text())
    assert selection["selection_metric"] == "eval/paloma/macro_loss"
    assert selection["selection_value"] == stage_a_evaluations[expected_step]["metrics"]["eval/paloma/macro_loss"]
    assert selection["tokens"] == expected_step * 8
    assert selection["requested_tokens"] == (8 if expected_step == 1 else 15)
    assert selection["step"] == expected_step
    assert selection["checkpoint_path"] == str(stage_a_path / "checkpoints" / f"step-{expected_step}")

    stage_b_path = tmp_path / "stage-b"
    stage_b_config = RecoveryJobConfig(
        **common,
        init_checkpoint_dir=str(stage_a_path / "checkpoints"),
        output_path=str(stage_b_path),
        run_id="test-stage-b",
        stage=RecoveryStage.PRESERVATION,
        trainable_scope=RecoveryTrainableScope.SHARED_BANK_AND_ROUTERS,
        initialization=RecoveryInitialization.LOCAL_RECOVERY,
        training_tokens=8,
        cross_entropy_weight=1.0,
        logit_kl_weight=0.1,
        logit_kl_vocab_chunk_size=8,
        checkpoint_token_milestones=(8,),
        initial_checkpoint_selection=RecoveryCheckpointSelection.BEST_VALIDATION,
    )
    run_recovery_local(stage_b_config)

    # Resuming exactly at a milestone must reconstruct the complete teacher-relative eval.
    run_recovery_local(stage_a_config)
    run_recovery_local(stage_b_config)

    capacity_split_path = tmp_path / "capacity-oracle-split"
    run_capacity_oracle_split_local(
        CapacityOracleSplitJobConfig(
            source=inputs.source,
            init_checkpoint_dir=str(stage_a_path / "checkpoints"),
            output_path=str(capacity_split_path),
            resources=ResourceConfig.with_cpu(),
            run_id="test-capacity-oracle-split",
            assignment_mode=AssignmentMode.SPECTRAL,
            prefit_applied=False,
            affected_layers=(1, 2),
        )
    )
    capacity_split_checkpoint = latest_checkpoint_path(str(capacity_split_path / "checkpoints"))
    capacity_split_manifest = read_merge_checkpoint_manifest(capacity_split_checkpoint)
    assert capacity_split_manifest.target_topology == tuple(range(4))
    assert capacity_split_manifest.capacity_oracle is not None
    assert capacity_split_manifest.capacity_oracle.source_checkpoint == selection["checkpoint_path"]

    capacity_recovery_path = tmp_path / "capacity-oracle-recovery"
    capacity_recovery_config = RecoveryJobConfig(
        **common,
        init_checkpoint_dir=str(capacity_split_path / "checkpoints"),
        output_path=str(capacity_recovery_path),
        run_id="test-capacity-oracle-recovery",
        stage=RecoveryStage.PRESERVATION,
        trainable_scope=RecoveryTrainableScope.AFFECTED_EXPERT_BANKS,
        initialization=RecoveryInitialization.CAPACITY_ORACLE_SPLIT,
        training_tokens=8,
        cross_entropy_weight=1.0,
        logit_kl_weight=0.1,
        logit_kl_vocab_chunk_size=8,
        checkpoint_token_milestones=(8,),
    )
    run_recovery_local(capacity_recovery_config)

    selected_stage_a_eval = stage_a_evaluations[expected_step]
    capacity_initial_eval = json.loads((capacity_recovery_path / "evaluations" / "tokens-0-step-0.json").read_text())
    assert capacity_initial_eval["metrics"]["student/loss"] == selected_stage_a_eval["metrics"]["student/loss"]

    for output_path, final_step, final_tokens in (
        (stage_a_path, 2, 16),
        (stage_b_path, 1, 8),
        (capacity_recovery_path, 1, 8),
    ):
        checkpoint = latest_checkpoint_path(str(output_path / "checkpoints"))
        manifest = read_merge_checkpoint_manifest(checkpoint)
        assert manifest.recovery_step == final_step
        assert manifest.optimizer_state_reset
        expected_config = (
            stage_a_config
            if output_path == stage_a_path
            else capacity_recovery_config if output_path == capacity_recovery_path else stage_b_config
        )
        assert manifest.recovery_stage is expected_config.stage
        assert manifest.recovery_trainable_scope is expected_config.trainable_scope
        assert manifest.recovery_cross_entropy_weight == expected_config.cross_entropy_weight
        assert manifest.recovery_moe_loss_weight == expected_config.moe_loss_weight
        assert manifest.recovery_logit_kl_weight == expected_config.logit_kl_weight
        if output_path == stage_b_path:
            assert manifest.recovery_initial_checkpoint == selection["checkpoint_path"]
        elif output_path == capacity_recovery_path:
            assert manifest.recovery_initialization is RecoveryInitialization.CAPACITY_ORACLE_SPLIT
            assert manifest.recovery_initial_checkpoint == capacity_split_checkpoint
        initial_eval = json.loads((output_path / "evaluations" / "tokens-0-step-0.json").read_text())
        milestone_eval = json.loads(
            (output_path / "evaluations" / f"tokens-{final_tokens}-step-{final_step}.json").read_text()
        )
        training_metrics = json.loads((output_path / "training_metrics" / f"step-{final_step}.json").read_text())
        assert initial_eval["max_eval_batches"] == 8
        assert initial_eval["trainable_scope"] == (
            RecoveryTrainableScope.SHARED_BANK.value
            if output_path == stage_a_path
            else (
                RecoveryTrainableScope.AFFECTED_EXPERT_BANKS.value
                if output_path == capacity_recovery_path
                else RecoveryTrainableScope.SHARED_BANK_AND_ROUTERS.value
            )
        )
        assert "merge/immediate_validation_loss_delta" in initial_eval["metrics"]
        assert "eval/paloma/macro_loss" in initial_eval["metrics"]
        assert milestone_eval["tokens"] == final_tokens
        assert milestone_eval["requested_tokens"] == (15 if output_path == stage_a_path else 8)
        assert "teacher/loss" in milestone_eval["metrics"]
        assert "merge/recovery_tokens_to_threshold" in milestone_eval["metrics"]
        assert "merge/moe_output_nrmse_by_layer/layer_1" in training_metrics["metrics"]
        assert "merge/block_output_nrmse_by_layer/layer_1" in training_metrics["metrics"]
        assert "merge/router_topk_agreement_with_teacher/layer_1" in training_metrics["metrics"]
        assert "throughput/tokens_per_second" in training_metrics["metrics"]
        assert training_metrics["metrics"]["throughput/total_tokens"] == final_tokens
        assert training_metrics["trainable_scope"] == initial_eval["trainable_scope"]
        assert "train/router/layer_1/routing_entropy" in training_metrics["metrics"]
        assert "train/router/layer_1/capacity_overflow" in training_metrics["metrics"]
        assert "train/router/layer_1/routing_count/expert_0" in training_metrics["metrics"]


def test_layer_adapter_augment_and_recovery_are_strict_and_resumable(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    converted_path = tmp_path / "adapter-converted"
    run_conversion_local(
        ConversionJobConfig(
            source=inputs.source,
            calibration_path=inputs.calibration_path,
            matching_path=inputs.matching_path,
            prefit_path=None,
            output_path=str(converted_path),
            resources=ResourceConfig.with_cpu(),
            run_id="test-adapter-conversion",
            assignment_mode=AssignmentMode.NATIVE,
            representative_layer=1,
            source_layer=2,
        )
    )
    common = {
        "source": inputs.source,
        "data": _data_config(),
        "matching_path": inputs.matching_path,
        "resources": ResourceConfig.with_cpu(),
        "assignment_mode": AssignmentMode.NATIVE,
        "prefit_applied": False,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "affected_layers": (1, 2),
        "checkpoint_every": 1,
    }
    stage_a_path = tmp_path / "adapter-stage-a"
    run_recovery_local(
        RecoveryJobConfig(
            **common,
            init_checkpoint_dir=str(converted_path / "checkpoints"),
            output_path=str(stage_a_path),
            run_id="test-adapter-stage-a",
            stage=RecoveryStage.LOCAL,
            trainable_scope=RecoveryTrainableScope.SHARED_BANK,
            initialization=RecoveryInitialization.CONVERTED_STEP_ZERO,
            training_tokens=8,
            cross_entropy_weight=0.05,
            moe_loss_weight=1.0,
            logit_kl_weight=0.1,
            logit_kl_vocab_chunk_size=8,
            checkpoint_token_milestones=(8,),
            select_best_validation_checkpoint=True,
        )
    )

    augment_path = tmp_path / "adapter-augment"
    augment_config = LayerAdapterAugmentJobConfig(
        source=inputs.source,
        init_checkpoint_dir=str(stage_a_path / "checkpoints"),
        output_path=str(augment_path),
        resources=ResourceConfig.with_cpu(),
        run_id="test-adapter-augment",
        assignment_mode=AssignmentMode.NATIVE,
        prefit_applied=False,
        adapter_rank=2,
        affected_layers=(1, 2),
    )
    run_layer_adapter_augment_local(augment_config)
    run_layer_adapter_augment_local(augment_config)
    augmented_checkpoint = latest_checkpoint_path(str(augment_path / "checkpoints"))
    augmented_manifest = read_merge_checkpoint_manifest(augmented_checkpoint)
    assert augmented_manifest.layer_adapter is not None
    assert augmented_manifest.layer_adapter.layer_index == 2
    assert augmented_manifest.layer_adapter.rank == 2
    assert augmented_manifest.layer_adapter.input_topology == (0, 1, 1, 2)
    assert augmented_manifest.recovery_step == 0

    with pytest.raises(ValueError, match="stale provenance"):
        run_layer_adapter_augment_local(dataclasses.replace(augment_config, adapter_rank=3))
    with pytest.raises(ValueError, match="merged layers"):
        run_layer_adapter_augment_local(dataclasses.replace(augment_config, affected_layers=(2, 1)))

    assert augmented_manifest.layer_adapter is not None
    stale_manifest = dataclasses.replace(
        augmented_manifest,
        layer_adapter=dataclasses.replace(
            augmented_manifest.layer_adapter,
            source_checkpoint="different-source",
            input_topology=(0, 0, 1, 2),
        ),
    )
    write_merge_checkpoint_manifest(augmented_checkpoint, stale_manifest)
    with pytest.raises(ValueError, match="stale provenance"):
        run_layer_adapter_augment_local(augment_config)
    write_merge_checkpoint_manifest(augmented_checkpoint, augmented_manifest)

    recovery_path = tmp_path / "adapter-stage-b"
    recovery_config = RecoveryJobConfig(
        **common,
        init_checkpoint_dir=str(augment_path / "checkpoints"),
        output_path=str(recovery_path),
        run_id="test-adapter-stage-b",
        stage=RecoveryStage.PRESERVATION,
        trainable_scope=RecoveryTrainableScope.SHARED_BANK_AND_LAYER_ADAPTERS,
        initialization=RecoveryInitialization.LAYER_ADAPTER_AUGMENTED,
        training_tokens=8,
        cross_entropy_weight=1.0,
        moe_loss_weight=1.0,
        logit_kl_weight=0.1,
        logit_kl_vocab_chunk_size=8,
        checkpoint_token_milestones=(8,),
        layer_adapter_rank=2,
        layer_adapter_source_checkpoint_dir=str(stage_a_path / "checkpoints"),
    )
    run_recovery_local(recovery_config)
    run_recovery_local(recovery_config)
    recovered_manifest = read_merge_checkpoint_manifest(latest_checkpoint_path(str(recovery_path / "checkpoints")))
    assert recovered_manifest.layer_adapter == augmented_manifest.layer_adapter
    assert recovered_manifest.recovery_initialization is RecoveryInitialization.LAYER_ADAPTER_AUGMENTED
    assert recovered_manifest.recovery_trainable_scope is RecoveryTrainableScope.SHARED_BANK_AND_LAYER_ADAPTERS

    wrong_scope = dataclasses.replace(
        recovery_config,
        output_path=str(tmp_path / "adapter-wrong-scope"),
        trainable_scope=RecoveryTrainableScope.SHARED_BANK,
    )
    with pytest.raises(ValueError, match="must train exactly"):
        run_recovery_local(wrong_scope)

    wrong_rank = dataclasses.replace(
        recovery_config,
        output_path=str(tmp_path / "adapter-wrong-rank"),
        layer_adapter_rank=3,
    )
    with pytest.raises(ValueError, match="stale provenance"):
        run_recovery_local(wrong_rank)

    assert augmented_manifest.layer_adapter is not None
    for suffix, stale_adapter in (
        (
            "source",
            dataclasses.replace(augmented_manifest.layer_adapter, source_checkpoint="different-source"),
        ),
        (
            "step",
            dataclasses.replace(augmented_manifest.layer_adapter, source_recovery_step=999),
        ),
    ):
        write_merge_checkpoint_manifest(
            augmented_checkpoint,
            dataclasses.replace(augmented_manifest, layer_adapter=stale_adapter),
        )
        with pytest.raises(ValueError, match="stale provenance"):
            run_recovery_local(
                dataclasses.replace(
                    recovery_config,
                    output_path=str(tmp_path / f"adapter-wrong-{suffix}"),
                )
            )
    write_merge_checkpoint_manifest(augmented_checkpoint, augmented_manifest)


def test_legacy_selected_stage_a_uses_artifact_config_without_weakening_modern_manifests(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    converted_path = tmp_path / "legacy-validation-converted"
    run_conversion_local(
        ConversionJobConfig(
            source=inputs.source,
            calibration_path=inputs.calibration_path,
            matching_path=inputs.matching_path,
            prefit_path=None,
            output_path=str(converted_path),
            resources=ResourceConfig.with_cpu(),
            run_id="test-legacy-validation-conversion",
            assignment_mode=AssignmentMode.NATIVE,
            representative_layer=1,
            source_layer=2,
        )
    )
    conversion_manifest = read_merge_checkpoint_manifest(latest_checkpoint_path(str(converted_path / "checkpoints")))
    converted_checkpoint = latest_checkpoint_path(str(converted_path / "checkpoints"))
    legacy_source_model = dataclasses.replace(
        inputs.source.model,
        num_layers=6,
        expert_bank_for_layer=tuple(range(6)),
    )
    legacy_source = dataclasses.replace(inputs.source, model=legacy_source_model)
    legacy_manifest = dataclasses.replace(
        conversion_manifest,
        spec=dataclasses.replace(conversion_manifest.spec, representative_layer=2, source_layer=3),
        source_topology=tuple(range(6)),
        target_topology=(0, 1, 2, 2, 3, 4),
        source_step=legacy_source.training_steps,
        recovery_step=382,
        recovery_initialization=RecoveryInitialization.CONVERTED_STEP_ZERO,
        recovery_initial_checkpoint=converted_checkpoint,
        format_version=2,
    )
    legacy_root = tmp_path / "legacy-stage-a"
    checkpoint_root = legacy_root / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    config = LayerAdapterAugmentJobConfig(
        source=legacy_source,
        init_checkpoint_dir=str(checkpoint_root),
        output_path=str(tmp_path / "legacy-adapter"),
        resources=ResourceConfig.with_cpu(),
        run_id="test-legacy-adapter",
        assignment_mode=AssignmentMode.NATIVE,
        prefit_applied=False,
        adapter_rank=2,
        affected_layers=(2, 3),
    )
    selection = {
        "format_version": 1,
        "checkpoint_path": str(checkpoint_root / "step-382"),
        "step": 382,
        "tokens": 50_069_504,
        "requested_tokens": 50_000_000,
        "selection_metric": "eval/paloma/macro_loss",
        "selection_value": 3.6,
    }
    artifact_config = {
        "run_id": "grug-xem-native_local_ce_kl-stage-a-d512-l2-l3",
        "stage": RecoveryStage.LOCAL.value,
        "initialization": RecoveryInitialization.CONVERTED_STEP_ZERO.value,
        "initial_checkpoint_selection": RecoveryCheckpointSelection.LATEST.value,
        "init_checkpoint_dir": str(converted_path / "checkpoints"),
        "affected_layers": [2, 3],
        "assignment_mode": AssignmentMode.NATIVE.value,
        "prefit_applied": False,
        "batch_size": 32,
        "training_tokens": 50_000_000,
        "cross_entropy_weight": 0.05,
        "moe_loss_weight": 1.0,
        "logit_kl_weight": 0.1,
        "select_best_validation_checkpoint": True,
        "source": {
            "checkpoint_dir": legacy_source.checkpoint_dir,
            "source_commit": legacy_source.source_commit,
            "training_steps": legacy_source.training_steps,
            "model": {"expert_bank_for_layer": list(legacy_source.model.resolved_expert_bank_for_layer)},
        },
    }
    artifact_path = legacy_root / ".artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "name": "grug/expert_merge/d512/native_local_ce_kl/stage-a",
                "version": "2026.08.06",
                "fingerprint": "62564720",
                "output_path": str(legacy_root),
                "config": artifact_config,
            }
        )
    )

    _validate_selected_local_recovery(config, legacy_manifest, selection)

    for stale_selection in (
        {**selection, "step": 381, "checkpoint_path": str(checkpoint_root / "step-381")},
        {**selection, "checkpoint_path": str(checkpoint_root / "step-381")},
        {**selection, "tokens": 50_069_503},
    ):
        with pytest.raises(ValueError, match=r"selected CE\+KL|stale provenance"):
            _validate_selected_local_recovery(config, legacy_manifest, stale_selection)

    for stale_manifest in (
        dataclasses.replace(legacy_manifest, source_topology=(0, 1, 2, 3, 3, 4)),
        dataclasses.replace(legacy_manifest, target_topology=(0, 1, 2, 3, 3, 4)),
        dataclasses.replace(legacy_manifest, source_step=legacy_source.training_steps - 1),
    ):
        with pytest.raises(ValueError, match="stale provenance"):
            _validate_selected_local_recovery(config, stale_manifest, selection)

    artifact_path.write_text(
        json.dumps(
            {
                "name": "grug/expert_merge/d512/native_local_ce_kl/stage-a",
                "version": "2026.08.06",
                "fingerprint": "62564720",
                "output_path": str(legacy_root),
                "config": {**artifact_config, "cross_entropy_weight": 0.0},
            }
        )
    )
    with pytest.raises(ValueError, match="stale provenance"):
        _validate_selected_local_recovery(config, legacy_manifest, selection)

    artifact_path.write_text(
        json.dumps(
            {
                "name": "grug/expert_merge/d512/native_local_ce_kl/stage-a",
                "version": "2026.08.06",
                "fingerprint": "62564720",
                "output_path": str(legacy_root),
                "config": artifact_config,
            }
        )
    )
    with pytest.raises(ValueError, match=r"requires the selected CE\+KL"):
        _validate_selected_local_recovery(config, dataclasses.replace(legacy_manifest, format_version=3), selection)


def test_native_joint_preservation_restores_converted_step_zero_strictly_and_records_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _runtime_inputs(tmp_path)
    converted_path = tmp_path / "native-joint-converted"
    run_conversion_local(
        ConversionJobConfig(
            source=inputs.source,
            calibration_path=inputs.calibration_path,
            matching_path=inputs.matching_path,
            prefit_path=None,
            output_path=str(converted_path),
            resources=ResourceConfig.with_cpu(),
            run_id="test-native-joint-conversion",
            assignment_mode=AssignmentMode.NATIVE,
            representative_layer=1,
            source_layer=2,
        )
    )
    concrete_conversion = latest_checkpoint_path(str(converted_path / "checkpoints"))
    converted_manifest = read_merge_checkpoint_manifest(concrete_conversion)
    assert converted_manifest.recovery_step == 0
    assert converted_manifest.recovery_initialization is None
    assert not converted_manifest.spec.prefit_applied

    strict_loads: list[str] = []
    original_load_checkpoint = load_checkpoint

    def recording_load_checkpoint(*args, **kwargs):
        if kwargs.get("allow_partial") is False:
            strict_loads.append(str(args[1]))
        return original_load_checkpoint(*args, **kwargs)

    monkeypatch.setattr(
        "experiments.grug.moe.merge_recovery_runtime.load_checkpoint",
        recording_load_checkpoint,
    )
    recovery_path = tmp_path / "native-joint-stage-b"
    run_recovery_local(
        RecoveryJobConfig(
            source=inputs.source,
            data=_data_config(),
            matching_path=inputs.matching_path,
            init_checkpoint_dir=str(converted_path / "checkpoints"),
            output_path=str(recovery_path),
            resources=ResourceConfig.with_cpu(),
            run_id="test-native-joint-stage-b",
            stage=RecoveryStage.PRESERVATION,
            trainable_scope=RecoveryTrainableScope.SHARED_BANK_AND_ROUTERS,
            initialization=RecoveryInitialization.CONVERTED_STEP_ZERO,
            assignment_mode=AssignmentMode.NATIVE,
            prefit_applied=False,
            training_tokens=8,
            cross_entropy_weight=1.0,
            batch_size=1,
            learning_rate=1e-3,
            moe_loss_weight=1.0,
            logit_kl_weight=0.1,
            logit_kl_vocab_chunk_size=8,
            affected_layers=(1, 2),
            checkpoint_every=1,
            checkpoint_token_milestones=(8,),
        )
    )

    assert strict_loads.count(concrete_conversion) == 1
    recovered_checkpoint = latest_checkpoint_path(str(recovery_path / "checkpoints"))
    recovered_manifest = read_merge_checkpoint_manifest(recovered_checkpoint)
    assert recovered_manifest.recovery_step == 1
    assert recovered_manifest.recovery_initialization is RecoveryInitialization.CONVERTED_STEP_ZERO
    assert recovered_manifest.recovery_initial_checkpoint == concrete_conversion
    assert recovered_manifest.spec.assignment_mode is AssignmentMode.NATIVE
    assert not recovered_manifest.spec.prefit_applied


def test_gradient_conflict_worker_reads_frozen_checkpoints_and_writes_only_scalar_json(tmp_path: Path) -> None:
    inputs = _runtime_inputs(tmp_path)
    gradient_source = dataclasses.replace(inputs.source, training_steps=7, source_commit="test-commit")
    converted_path = tmp_path / "gradient-converted"
    run_conversion_local(
        ConversionJobConfig(
            source=gradient_source,
            calibration_path=inputs.calibration_path,
            matching_path=inputs.matching_path,
            prefit_path=None,
            output_path=str(converted_path),
            resources=ResourceConfig.with_cpu(),
            run_id="test-gradient-conversion",
            assignment_mode=AssignmentMode.NATIVE,
            representative_layer=1,
            source_layer=2,
        )
    )
    converted_checkpoint = latest_checkpoint_path(str(converted_path / "checkpoints"))
    converted_manifest = read_merge_checkpoint_manifest(converted_checkpoint)
    converted_manifest = dataclasses.replace(
        converted_manifest,
        spec=dataclasses.replace(converted_manifest.spec, source_commit="test-commit"),
    )
    student_config = dataclasses.replace(
        gradient_source.model,
        expert_bank_for_layer=converted_manifest.target_topology,
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        student = Transformer.init(student_config, key=jax.random.key(9))
        loaded = load_checkpoint(
            {
                "params": student,
                "pending_qb_betas": jnp.zeros((4, 2), dtype=jnp.float32),
            },
            converted_checkpoint,
            allow_partial=False,
        )
        recovered_params = eqx.tree_at(
            lambda model: model.expert_banks[1].w_down,
            loaded["params"],
            loaded["params"].expert_banks[1].w_down * 0.8,
        )
    stage_a_root = tmp_path / "gradient-stage-a"
    control_root = tmp_path / "gradient-control"
    stage_a_checkpoint = stage_a_root / "checkpoints" / "step-1"
    midpoint_checkpoint = control_root / "checkpoints" / "step-1"
    endpoint_checkpoint = control_root / "checkpoints" / "step-2"
    for step, checkpoint in ((1, stage_a_checkpoint), (1, midpoint_checkpoint), (2, endpoint_checkpoint)):
        save_checkpoint(
            {
                "step": jnp.asarray(step, dtype=jnp.int32),
                "params": recovered_params,
                "pending_qb_betas": loaded["pending_qb_betas"],
            },
            step=step,
            checkpoint_path=checkpoint,
            is_temporary=False,
        )

    stage_a_manifest = dataclasses.replace(
        converted_manifest,
        recovery_step=1,
        recovery_initialization=RecoveryInitialization.CONVERTED_STEP_ZERO,
        recovery_initial_checkpoint=str(converted_checkpoint),
        recovery_stage=RecoveryStage.LOCAL,
        recovery_trainable_scope=RecoveryTrainableScope.SHARED_BANK,
        recovery_cross_entropy_weight=0.05,
        recovery_moe_loss_weight=1.0,
        recovery_logit_kl_weight=0.1,
    )
    write_merge_checkpoint_manifest(str(stage_a_checkpoint), stage_a_manifest)
    for step, checkpoint in ((1, midpoint_checkpoint), (2, endpoint_checkpoint)):
        write_merge_checkpoint_manifest(
            str(checkpoint),
            dataclasses.replace(
                converted_manifest,
                recovery_step=step,
                recovery_initialization=RecoveryInitialization.LOCAL_RECOVERY,
                recovery_initial_checkpoint=str(stage_a_checkpoint),
                recovery_stage=RecoveryStage.PRESERVATION,
                recovery_trainable_scope=RecoveryTrainableScope.SHARED_BANK,
                recovery_cross_entropy_weight=1.0,
                recovery_moe_loss_weight=1.0,
                recovery_logit_kl_weight=0.1,
            ),
        )

    (stage_a_root / "checkpoints" / "selected_checkpoint.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "checkpoint_path": str(stage_a_checkpoint),
                "step": 1,
                "tokens": 8,
                "requested_tokens": 8,
                "selection_metric": "eval/paloma/macro_loss",
                "selection_value": 1.0,
            }
        )
    )
    data = _data_config()
    teacher_root = tmp_path / "teacher"
    teacher_artifact = GradientConflictArtifactReference(
        name="test/teacher",
        version="dev",
        root=str(teacher_root),
        fingerprint="teacher-fingerprint",
    )
    stage_a_artifact = GradientConflictArtifactReference(
        name="test/stage-a",
        version="dev",
        root=str(stage_a_root),
        fingerprint="stage-a-fingerprint",
    )
    control_artifact = GradientConflictArtifactReference(
        name="test/control",
        version="dev",
        root=str(control_root),
        fingerprint="control-fingerprint",
    )
    for artifact, config in (
        (teacher_artifact, {}),
        (stage_a_artifact, {}),
        (control_artifact, {"data": json.loads(canonical_json(data))}),
    ):
        write_record(
            ArtifactRecord(
                name=artifact.name,
                version=artifact.version,
                fingerprint=artifact.fingerprint,
                output_path=artifact.root,
                config=config,
            )
        )

    output_path = tmp_path / "gradient-output"
    checkpoints = tuple(
        GradientConflictCheckpointConfig(
            label=label,
            artifact=artifact,
            checkpoint_path=str(checkpoint),
            expected_step=step,
            continuation_tokens=index * 8,
        )
        for index, (label, artifact, checkpoint, step) in enumerate(
            (
                ("selected_stage_a", stage_a_artifact, stage_a_checkpoint, 1),
                ("shared_control_midpoint", control_artifact, midpoint_checkpoint, 1),
                ("shared_control_endpoint", control_artifact, endpoint_checkpoint, 2),
            )
        )
    )
    config = GradientConflictJobConfig(
        source=gradient_source,
        teacher_artifact=teacher_artifact,
        data=data,
        checkpoints=checkpoints,
        output_path=str(output_path),
        resources=ResourceConfig.with_cpu(),
        run_id="test-gradient-conflict",
        affected_layers=(1, 2),
        batch_size=1,
        num_batches=1,
        loader_start_step=1,
        bootstrap_samples=32,
    )
    run_gradient_conflict_local(config)

    assert [path.name for path in output_path.iterdir()] == ["gradient_conflict.json"]
    payload = json.loads((output_path / "gradient_conflict.json").read_text())
    assert payload["teacher_checkpoint"] == str(tmp_path / "teacher" / "checkpoints" / "step-7")
    assert payload["data"]["loader_start_step"] == 1
    assert payload["data"]["loader_stop_step_exclusive"] == 2
    assert payload["data"]["positions_per_checkpoint"] == 8
    assert [checkpoint["label"] for checkpoint in payload["checkpoints"]] == [
        "selected_stage_a",
        "shared_control_midpoint",
        "shared_control_endpoint",
    ]
    for checkpoint in payload["checkpoints"]:
        assert checkpoint["controls"]["capacity_overflow_by_layer"] == {"layer_1": 0.0, "layer_2": 0.0}
        assert checkpoint["controls"]["active_experts_by_layer"] == {"layer_1": 2, "layer_2": 2}
        assert checkpoint["controls"]["router_selected_ids_exact_by_layer"] == {"layer_1": 1.0, "layer_2": 1.0}
        for layer in (1, 2):
            assert checkpoint["mean_moe_nrmse_by_layer"][f"layer_{layer}"] == pytest.approx(
                np.sqrt(checkpoint["mean_layer_losses"][f"layer_{layer}"])
            )
            assert checkpoint["batches"][0]["moe_nrmse_by_layer"][f"layer_{layer}"] == pytest.approx(
                np.sqrt(checkpoint["batches"][0]["layer_errors"][f"layer_{layer}"])
            )
        assert len(checkpoint["batches"]) == 1
        assert len(checkpoint["experts"]) == 2
    assert not (output_path / "checkpoints").exists()

    write_record(
        ArtifactRecord(
            name=control_artifact.name,
            version=control_artifact.version,
            fingerprint="wrong-fingerprint",
            output_path=control_artifact.root,
            config={"data": json.loads(canonical_json(data))},
        )
    )
    with pytest.raises(ValueError, match="stale provenance"):
        run_gradient_conflict_local(dataclasses.replace(config, output_path=str(tmp_path / "stale-fingerprint")))

    write_record(
        ArtifactRecord(
            name=control_artifact.name,
            version=control_artifact.version,
            fingerprint=control_artifact.fingerprint,
            output_path=control_artifact.root,
            config={"data": {}},
        )
    )
    with pytest.raises(ValueError, match="data config differs"):
        run_gradient_conflict_local(dataclasses.replace(config, output_path=str(tmp_path / "stale-data")))

    write_record(
        ArtifactRecord(
            name=control_artifact.name,
            version=control_artifact.version,
            fingerprint=control_artifact.fingerprint,
            output_path=control_artifact.root,
            config={"data": json.loads(canonical_json(data))},
        )
    )
    endpoint_manifest = read_merge_checkpoint_manifest(str(endpoint_checkpoint))
    write_merge_checkpoint_manifest(
        str(endpoint_checkpoint),
        dataclasses.replace(endpoint_manifest, recovery_stage=RecoveryStage.LOCAL),
    )
    with pytest.raises(ValueError, match="S-prime recovery role"):
        run_gradient_conflict_local(dataclasses.replace(config, output_path=str(tmp_path / "stale-role")))
