# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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

from experiments.grug.moe.expert_merge import (
    AssignmentMode,
    ExpertCostMatrix,
    ExpertProbeSet,
    ExpertReservoirCollection,
    MoeLayerTrace,
    SpectralProbeConfig,
)
from experiments.grug.moe.expert_prefit import PrefitConfig
from experiments.grug.moe.merge_artifacts import (
    CalibrationArtifactManifest,
    MatchingArtifactManifest,
    write_calibration_artifact,
    write_matching_artifact,
)
from experiments.grug.moe.merge_checkpoint import read_merge_checkpoint_manifest
from experiments.grug.moe.merge_jobs import (
    ConversionJobConfig,
    PrefitJobConfig,
    RecoveryJobConfig,
    SourceCheckpointConfig,
)
from experiments.grug.moe.merge_recovery import RecoveryStage
from experiments.grug.moe.merge_recovery_runtime import (
    PrefitRuntimeState,
    evaluate_prefit_checkpoint_local,
    run_conversion_local,
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
    assignments = {mode: (0, 1) for mode in AssignmentMode}
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
from experiments.grug.moe.expert_prefit import aggregate_routed_moe_nrmse
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
    expected_nrmse = np.sqrt(
        np.sum(np.square(shared_output - teacher_output)) / np.sum(np.square(teacher_output))
    )
np.testing.assert_allclose(closed, explicit, rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(aggregate_nrmse, expected_nrmse, rtol=5e-3, atol=1e-6)
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

    with pytest.raises(ValueError, match="stale provenance"):
        run_prefit_local(dataclasses.replace(config, config=dataclasses.replace(config.config, steps=2)))


def _data_config() -> LmDataConfig:
    examples = [GrugLmExample.causal((jnp.arange(8, dtype=jnp.int32) + offset) % 32) for offset in range(4)]
    return LmDataConfig(
        components={
            "direct": DirectDatasetComponent(
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
        "training_tokens": 8,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "affected_layers": (1, 2),
        "checkpoint_every": 1,
        "checkpoint_token_milestones": (8,),
    }
    stage_a_config = RecoveryJobConfig(
        **common,
        init_checkpoint_dir=str(converted_path / "checkpoints"),
        output_path=str(stage_a_path),
        run_id="test-stage-a",
        stage=RecoveryStage.LOCAL,
    )
    run_recovery_local(stage_a_config)

    stage_b_path = tmp_path / "stage-b"
    stage_b_config = RecoveryJobConfig(
        **common,
        init_checkpoint_dir=str(stage_a_path / "checkpoints"),
        output_path=str(stage_b_path),
        run_id="test-stage-b",
        stage=RecoveryStage.PRESERVATION,
        logit_kl_weight=0.1,
        logit_kl_vocab_chunk_size=8,
    )
    run_recovery_local(stage_b_config)

    # Resuming exactly at a milestone must reconstruct the complete teacher-relative eval.
    run_recovery_local(stage_a_config)
    run_recovery_local(stage_b_config)

    for output_path in (stage_a_path, stage_b_path):
        checkpoint = latest_checkpoint_path(str(output_path / "checkpoints"))
        manifest = read_merge_checkpoint_manifest(checkpoint)
        assert manifest.recovery_step == 1
        assert manifest.optimizer_state_reset
        initial_eval = json.loads((output_path / "evaluations" / "tokens-0-step-0.json").read_text())
        milestone_eval = json.loads((output_path / "evaluations" / "tokens-8-step-1.json").read_text())
        training_metrics = json.loads((output_path / "training_metrics" / "step-1.json").read_text())
        assert initial_eval["max_eval_batches"] == 8
        assert "merge/immediate_validation_loss_delta" in initial_eval["metrics"]
        assert "eval/paloma/macro_loss" in initial_eval["metrics"]
        assert milestone_eval["tokens"] == 8
        assert "teacher/loss" in milestone_eval["metrics"]
        assert "merge/recovery_tokens_to_threshold" in milestone_eval["metrics"]
        assert "merge/moe_output_nrmse_by_layer/layer_1" in training_metrics["metrics"]
        assert "merge/block_output_nrmse_by_layer/layer_1" in training_metrics["metrics"]
        assert "merge/router_topk_agreement_with_teacher/layer_1" in training_metrics["metrics"]
        assert "throughput/tokens_per_second" in training_metrics["metrics"]
        assert training_metrics["metrics"]["throughput/total_tokens"] == 8
        assert "train/router/layer_1/routing_entropy" in training_metrics["metrics"]
        assert "train/router/layer_1/capacity_overflow" in training_metrics["metrics"]
        assert "train/router/layer_1/routing_count/expert_0" in training_metrics["metrics"]
