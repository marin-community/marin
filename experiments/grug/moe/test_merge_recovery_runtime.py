# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
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
        ),
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

    checkpoint = latest_checkpoint_path(str(output_path / "checkpoints"))
    manifest = json.loads((Path(checkpoint) / "prefit_manifest.json").read_text())
    assert manifest["step"] == 1
    assert len(manifest["nrmse_by_source"]) == 4
    assert set(manifest["merge/expert_holdout_nrmse_by_cluster"]) == {"shared_0000", "shared_0001"}

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
