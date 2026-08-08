# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accelerator workers for expert-bank prefit, conversion, and recovery."""

import dataclasses
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from fray.types import CpuConfig
from haliax.partitioning import set_mesh
from jax.experimental import multihost_utils
from jax.tree_util import register_dataclass
from levanter.checkpoint import discover_latest_checkpoint, latest_checkpoint_path, load_checkpoint, save_checkpoint
from levanter.distributed import DistributedConfig
from levanter.grug.grug_moe import MoEExpertMlp
from levanter.grug.sharding import compact_grug_mesh
from levanter.schedule import BatchSchedule
from rigging.filesystem import StoragePath, check_gcs_paths_same_region, prefix_join

from experiments.grug.moe.expert_merge import AssignmentMode, ExpertProbeSet, build_spectral_probe_set, eval_expert
from experiments.grug.moe.expert_prefit import (
    AggregatePrefitBatch,
    AggregatePrefitDataset,
    PrefitBatch,
    PrefitDataset,
    PrefitObjective,
    PrefitSplit,
    aggregate_prefit_loss,
    aggregate_routed_moe_nrmse,
    make_aggregate_prefit_dataset,
    prefit_loss,
    sample_aggregate_prefit_batch,
    sample_prefit_batch,
)
from experiments.grug.moe.merge_artifacts import (
    MatchingArtifactManifest,
    read_calibration_manifest,
    read_expert_calibration,
    read_expert_probe,
    read_layer_calibration_trace,
    read_matching_manifest,
)
from experiments.grug.moe.merge_checkpoint import (
    CapacityOracleKind,
    LayerAdapterKind,
    MergeCheckpointManifest,
    OnePairMergeCheckpointSpec,
    convert_grug_state_for_capacity_oracle_split,
    convert_grug_state_for_layer_adapter,
    convert_grug_state_for_one_pair_merge,
    read_merge_checkpoint_manifest,
    write_merge_checkpoint_manifest,
)
from experiments.grug.moe.merge_jobs import (
    CapacityOracleSplitJobConfig,
    ConversionJobConfig,
    LayerAdapterAugmentJobConfig,
    PrefitJobConfig,
    RecoveryJobConfig,
    SourceCheckpointConfig,
)
from experiments.grug.moe.merge_recovery import (
    MergeRecoveryConfig,
    MergeRecoveryState,
    RecoveryCheckpointSelection,
    RecoveryInitialization,
    RecoveryLosses,
    RecoveryStage,
    RecoveryTrainableScope,
    initial_recovery_state,
    make_chunked_logit_kl,
    make_recovery_train_step,
)
from experiments.grug.moe.model import Transformer
from experiments.grug.moe.train import (
    GrugEvalConfig,
    GrugTrainState,
    build_tagged_evaluator,
    build_train_dataset,
    build_train_loader,
)

logger = logging.getLogger(__name__)

_CHECKPOINTS_DIRECTORY = "checkpoints"
_PREFIT_MANIFEST_FILENAME = "prefit_manifest.json"
_PREFIT_FORMAT_VERSION = 2
_RECOVERY_THRESHOLD_FILENAME = "recovery_threshold.json"
_RECOVERY_SELECTION_FILENAME = "selected_checkpoint.json"
_LEGACY_CE_KL_STAGE_A_NAME = "grug/expert_merge/d512/native_local_ce_kl/stage-a"
_LEGACY_CE_KL_STAGE_A_VERSION = "2026.08.06"
_LEGACY_CE_KL_STAGE_A_FINGERPRINT = "62564720"
_LEGACY_CE_KL_STAGE_A_STEP = 382
_LEGACY_CE_KL_STAGE_A_TOKENS = 50_069_504
_LEGACY_CE_KL_TARGET_TOPOLOGY = (0, 1, 2, 2, 3, 4)


@register_dataclass
@dataclass(frozen=True)
class PrefitRuntimeState:
    step: jax.Array
    bank: MoEExpertMlp
    opt_state: optax.OptState
    best_bank: MoEExpertMlp
    best_loss: jax.Array
    stale_evaluations: jax.Array


def _is_local(
    config: (
        PrefitJobConfig
        | ConversionJobConfig
        | CapacityOracleSplitJobConfig
        | LayerAdapterAugmentJobConfig
        | RecoveryJobConfig
    ),
) -> bool:
    return isinstance(config.resources.device, CpuConfig)


def _validate_runtime_paths(
    config: (
        PrefitJobConfig
        | ConversionJobConfig
        | CapacityOracleSplitJobConfig
        | LayerAdapterAugmentJobConfig
        | RecoveryJobConfig
    ),
    paths: dict[str, str | None],
) -> None:
    """Validate material stage paths before initializing JAX devices."""
    material = {name: path for name, path in paths.items() if path is not None}
    local_ok = _is_local(config)
    if not local_ok:
        for name, path in material.items():
            if not path.startswith("gs://"):
                raise ValueError(f"{name} must be a GCS path on accelerator workers, got {path}")
    else:
        gcs_names = [name for name, path in material.items() if path.startswith("gs://")]
        local_names = [name for name, path in material.items() if not path.startswith("gs://")]
        if gcs_names and local_names:
            raise ValueError(
                f"local merge smoke paths must be all local or all GCS; GCS={gcs_names}, local={local_names}"
            )

    payload: dict[str, Any] = {"paths": material}
    if isinstance(config, RecoveryJobConfig):
        payload["data"] = config.data
    check_gcs_paths_same_region(payload, local_ok=local_ok, skip_if_prefix_contains=())


def initialize_merge_worker() -> None:
    """Initialize logging and JAX distributed state before constructing a mesh."""
    logging.basicConfig(level=logging.INFO)
    DistributedConfig().initialize()


def compact_merge_mesh(*, expert_axis_size: int = 1, replica_axis_size: int | None = None) -> jax.sharding.Mesh:
    """Build a merge mesh that keeps non-expert parameter dimensions whole.

    Merge workers may expose a whole TPU slice as local devices in one process,
    unlike the process-per-device training layout. Put every device not used for
    expert parallelism on the replica axis so the local MoE fallback does not
    shard its hidden/output dimensions over ``data``.
    """
    if replica_axis_size is None:
        global_device_count = jax.device_count()
        if global_device_count % expert_axis_size != 0:
            raise ValueError(
                f"global device count {global_device_count} must be divisible by expert axis size {expert_axis_size}"
            )
        replica_axis_size = global_device_count // expert_axis_size
    return compact_grug_mesh(
        expert_axis_size=expert_axis_size,
        replica_axis_size=replica_axis_size,
    )


def _checkpoint_root(output_path: str) -> str:
    return prefix_join(output_path, _CHECKPOINTS_DIRECTORY)


def _sync(name: str) -> None:
    multihost_utils.sync_global_devices(name)


def _write_json_process_zero(path: str, payload: dict[str, Any], *, sync_name: str) -> None:
    _sync(f"{sync_name}_before_json")
    if jax.process_index() == 0:
        StoragePath(path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    _sync(f"{sync_name}_after_json")


def _save_permanent_checkpoint(
    tree,
    *,
    checkpoint_root: str,
    step: int,
    sync_name: str,
    merge_manifest: MergeCheckpointManifest | None = None,
    extra_manifest: tuple[str, dict[str, Any]] | None = None,
) -> str:
    """Collectively save a permanent checkpoint, then commit process-0 manifests."""
    checkpoint_path = prefix_join(checkpoint_root, f"step-{step}")
    save_checkpoint(tree, step=step, checkpoint_path=checkpoint_path, is_temporary=False)
    _sync(f"{sync_name}_checkpoint_saved")
    if merge_manifest is not None:
        if jax.process_index() == 0:
            write_merge_checkpoint_manifest(checkpoint_path, merge_manifest)
        _sync(f"{sync_name}_merge_manifest")
    if extra_manifest is not None:
        filename, payload = extra_manifest
        _write_json_process_zero(
            prefix_join(checkpoint_path, filename),
            payload,
            sync_name=f"{sync_name}_extra_manifest",
        )
    return checkpoint_path


def _source_state(
    source: SourceCheckpointConfig,
    *,
    key: jax.Array,
    mesh: jax.sharding.Mesh,
) -> tuple[GrugTrainState, str]:
    """Load exact untied model and QB state without the source optimizer."""
    expected_topology = tuple(range(source.model.num_layers))
    if source.model.resolved_expert_bank_for_layer != expected_topology:
        raise ValueError(
            f"expert merge source must be fully untied; expected {expected_topology}, "
            f"got {source.model.resolved_expert_bank_for_layer}"
        )
    concrete_path = latest_checkpoint_path(source.checkpoint_dir)
    params = Transformer.init(source.model, key=key)
    pending_qb_betas = jnp.zeros((len(params.blocks), params.config.num_experts), dtype=jnp.float32)
    exemplar: dict[str, Any] = {
        "step": jnp.array(0, dtype=jnp.int32),
        "params": params,
        "pending_qb_betas": pending_qb_betas,
    }
    loaded = cast(
        "dict[str, Any]",
        load_checkpoint(exemplar, concrete_path, mesh=mesh, allow_partial=False),
    )
    return (
        GrugTrainState(
            step=loaded["step"],
            params=loaded["params"],
            opt_state=optax.EmptyState(),
            ema_params=None,
            pending_qb_betas=loaded["pending_qb_betas"],
        ),
        concrete_path,
    )


def _matching_manifest(
    path: str,
    *,
    representative_layer: int,
    source_layer: int,
    num_experts: int,
) -> MatchingArtifactManifest:
    manifest = read_matching_manifest(path)
    if (manifest.representative_layer, manifest.source_layer) != (representative_layer, source_layer):
        raise ValueError(
            "matching layers do not match the worker config: "
            f"{(manifest.representative_layer, manifest.source_layer)} != "
            f"{(representative_layer, source_layer)}"
        )
    if manifest.num_experts != num_experts:
        raise ValueError(f"matching artifact has {manifest.num_experts} experts, expected {num_experts}")
    return manifest


def _validate_matching_calibration(manifest: MatchingArtifactManifest, calibration_path: str) -> None:
    if manifest.calibration_path != calibration_path:
        raise ValueError(
            f"matching artifact refers to calibration {manifest.calibration_path}, expected {calibration_path}"
        )


def _prefit_inputs(
    calibration_path: str,
    matching_path: str,
    *,
    layer: int,
    expert: int,
    include_spectral: bool,
) -> tuple[np.ndarray, np.ndarray]:
    calibration = read_expert_calibration(calibration_path, layer, expert)
    train_inputs = [calibration.train.states]
    heldout_inputs = [calibration.heldout.states]
    if include_spectral:
        probe = read_expert_probe(matching_path, expert)
        train_inputs.extend(
            [
                probe.centers,
                probe.spectral_pairs.reshape(-1, probe.spectral_pairs.shape[-1]),
            ]
        )
        heldout_inputs.append(probe.ordinary_inputs)
    return np.concatenate(train_inputs), np.concatenate(heldout_inputs)


def _inputs_with_probe(calibration, probe: ExpertProbeSet) -> tuple[np.ndarray, np.ndarray]:
    train_inputs = np.concatenate(
        [
            calibration.train.states,
            probe.centers,
            probe.spectral_pairs.reshape(-1, probe.spectral_pairs.shape[-1]),
        ]
    )
    heldout_inputs = np.concatenate([calibration.heldout.states, probe.ordinary_inputs])
    return train_inputs, heldout_inputs


def _distributed_prefit_dataset(
    source_bank: MoEExpertMlp,
    *,
    source_layer: int,
    source_expert: int,
    shared_expert: int,
    train_inputs: np.ndarray,
    heldout_inputs: np.ndarray,
) -> PrefitDataset:
    """Build source targets collectively when expert outputs span multiple hosts."""
    train_targets = multihost_utils.process_allgather(
        eval_expert(source_bank, source_expert, train_inputs),
        tiled=True,
    )
    heldout_targets = multihost_utils.process_allgather(
        eval_expert(source_bank, source_expert, heldout_inputs),
        tiled=True,
    )
    return PrefitDataset(
        source_layer=source_layer,
        source_expert=source_expert,
        shared_expert=shared_expert,
        train_inputs=np.asarray(train_inputs),
        train_targets=np.asarray(train_targets),
        heldout_inputs=np.asarray(heldout_inputs),
        heldout_targets=np.asarray(heldout_targets),
    )


def _prefit_datasets(
    teacher: Transformer,
    config: PrefitJobConfig,
    matching: MatchingArtifactManifest,
) -> tuple[PrefitDataset, ...]:
    assignment = np.asarray(matching.assignments[config.assignment_mode], dtype=np.int32)
    representative_bank = teacher.expert_banks[teacher.blocks[config.representative_layer].expert_bank_index]
    source_bank = teacher.expert_banks[teacher.blocks[config.source_layer].expert_bank_index]
    datasets: list[PrefitDataset] = []
    for shared_expert in range(teacher.config.num_experts):
        calibration = read_expert_calibration(
            config.calibration_path,
            config.representative_layer,
            shared_expert,
        )
        probe = build_spectral_probe_set(
            representative_bank,
            shared_expert,
            calibration.train,
            calibration.heldout,
            config=config.probe,
            seed=config.seed + shared_expert,
        )
        train_inputs, heldout_inputs = _inputs_with_probe(calibration, probe)
        datasets.append(
            _distributed_prefit_dataset(
                representative_bank,
                source_layer=config.representative_layer,
                source_expert=shared_expert,
                shared_expert=shared_expert,
                train_inputs=train_inputs,
                heldout_inputs=heldout_inputs,
            )
        )

    for source_expert, shared_expert in enumerate(assignment):
        train_inputs, heldout_inputs = _prefit_inputs(
            config.calibration_path,
            config.matching_path,
            layer=config.source_layer,
            expert=source_expert,
            include_spectral=True,
        )
        datasets.append(
            _distributed_prefit_dataset(
                source_bank,
                source_layer=config.source_layer,
                source_expert=source_expert,
                shared_expert=int(shared_expert),
                train_inputs=train_inputs,
                heldout_inputs=heldout_inputs,
            )
        )
    return tuple(datasets)


def _aggregate_prefit_datasets(
    config: PrefitJobConfig,
    matching: MatchingArtifactManifest,
) -> tuple[AggregatePrefitDataset, ...]:
    identity = tuple(range(matching.num_experts))
    source_assignment = tuple(int(index) for index in matching.assignments[config.assignment_mode])
    return (
        make_aggregate_prefit_dataset(
            read_layer_calibration_trace(config.calibration_path, config.representative_layer),
            source_layer=config.representative_layer,
            source_to_shared=identity,
            heldout_fraction=config.config.aggregate_trace_heldout_fraction,
            seed=config.seed + config.representative_layer,
        ),
        make_aggregate_prefit_dataset(
            read_layer_calibration_trace(config.calibration_path, config.source_layer),
            source_layer=config.source_layer,
            source_to_shared=source_assignment,
            heldout_fraction=config.config.aggregate_trace_heldout_fraction,
            seed=config.seed + config.source_layer,
        ),
    )


def _initial_prefit_state(bank: MoEExpertMlp, config: PrefitJobConfig) -> PrefitRuntimeState:
    optimizer = optax.adamw(config.config.learning_rate, weight_decay=config.config.weight_decay)
    return PrefitRuntimeState(
        step=jnp.array(0, dtype=jnp.int32),
        bank=bank,
        opt_state=optimizer.init(bank),
        best_bank=bank,
        best_loss=jnp.array(jnp.inf, dtype=jnp.float32),
        stale_evaluations=jnp.array(0, dtype=jnp.int32),
    )


def _prefit_step(
    state: PrefitRuntimeState,
    batch: PrefitBatch,
    optimizer: optax.GradientTransformation,
    epsilon: float,
) -> tuple[PrefitRuntimeState, jax.Array]:
    loss, grads = jax.value_and_grad(lambda bank: prefit_loss(bank, batch, epsilon=epsilon)[0])(state.bank)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.bank)
    return (
        dataclasses.replace(
            state,
            step=state.step + jnp.array(1, dtype=jnp.int32),
            bank=optax.apply_updates(state.bank, updates),
            opt_state=opt_state,
        ),
        loss,
    )


def _evaluate_prefit(
    state: PrefitRuntimeState,
    heldout_batch: PrefitBatch,
    *,
    epsilon: float,
) -> tuple[PrefitRuntimeState, float, np.ndarray]:
    loss, nrmse = prefit_loss(state.bank, heldout_batch, epsilon=epsilon)
    loss_value = float(jax.device_get(loss))
    if loss_value < float(jax.device_get(state.best_loss)):
        state = dataclasses.replace(
            state,
            best_bank=state.bank,
            best_loss=jnp.asarray(loss, dtype=jnp.float32),
            stale_evaluations=jnp.array(0, dtype=jnp.int32),
        )
    else:
        state = dataclasses.replace(state, stale_evaluations=state.stale_evaluations + 1)
    return state, loss_value, np.asarray(jax.device_get(nrmse))


def _aggregate_prefit_step(
    state: PrefitRuntimeState,
    batch: AggregatePrefitBatch,
    optimizer: optax.GradientTransformation,
    epsilon: float,
) -> tuple[PrefitRuntimeState, jax.Array]:
    loss, grads = jax.value_and_grad(lambda bank: aggregate_prefit_loss(bank, batch, epsilon=epsilon)[0])(state.bank)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.bank)
    return (
        dataclasses.replace(
            state,
            step=state.step + jnp.array(1, dtype=jnp.int32),
            bank=optax.apply_updates(state.bank, updates),
            opt_state=opt_state,
        ),
        loss,
    )


def _evaluate_aggregate_prefit(
    state: PrefitRuntimeState,
    heldout_batch: AggregatePrefitBatch,
    *,
    epsilon: float,
) -> tuple[PrefitRuntimeState, float, np.ndarray]:
    loss, nrmse = aggregate_prefit_loss(state.bank, heldout_batch, epsilon=epsilon)
    loss_value = float(jax.device_get(loss))
    if loss_value < float(jax.device_get(state.best_loss)):
        state = dataclasses.replace(
            state,
            best_bank=state.bank,
            best_loss=jnp.asarray(loss, dtype=jnp.float32),
            stale_evaluations=jnp.array(0, dtype=jnp.int32),
        )
    else:
        state = dataclasses.replace(state, stale_evaluations=state.stale_evaluations + 1)
    return state, loss_value, np.asarray(jax.device_get(nrmse))


def evaluate_prefit_routed_nrmse(
    bank: MoEExpertMlp,
    config: PrefitJobConfig,
    matching: MatchingArtifactManifest,
) -> dict[int, float]:
    """Evaluate aggregate persisted calibration traces for both merged layers."""
    assignment = tuple(int(index) for index in matching.assignments[config.assignment_mode])
    identity = tuple(range(matching.num_experts))
    assignments = {
        config.representative_layer: identity,
        config.source_layer: assignment,
    }
    metrics = {}
    for layer, source_to_shared in assignments.items():
        trace = read_layer_calibration_trace(config.calibration_path, layer)
        nrmse = aggregate_routed_moe_nrmse(
            bank,
            trace,
            source_to_shared,
            epsilon=config.config.epsilon,
        )
        metrics[layer] = float(jax.device_get(nrmse))
    return metrics


def evaluate_prefit_checkpoint_local(config: PrefitJobConfig) -> dict[int, float]:
    """Score a saved prefit bank from persisted traces without resuming training."""
    _validate_runtime_paths(
        config,
        {
            "teacher_checkpoint": config.source.checkpoint_dir,
            "calibration": config.calibration_path,
            "matching": config.matching_path,
            "prefit_output": _checkpoint_root(config.output_path),
        },
    )
    initialize_merge_worker()
    mesh = compact_merge_mesh(
        expert_axis_size=config.expert_axis_size,
        replica_axis_size=config.replica_axis_size,
    )
    with set_mesh(mesh):
        source_state, source_checkpoint = _source_state(config.source, key=jax.random.key(config.seed), mesh=mesh)
        matching = _matching_manifest(
            config.matching_path,
            representative_layer=config.representative_layer,
            source_layer=config.source_layer,
            num_experts=config.source.model.num_experts,
        )
        _validate_matching_calibration(matching, config.calibration_path)
        checkpoint = discover_latest_checkpoint(_checkpoint_root(config.output_path))
        if checkpoint is None:
            raise ValueError(f"no prefit checkpoint exists under {_checkpoint_root(config.output_path)}")
        _validate_prefit_provenance(config, checkpoint=checkpoint, source_checkpoint=source_checkpoint)
        initial_bank = source_state.params.expert_banks[
            source_state.params.blocks[config.representative_layer].expert_bank_index
        ]
        state = load_checkpoint(_initial_prefit_state(initial_bank, config), checkpoint, mesh=mesh)
        if config.objective is PrefitObjective.AGGREGATE_ROUTED:
            datasets = _aggregate_prefit_datasets(config, matching)
            heldout_batch = sample_aggregate_prefit_batch(
                datasets,
                examples_per_layer=config.config.aggregate_heldout_examples_per_layer,
                split=PrefitSplit.HELDOUT,
                rng=np.random.default_rng(config.seed),
            )
            _, nrmse = aggregate_prefit_loss(state.best_bank, heldout_batch, epsilon=config.config.epsilon)
            values = np.asarray(jax.device_get(nrmse))
            metrics = {dataset.source_layer: float(value) for dataset, value in zip(datasets, values, strict=True)}
            logger.info(
                "Prefit checkpoint %s held-out aggregate routed-MoE NRMSE by layer: %s",
                checkpoint,
                metrics,
            )
            return metrics
        metrics = evaluate_prefit_routed_nrmse(state.best_bank, config, matching)
        logger.info(
            "Prefit checkpoint %s aggregate routed-MoE NRMSE by layer: %s",
            checkpoint,
            metrics,
        )
        return metrics


def _prefit_manifest(
    config: PrefitJobConfig,
    *,
    source_checkpoint: str,
    step: int,
    best_loss: float,
    stopped_early: bool,
    nrmse_by_source: np.ndarray,
    routed_nrmse_by_layer: dict[int, float],
    datasets: tuple[PrefitDataset, ...],
) -> dict[str, Any]:
    nrmse_by_cluster: dict[str, list[dict[str, int | float]]] = {}
    for dataset, nrmse in zip(datasets, nrmse_by_source, strict=True):
        cluster = nrmse_by_cluster.setdefault(f"shared_{dataset.shared_expert:04d}", [])
        cluster.append(
            {
                "source_layer": dataset.source_layer,
                "source_expert": dataset.source_expert,
                "nrmse": float(nrmse),
            }
        )
    return {
        "format_version": _PREFIT_FORMAT_VERSION,
        "assignment_mode": config.assignment_mode.value,
        "objective": config.objective.value,
        "source_checkpoint": source_checkpoint,
        "calibration_path": config.calibration_path,
        "matching_path": config.matching_path,
        "representative_layer": config.representative_layer,
        "source_layer": config.source_layer,
        "step": step,
        "best_loss": best_loss,
        "stopped_early": stopped_early,
        "nrmse_by_source": nrmse_by_source.tolist(),
        "merge/expert_holdout_nrmse_by_cluster": nrmse_by_cluster,
        "merge/prefit_routed_moe_nrmse_by_layer": {
            f"layer_{layer}": value for layer, value in routed_nrmse_by_layer.items()
        },
        "probe_config": json.loads(json.dumps(dataclasses.asdict(config.probe))),
        "prefit_config": _prefit_config_payload(config),
    }


def _aggregate_prefit_manifest(
    config: PrefitJobConfig,
    *,
    source_checkpoint: str,
    step: int,
    best_loss: float,
    stopped_early: bool,
    nrmse_by_layer: np.ndarray,
    datasets: tuple[AggregatePrefitDataset, ...],
) -> dict[str, Any]:
    return {
        "format_version": _PREFIT_FORMAT_VERSION,
        "assignment_mode": config.assignment_mode.value,
        "objective": config.objective.value,
        "source_checkpoint": source_checkpoint,
        "calibration_path": config.calibration_path,
        "matching_path": config.matching_path,
        "representative_layer": config.representative_layer,
        "source_layer": config.source_layer,
        "step": step,
        "best_loss": best_loss,
        "stopped_early": stopped_early,
        "merge/prefit_heldout_routed_moe_nrmse_by_layer": {
            f"layer_{dataset.source_layer}": float(nrmse)
            for dataset, nrmse in zip(datasets, nrmse_by_layer, strict=True)
        },
        "aggregate_trace_split": {
            f"layer_{dataset.source_layer}": {
                "train_tokens": int(dataset.train.mlp_input.shape[0]),
                "heldout_tokens": int(dataset.heldout.mlp_input.shape[0]),
            }
            for dataset in datasets
        },
        "prefit_config": _prefit_config_payload(config),
    }


def _prefit_config_payload(config: PrefitJobConfig) -> dict[str, Any]:
    payload = json.loads(json.dumps(dataclasses.asdict(config.config)))
    if config.objective is PrefitObjective.PER_EXPERT:
        payload.pop("aggregate_examples_per_layer")
        payload.pop("aggregate_heldout_examples_per_layer")
        payload.pop("aggregate_trace_heldout_fraction")
    else:
        payload.pop("examples_per_source")
        payload.pop("heldout_examples_per_source")
    return payload


def _validate_prefit_provenance(
    config: PrefitJobConfig,
    *,
    checkpoint: str,
    source_checkpoint: str,
) -> None:
    manifest = json.loads(StoragePath(prefix_join(checkpoint, _PREFIT_MANIFEST_FILENAME)).read_text())
    actual = {
        **manifest,
        "assignment_mode": manifest.get("assignment_mode", AssignmentMode.SPECTRAL.value),
        "objective": manifest.get("objective", PrefitObjective.PER_EXPERT.value),
    }
    expected = {
        "assignment_mode": config.assignment_mode.value,
        "objective": config.objective.value,
        "source_checkpoint": source_checkpoint,
        "calibration_path": config.calibration_path,
        "matching_path": config.matching_path,
        "representative_layer": config.representative_layer,
        "source_layer": config.source_layer,
        "prefit_config": _prefit_config_payload(config),
    }
    if config.objective is PrefitObjective.PER_EXPERT:
        expected["probe_config"] = json.loads(json.dumps(dataclasses.asdict(config.probe)))
    mismatches = {key: (actual.get(key), value) for key, value in expected.items() if actual.get(key) != value}
    if mismatches:
        raise ValueError(f"existing prefit checkpoint has stale provenance: {mismatches}")


def _run_aggregate_prefit(
    config: PrefitJobConfig,
    *,
    matching: MatchingArtifactManifest,
    state: PrefitRuntimeState,
    optimizer: optax.GradientTransformation,
    checkpoint_root: str,
    source_checkpoint: str,
) -> None:
    datasets = _aggregate_prefit_datasets(config, matching)
    heldout_batch = sample_aggregate_prefit_batch(
        datasets,
        examples_per_layer=config.config.aggregate_heldout_examples_per_layer,
        split=PrefitSplit.HELDOUT,
        rng=np.random.default_rng(config.seed),
    )
    if int(state.step) == 0 and math.isinf(float(jax.device_get(state.best_loss))):
        state, _, _ = _evaluate_aggregate_prefit(state, heldout_batch, epsilon=config.config.epsilon)

    step_fn = jax.jit(_aggregate_prefit_step, static_argnums=(2, 3))
    stopped_early = int(state.stale_evaluations) >= config.config.early_stopping_patience
    while int(state.step) < config.config.steps and not stopped_early:
        next_step = int(state.step) + 1
        batch = sample_aggregate_prefit_batch(
            datasets,
            examples_per_layer=config.config.aggregate_examples_per_layer,
            split=PrefitSplit.TRAIN,
            rng=np.random.default_rng(config.seed + next_step),
        )
        state, loss = step_fn(state, batch, optimizer, config.config.epsilon)
        if next_step % config.config.eval_every != 0 and next_step != config.config.steps:
            continue
        state, heldout_loss, _ = _evaluate_aggregate_prefit(
            state,
            heldout_batch,
            epsilon=config.config.epsilon,
        )
        _, best_nrmse_by_layer = aggregate_prefit_loss(
            state.best_bank,
            heldout_batch,
            epsilon=config.config.epsilon,
        )
        nrmse_by_layer = np.asarray(jax.device_get(best_nrmse_by_layer))
        stopped_early = int(state.stale_evaluations) >= config.config.early_stopping_patience
        manifest = _aggregate_prefit_manifest(
            config,
            source_checkpoint=source_checkpoint,
            step=int(state.step),
            best_loss=float(jax.device_get(state.best_loss)),
            stopped_early=stopped_early,
            nrmse_by_layer=nrmse_by_layer,
            datasets=datasets,
        )
        _save_permanent_checkpoint(
            state,
            checkpoint_root=checkpoint_root,
            step=int(state.step),
            sync_name=f"aggregate_prefit_{int(state.step)}",
            extra_manifest=(_PREFIT_MANIFEST_FILENAME, manifest),
        )
        if jax.process_index() == 0:
            logger.info(
                "aggregate prefit step=%d train_loss=%g heldout_loss=%g best_loss=%g stale=%d "
                "heldout_routed_moe_nrmse=%s",
                int(state.step),
                float(jax.device_get(loss)),
                heldout_loss,
                float(jax.device_get(state.best_loss)),
                int(state.stale_evaluations),
                {dataset.source_layer: float(value) for dataset, value in zip(datasets, nrmse_by_layer, strict=True)},
            )


def run_prefit_local(config: PrefitJobConfig) -> None:
    if config.config.steps <= 0 or config.config.eval_every <= 0 or config.config.early_stopping_patience <= 0:
        raise ValueError("prefit steps, eval_every, and early_stopping_patience must be positive")
    if config.objective is PrefitObjective.AGGREGATE_ROUTED and (
        config.config.aggregate_examples_per_layer <= 0
        or config.config.aggregate_heldout_examples_per_layer <= 0
        or not 0.0 < config.config.aggregate_trace_heldout_fraction < 1.0
    ):
        raise ValueError("aggregate prefit batch sizes must be positive and heldout fraction must lie in (0, 1)")
    if config.objective is PrefitObjective.AGGREGATE_ROUTED and config.assignment_mode is not AssignmentMode.NATIVE:
        raise ValueError("aggregate routed prefit requires the native Hungarian assignment")
    _validate_runtime_paths(
        config,
        {
            "teacher_checkpoint": config.source.checkpoint_dir,
            "calibration": config.calibration_path,
            "matching": config.matching_path,
            "prefit_output": _checkpoint_root(config.output_path),
        },
    )
    initialize_merge_worker()
    mesh = compact_merge_mesh(
        expert_axis_size=config.expert_axis_size,
        replica_axis_size=config.replica_axis_size,
    )
    with set_mesh(mesh):
        source_state, source_checkpoint = _source_state(config.source, key=jax.random.key(config.seed), mesh=mesh)
        calibration = read_calibration_manifest(config.calibration_path)
        if set(calibration.layers) != {config.representative_layer, config.source_layer}:
            raise ValueError(f"calibration layers {calibration.layers} do not match the prefit pair")
        if calibration.source_checkpoint != source_checkpoint:
            raise ValueError(f"calibration was collected from {calibration.source_checkpoint}, not {source_checkpoint}")
        matching = _matching_manifest(
            config.matching_path,
            representative_layer=config.representative_layer,
            source_layer=config.source_layer,
            num_experts=config.source.model.num_experts,
        )
        _validate_matching_calibration(matching, config.calibration_path)
        initial_bank = source_state.params.expert_banks[
            source_state.params.blocks[config.representative_layer].expert_bank_index
        ]
        optimizer = optax.adamw(config.config.learning_rate, weight_decay=config.config.weight_decay)
        state = _initial_prefit_state(initial_bank, config)
        checkpoint_root = _checkpoint_root(config.output_path)
        own_checkpoint = discover_latest_checkpoint(checkpoint_root)
        if own_checkpoint is not None:
            _validate_prefit_provenance(
                config,
                checkpoint=own_checkpoint,
                source_checkpoint=source_checkpoint,
            )
            state = load_checkpoint(state, own_checkpoint, mesh=mesh)
            if (
                int(state.step) >= config.config.steps
                or int(state.stale_evaluations) >= config.config.early_stopping_patience
            ):
                if config.objective is PrefitObjective.AGGREGATE_ROUTED:
                    datasets = _aggregate_prefit_datasets(config, matching)
                    heldout_batch = sample_aggregate_prefit_batch(
                        datasets,
                        examples_per_layer=config.config.aggregate_heldout_examples_per_layer,
                        split=PrefitSplit.HELDOUT,
                        rng=np.random.default_rng(config.seed),
                    )
                    _, nrmse = aggregate_prefit_loss(
                        state.best_bank,
                        heldout_batch,
                        epsilon=config.config.epsilon,
                    )
                    logger.info(
                        "Loaded completed aggregate prefit output at step %d; held-out NRMSE by layer: %s",
                        int(state.step),
                        {
                            dataset.source_layer: float(value)
                            for dataset, value in zip(
                                datasets,
                                np.asarray(jax.device_get(nrmse)),
                                strict=True,
                            )
                        },
                    )
                    return
                routed_nrmse_by_layer = evaluate_prefit_routed_nrmse(state.best_bank, config, matching)
                logger.info(
                    "Loaded completed prefit output at step %d; aggregate routed-MoE NRMSE by layer: %s",
                    int(state.step),
                    routed_nrmse_by_layer,
                )
                return

        if config.objective is PrefitObjective.AGGREGATE_ROUTED:
            _run_aggregate_prefit(
                config,
                matching=matching,
                state=state,
                optimizer=optimizer,
                checkpoint_root=checkpoint_root,
                source_checkpoint=source_checkpoint,
            )
            return
        if config.objective is not PrefitObjective.PER_EXPERT:
            raise AssertionError(f"unhandled prefit objective {config.objective}")

        datasets = _prefit_datasets(source_state.params, config, matching)
        heldout_batch = sample_prefit_batch(
            datasets,
            examples_per_source=config.config.heldout_examples_per_source,
            split=PrefitSplit.HELDOUT,
            rng=np.random.default_rng(config.seed),
        )
        if int(state.step) == 0 and math.isinf(float(jax.device_get(state.best_loss))):
            state, _, _ = _evaluate_prefit(state, heldout_batch, epsilon=config.config.epsilon)

        step_fn = jax.jit(_prefit_step, static_argnums=(2, 3))
        last_nrmse = np.zeros((len(datasets),), dtype=np.float32)
        stopped_early = int(state.stale_evaluations) >= config.config.early_stopping_patience
        while int(state.step) < config.config.steps and not stopped_early:
            next_step = int(state.step) + 1
            batch = sample_prefit_batch(
                datasets,
                examples_per_source=config.config.examples_per_source,
                split=PrefitSplit.TRAIN,
                rng=np.random.default_rng(config.seed + next_step),
            )
            state, loss = step_fn(state, batch, optimizer, config.config.epsilon)
            if next_step % config.config.eval_every != 0 and next_step != config.config.steps:
                continue
            state, heldout_loss, last_nrmse = _evaluate_prefit(
                state,
                heldout_batch,
                epsilon=config.config.epsilon,
            )
            routed_nrmse_by_layer = evaluate_prefit_routed_nrmse(state.best_bank, config, matching)
            stopped_early = int(state.stale_evaluations) >= config.config.early_stopping_patience
            manifest = _prefit_manifest(
                config,
                source_checkpoint=source_checkpoint,
                step=int(state.step),
                best_loss=float(jax.device_get(state.best_loss)),
                stopped_early=stopped_early,
                nrmse_by_source=last_nrmse,
                routed_nrmse_by_layer=routed_nrmse_by_layer,
                datasets=datasets,
            )
            _save_permanent_checkpoint(
                state,
                checkpoint_root=checkpoint_root,
                step=int(state.step),
                sync_name=f"prefit_{int(state.step)}",
                extra_manifest=(_PREFIT_MANIFEST_FILENAME, manifest),
            )
            if jax.process_index() == 0:
                logger.info(
                    "prefit step=%d train_loss=%g heldout_loss=%g best_loss=%g stale=%d "
                    "aggregate_routed_moe_nrmse=%s",
                    int(state.step),
                    float(jax.device_get(loss)),
                    heldout_loss,
                    float(jax.device_get(state.best_loss)),
                    int(state.stale_evaluations),
                    routed_nrmse_by_layer,
                )


def _load_prefitted_bank(
    config: ConversionJobConfig,
    initial_bank: MoEExpertMlp,
    *,
    mesh: jax.sharding.Mesh,
) -> tuple[MoEExpertMlp | None, str | None, PrefitObjective | None]:
    if config.prefit_path is None:
        return None, None, None
    checkpoint = latest_checkpoint_path(config.prefit_path)
    manifest = json.loads(StoragePath(prefix_join(checkpoint, _PREFIT_MANIFEST_FILENAME)).read_text())
    if int(manifest["format_version"]) != _PREFIT_FORMAT_VERSION:
        raise ValueError(f"unsupported prefit checkpoint format {manifest['format_version']}")
    if manifest["matching_path"] != config.matching_path:
        raise ValueError(f"prefit checkpoint refers to matching artifact {manifest['matching_path']}")
    if (int(manifest["representative_layer"]), int(manifest["source_layer"])) != (
        config.representative_layer,
        config.source_layer,
    ):
        raise ValueError("prefit checkpoint layers do not match the conversion")
    assignment_mode = AssignmentMode(manifest.get("assignment_mode", AssignmentMode.SPECTRAL.value))
    objective = PrefitObjective(manifest.get("objective", PrefitObjective.PER_EXPERT.value))
    if assignment_mode is not config.assignment_mode:
        raise ValueError(
            f"prefit checkpoint uses {assignment_mode.value} assignment, conversion uses {config.assignment_mode.value}"
        )
    optimizer = optax.adamw(1e-4, weight_decay=0.0)
    template = PrefitRuntimeState(
        step=jnp.array(0, dtype=jnp.int32),
        bank=initial_bank,
        opt_state=optimizer.init(initial_bank),
        best_bank=initial_bank,
        best_loss=jnp.array(jnp.inf, dtype=jnp.float32),
        stale_evaluations=jnp.array(0, dtype=jnp.int32),
    )
    restored = load_checkpoint(template, checkpoint, mesh=mesh)
    return restored.best_bank, checkpoint, objective


def _validate_conversion_provenance(
    config: ConversionJobConfig,
    manifest: MergeCheckpointManifest,
    matching: MatchingArtifactManifest,
) -> None:
    spec = manifest.spec
    expected_prefit = latest_checkpoint_path(config.prefit_path) if config.prefit_path is not None else None
    expected_prefit_objective = None
    if expected_prefit is not None:
        prefit_manifest = json.loads(StoragePath(prefix_join(expected_prefit, _PREFIT_MANIFEST_FILENAME)).read_text())
        expected_prefit_objective = PrefitObjective(prefit_manifest.get("objective", PrefitObjective.PER_EXPERT.value))
    expected = {
        "source_checkpoint": latest_checkpoint_path(config.source.checkpoint_dir),
        "calibration_path": config.calibration_path,
        "cost_matrix_path": prefix_join(config.matching_path, "cost_matrix.npz"),
        "probe_path": prefix_join(config.matching_path, "probes"),
        "prefit_checkpoint": expected_prefit,
        "prefit_objective": expected_prefit_objective,
    }
    actual = {name: getattr(spec, name) for name in expected}
    mismatches = {name: (actual[name], value) for name, value in expected.items() if actual[name] != value}
    if mismatches:
        raise ValueError(f"existing conversion checkpoint has stale provenance: {mismatches}")
    _validate_assignment_provenance(manifest, matching)


def _validate_merge_manifest(
    manifest: MergeCheckpointManifest,
    *,
    assignment_mode: AssignmentMode,
    prefit_applied: bool,
    affected_layers: tuple[int, int],
) -> None:
    spec = manifest.spec
    if spec.assignment_mode is not assignment_mode:
        raise ValueError(f"checkpoint assignment mode is {spec.assignment_mode}, expected {assignment_mode}")
    if spec.prefit_applied != prefit_applied:
        raise ValueError(f"checkpoint prefit_applied={spec.prefit_applied}, expected {prefit_applied}")
    if (spec.representative_layer, spec.source_layer) != affected_layers:
        raise ValueError(
            f"checkpoint merged layers {(spec.representative_layer, spec.source_layer)}, expected {affected_layers}"
        )


def _validate_assignment_provenance(
    manifest: MergeCheckpointManifest,
    matching: MatchingArtifactManifest,
) -> None:
    expected = matching.assignments[manifest.spec.assignment_mode]
    if manifest.spec.source_to_shared != expected:
        raise ValueError("merge checkpoint assignment does not match the persisted matching artifact")


def run_conversion_local(config: ConversionJobConfig) -> None:
    checkpoint_root = _checkpoint_root(config.output_path)
    _validate_runtime_paths(
        config,
        {
            "teacher_checkpoint": config.source.checkpoint_dir,
            "calibration": config.calibration_path,
            "matching": config.matching_path,
            "prefit_checkpoint": config.prefit_path,
            "converted_output": checkpoint_root,
        },
    )
    initialize_merge_worker()
    mesh = compact_merge_mesh(
        expert_axis_size=config.expert_axis_size,
        replica_axis_size=config.replica_axis_size,
    )
    with set_mesh(mesh):
        matching = _matching_manifest(
            config.matching_path,
            representative_layer=config.representative_layer,
            source_layer=config.source_layer,
            num_experts=config.source.model.num_experts,
        )
        _validate_matching_calibration(matching, config.calibration_path)
        own_checkpoint = discover_latest_checkpoint(checkpoint_root)
        if own_checkpoint is not None:
            manifest = read_merge_checkpoint_manifest(own_checkpoint)
            _validate_merge_manifest(
                manifest,
                assignment_mode=config.assignment_mode,
                prefit_applied=config.prefit_path is not None,
                affected_layers=(config.representative_layer, config.source_layer),
            )
            _validate_conversion_provenance(config, manifest, matching)
            logger.info("Converted output already exists at %s", own_checkpoint)
            return

        source_state, source_checkpoint = _source_state(config.source, key=jax.random.key(0), mesh=mesh)
        calibration = read_calibration_manifest(config.calibration_path)
        if calibration.source_checkpoint != source_checkpoint:
            raise ValueError(f"calibration was collected from {calibration.source_checkpoint}, not {source_checkpoint}")
        assignment = matching.assignments[config.assignment_mode]
        representative_bank = source_state.params.expert_banks[
            source_state.params.blocks[config.representative_layer].expert_bank_index
        ]
        prefitted_bank, prefit_checkpoint, prefit_objective = _load_prefitted_bank(
            config,
            representative_bank,
            mesh=mesh,
        )
        spec = OnePairMergeCheckpointSpec(
            representative_layer=config.representative_layer,
            source_layer=config.source_layer,
            source_to_shared=assignment,
            assignment_mode=config.assignment_mode,
            source_checkpoint=source_checkpoint,
            source_commit=calibration.source_commit,
            calibration_path=config.calibration_path,
            cost_matrix_path=prefix_join(config.matching_path, "cost_matrix.npz"),
            probe_path=prefix_join(config.matching_path, "probes"),
            prefit_applied=prefitted_bank is not None,
            prefit_checkpoint=prefit_checkpoint,
            prefit_objective=prefit_objective,
        )
        converted = convert_grug_state_for_one_pair_merge(
            source_state,
            spec=spec,
            init_optimizer_state=lambda _: optax.EmptyState(),
            shared_bank=prefitted_bank,
        )
        _save_permanent_checkpoint(
            converted.state,
            checkpoint_root=checkpoint_root,
            step=0,
            sync_name="conversion_0",
            merge_manifest=converted.manifest,
        )


def run_capacity_oracle_split_local(config: CapacityOracleSplitJobConfig) -> None:
    """Duplicate the selected recovered bank into an untied, function-identical diagnostic checkpoint."""
    checkpoint_root = _checkpoint_root(config.output_path)
    _validate_runtime_paths(
        config,
        {
            "teacher_checkpoint": config.source.checkpoint_dir,
            "selected_recovery_checkpoint": config.init_checkpoint_dir,
            "capacity_oracle_output": checkpoint_root,
        },
    )
    initialize_merge_worker()
    mesh = compact_merge_mesh(
        expert_axis_size=config.expert_axis_size,
        replica_axis_size=config.replica_axis_size,
    )
    with set_mesh(mesh):
        selection = _best_validation_selection(config.init_checkpoint_dir)
        source_checkpoint = str(selection["checkpoint_path"])
        _validate_runtime_paths(config, {"resolved_recovery_checkpoint": source_checkpoint})
        own_checkpoint = discover_latest_checkpoint(checkpoint_root)
        if own_checkpoint is not None:
            manifest = read_merge_checkpoint_manifest(own_checkpoint)
            _validate_merge_manifest(
                manifest,
                assignment_mode=config.assignment_mode,
                prefit_applied=config.prefit_applied,
                affected_layers=config.affected_layers,
            )
            expected_topology = tuple(range(config.source.model.num_layers))
            if (
                manifest.capacity_oracle is None
                or manifest.capacity_oracle.kind is not CapacityOracleKind.UNTIED_IDENTICAL_START_DIAGNOSTIC
                or manifest.capacity_oracle.source_checkpoint != source_checkpoint
                or manifest.target_topology != expected_topology
                or manifest.recovery_step != 0
            ):
                raise ValueError("existing capacity-oracle checkpoint has stale provenance")
            logger.info("Capacity-oracle split already exists at %s", own_checkpoint)
            return

        source_manifest = read_merge_checkpoint_manifest(source_checkpoint)
        _validate_merge_manifest(
            source_manifest,
            assignment_mode=config.assignment_mode,
            prefit_applied=config.prefit_applied,
            affected_layers=config.affected_layers,
        )
        model_config = dataclasses.replace(config.source.model, expert_bank_for_layer=source_manifest.target_topology)
        params = Transformer.init(model_config, key=jax.random.key(0))
        pending_qb_betas = jnp.zeros(
            (len(params.blocks), params.config.num_experts),
            dtype=jnp.float32,
        )
        loaded = cast(
            "dict[str, Any]",
            load_checkpoint(
                {
                    "step": jnp.asarray(source_manifest.recovery_step, dtype=jnp.int32),
                    "params": params,
                    "pending_qb_betas": pending_qb_betas,
                },
                source_checkpoint,
                mesh=mesh,
                allow_partial=False,
            ),
        )
        loaded_step = int(jax.device_get(loaded["step"]))
        if loaded_step != source_manifest.recovery_step or loaded_step != int(selection["step"]):
            raise ValueError(
                "selected recovery checkpoint step disagrees with selector or manifest: "
                f"checkpoint={loaded_step}, selector={selection['step']}, manifest={source_manifest.recovery_step}"
            )
        recovered = GrugTrainState(
            step=loaded["step"],
            params=loaded["params"],
            opt_state=optax.EmptyState(),
            ema_params=None,
            pending_qb_betas=loaded["pending_qb_betas"],
        )
        oracle = convert_grug_state_for_capacity_oracle_split(
            recovered,
            source_manifest=source_manifest,
            source_checkpoint=source_checkpoint,
            init_optimizer_state=lambda _: optax.EmptyState(),
        )
        _save_permanent_checkpoint(
            oracle.state,
            checkpoint_root=checkpoint_root,
            step=0,
            sync_name="capacity_oracle_split_0",
            merge_manifest=oracle.manifest,
        )


def _validate_selected_local_recovery(
    config: LayerAdapterAugmentJobConfig,
    manifest: MergeCheckpointManifest,
    selection: dict[str, Any],
) -> None:
    """Validate modern provenance or the one known legacy CE+KL Stage-A artifact."""
    selected_step = int(selection["step"])
    expected = {
        "recovery_step": selected_step,
        "recovery_initialization": RecoveryInitialization.CONVERTED_STEP_ZERO,
        "recovery_stage": RecoveryStage.LOCAL,
        "recovery_trainable_scope": RecoveryTrainableScope.SHARED_BANK,
        "recovery_cross_entropy_weight": 0.05,
        "recovery_moe_loss_weight": 1.0,
        "recovery_logit_kl_weight": 0.1,
    }
    manifest_mismatches = {
        name: (getattr(manifest, name), value) for name, value in expected.items() if getattr(manifest, name) != value
    }
    if manifest.capacity_oracle is not None or manifest.layer_adapter is not None:
        raise ValueError("adapter augmentation requires an ordinary adapter-free tied recovery checkpoint")
    if not manifest_mismatches:
        return

    legacy_fields = (
        manifest.recovery_stage,
        manifest.recovery_trainable_scope,
        manifest.recovery_cross_entropy_weight,
        manifest.recovery_moe_loss_weight,
        manifest.recovery_logit_kl_weight,
    )
    if (
        manifest.format_version != 2
        or selected_step != _LEGACY_CE_KL_STAGE_A_STEP
        or manifest.recovery_step != _LEGACY_CE_KL_STAGE_A_STEP
        or any(value is not None for value in legacy_fields)
    ):
        raise ValueError(f"adapter augmentation requires the selected CE+KL local recovery: {manifest_mismatches}")
    if manifest.recovery_initialization is not RecoveryInitialization.CONVERTED_STEP_ZERO:
        raise ValueError(f"adapter augmentation requires the selected CE+KL local recovery: {manifest_mismatches}")
    _validate_legacy_ce_kl_stage_a_artifact(config, manifest=manifest, selection=selection)


def _validate_legacy_ce_kl_stage_a_artifact(
    config: LayerAdapterAugmentJobConfig,
    *,
    manifest: MergeCheckpointManifest,
    selection: dict[str, Any],
) -> None:
    artifact_root = str(StoragePath(config.init_checkpoint_dir).parent)
    artifact_path = prefix_join(artifact_root, ".artifact.json")
    artifact = StoragePath(artifact_path)
    if not artifact.exists():
        raise ValueError(f"legacy CE+KL Stage-A provenance is missing at {artifact_path}")
    payload = json.loads(artifact.read_text())
    expected_record = {
        "name": _LEGACY_CE_KL_STAGE_A_NAME,
        "version": _LEGACY_CE_KL_STAGE_A_VERSION,
        "fingerprint": _LEGACY_CE_KL_STAGE_A_FINGERPRINT,
        "output_path": artifact_root,
    }
    record_mismatches: dict[str, tuple[Any, Any]] = {
        name: (payload.get(name), value) for name, value in expected_record.items() if payload.get(name) != value
    }
    artifact_config = payload.get("config")
    if not isinstance(artifact_config, dict):
        raise ValueError(f"legacy CE+KL Stage-A artifact config is missing at {artifact_path}")
    expected_config = {
        "run_id": "grug-xem-native_local_ce_kl-stage-a-d512-l2-l3",
        "stage": RecoveryStage.LOCAL.value,
        "initialization": RecoveryInitialization.CONVERTED_STEP_ZERO.value,
        "initial_checkpoint_selection": RecoveryCheckpointSelection.LATEST.value,
        "affected_layers": list(config.affected_layers),
        "assignment_mode": config.assignment_mode.value,
        "prefit_applied": config.prefit_applied,
        "batch_size": 32,
        "training_tokens": 50_000_000,
        "cross_entropy_weight": 0.05,
        "moe_loss_weight": 1.0,
        "logit_kl_weight": 0.1,
        "select_best_validation_checkpoint": True,
    }
    config_mismatches: dict[str, tuple[Any, Any]] = {
        name: (artifact_config.get(name), value)
        for name, value in expected_config.items()
        if artifact_config.get(name) != value
    }
    if "trainable_scope" in artifact_config:
        config_mismatches["trainable_scope"] = (artifact_config["trainable_scope"], "absent in legacy schema")

    expected_source_topology = config.source.model.resolved_expert_bank_for_layer
    manifest_mismatches: dict[str, tuple[Any, Any]] = {}
    if manifest.source_topology != expected_source_topology:
        manifest_mismatches["source_topology"] = (manifest.source_topology, expected_source_topology)
    if manifest.target_topology != _LEGACY_CE_KL_TARGET_TOPOLOGY:
        manifest_mismatches["target_topology"] = (manifest.target_topology, _LEGACY_CE_KL_TARGET_TOPOLOGY)
    if manifest.source_step != config.source.training_steps:
        manifest_mismatches["source_step"] = (manifest.source_step, config.source.training_steps)
    converted_checkpoint_dir = artifact_config.get("init_checkpoint_dir")
    expected_converted_checkpoint = (
        prefix_join(converted_checkpoint_dir, "step-0") if isinstance(converted_checkpoint_dir, str) else None
    )
    if manifest.recovery_initial_checkpoint != expected_converted_checkpoint:
        config_mismatches["recovery_initial_checkpoint"] = (
            manifest.recovery_initial_checkpoint,
            expected_converted_checkpoint,
        )

    source = artifact_config.get("source")
    expected_source = {
        "checkpoint_dir": config.source.checkpoint_dir,
        "source_commit": config.source.source_commit,
        "training_steps": config.source.training_steps,
    }
    if not isinstance(source, dict):
        config_mismatches["source"] = (source, expected_source)
    else:
        for name, value in expected_source.items():
            if source.get(name) != value:
                config_mismatches[f"source.{name}"] = (source.get(name), value)
        source_model = source.get("model")
        expected_topology = list(config.source.model.resolved_expert_bank_for_layer)
        if not isinstance(source_model, dict) or source_model.get("expert_bank_for_layer") != expected_topology:
            config_mismatches["source.model.expert_bank_for_layer"] = (
                source_model.get("expert_bank_for_layer") if isinstance(source_model, dict) else None,
                expected_topology,
            )

    expected_selection = {
        "step": _LEGACY_CE_KL_STAGE_A_STEP,
        "checkpoint_path": prefix_join(config.init_checkpoint_dir, f"step-{_LEGACY_CE_KL_STAGE_A_STEP}"),
        "tokens": _LEGACY_CE_KL_STAGE_A_TOKENS,
        "requested_tokens": 50_000_000,
        "selection_metric": "eval/paloma/macro_loss",
    }
    selection_mismatches: dict[str, tuple[Any, Any]] = {
        name: (selection.get(name), value) for name, value in expected_selection.items() if selection.get(name) != value
    }
    if record_mismatches or config_mismatches or manifest_mismatches or selection_mismatches:
        raise ValueError(
            "legacy CE+KL Stage-A artifact has stale provenance: "
            f"record={record_mismatches}, config={config_mismatches}, manifest={manifest_mismatches}, "
            f"selection={selection_mismatches}"
        )


def run_layer_adapter_augment_local(config: LayerAdapterAugmentJobConfig) -> None:
    """Add a function-identical adapter to the configured merged source layer."""
    if config.adapter_rank <= 0:
        raise ValueError(f"adapter_rank must be positive, got {config.adapter_rank}")
    checkpoint_root = _checkpoint_root(config.output_path)
    _validate_runtime_paths(
        config,
        {
            "teacher_checkpoint": config.source.checkpoint_dir,
            "selected_recovery_checkpoint": config.init_checkpoint_dir,
            "layer_adapter_output": checkpoint_root,
        },
    )
    initialize_merge_worker()
    mesh = compact_merge_mesh(
        expert_axis_size=config.expert_axis_size,
        replica_axis_size=config.replica_axis_size,
    )
    with set_mesh(mesh):
        selection = _best_validation_selection(config.init_checkpoint_dir)
        source_checkpoint = str(selection["checkpoint_path"])
        selected_step = int(selection["step"])
        _validate_runtime_paths(config, {"resolved_recovery_checkpoint": source_checkpoint})

        source_manifest = read_merge_checkpoint_manifest(source_checkpoint)
        _validate_merge_manifest(
            source_manifest,
            assignment_mode=config.assignment_mode,
            prefit_applied=config.prefit_applied,
            affected_layers=config.affected_layers,
        )
        _validate_selected_local_recovery(config, source_manifest, selection)
        teacher_checkpoint = latest_checkpoint_path(config.source.checkpoint_dir)
        if source_manifest.spec.source_checkpoint != teacher_checkpoint:
            raise ValueError(
                f"merge checkpoint refers to teacher {source_manifest.spec.source_checkpoint}, not {teacher_checkpoint}"
            )

        own_checkpoint = discover_latest_checkpoint(checkpoint_root)
        if own_checkpoint is not None:
            manifest = read_merge_checkpoint_manifest(own_checkpoint)
            _validate_merge_manifest(
                manifest,
                assignment_mode=config.assignment_mode,
                prefit_applied=config.prefit_applied,
                affected_layers=config.affected_layers,
            )
            adapter = manifest.layer_adapter
            expected_layer = config.affected_layers[1]
            if (
                adapter is None
                or adapter.kind is not LayerAdapterKind.ZERO_INITIALIZED_INPUT_OUTPUT
                or adapter.source_checkpoint != source_checkpoint
                or adapter.layer_index != expected_layer
                or adapter.rank != config.adapter_rank
                or adapter.input_topology != source_manifest.target_topology
                or adapter.source_recovery_step != selected_step
                or adapter.output_step != 0
                or manifest.target_topology != source_manifest.target_topology
                or manifest.recovery_step != 0
                or manifest.capacity_oracle is not None
            ):
                raise ValueError("existing layer-adapter checkpoint has stale provenance")
            logger.info("Layer-adapter checkpoint already exists at %s", own_checkpoint)
            return

        model_config = dataclasses.replace(
            config.source.model,
            expert_bank_for_layer=source_manifest.target_topology,
            expert_adapter_rank_for_layer=None,
        )
        params = Transformer.init(model_config, key=jax.random.key(config.seed))
        pending_qb_betas = jnp.zeros((len(params.blocks), params.config.num_experts), dtype=jnp.float32)
        loaded = cast(
            "dict[str, Any]",
            load_checkpoint(
                {
                    "step": jnp.asarray(selected_step, dtype=jnp.int32),
                    "params": params,
                    "pending_qb_betas": pending_qb_betas,
                },
                source_checkpoint,
                mesh=mesh,
                allow_partial=False,
            ),
        )
        loaded_step = int(jax.device_get(loaded["step"]))
        if loaded_step != selected_step or loaded_step != source_manifest.recovery_step:
            raise ValueError(
                "selected recovery checkpoint step disagrees with selector or manifest: "
                f"checkpoint={loaded_step}, selector={selected_step}, manifest={source_manifest.recovery_step}"
            )
        recovered = GrugTrainState(
            step=loaded["step"],
            params=loaded["params"],
            opt_state=optax.EmptyState(),
            ema_params=None,
            pending_qb_betas=loaded["pending_qb_betas"],
        )
        augmented = convert_grug_state_for_layer_adapter(
            recovered,
            source_manifest=source_manifest,
            source_checkpoint=source_checkpoint,
            layer_index=config.affected_layers[1],
            rank=config.adapter_rank,
            key=jax.random.fold_in(jax.random.key(config.seed), 1),
            init_optimizer_state=lambda _: optax.EmptyState(),
        )
        _save_permanent_checkpoint(
            augmented.state,
            checkpoint_root=checkpoint_root,
            step=0,
            sync_name="layer_adapter_augment_0",
            merge_manifest=augmented.manifest,
        )


def _recovery_template(
    config: RecoveryJobConfig,
    manifest: MergeCheckpointManifest,
    *,
    key: jax.Array,
) -> tuple[MergeRecoveryState, optax.GradientTransformation, MergeRecoveryConfig]:
    adapter_ranks = [0] * config.source.model.num_layers
    if manifest.layer_adapter is not None:
        adapter_ranks[manifest.layer_adapter.layer_index] = manifest.layer_adapter.rank
    model_config = dataclasses.replace(
        config.source.model,
        expert_bank_for_layer=manifest.target_topology,
        expert_adapter_rank_for_layer=tuple(adapter_ranks),
    )
    params = Transformer.init(model_config, key=key)
    optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    recovery_config = MergeRecoveryConfig(
        affected_layers=config.affected_layers,
        stage=config.stage,
        trainable_scope=config.trainable_scope,
        cross_entropy_weight=config.cross_entropy_weight,
        moe_loss_weight=config.moe_loss_weight,
        logit_kl_weight=config.logit_kl_weight,
        source_to_shared=manifest.spec.source_to_shared,
    )
    state = initial_recovery_state(
        params,
        optimizer=optimizer,
        pending_qb_betas=jnp.zeros((len(params.blocks), params.config.num_experts), dtype=jnp.float32),
        config=recovery_config,
    )
    return state, optimizer, recovery_config


def _load_recovery_initial_weights(
    state: MergeRecoveryState,
    checkpoint_path: str,
    *,
    mesh: jax.sharding.Mesh,
) -> MergeRecoveryState:
    exemplar: dict[str, Any] = {
        "params": state.params,
        "pending_qb_betas": state.pending_qb_betas,
    }
    loaded = cast("dict[str, Any]", load_checkpoint(exemplar, checkpoint_path, mesh=mesh, allow_partial=False))
    return dataclasses.replace(
        state,
        params=loaded["params"],
        pending_qb_betas=loaded["pending_qb_betas"],
    )


def _recovery_initial_checkpoint(config: RecoveryJobConfig) -> str:
    if config.initial_checkpoint_selection is RecoveryCheckpointSelection.LATEST:
        return latest_checkpoint_path(config.init_checkpoint_dir)
    if config.initial_checkpoint_selection is not RecoveryCheckpointSelection.BEST_VALIDATION:
        raise ValueError(f"unknown recovery checkpoint selection {config.initial_checkpoint_selection}")
    if config.initialization is not RecoveryInitialization.LOCAL_RECOVERY:
        raise ValueError("best-validation checkpoint selection requires a local-recovery initializer")
    return _best_validation_checkpoint(config.init_checkpoint_dir)


def _best_validation_checkpoint(checkpoint_dir: str) -> str:
    return str(_best_validation_selection(checkpoint_dir)["checkpoint_path"])


def _best_validation_selection(checkpoint_dir: str) -> dict[str, Any]:
    selection_path = prefix_join(checkpoint_dir, _RECOVERY_SELECTION_FILENAME)
    selection = StoragePath(selection_path)
    if not selection.exists():
        raise ValueError(f"best-validation checkpoint selection is missing at {selection_path}")
    payload = json.loads(selection.read_text())
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported recovery checkpoint selection at {selection_path}")
    checkpoint_path = payload.get("checkpoint_path")
    if not isinstance(checkpoint_path, str):
        raise ValueError(f"recovery checkpoint selection at {selection_path} has no checkpoint_path")
    step = payload.get("step")
    if not isinstance(step, int) or step <= 0:
        raise ValueError(f"recovery checkpoint selection at {selection_path} has invalid step {step!r}")
    return payload


def _record_best_validation_checkpoint(
    config: RecoveryJobConfig,
    *,
    checkpoint_path: str,
    step: int,
    tokens: int,
    requested_tokens: int,
    student_macro_loss: float,
) -> None:
    selection_path = prefix_join(_checkpoint_root(config.output_path), _RECOVERY_SELECTION_FILENAME)
    _sync(f"recovery_{config.stage.value}_selection_before_{step}")
    if jax.process_index() == 0:
        selection = StoragePath(selection_path)
        current = json.loads(selection.read_text()) if selection.exists() else None
        if current is None or student_macro_loss < float(current["selection_value"]):
            selection.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "checkpoint_path": checkpoint_path,
                        "step": step,
                        "tokens": tokens,
                        "requested_tokens": requested_tokens,
                        "selection_metric": "eval/paloma/macro_loss",
                        "selection_value": student_macro_loss,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    _sync(f"recovery_{config.stage.value}_selection_after_{step}")


def _recovery_manifest_for_run(
    config: RecoveryJobConfig,
    manifest: MergeCheckpointManifest,
    *,
    initialization_checkpoint: str,
    expected_adapter_source_checkpoint: str | None,
    expected_adapter_source_step: int | None,
    resuming: bool,
) -> MergeCheckpointManifest:
    if config.stage is RecoveryStage.LOCAL and config.initialization is not RecoveryInitialization.CONVERTED_STEP_ZERO:
        raise ValueError("local recovery must initialize from a converted step-0 checkpoint")
    if (
        config.initialization is not RecoveryInitialization.LAYER_ADAPTER_AUGMENTED
        and manifest.layer_adapter is not None
    ):
        raise ValueError("an adapter checkpoint requires layer-adapter recovery initialization")

    if resuming:
        if manifest.recovery_initialization is not config.initialization:
            raise ValueError(
                "recovery checkpoint initialization is "
                f"{manifest.recovery_initialization}, expected {config.initialization}"
            )
        if manifest.recovery_initial_checkpoint != initialization_checkpoint:
            raise ValueError(
                "recovery checkpoint initializer is "
                f"{manifest.recovery_initial_checkpoint}, expected {initialization_checkpoint}"
            )
        expected_recovery = {
            "recovery_stage": config.stage,
            "recovery_trainable_scope": config.trainable_scope,
            "recovery_cross_entropy_weight": config.cross_entropy_weight,
            "recovery_moe_loss_weight": config.moe_loss_weight,
            "recovery_logit_kl_weight": config.logit_kl_weight,
        }
        mismatches = {
            name: (getattr(manifest, name), expected)
            for name, expected in expected_recovery.items()
            if getattr(manifest, name) != expected
        }
        if mismatches:
            raise ValueError(f"recovery checkpoint objective or trainable scope changed: {mismatches}")
        if config.initialization is RecoveryInitialization.LAYER_ADAPTER_AUGMENTED:
            _validate_layer_adapter_recovery(
                config,
                manifest,
                expected_source_checkpoint=expected_adapter_source_checkpoint,
                expected_source_step=expected_adapter_source_step,
            )
        return manifest

    if config.initialization is RecoveryInitialization.CONVERTED_STEP_ZERO:
        if manifest.recovery_step != 0:
            raise ValueError("converted-step-zero recovery requires recovery_step=0, " f"got {manifest.recovery_step}")
        if manifest.recovery_initialization is not None or manifest.recovery_initial_checkpoint is not None:
            raise ValueError("converted-step-zero recovery cannot initialize from a prior recovery checkpoint")
    elif config.initialization is RecoveryInitialization.LOCAL_RECOVERY:
        if config.stage is not RecoveryStage.PRESERVATION:
            raise ValueError("a local-recovery initializer is valid only for preservation recovery")
        if manifest.recovery_step <= 0:
            raise ValueError("local-recovery initialization requires a checkpoint with recovery_step > 0")
    elif config.initialization is RecoveryInitialization.CAPACITY_ORACLE_SPLIT:
        if config.stage is not RecoveryStage.PRESERVATION:
            raise ValueError("a capacity-oracle initializer is valid only for preservation recovery")
        if config.trainable_scope is not RecoveryTrainableScope.AFFECTED_EXPERT_BANKS:
            raise ValueError("a capacity-oracle initializer must train the two affected expert banks")
        if manifest.recovery_step != 0 or manifest.capacity_oracle is None:
            raise ValueError("capacity-oracle recovery requires an explicit step-0 split checkpoint")
    elif config.initialization is RecoveryInitialization.LAYER_ADAPTER_AUGMENTED:
        if manifest.recovery_initialization is not None or manifest.recovery_step != 0:
            raise ValueError("layer-adapter recovery must initialize from the augmented step-0 checkpoint")
        _validate_layer_adapter_recovery(
            config,
            manifest,
            expected_source_checkpoint=expected_adapter_source_checkpoint,
            expected_source_step=expected_adapter_source_step,
        )
    else:
        raise ValueError(f"unknown recovery initialization {config.initialization}")

    return dataclasses.replace(
        manifest,
        recovery_initialization=config.initialization,
        recovery_initial_checkpoint=initialization_checkpoint,
        recovery_stage=config.stage,
        recovery_trainable_scope=config.trainable_scope,
        recovery_cross_entropy_weight=config.cross_entropy_weight,
        recovery_moe_loss_weight=config.moe_loss_weight,
        recovery_logit_kl_weight=config.logit_kl_weight,
    )


def _validate_layer_adapter_recovery(
    config: RecoveryJobConfig,
    manifest: MergeCheckpointManifest,
    *,
    expected_source_checkpoint: str | None,
    expected_source_step: int | None,
) -> None:
    if config.stage is not RecoveryStage.PRESERVATION:
        raise ValueError("layer-adapter initialization is valid only for preservation recovery")
    if config.trainable_scope is not RecoveryTrainableScope.SHARED_BANK_AND_LAYER_ADAPTERS:
        raise ValueError("layer-adapter initialization must train exactly the shared bank and layer adapters")
    adapter = manifest.layer_adapter
    if adapter is None:
        raise ValueError("layer-adapter recovery requires explicit adapter provenance")
    if config.layer_adapter_rank is None or expected_source_checkpoint is None or expected_source_step is None:
        raise ValueError("layer-adapter recovery requires an expected source checkpoint, step, and rank")
    if (
        adapter.kind is not LayerAdapterKind.ZERO_INITIALIZED_INPUT_OUTPUT
        or adapter.layer_index != config.affected_layers[1]
        or adapter.rank != config.layer_adapter_rank
        or adapter.source_checkpoint != expected_source_checkpoint
        or adapter.source_recovery_step != expected_source_step
        or adapter.input_topology != manifest.target_topology
        or adapter.output_step != 0
        or manifest.capacity_oracle is not None
    ):
        raise ValueError("layer-adapter recovery checkpoint has stale provenance")


def _recovery_metrics(
    losses: RecoveryLosses,
    affected_layers: tuple[int, int],
    *,
    tokens_per_second: float | None = None,
    total_tokens: int | None = None,
) -> dict[str, float]:
    metrics = {
        "train/loss": float(jax.device_get(losses.total)),
        "train/cross_entropy_loss": float(jax.device_get(losses.cross_entropy)),
        "merge/moe_loss": float(jax.device_get(losses.moe)),
        "merge/logit_kl_loss": float(jax.device_get(losses.logit_kl)),
    }
    nrmse = np.asarray(jax.device_get(losses.moe_output_nrmse))
    block_nrmse = np.asarray(jax.device_get(losses.block_output_nrmse))
    top1_agreement = np.asarray(jax.device_get(losses.router_top1_agreement_with_teacher))
    topk_agreement = np.asarray(jax.device_get(losses.router_topk_agreement_with_teacher))
    routing_entropy = np.asarray(jax.device_get(losses.routing_entropy_by_layer))
    routing_counts = np.asarray(jax.device_get(losses.routing_counts_by_layer))
    capacity_overflow = np.asarray(jax.device_get(losses.capacity_overflow_by_layer))
    if tokens_per_second is not None:
        metrics["throughput/tokens_per_second"] = tokens_per_second
    if total_tokens is not None:
        metrics["throughput/total_tokens"] = float(total_tokens)
    for index, layer in enumerate(affected_layers):
        metrics[f"merge/moe_output_nrmse_by_layer/layer_{layer}"] = float(nrmse[index])
        metrics[f"merge/block_output_nrmse_by_layer/layer_{layer}"] = float(block_nrmse[index])
        metrics[f"merge/router_top1_agreement_with_teacher/layer_{layer}"] = float(top1_agreement[index])
        metrics[f"merge/router_topk_agreement_with_teacher/layer_{layer}"] = float(topk_agreement[index])
        metrics[f"train/router/layer_{layer}/routing_entropy"] = float(routing_entropy[index])
        metrics[f"train/router/layer_{layer}/capacity_overflow"] = float(capacity_overflow[index])
        for expert, count in enumerate(routing_counts[index]):
            metrics[f"train/router/layer_{layer}/routing_count/expert_{expert}"] = float(count)
    return metrics


def _eval_result_metrics(result, *, prefix: str) -> dict[str, float]:
    metrics = {
        f"{prefix}/loss": float(result.micro_avg_loss),
        f"{prefix}/macro_loss": float(result.macro_avg_loss),
    }
    for tag, loss in result.tag_micro_losses.items():
        if tag:
            metrics[f"{prefix}/{tag}/loss"] = float(loss)
    for tag, loss in result.tag_macro_losses.items():
        if tag:
            metrics[f"{prefix}/{tag}/macro_loss"] = float(loss)
    return metrics


def _paloma_macro_loss(result) -> float:
    try:
        return float(result.tag_macro_losses["paloma"])
    except KeyError as error:
        raise ValueError("recovery checkpoint selection requires a paloma validation tag") from error


def _evaluate_recovery_checkpoint(
    config: RecoveryJobConfig,
    *,
    evaluator,
    student: Transformer,
    teacher: Transformer | None,
    step: int,
    tokens: int,
    requested_tokens: int,
) -> float:
    student_result = evaluator.evaluate(student)
    metrics = _eval_result_metrics(student_result, prefix="student")
    student_paloma_macro_loss = _paloma_macro_loss(student_result)
    metrics["eval/paloma/macro_loss"] = student_paloma_macro_loss
    if teacher is not None:
        teacher_result = evaluator.evaluate(teacher)
        metrics.update(_eval_result_metrics(teacher_result, prefix="teacher"))
        metrics["merge/immediate_validation_loss_delta"] = float(
            student_result.micro_avg_loss - teacher_result.micro_avg_loss
        )
        metrics["merge/immediate_macro_loss_delta"] = float(
            student_result.macro_avg_loss - teacher_result.macro_avg_loss
        )
        metrics["merge/immediate_paloma_macro_loss_delta"] = student_paloma_macro_loss - _paloma_macro_loss(
            teacher_result
        )
        threshold = float(teacher_result.micro_avg_loss) + config.recovery_loss_threshold_delta
        reached = float(student_result.micro_avg_loss) <= threshold
        threshold_path = prefix_join(prefix_join(config.output_path, "evaluations"), _RECOVERY_THRESHOLD_FILENAME)
        _sync(f"recovery_{config.stage.value}_threshold_before")
        recovery_tokens = -1
        if jax.process_index() == 0:
            stored = StoragePath(threshold_path)
            if stored.exists():
                recovery_tokens = int(json.loads(stored.read_text())["merge/recovery_tokens_to_threshold"])
            elif reached:
                recovery_tokens = tokens
                stored.write_text(
                    json.dumps(
                        {
                            "format_version": 1,
                            "loss_threshold": threshold,
                            "merge/recovery_tokens_to_threshold": recovery_tokens,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
        recovery_tokens = int(multihost_utils.broadcast_one_to_all(np.asarray(recovery_tokens)))
        _sync(f"recovery_{config.stage.value}_threshold_after")
        metrics["merge/recovery_tokens_to_threshold"] = float(recovery_tokens)
    payload = {
        "format_version": 1,
        "stage": config.stage.value,
        "trainable_scope": config.trainable_scope.value,
        "step": step,
        "tokens": tokens,
        "requested_tokens": requested_tokens,
        "max_eval_batches": 8,
        "metrics": metrics,
    }
    _write_json_process_zero(
        prefix_join(prefix_join(config.output_path, "evaluations"), f"tokens-{tokens}-step-{step}.json"),
        payload,
        sync_name=f"recovery_{config.stage.value}_eval_{step}",
    )
    return student_paloma_macro_loss


def run_recovery_local(config: RecoveryJobConfig) -> None:
    if (
        config.training_tokens <= 0
        or config.batch_size <= 0
        or config.checkpoint_every <= 0
        or config.logit_kl_vocab_chunk_size <= 0
    ):
        raise ValueError("training_tokens, batch_size, checkpoint_every, and KL chunk size must be positive")
    if any(tokens <= 0 for tokens in config.checkpoint_token_milestones):
        raise ValueError("checkpoint token milestones must be positive")
    if config.select_best_validation_checkpoint and config.stage is not RecoveryStage.LOCAL:
        raise ValueError("best-validation checkpoint selection is only valid for local recovery")
    uses_layer_adapter = config.initialization is RecoveryInitialization.LAYER_ADAPTER_AUGMENTED
    if uses_layer_adapter:
        if config.layer_adapter_rank is None or config.layer_adapter_rank <= 0:
            raise ValueError("layer-adapter recovery requires a positive expected adapter rank")
        if config.layer_adapter_source_checkpoint_dir is None:
            raise ValueError("layer-adapter recovery requires the selected source checkpoint directory")
    elif config.layer_adapter_rank is not None or config.layer_adapter_source_checkpoint_dir is not None:
        raise ValueError("layer-adapter expectations are valid only for layer-adapter recovery")
    checkpoint_root = _checkpoint_root(config.output_path)
    _validate_runtime_paths(
        config,
        {
            "teacher_checkpoint": config.source.checkpoint_dir,
            "matching": config.matching_path,
            "initial_checkpoint": config.init_checkpoint_dir,
            "layer_adapter_source_checkpoint": config.layer_adapter_source_checkpoint_dir,
            "recovery_output": checkpoint_root,
        },
    )
    initialize_merge_worker()
    mesh = compact_merge_mesh(
        expert_axis_size=config.expert_axis_size,
        replica_axis_size=config.replica_axis_size,
    )
    with set_mesh(mesh):
        expected_adapter_source_checkpoint = None
        expected_adapter_source_step = None
        if uses_layer_adapter:
            assert config.layer_adapter_source_checkpoint_dir is not None
            selection = _best_validation_selection(config.layer_adapter_source_checkpoint_dir)
            expected_adapter_source_checkpoint = str(selection["checkpoint_path"])
            expected_adapter_source_step = int(selection["step"])
            _validate_runtime_paths(
                config,
                {"resolved_layer_adapter_source_checkpoint": expected_adapter_source_checkpoint},
            )
        matching = _matching_manifest(
            config.matching_path,
            representative_layer=config.affected_layers[0],
            source_layer=config.affected_layers[1],
            num_experts=config.source.model.num_experts,
        )
        initialization_checkpoint = _recovery_initial_checkpoint(config)
        _validate_runtime_paths(config, {"resolved_initial_checkpoint": initialization_checkpoint})
        own_checkpoint = discover_latest_checkpoint(checkpoint_root)
        if own_checkpoint is None:
            manifest = read_merge_checkpoint_manifest(initialization_checkpoint)
            manifest_checkpoint = initialization_checkpoint
        else:
            manifest = read_merge_checkpoint_manifest(own_checkpoint)
            manifest_checkpoint = own_checkpoint
        _validate_merge_manifest(
            manifest,
            assignment_mode=config.assignment_mode,
            prefit_applied=config.prefit_applied,
            affected_layers=config.affected_layers,
        )
        _validate_assignment_provenance(manifest, matching)
        manifest = _recovery_manifest_for_run(
            config,
            manifest,
            initialization_checkpoint=initialization_checkpoint,
            expected_adapter_source_checkpoint=expected_adapter_source_checkpoint,
            expected_adapter_source_step=expected_adapter_source_step,
            resuming=own_checkpoint is not None,
        )

        state, optimizer, recovery_config = _recovery_template(
            config,
            manifest,
            key=jax.random.key(config.seed),
        )
        if own_checkpoint is not None:
            state = load_checkpoint(state, own_checkpoint, mesh=mesh)
        else:
            state = _load_recovery_initial_weights(state, manifest_checkpoint, mesh=mesh)

        teacher_state, source_checkpoint = _source_state(
            config.source,
            key=jax.random.fold_in(jax.random.key(config.seed), 1),
            mesh=mesh,
        )
        if manifest.spec.source_checkpoint != source_checkpoint:
            raise ValueError(
                f"merge checkpoint refers to teacher {manifest.spec.source_checkpoint}, not {source_checkpoint}"
            )

        tokens_per_step = config.batch_size * config.source.model.max_seq_len
        target_steps = math.ceil(config.training_tokens / tokens_per_step)
        milestone_requested_tokens_by_step = {
            math.ceil(tokens / tokens_per_step): tokens
            for tokens in config.checkpoint_token_milestones
            if tokens <= config.training_tokens
        }
        milestone_steps = set(milestone_requested_tokens_by_step)
        evaluator = build_tagged_evaluator(
            data_config=config.data,
            max_seq_len=config.source.model.max_seq_len,
            mesh=mesh,
            eval_cfg=GrugEvalConfig(
                eval_batch_size=config.batch_size,
                max_eval_batches=8,
                compute_bpb=False,
                eval_ema=False,
            ),
        )
        if evaluator is None:
            raise ValueError("recovery requires at least one validation dataset")
        initial_step = int(state.step)
        if initial_step == 0:
            _evaluate_recovery_checkpoint(
                config,
                evaluator=evaluator,
                student=state.params,
                teacher=teacher_state.params,
                step=0,
                tokens=0,
                requested_tokens=0,
            )
        elif initial_step in milestone_steps or (
            config.select_best_validation_checkpoint and initial_step == target_steps
        ):
            requested_tokens = milestone_requested_tokens_by_step.get(
                initial_step,
                config.training_tokens,
            )
            actual_tokens = initial_step * tokens_per_step
            student_macro_loss = _evaluate_recovery_checkpoint(
                config,
                evaluator=evaluator,
                student=state.params,
                teacher=teacher_state.params,
                step=initial_step,
                tokens=actual_tokens,
                requested_tokens=requested_tokens,
            )
            if config.select_best_validation_checkpoint:
                if own_checkpoint is None:
                    raise ValueError("resumed local recovery has no checkpoint to select")
                _record_best_validation_checkpoint(
                    config,
                    checkpoint_path=own_checkpoint,
                    step=initial_step,
                    tokens=actual_tokens,
                    requested_tokens=requested_tokens,
                    student_macro_loss=student_macro_loss,
                )
        if int(state.step) >= target_steps:
            logger.info("Recovery output already complete at step %d", int(state.step))
            return

        data_key = jax.random.fold_in(jax.random.key(config.seed), 2)
        batch_schedule = BatchSchedule(config.batch_size)
        train_dataset = build_train_dataset(
            config.data,
            max_seq_len=config.source.model.max_seq_len,
            batch_schedule=batch_schedule,
            key=data_key,
        )
        train_loader = build_train_loader(train_dataset, batch_schedule=batch_schedule, mesh=mesh)
        iterator = train_loader.iter_from_step(int(state.step))
        logit_kl_loss = make_chunked_logit_kl(config.logit_kl_vocab_chunk_size) if config.logit_kl_weight > 0 else None
        train_step = jax.jit(
            make_recovery_train_step(optimizer, recovery_config, logit_kl_loss=logit_kl_loss),
            donate_argnums=(0,),
        )
        metrics_started = time.monotonic()
        metrics_start_step = int(state.step)

        while int(state.step) < target_steps:
            batch = next(iterator)
            state, losses = train_step(
                state,
                teacher_state.params,
                batch.tokens,
                batch.loss_weight,
                batch.attn_mask,
            )
            jax.block_until_ready(losses.total)
            step = int(state.step)
            if jax.process_index() == 0 and (step == 1 or step % 10 == 0 or step == target_steps):
                logger.info(
                    "recovery stage=%s step=%d metrics=%s",
                    config.stage.value,
                    step,
                    _recovery_metrics(losses, config.affected_layers),
                )

            if step % config.checkpoint_every != 0 and step not in milestone_steps and step != target_steps:
                continue
            elapsed = max(time.monotonic() - metrics_started, 1e-9)
            tokens_per_second = (step - metrics_start_step) * tokens_per_step / elapsed
            checkpoint_metrics = _recovery_metrics(
                losses,
                config.affected_layers,
                tokens_per_second=tokens_per_second,
                total_tokens=step * tokens_per_step,
            )
            recovery_manifest = dataclasses.replace(manifest, recovery_step=step, optimizer_state_reset=True)
            checkpoint_path = _save_permanent_checkpoint(
                state,
                checkpoint_root=checkpoint_root,
                step=step,
                sync_name=f"recovery_{config.stage.value}_{step}",
                merge_manifest=recovery_manifest,
            )
            _write_json_process_zero(
                prefix_join(prefix_join(config.output_path, "training_metrics"), f"step-{step}.json"),
                {
                    "format_version": 1,
                    "stage": config.stage.value,
                    "trainable_scope": config.trainable_scope.value,
                    "step": step,
                    "tokens": step * tokens_per_step,
                    "metrics": checkpoint_metrics,
                },
                sync_name=f"recovery_{config.stage.value}_training_metrics_{step}",
            )
            metrics_started = time.monotonic()
            metrics_start_step = step
            should_evaluate = step in milestone_steps or (
                config.select_best_validation_checkpoint and step == target_steps
            )
            if should_evaluate:
                requested_tokens = milestone_requested_tokens_by_step.get(
                    step,
                    config.training_tokens,
                )
                actual_tokens = step * tokens_per_step
                student_macro_loss = _evaluate_recovery_checkpoint(
                    config,
                    evaluator=evaluator,
                    student=state.params,
                    teacher=teacher_state.params,
                    step=step,
                    tokens=actual_tokens,
                    requested_tokens=requested_tokens,
                )
                if config.select_best_validation_checkpoint:
                    _record_best_validation_checkpoint(
                        config,
                        checkpoint_path=checkpoint_path,
                        step=step,
                        tokens=actual_tokens,
                        requested_tokens=requested_tokens,
                        student_macro_loss=student_macro_loss,
                    )
