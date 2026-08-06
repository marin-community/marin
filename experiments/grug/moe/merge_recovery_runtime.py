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
    PrefitBatch,
    PrefitDataset,
    PrefitSplit,
    prefit_loss,
    sample_prefit_batch,
)
from experiments.grug.moe.merge_artifacts import (
    MatchingArtifactManifest,
    read_calibration_manifest,
    read_expert_calibration,
    read_expert_probe,
    read_matching_manifest,
)
from experiments.grug.moe.merge_checkpoint import (
    MergeCheckpointManifest,
    OnePairMergeCheckpointSpec,
    convert_grug_state_for_one_pair_merge,
    read_merge_checkpoint_manifest,
    write_merge_checkpoint_manifest,
)
from experiments.grug.moe.merge_jobs import (
    ConversionJobConfig,
    PrefitJobConfig,
    RecoveryJobConfig,
    SourceCheckpointConfig,
)
from experiments.grug.moe.merge_recovery import (
    MergeRecoveryConfig,
    MergeRecoveryState,
    RecoveryLosses,
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


@register_dataclass
@dataclass(frozen=True)
class PrefitRuntimeState:
    step: jax.Array
    bank: MoEExpertMlp
    opt_state: optax.OptState
    best_bank: MoEExpertMlp
    best_loss: jax.Array
    stale_evaluations: jax.Array


def _is_local(config: PrefitJobConfig | ConversionJobConfig | RecoveryJobConfig) -> bool:
    return isinstance(config.resources.device, CpuConfig)


def _validate_runtime_paths(
    config: PrefitJobConfig | ConversionJobConfig | RecoveryJobConfig,
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
    assignment = np.asarray(matching.assignments[AssignmentMode.SPECTRAL], dtype=np.int32)
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


def _prefit_manifest(
    config: PrefitJobConfig,
    *,
    source_checkpoint: str,
    step: int,
    best_loss: float,
    stopped_early: bool,
    nrmse_by_source: np.ndarray,
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
        "probe_config": json.loads(json.dumps(dataclasses.asdict(config.probe))),
        "prefit_config": json.loads(json.dumps(dataclasses.asdict(config.config))),
    }


def _validate_prefit_provenance(
    config: PrefitJobConfig,
    *,
    checkpoint: str,
    source_checkpoint: str,
) -> None:
    manifest = json.loads(StoragePath(prefix_join(checkpoint, _PREFIT_MANIFEST_FILENAME)).read_text())
    expected = {
        "source_checkpoint": source_checkpoint,
        "calibration_path": config.calibration_path,
        "matching_path": config.matching_path,
        "representative_layer": config.representative_layer,
        "source_layer": config.source_layer,
        "probe_config": json.loads(json.dumps(dataclasses.asdict(config.probe))),
        "prefit_config": json.loads(json.dumps(dataclasses.asdict(config.config))),
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"existing prefit checkpoint has stale provenance: {mismatches}")


def run_prefit_local(config: PrefitJobConfig) -> None:
    if config.config.steps <= 0 or config.config.eval_every <= 0 or config.config.early_stopping_patience <= 0:
        raise ValueError("prefit steps, eval_every, and early_stopping_patience must be positive")
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
        datasets = _prefit_datasets(source_state.params, config, matching)
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
            stopped_early = int(state.stale_evaluations) >= config.config.early_stopping_patience
            manifest = _prefit_manifest(
                config,
                source_checkpoint=source_checkpoint,
                step=int(state.step),
                best_loss=float(jax.device_get(state.best_loss)),
                stopped_early=stopped_early,
                nrmse_by_source=last_nrmse,
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
                    "prefit step=%d train_loss=%g heldout_loss=%g best_loss=%g stale=%d",
                    int(state.step),
                    float(jax.device_get(loss)),
                    heldout_loss,
                    float(jax.device_get(state.best_loss)),
                    int(state.stale_evaluations),
                )

        if own_checkpoint is not None and int(state.step) >= config.config.steps:
            logger.info("Prefit output already complete at step %d", int(state.step))


def _load_prefitted_bank(
    config: ConversionJobConfig,
    initial_bank: MoEExpertMlp,
    *,
    mesh: jax.sharding.Mesh,
) -> tuple[MoEExpertMlp | None, str | None]:
    if config.prefit_path is None:
        return None, None
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
    return restored.best_bank, checkpoint


def _validate_conversion_provenance(
    config: ConversionJobConfig,
    manifest: MergeCheckpointManifest,
    matching: MatchingArtifactManifest,
) -> None:
    spec = manifest.spec
    expected_prefit = latest_checkpoint_path(config.prefit_path) if config.prefit_path is not None else None
    expected = {
        "source_checkpoint": latest_checkpoint_path(config.source.checkpoint_dir),
        "calibration_path": config.calibration_path,
        "cost_matrix_path": prefix_join(config.matching_path, "cost_matrix.npz"),
        "probe_path": prefix_join(config.matching_path, "probes"),
        "prefit_checkpoint": expected_prefit,
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
        prefitted_bank, prefit_checkpoint = _load_prefitted_bank(config, representative_bank, mesh=mesh)
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


def _recovery_template(
    config: RecoveryJobConfig,
    manifest: MergeCheckpointManifest,
    *,
    key: jax.Array,
) -> tuple[MergeRecoveryState, optax.GradientTransformation, MergeRecoveryConfig]:
    model_config = dataclasses.replace(config.source.model, expert_bank_for_layer=manifest.target_topology)
    params = Transformer.init(model_config, key=key)
    optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    recovery_config = MergeRecoveryConfig(
        affected_layers=config.affected_layers,
        stage=config.stage,
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


def _evaluate_recovery_checkpoint(
    config: RecoveryJobConfig,
    *,
    evaluator,
    student: Transformer,
    teacher: Transformer | None,
    step: int,
    tokens: int,
) -> None:
    student_result = evaluator.evaluate(student)
    metrics = _eval_result_metrics(student_result, prefix="student")
    metrics["eval/paloma/macro_loss"] = float(student_result.macro_avg_loss)
    if teacher is not None:
        teacher_result = evaluator.evaluate(teacher)
        metrics.update(_eval_result_metrics(teacher_result, prefix="teacher"))
        metrics["merge/immediate_validation_loss_delta"] = float(
            student_result.micro_avg_loss - teacher_result.micro_avg_loss
        )
        metrics["merge/immediate_macro_loss_delta"] = float(
            student_result.macro_avg_loss - teacher_result.macro_avg_loss
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
        "step": step,
        "tokens": tokens,
        "max_eval_batches": 8,
        "metrics": metrics,
    }
    _write_json_process_zero(
        prefix_join(prefix_join(config.output_path, "evaluations"), f"tokens-{tokens}-step-{step}.json"),
        payload,
        sync_name=f"recovery_{config.stage.value}_eval_{step}",
    )


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
    checkpoint_root = _checkpoint_root(config.output_path)
    _validate_runtime_paths(
        config,
        {
            "teacher_checkpoint": config.source.checkpoint_dir,
            "matching": config.matching_path,
            "initial_checkpoint": config.init_checkpoint_dir,
            "recovery_output": checkpoint_root,
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
            representative_layer=config.affected_layers[0],
            source_layer=config.affected_layers[1],
            num_experts=config.source.model.num_experts,
        )
        own_checkpoint = discover_latest_checkpoint(checkpoint_root)
        if own_checkpoint is None:
            initialization_checkpoint = latest_checkpoint_path(config.init_checkpoint_dir)
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
        milestone_tokens_by_step = {
            math.ceil(tokens / tokens_per_step): tokens
            for tokens in config.checkpoint_token_milestones
            if tokens <= config.training_tokens
        }
        milestone_steps = set(milestone_tokens_by_step)
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
            )
        elif initial_step in milestone_steps:
            _evaluate_recovery_checkpoint(
                config,
                evaluator=evaluator,
                student=state.params,
                teacher=teacher_state.params,
                step=initial_step,
                tokens=milestone_tokens_by_step[initial_step],
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
                total_tokens=min(step * tokens_per_step, config.training_tokens),
            )
            recovery_manifest = dataclasses.replace(manifest, recovery_step=step, optimizer_state_reset=True)
            _save_permanent_checkpoint(
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
                    "step": step,
                    "tokens": min(step * tokens_per_step, config.training_tokens),
                    "metrics": checkpoint_metrics,
                },
                sync_name=f"recovery_{config.stage.value}_training_metrics_{step}",
            )
            metrics_started = time.monotonic()
            metrics_start_step = step
            if step in milestone_steps:
                _evaluate_recovery_checkpoint(
                    config,
                    evaluator=evaluator,
                    student=state.params,
                    teacher=teacher_state.params,
                    step=step,
                    tokens=milestone_tokens_by_step[step],
                )
