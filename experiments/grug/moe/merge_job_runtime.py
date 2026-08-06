# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""On-accelerator implementations for expert calibration, matching, conversion, and recovery."""

import dataclasses
import logging
import os

import jax
import jmp
import numpy as np
from fray.types import CpuConfig
from haliax.partitioning import set_mesh
from jax.experimental import multihost_utils
from levanter.checkpoint import latest_checkpoint_path
from levanter.grug.sharding import compact_grug_mesh
from levanter.schedule import BatchSchedule
from rigging.filesystem import check_gcs_paths_same_region

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.moe.expert_merge import (
    AssignmentMode,
    ExpertCalibration,
    ExpertReservoirCollection,
    MoeLayerTrace,
    build_spectral_probe_set,
    forward_with_moe_traces,
    functional_cost_matrix,
    solve_expert_assignment,
)
from experiments.grug.moe.merge_artifacts import (
    CalibrationArtifactManifest,
    MatchingArtifactManifest,
    read_calibration_manifest,
    read_expert_calibration,
    write_calibration_artifact,
    write_matching_artifact,
)
from experiments.grug.moe.merge_jobs import (
    CalibrationJobConfig,
    MatchingJobConfig,
    SourceCheckpointConfig,
)
from experiments.grug.moe.merge_recovery_runtime import initialize_merge_worker
from experiments.grug.moe.merge_storage import MergeStoragePaths, validate_merge_storage_region
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import build_train_dataset, build_train_loader, initial_state

logger = logging.getLogger(__name__)

_SOURCE_MP_POLICY = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")


def _validate_untied_source(source: SourceCheckpointConfig) -> None:
    expected = tuple(range(source.model.num_layers))
    actual = source.model.resolved_expert_bank_for_layer
    if actual != expected:
        raise ValueError(f"expert merge source must be fully untied; expected {expected}, got {actual}")


def _source_optimizer(source: SourceCheckpointConfig):
    optimizer_config = source.optimizer
    if isinstance(optimizer_config, GrugMoeMuonHConfig):
        group_sizes = source.model.expert_bank_group_sizes
        if optimizer_config.expert_bank_group_sizes not in (None, group_sizes):
            raise ValueError(
                "source optimizer expert-bank topology does not match the source model: "
                f"{optimizer_config.expert_bank_group_sizes} != {group_sizes}"
            )
        optimizer_config = dataclasses.replace(optimizer_config, expert_bank_group_sizes=group_sizes)
    return optimizer_config.build(source.training_steps)


def _load_source_state(source: SourceCheckpointConfig, mesh: jax.sharding.Mesh):
    """Load one exact untied trainer state without partial topology restoration."""
    _validate_untied_source(source)
    optimizer = _source_optimizer(source)

    @jax.jit
    def init_state(key):
        return initial_state(
            source.model,
            optimizer=optimizer,
            mp=_SOURCE_MP_POLICY,
            key=key,
            ema_beta=None,
        )

    state = init_state(jax.random.PRNGKey(0))
    checkpoint_path = latest_checkpoint_path(source.checkpoint_dir)
    state = restore_grug_state_from_checkpoint(
        state,
        checkpoint_search_paths=(checkpoint_path,),
        load_checkpoint_setting=True,
        mesh=mesh,
        allow_partial=False,
    )
    if state.params.config.resolved_expert_bank_for_layer != tuple(range(source.model.num_layers)):
        raise ValueError("loaded source checkpoint is not an untied Grug MoE")
    return state, checkpoint_path


def _validate_stage_storage(
    *,
    source_checkpoint: str,
    output_path: str,
    resources,
    calibration_path: str | None = None,
    data=None,
) -> None:
    """Apply the strict merge storage contract to one materialization stage."""
    paths = MergeStoragePaths(
        teacher_checkpoint=source_checkpoint,
        calibration=calibration_path or output_path,
        converted_checkpoint=output_path,
        recovery_output=output_path,
        matching=output_path if calibration_path is not None else None,
    )
    local_ok = isinstance(resources.device, CpuConfig)
    validate_merge_storage_region(paths, local_ok=local_ok)
    if data is not None:
        check_gcs_paths_same_region(
            {"paths": paths, "data": data},
            local_ok=local_ok,
            skip_if_prefix_contains=(),
        )


def _gather_trace_arrays(traces):
    """Collect globally sharded traces onto every host in one lockstep collective."""
    gathered = multihost_utils.process_allgather(traces, tiled=True)
    return jax.tree.map(np.asarray, gathered)


def _add_trace_arrays(
    reservoirs: ExpertReservoirCollection,
    trace_arrays,
    *,
    token_limit: int,
) -> None:
    mlp_input, selected_experts, combine_weights, _routed_output = trace_arrays
    flat_inputs = np.asarray(mlp_input).reshape(-1, reservoirs.state_dim)[:token_limit]
    reservoirs.add_routes(
        flat_inputs,
        np.asarray(selected_experts).reshape(-1, selected_experts.shape[-1])[:token_limit],
        np.asarray(combine_weights).reshape(-1, combine_weights.shape[-1])[:token_limit],
    )


def _append_bounded_trace(
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    trace_arrays,
    *,
    remaining_capacity: int,
    token_limit: int,
) -> int:
    take = min(remaining_capacity, token_limit)
    if take <= 0:
        return 0
    mlp_input, selected_experts, combine_weights, routed_output = trace_arrays
    state_dim = mlp_input.shape[-1]
    chunks.append(
        (
            np.asarray(mlp_input).reshape(-1, state_dim)[:take],
            np.asarray(selected_experts).reshape(-1, selected_experts.shape[-1])[:take],
            np.asarray(combine_weights).reshape(-1, combine_weights.shape[-1])[:take],
            np.asarray(routed_output).reshape(-1, state_dim)[:take],
        )
    )
    return take


def _merge_trace_chunks(chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> MoeLayerTrace:
    fields = tuple(np.concatenate(values, axis=0) for values in zip(*chunks, strict=True))
    return MoeLayerTrace(*fields)


def _validate_expert_calibration(calibration: ExpertCalibration, *, layer: int, expert: int) -> None:
    for split_name, sample in (("train", calibration.train), ("heldout", calibration.heldout)):
        if sample.states.shape[0] == 0 or float(np.sum(sample.weights)) <= 0:
            raise ValueError(f"layer {layer} expert {expert} has no positive-weight {split_name} calibration states")


def _sync_before_and_after_write(run_id: str, stage: str, write) -> None:
    multihost_utils.sync_global_devices(f"{run_id}-{stage}-ready")
    if jax.process_index() == 0:
        write()
    multihost_utils.sync_global_devices(f"{run_id}-{stage}-committed")


def run_calibration_local(config: CalibrationJobConfig) -> None:
    _validate_stage_storage(
        source_checkpoint=config.source.checkpoint_dir,
        output_path=config.output_path,
        resources=config.resources,
        data=config.data,
    )
    if config.calibration_tokens <= 0:
        raise ValueError(f"calibration_tokens must be positive, got {config.calibration_tokens}")
    if len(set(config.layers)) != len(config.layers):
        raise ValueError(f"calibration layers must be distinct, got {config.layers}")
    if config.trace_capacity <= 0:
        raise ValueError(f"trace_capacity must be positive, got {config.trace_capacity}")

    initialize_merge_worker()
    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        state, source_checkpoint = _load_source_state(config.source, mesh)
        if any(layer < 0 or layer >= config.source.model.num_layers for layer in config.layers):
            raise ValueError(f"calibration layers are outside the source model: {config.layers}")

        reservoirs_by_layer = {
            layer: ExpertReservoirCollection(
                num_experts=config.source.model.num_experts,
                state_dim=config.source.model.hidden_dim,
                capacity_per_expert=config.capacity_per_expert,
                heldout_fraction=config.heldout_fraction,
                seed=config.seed + layer,
            )
            for layer in config.layers
        }
        trace_chunks_by_layer: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {
            layer: [] for layer in config.layers
        }
        trace_tokens_by_layer = {layer: 0 for layer in config.layers}
        batch_schedule = BatchSchedule(config.batch_size)
        data_key = jax.random.PRNGKey(config.seed)
        train_dataset = build_train_dataset(
            config.data,
            max_seq_len=config.source.model.max_seq_len,
            batch_schedule=batch_schedule,
            key=data_key,
        )
        train_loader = build_train_loader(train_dataset, batch_schedule=batch_schedule, mesh=mesh)

        layers = config.layers

        @jax.jit
        def trace_batch(tokens, attn_mask):
            _, traces = forward_with_moe_traces(
                state.params,
                tokens,
                target_layers=layers,
                mask=attn_mask,
            )
            return tuple(
                (
                    traces[layer].mlp_input,
                    traces[layer].selected_experts,
                    traces[layer].combine_weights,
                    traces[layer].routed_output,
                )
                for layer in layers
            )

        processed_tokens = 0
        iterator = train_loader.iter_from_step(0)
        while processed_tokens < config.calibration_tokens:
            batch = next(iterator)
            gathered_traces = _gather_trace_arrays(trace_batch(batch.tokens, batch.attn_mask))
            remaining = config.calibration_tokens - processed_tokens
            batch_tokens = int(np.prod(batch.tokens.shape))
            token_limit = min(remaining, batch_tokens)
            for layer, trace_arrays in zip(layers, gathered_traces, strict=True):
                _add_trace_arrays(reservoirs_by_layer[layer], trace_arrays, token_limit=token_limit)
                trace_tokens_by_layer[layer] += _append_bounded_trace(
                    trace_chunks_by_layer[layer],
                    trace_arrays,
                    remaining_capacity=config.trace_capacity - trace_tokens_by_layer[layer],
                    token_limit=token_limit,
                )
            processed_tokens += token_limit
            logger.info(
                "Calibration %s collected %d/%d tokens",
                config.run_id,
                processed_tokens,
                config.calibration_tokens,
            )

        manifest = CalibrationArtifactManifest(
            source_checkpoint=source_checkpoint,
            source_commit=config.source.source_commit or os.environ.get("GIT_COMMIT"),
            layers=layers,
            num_experts=config.source.model.num_experts,
            state_dim=config.source.model.hidden_dim,
            capacity_per_expert=config.capacity_per_expert,
            heldout_fraction=config.heldout_fraction,
            calibration_tokens=processed_tokens,
            trace_capacity=config.trace_capacity,
        )
        traces_by_layer = {layer: _merge_trace_chunks(trace_chunks_by_layer[layer]) for layer in config.layers}
        _sync_before_and_after_write(
            config.run_id,
            "calibration",
            lambda: write_calibration_artifact(
                config.output_path,
                reservoirs_by_layer,
                manifest,
                traces_by_layer=traces_by_layer,
            ),
        )


def run_matching_local(config: MatchingJobConfig) -> None:
    _validate_stage_storage(
        source_checkpoint=config.source.checkpoint_dir,
        calibration_path=config.calibration_path,
        output_path=config.output_path,
        resources=config.resources,
    )
    if config.representative_layer == config.source_layer:
        raise ValueError("representative_layer and source_layer must differ")

    initialize_merge_worker()
    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        state, source_checkpoint = _load_source_state(config.source, mesh)
        calibration_manifest = read_calibration_manifest(config.calibration_path)
        expected_layers = {config.representative_layer, config.source_layer}
        if not expected_layers.issubset(calibration_manifest.layers):
            raise ValueError(
                f"calibration artifact layers {calibration_manifest.layers} do not cover {sorted(expected_layers)}"
            )
        if calibration_manifest.source_checkpoint != source_checkpoint:
            raise ValueError(
                "calibration artifact was collected from a different source checkpoint: "
                f"{calibration_manifest.source_checkpoint} != {source_checkpoint}"
            )
        if (
            calibration_manifest.num_experts != config.source.model.num_experts
            or calibration_manifest.state_dim != config.source.model.hidden_dim
        ):
            raise ValueError("calibration artifact geometry does not match the source model")

        representative_bank_index = state.params.blocks[config.representative_layer].expert_bank_index
        source_bank_index = state.params.blocks[config.source_layer].expert_bank_index
        representative_bank = state.params.expert_banks[representative_bank_index]
        source_bank = state.params.expert_banks[source_bank_index]

        probes = []
        for expert in range(config.source.model.num_experts):
            calibration = read_expert_calibration(
                config.calibration_path,
                config.source_layer,
                expert,
                manifest=calibration_manifest,
            )
            _validate_expert_calibration(calibration, layer=config.source_layer, expert=expert)
            probes.append(
                build_spectral_probe_set(
                    source_bank,
                    expert,
                    calibration.train,
                    calibration.heldout,
                    config=config.probe,
                    seed=config.seed + expert,
                )
            )
            logger.info("Matching %s built probes for expert %d", config.run_id, expert)
        probe_tuple = tuple(probes)
        costs = functional_cost_matrix(
            source_bank,
            representative_bank,
            probe_tuple,
            eta=config.eta,
            expert_chunk_size=config.expert_chunk_size,
        )
        assignments = {
            mode: tuple(int(index) for index in solve_expert_assignment(costs, mode)) for mode in AssignmentMode
        }
        manifest = MatchingArtifactManifest(
            calibration_path=config.calibration_path,
            representative_layer=config.representative_layer,
            source_layer=config.source_layer,
            num_experts=config.source.model.num_experts,
            eta=config.eta,
            assignments=assignments,
        )
        _sync_before_and_after_write(
            config.run_id,
            "matching",
            lambda: write_matching_artifact(config.output_path, probe_tuple, costs, manifest),
        )
