# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Central-region accelerator workers for June expert calibration and matching."""

import logging
import os
from typing import cast

import jax
import jax.numpy as jnp
import jmp
import numpy as np
from fray.types import CpuConfig
from haliax.nn import ArrayStacked
from haliax.partitioning import set_mesh
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import latest_checkpoint_path
from levanter.distributed import DistributedConfig
from levanter.grug.grug_moe import MoEExpertMlp
from levanter.grug.sharding import compact_grug_mesh
from levanter.schedule import BatchSchedule
from levanter.utils.activation import ActivationFunctionEnum
from rigging.filesystem import check_gcs_paths_same_region

from experiments.grug.moe.expert_merge import (
    AssignmentMode,
    ExpertCalibration,
    ExpertReservoirCollection,
    InputManifold,
    MoeLayerTrace,
    SpectralProbeConfig,
    SpectralProbePreparation,
    finalize_spectral_probe_set,
    functional_cost_matrix,
    prepare_spectral_probe_set,
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
from experiments.grug.moe.merge_storage import MergeStoragePaths, validate_merge_storage_region
from experiments.june_tpu_67b_a2b.checkpointing import load_june_checkpoint, load_legacy_june_stacked_experts
from experiments.june_tpu_67b_a2b.moe.expert_merge import forward_with_moe_traces, stacked_expert_bank_at
from experiments.june_tpu_67b_a2b.moe.merge_jobs import (
    JuneCalibrationJobConfig,
    JuneMatchingJobConfig,
    JuneSourceCheckpointConfig,
)
from experiments.june_tpu_67b_a2b.moe.model import Transformer
from experiments.june_tpu_67b_a2b.moe.train import build_train_dataset, build_train_loader

logger = logging.getLogger(__name__)

_SOURCE_MP_POLICY = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")


def _validate_untied_source(source: JuneSourceCheckpointConfig) -> None:
    expected = tuple(range(source.model.num_layers))
    actual = source.model.resolved_expert_bank_for_layer
    if actual != expected:
        raise ValueError(f"expert merge source must be fully untied; expected {expected}, got {actual}")


def _merge_mesh(config: JuneCalibrationJobConfig | JuneMatchingJobConfig) -> jax.sharding.Mesh:
    return compact_grug_mesh(
        expert_axis_size=config.expert_axis_size,
        replica_axis_size=config.replica_axis_size,
        model_axis_size=config.model_axis_size,
    )


def _load_source_params(source: JuneSourceCheckpointConfig, mesh: jax.sharding.Mesh) -> tuple[Transformer, str]:
    """Load legacy June weights and QB state without reading the source optimizer."""
    _validate_untied_source(source)

    @jax.jit
    def init_model(key):
        return _SOURCE_MP_POLICY.cast_to_param(Transformer.init(source.model, key=key))

    params = init_model(jax.random.PRNGKey(0))
    pending_qb_betas = jax.numpy.zeros((source.model.num_layers, source.model.num_experts), dtype=jax.numpy.float32)
    checkpoint_path = latest_checkpoint_path(source.checkpoint_dir)
    loaded = load_june_checkpoint(
        {"params": params, "pending_qb_betas": pending_qb_betas},
        checkpoint_path,
        mesh=mesh,
        allow_partial=False,
    )
    loaded_params = cast(Transformer, loaded["params"])
    if loaded_params.config.resolved_expert_bank_for_layer != tuple(range(source.model.num_layers)):
        raise ValueError("loaded source checkpoint is not an untied June Grug MoE")
    return loaded_params, checkpoint_path


def _load_source_expert_banks(
    source: JuneSourceCheckpointConfig,
    mesh: jax.sharding.Mesh,
) -> tuple[ArrayStacked[MoEExpertMlp], str]:
    """Load only the legacy routed-expert arrays, sharded for matching."""
    _validate_untied_source(source)
    gate_sharding = NamedSharding(mesh, P(None, "expert", "data", "model"))
    down_sharding = NamedSharding(mesh, P(None, "expert", "model", "data"))
    gate_shape = (
        source.model.num_layers,
        source.model.num_experts,
        source.model.hidden_dim,
        source.model.intermediate_dim,
    )
    down_shape = (
        source.model.num_layers,
        source.model.num_experts,
        source.model.intermediate_dim,
        source.model.hidden_dim,
    )
    exemplar = MoEExpertMlp(
        w_gate=jax.ShapeDtypeStruct(gate_shape, jnp.float32, sharding=gate_sharding),
        w_up=jax.ShapeDtypeStruct(gate_shape, jnp.float32, sharding=gate_sharding),
        w_down=jax.ShapeDtypeStruct(down_shape, jnp.float32, sharding=down_sharding),
        implementation=source.model.moe_implementation,
        activation=ActivationFunctionEnum.silu,
        capacity_factor=1.0,
    )
    checkpoint_path = latest_checkpoint_path(source.checkpoint_dir)
    loaded = load_legacy_june_stacked_experts(exemplar, checkpoint_path, mesh=mesh)
    return (
        ArrayStacked(stacked=loaded, num_layers=source.model.num_layers, gradient_checkpointing=False),
        checkpoint_path,
    )


def _validate_stage_storage(
    *,
    source_checkpoint: str,
    output_path: str,
    resources,
    calibration_path: str | None = None,
    data=None,
) -> None:
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
    return multihost_utils.process_allgather(traces, tiled=True)


def _append_bounded_trace(
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    trace_arrays,
    *,
    remaining_capacity: int,
) -> None:
    take = min(remaining_capacity, int(trace_arrays[0].shape[0]))
    if take <= 0:
        return
    chunks.append(tuple(np.asarray(value)[:take] for value in trace_arrays))


def _merge_trace_chunks(chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> MoeLayerTrace:
    fields = tuple(np.concatenate(values, axis=0) for values in zip(*chunks, strict=True))
    return MoeLayerTrace(*fields)


def _validate_expert_calibration(calibration: ExpertCalibration, *, layer: int, expert: int) -> None:
    for split_name, sample in (("train", calibration.train), ("heldout", calibration.heldout)):
        if sample.states.shape[0] == 0 or float(np.sum(sample.weights)) <= 0:
            raise ValueError(f"layer {layer} expert {expert} has no positive-weight {split_name} calibration states")


def _stack_probe_preparations(preparations: list[SpectralProbePreparation]) -> tuple[np.ndarray, ...]:
    return (
        np.stack([preparation.manifold.mean for preparation in preparations]),
        np.stack([preparation.manifold.eigenvectors for preparation in preparations]),
        np.stack([preparation.manifold.eigenvalues for preparation in preparations]),
        np.asarray([preparation.manifold.mahalanobis_radius for preparation in preparations], dtype=np.float32),
        np.stack([preparation.centers for preparation in preparations]),
        np.stack([preparation.ordinary_inputs for preparation in preparations]),
        np.stack([preparation.ordinary_weights for preparation in preparations]),
    )


def _probe_preparation_at(
    stacked: tuple[np.ndarray, ...],
    expert: int,
    *,
    config: SpectralProbeConfig,
) -> SpectralProbePreparation:
    means, eigenvectors, eigenvalues, radii, centers, ordinary_inputs, ordinary_weights = stacked
    return SpectralProbePreparation(
        manifold=InputManifold(
            mean=means[expert],
            eigenvectors=eigenvectors[expert],
            eigenvalues=eigenvalues[expert],
            mahalanobis_radius=float(radii[expert]),
        ),
        centers=centers[expert],
        ordinary_inputs=ordinary_inputs[expert],
        ordinary_weights=ordinary_weights[expert],
        config=config,
    )


def _sync_write(run_id: str, stage: str, write) -> None:
    multihost_utils.sync_global_devices(f"{run_id}-{stage}-ready")
    if jax.process_index() == 0:
        write()
    multihost_utils.sync_global_devices(f"{run_id}-{stage}-committed")


def _initialize_worker() -> None:
    logging.basicConfig(level=logging.INFO)
    DistributedConfig().initialize()


def run_calibration_local(config: JuneCalibrationJobConfig) -> None:
    _validate_stage_storage(
        source_checkpoint=config.source.checkpoint_dir,
        output_path=config.output_path,
        resources=config.resources,
        data=config.data,
    )
    if config.calibration_tokens <= 0:
        raise ValueError(f"calibration_tokens must be positive, got {config.calibration_tokens}")
    if config.trace_sample_size <= 0:
        raise ValueError(f"trace_sample_size must be positive, got {config.trace_sample_size}")
    if config.trace_capacity <= 0:
        raise ValueError(f"trace_capacity must be positive, got {config.trace_capacity}")
    if len(set(config.layers)) != len(config.layers):
        raise ValueError(f"calibration layers must be distinct, got {config.layers}")

    _initialize_worker()
    mesh = _merge_mesh(config)
    with set_mesh(mesh):
        params, source_checkpoint = _load_source_params(config.source, mesh)
        if any(layer < 0 or layer >= config.source.model.num_layers for layer in config.layers):
            raise ValueError(f"calibration layers are outside the source model: {config.layers}")
        compute_params = _SOURCE_MP_POLICY.cast_to_compute(params)

        is_writer = jax.process_index() == 0
        reservoirs_by_layer = (
            {
                layer: ExpertReservoirCollection(
                    num_experts=config.source.model.num_experts,
                    state_dim=config.source.model.hidden_dim,
                    capacity_per_expert=config.capacity_per_expert,
                    heldout_fraction=config.heldout_fraction,
                    seed=config.seed + layer,
                    dtype=np.dtype(np.float32),
                )
                for layer in config.layers
            }
            if is_writer
            else {}
        )
        trace_chunks_by_layer: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = (
            {layer: [] for layer in config.layers} if is_writer else {}
        )
        trace_tokens_by_layer = {layer: 0 for layer in config.layers} if is_writer else {}
        batch_schedule = BatchSchedule(config.batch_size)
        train_dataset = build_train_dataset(
            config.data,
            max_seq_len=config.source.model.max_seq_len,
            batch_schedule=batch_schedule,
            key=jax.random.PRNGKey(config.seed),
        )
        train_loader = build_train_loader(train_dataset, batch_schedule=batch_schedule, mesh=mesh)
        layers = config.layers

        @jax.jit
        def trace_batch(tokens, attn_mask):
            total_tokens = tokens.shape[0] * tokens.shape[1]
            token_indices = (
                jax.numpy.arange(config.trace_sample_size, dtype=jax.numpy.int32)
                * total_tokens
                // config.trace_sample_size
            )
            _, traces, capacity_overflow = forward_with_moe_traces(
                compute_params,
                tokens,
                target_layers=layers,
                token_indices=token_indices,
                mask=attn_mask,
            )
            trace_fields = tuple(
                (
                    traces[layer].mlp_input,
                    traces[layer].selected_experts,
                    traces[layer].combine_weights,
                    traces[layer].routed_output,
                )
                for layer in layers
            )
            return trace_fields, capacity_overflow

        processed_tokens = 0
        iterator = train_loader.iter_from_step(0)
        while processed_tokens < config.calibration_tokens:
            batch = next(iterator)
            local_traces, capacity_overflow = trace_batch(batch.tokens, batch.attn_mask)
            gathered_overflow = multihost_utils.process_allgather(capacity_overflow, tiled=True)
            if np.any(np.asarray(gathered_overflow) != 0):
                raise ValueError("source calibration changed the teacher function through MoE capacity overflow")
            gathered_traces = _gather_trace_arrays(local_traces)
            batch_tokens = int(np.prod(batch.tokens.shape))
            processed_tokens += batch_tokens
            if is_writer:
                for layer, trace_arrays in zip(layers, gathered_traces, strict=True):
                    reservoirs_by_layer[layer].add_routes(
                        trace_arrays[0],
                        trace_arrays[1],
                        trace_arrays[2],
                    )
                    remaining_capacity = config.trace_capacity - trace_tokens_by_layer[layer]
                    _append_bounded_trace(
                        trace_chunks_by_layer[layer],
                        trace_arrays,
                        remaining_capacity=remaining_capacity,
                    )
                    trace_tokens_by_layer[layer] += min(remaining_capacity, int(trace_arrays[0].shape[0]))
            logger.info(
                "Calibration %s processed %d/%d native tokens and sampled %d states",
                config.run_id,
                processed_tokens,
                config.calibration_tokens,
                config.trace_sample_size,
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
        traces_by_layer = (
            {layer: _merge_trace_chunks(trace_chunks_by_layer[layer]) for layer in layers} if is_writer else {}
        )
        _sync_write(
            config.run_id,
            "calibration",
            lambda: write_calibration_artifact(
                config.output_path,
                reservoirs_by_layer,
                manifest,
                traces_by_layer=traces_by_layer,
            ),
        )


def run_matching_local(config: JuneMatchingJobConfig) -> None:
    _validate_stage_storage(
        source_checkpoint=config.source.checkpoint_dir,
        calibration_path=config.calibration_path,
        output_path=config.output_path,
        resources=config.resources,
    )
    if config.representative_layer == config.source_layer:
        raise ValueError("representative_layer and source_layer must differ")

    _initialize_worker()
    mesh = _merge_mesh(config)
    with set_mesh(mesh):
        expert_banks, source_checkpoint = _load_source_expert_banks(config.source, mesh)
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

        representative_bank = jax.tree.map(
            lambda value: value.astype(jnp.bfloat16),
            stacked_expert_bank_at(expert_banks, config.representative_layer),
        )
        source_bank = jax.tree.map(
            lambda value: value.astype(jnp.bfloat16),
            stacked_expert_bank_at(expert_banks, config.source_layer),
        )
        jax.block_until_ready((representative_bank, source_bank))
        del expert_banks
        num_experts = config.source.model.num_experts
        if num_experts % jax.process_count() != 0:
            raise ValueError(f"num_experts={num_experts} must be divisible by process_count={jax.process_count()}")
        experts_per_process = num_experts // jax.process_count()
        local_start = jax.process_index() * experts_per_process
        local_stop = local_start + experts_per_process
        local_preparations = []
        for expert in range(local_start, local_stop):
            calibration = read_expert_calibration(
                config.calibration_path,
                config.source_layer,
                expert,
                manifest=calibration_manifest,
            )
            _validate_expert_calibration(calibration, layer=config.source_layer, expert=expert)
            if calibration.train.states.shape[0] < max(config.probe.num_centers, config.probe.covariance_rank):
                raise ValueError(f"expert {expert} has too few train states for the configured spectral probe")
            if calibration.heldout.states.shape[0] < config.probe.ordinary_samples:
                raise ValueError(f"expert {expert} has too few held-out states for native matching")
            preparation = prepare_spectral_probe_set(
                calibration.train,
                calibration.heldout,
                config=config.probe,
                seed=config.seed + expert,
            )
            if preparation.manifold.eigenvalues.shape != (config.probe.covariance_rank,):
                raise ValueError(f"expert {expert} routed-state manifold is below the configured covariance rank")
            local_preparations.append(preparation)
            logger.info("Matching %s prepared probes for expert %d", config.run_id, expert)
        gathered_preparations = tuple(
            np.asarray(value)
            for value in multihost_utils.process_allgather(_stack_probe_preparations(local_preparations), tiled=True)
        )
        probes = []
        for expert in range(num_experts):
            preparation = _probe_preparation_at(gathered_preparations, expert, config=config.probe)
            probes.append(finalize_spectral_probe_set(source_bank, expert, preparation))
            logger.info("Matching %s finalized probes for expert %d", config.run_id, expert)
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
        _sync_write(
            config.run_id,
            "matching",
            lambda: write_matching_artifact(config.output_path, probe_tuple, costs, manifest),
        )


__all__ = ["run_calibration_local", "run_matching_local"]
