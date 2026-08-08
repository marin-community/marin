# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-state conversion for one-pair Grug expert merges."""

import dataclasses
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.grug.grug_moe import MoEExpertMlp
from rigging.filesystem import StoragePath, prefix_join

from experiments.grug.moe.expert_merge import AssignmentMode, convert_one_expert_pair, permute_pending_qb_beta
from experiments.grug.moe.expert_prefit import PrefitObjective
from experiments.grug.moe.merge_recovery import (
    RecoveryInitialization,
    RecoveryStage,
    RecoveryTrainableScope,
)
from experiments.grug.moe.model import RoutedExpertAdapter, Transformer
from experiments.grug.moe.train import GrugTrainState

MERGE_CHECKPOINT_MANIFEST_FILENAME = "merge_manifest.json"
_LEGACY_MERGE_CHECKPOINT_FORMAT_VERSION = 2
_MERGE_CHECKPOINT_FORMAT_VERSION = 3

type OptimizerStateInitializer = Callable[[Transformer], optax.OptState]


@dataclass(frozen=True)
class OnePairMergeCheckpointSpec:
    """Reproducible inputs for converting one untied source layer."""

    representative_layer: int
    source_layer: int
    source_to_shared: tuple[int, ...]
    assignment_mode: AssignmentMode
    source_checkpoint: str | None = None
    source_commit: str | None = None
    calibration_path: str | None = None
    cost_matrix_path: str | None = None
    probe_path: str | None = None
    prefit_applied: bool = False
    prefit_checkpoint: str | None = None
    prefit_objective: PrefitObjective | None = None


class CapacityOracleKind(StrEnum):
    """Machine-readable purpose of a capacity-oracle checkpoint."""

    UNTIED_IDENTICAL_START_DIAGNOSTIC = "untied_identical_start_diagnostic"


@dataclass(frozen=True)
class CapacityOracleProvenance:
    """Provenance for splitting one recovered shared bank without changing its function."""

    kind: CapacityOracleKind
    source_checkpoint: str
    representative_layer: int
    source_layer: int
    input_topology: tuple[int, ...]
    source_shared_bank_index: int
    duplicated_bank_index: int
    source_recovery_step: int
    output_step: int


class LayerAdapterKind(StrEnum):
    """Machine-readable form of a function-preserving layer adapter."""

    ZERO_INITIALIZED_INPUT_OUTPUT = "zero_initialized_input_output"


@dataclass(frozen=True)
class LayerAdapterProvenance:
    """Provenance for adding one zero-initialized routed-expert adapter."""

    kind: LayerAdapterKind
    source_checkpoint: str
    layer_index: int
    rank: int
    input_topology: tuple[int, ...]
    source_recovery_step: int
    output_step: int


@dataclass(frozen=True)
class MergeCheckpointManifest:
    """Metadata required to reconstruct a converted checkpoint's static topology."""

    spec: OnePairMergeCheckpointSpec
    source_topology: tuple[int, ...]
    target_topology: tuple[int, ...]
    source_step: int
    recovery_step: int
    ema_converted: bool
    optimizer_state_reset: bool = True
    recovery_initialization: RecoveryInitialization | None = None
    recovery_initial_checkpoint: str | None = None
    recovery_stage: RecoveryStage | None = None
    recovery_trainable_scope: RecoveryTrainableScope | None = None
    recovery_cross_entropy_weight: float | None = None
    recovery_moe_loss_weight: float | None = None
    recovery_logit_kl_weight: float | None = None
    capacity_oracle: CapacityOracleProvenance | None = None
    layer_adapter: LayerAdapterProvenance | None = None
    format_version: int = _MERGE_CHECKPOINT_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MergeCheckpointManifest":
        format_version = int(payload["format_version"])
        if format_version not in {_LEGACY_MERGE_CHECKPOINT_FORMAT_VERSION, _MERGE_CHECKPOINT_FORMAT_VERSION}:
            raise ValueError(
                f"unsupported merge checkpoint format version {format_version}; "
                f"expected one of {_LEGACY_MERGE_CHECKPOINT_FORMAT_VERSION}, {_MERGE_CHECKPOINT_FORMAT_VERSION}"
            )
        spec_payload = payload["spec"]
        if not isinstance(spec_payload, Mapping):
            raise ValueError("merge checkpoint spec must be an object")
        spec = OnePairMergeCheckpointSpec(
            representative_layer=int(spec_payload["representative_layer"]),
            source_layer=int(spec_payload["source_layer"]),
            source_to_shared=tuple(int(index) for index in spec_payload["source_to_shared"]),
            assignment_mode=AssignmentMode(spec_payload["assignment_mode"]),
            source_checkpoint=spec_payload.get("source_checkpoint"),
            source_commit=spec_payload.get("source_commit"),
            calibration_path=spec_payload.get("calibration_path"),
            cost_matrix_path=spec_payload.get("cost_matrix_path"),
            probe_path=spec_payload.get("probe_path"),
            prefit_applied=bool(spec_payload.get("prefit_applied", False)),
            prefit_checkpoint=spec_payload["prefit_checkpoint"],
            prefit_objective=(
                PrefitObjective(spec_payload["prefit_objective"])
                if spec_payload.get("prefit_objective") is not None
                else None
            ),
        )
        capacity_oracle_payload = payload.get("capacity_oracle")
        if capacity_oracle_payload is not None and not isinstance(capacity_oracle_payload, Mapping):
            raise ValueError("capacity_oracle must be an object")
        capacity_oracle = (
            CapacityOracleProvenance(
                kind=CapacityOracleKind(capacity_oracle_payload["kind"]),
                source_checkpoint=str(capacity_oracle_payload["source_checkpoint"]),
                representative_layer=int(capacity_oracle_payload["representative_layer"]),
                source_layer=int(capacity_oracle_payload["source_layer"]),
                input_topology=tuple(int(index) for index in capacity_oracle_payload["input_topology"]),
                source_shared_bank_index=int(capacity_oracle_payload["source_shared_bank_index"]),
                duplicated_bank_index=int(capacity_oracle_payload["duplicated_bank_index"]),
                source_recovery_step=int(capacity_oracle_payload["source_recovery_step"]),
                output_step=int(capacity_oracle_payload["output_step"]),
            )
            if capacity_oracle_payload is not None
            else None
        )
        layer_adapter_payload = payload.get("layer_adapter")
        if layer_adapter_payload is not None and not isinstance(layer_adapter_payload, Mapping):
            raise ValueError("layer_adapter must be an object")
        if format_version == _LEGACY_MERGE_CHECKPOINT_FORMAT_VERSION and layer_adapter_payload is not None:
            raise ValueError("layer_adapter provenance requires merge checkpoint format version 3")
        layer_adapter = (
            LayerAdapterProvenance(
                kind=LayerAdapterKind(layer_adapter_payload["kind"]),
                source_checkpoint=str(layer_adapter_payload["source_checkpoint"]),
                layer_index=int(layer_adapter_payload["layer_index"]),
                rank=int(layer_adapter_payload["rank"]),
                input_topology=tuple(int(index) for index in layer_adapter_payload["input_topology"]),
                source_recovery_step=int(layer_adapter_payload["source_recovery_step"]),
                output_step=int(layer_adapter_payload["output_step"]),
            )
            if layer_adapter_payload is not None
            else None
        )
        return cls(
            spec=spec,
            source_topology=tuple(int(index) for index in payload["source_topology"]),
            target_topology=tuple(int(index) for index in payload["target_topology"]),
            source_step=int(payload["source_step"]),
            recovery_step=int(payload["recovery_step"]),
            ema_converted=bool(payload["ema_converted"]),
            optimizer_state_reset=bool(payload["optimizer_state_reset"]),
            recovery_initialization=(
                RecoveryInitialization(payload["recovery_initialization"])
                if payload.get("recovery_initialization") is not None
                else None
            ),
            recovery_initial_checkpoint=payload.get("recovery_initial_checkpoint"),
            recovery_stage=(
                RecoveryStage(payload["recovery_stage"]) if payload.get("recovery_stage") is not None else None
            ),
            recovery_trainable_scope=(
                RecoveryTrainableScope(payload["recovery_trainable_scope"])
                if payload.get("recovery_trainable_scope") is not None
                else None
            ),
            recovery_cross_entropy_weight=(
                float(payload["recovery_cross_entropy_weight"])
                if payload.get("recovery_cross_entropy_weight") is not None
                else None
            ),
            recovery_moe_loss_weight=(
                float(payload["recovery_moe_loss_weight"])
                if payload.get("recovery_moe_loss_weight") is not None
                else None
            ),
            recovery_logit_kl_weight=(
                float(payload["recovery_logit_kl_weight"])
                if payload.get("recovery_logit_kl_weight") is not None
                else None
            ),
            capacity_oracle=capacity_oracle,
            layer_adapter=layer_adapter,
            format_version=format_version,
        )


@dataclass(frozen=True)
class ConvertedMergeCheckpoint:
    state: GrugTrainState
    manifest: MergeCheckpointManifest


def _scalar_like(value: int, exemplar: jax.Array) -> jax.Array:
    scalar = np.asarray(value, dtype=exemplar.dtype)
    sharding = getattr(exemplar, "sharding", None)
    return jax.device_put(scalar, sharding) if sharding is not None else jnp.asarray(scalar)


def convert_grug_state_for_one_pair_merge(
    state: GrugTrainState,
    *,
    spec: OnePairMergeCheckpointSpec,
    init_optimizer_state: OptimizerStateInitializer,
    recovery_step: int = 0,
    shared_bank: MoEExpertMlp | None = None,
) -> ConvertedMergeCheckpoint:
    """Convert model state and initialize a fresh optimizer state for recovery."""
    if recovery_step < 0:
        raise ValueError(f"recovery_step must be non-negative, got {recovery_step}")
    expected_qb_shape = (len(state.params.blocks), state.params.config.num_experts)
    if state.pending_qb_betas.shape != expected_qb_shape:
        raise ValueError(f"pending_qb_betas must have shape {expected_qb_shape}, got {state.pending_qb_betas.shape}")
    if spec.prefit_applied != (shared_bank is not None):
        raise ValueError("prefit_applied must be true exactly when a prefitted shared bank is supplied")

    source_topology = state.params.config.resolved_expert_bank_for_layer
    converted_params = convert_one_expert_pair(
        state.params,
        representative_layer=spec.representative_layer,
        source_layer=spec.source_layer,
        source_to_shared=np.asarray(spec.source_to_shared, dtype=np.int32),
        shared_bank=shared_bank,
    )
    if state.ema_params is None:
        converted_ema_params = None
    elif shared_bank is None:
        converted_ema_params = convert_one_expert_pair(
            state.ema_params,
            representative_layer=spec.representative_layer,
            source_layer=spec.source_layer,
            source_to_shared=np.asarray(spec.source_to_shared, dtype=np.int32),
        )
    else:
        # A prefitted bank has no historical EMA. Start recovery EMA from the
        # complete converted parameter tree instead of mixing stale and fresh leaves.
        converted_ema_params = converted_params

    pending_qb_betas = state.pending_qb_betas.at[spec.source_layer].set(
        permute_pending_qb_beta(
            state.pending_qb_betas[spec.source_layer],
            np.asarray(spec.source_to_shared, dtype=np.int32),
        )
    )
    converted_state = dataclasses.replace(
        state,
        step=_scalar_like(recovery_step, state.step),
        params=converted_params,
        opt_state=init_optimizer_state(converted_params),
        ema_params=converted_ema_params,
        pending_qb_betas=pending_qb_betas,
    )
    manifest = MergeCheckpointManifest(
        spec=spec,
        source_topology=source_topology,
        target_topology=converted_params.config.resolved_expert_bank_for_layer,
        source_step=int(jax.device_get(state.step)),
        recovery_step=recovery_step,
        ema_converted=converted_ema_params is not None,
    )
    return ConvertedMergeCheckpoint(state=converted_state, manifest=manifest)


def _copy_expert_bank(bank: MoEExpertMlp) -> MoEExpertMlp:
    return jax.tree.map(lambda value: jnp.array(value, copy=True), bank)


def _add_layer_adapter(
    model: Transformer,
    *,
    layer_index: int,
    rank: int,
    key: jax.Array,
) -> Transformer:
    if not 0 <= layer_index < len(model.blocks):
        raise IndexError(f"layer_index must be in [0, {len(model.blocks)}), got {layer_index}")
    if rank <= 0:
        raise ValueError(f"adapter rank must be positive, got {rank}")
    input_ranks = model.config.resolved_expert_adapter_rank_for_layer
    if any(input_ranks):
        raise ValueError(f"layer-adapter conversion requires an adapter-free model, got ranks {input_ranks}")

    output_ranks = tuple(rank if index == layer_index else 0 for index in range(len(model.blocks)))
    converted_config = dataclasses.replace(model.config, expert_adapter_rank_for_layer=output_ranks)
    adapter = RoutedExpertAdapter.init(model.config.hidden_dim, rank, key=key)
    converted_blocks = tuple(
        dataclasses.replace(
            block,
            attn=dataclasses.replace(block.attn, cfg=converted_config),
            mlp=dataclasses.replace(block.mlp, cfg=converted_config),
            routed_expert_adapter=adapter if index == layer_index else None,
        )
        for index, block in enumerate(model.blocks)
    )
    return dataclasses.replace(model, blocks=converted_blocks, config=converted_config)


def convert_grug_state_for_layer_adapter(
    state: GrugTrainState,
    *,
    source_manifest: MergeCheckpointManifest,
    source_checkpoint: str,
    layer_index: int,
    rank: int,
    key: jax.Array,
    init_optimizer_state: OptimizerStateInitializer,
    adapter_step: int = 0,
) -> ConvertedMergeCheckpoint:
    """Add one zero-initialized layer adapter without changing the checkpoint function."""
    if adapter_step < 0:
        raise ValueError(f"adapter_step must be non-negative, got {adapter_step}")
    if not source_checkpoint:
        raise ValueError("source_checkpoint must identify the recovered tied checkpoint")
    if source_manifest.layer_adapter is not None:
        raise ValueError("layer-adapter conversion cannot augment a checkpoint that already has an adapter")

    input_topology = state.params.config.resolved_expert_bank_for_layer
    if input_topology != source_manifest.target_topology:
        raise ValueError(
            f"checkpoint topology {input_topology} does not match merge manifest target "
            f"{source_manifest.target_topology}"
        )
    if layer_index != source_manifest.spec.source_layer:
        raise ValueError(
            f"layer adapter must be installed on merged source layer {source_manifest.spec.source_layer}, "
            f"got {layer_index}"
        )
    representative_layer = source_manifest.spec.representative_layer
    shared_bank_index = state.params.blocks[representative_layer].expert_bank_index
    if state.params.blocks[layer_index].expert_bank_index != shared_bank_index:
        raise ValueError("layer-adapter conversion requires the representative and source layers to share a bank")

    converted_params = _add_layer_adapter(state.params, layer_index=layer_index, rank=rank, key=key)
    converted_ema_params = (
        None
        if state.ema_params is None
        else _add_layer_adapter(state.ema_params, layer_index=layer_index, rank=rank, key=key)
    )
    converted_state = dataclasses.replace(
        state,
        step=_scalar_like(adapter_step, state.step),
        params=converted_params,
        opt_state=init_optimizer_state(converted_params),
        ema_params=converted_ema_params,
    )
    provenance = LayerAdapterProvenance(
        kind=LayerAdapterKind.ZERO_INITIALIZED_INPUT_OUTPUT,
        source_checkpoint=source_checkpoint,
        layer_index=layer_index,
        rank=rank,
        input_topology=input_topology,
        source_recovery_step=int(jax.device_get(state.step)),
        output_step=adapter_step,
    )
    manifest = dataclasses.replace(
        source_manifest,
        recovery_step=adapter_step,
        ema_converted=converted_ema_params is not None,
        optimizer_state_reset=True,
        recovery_initialization=None,
        recovery_initial_checkpoint=None,
        recovery_stage=None,
        recovery_trainable_scope=None,
        recovery_cross_entropy_weight=None,
        recovery_moe_loss_weight=None,
        recovery_logit_kl_weight=None,
        layer_adapter=provenance,
        format_version=_MERGE_CHECKPOINT_FORMAT_VERSION,
    )
    return ConvertedMergeCheckpoint(state=converted_state, manifest=manifest)


def _split_recovered_shared_bank(
    model: Transformer,
    *,
    representative_layer: int,
    source_layer: int,
) -> tuple[Transformer, int, int]:
    num_layers = len(model.blocks)
    if representative_layer == source_layer:
        raise ValueError("representative_layer and source_layer must be different")
    if not 0 <= representative_layer < num_layers or not 0 <= source_layer < num_layers:
        raise IndexError(f"layer indices must be in [0, {num_layers})")

    input_topology = model.config.resolved_expert_bank_for_layer
    shared_bank_index = model.blocks[representative_layer].expert_bank_index
    if model.blocks[source_layer].expert_bank_index != shared_bank_index:
        raise ValueError("capacity oracle requires the representative and source layers to share a bank")
    shared_layers = tuple(index for index, bank_index in enumerate(input_topology) if bank_index == shared_bank_index)
    if shared_layers != tuple(sorted((representative_layer, source_layer))):
        raise ValueError(
            "capacity oracle requires a recovered one-pair bank used only by the representative and source layers; "
            f"bank {shared_bank_index} is used by layers {shared_layers}"
        )

    duplicated_bank_index = shared_bank_index + 1
    output_topology = tuple(
        duplicated_bank_index if layer_index == source_layer else bank_index + int(bank_index >= duplicated_bank_index)
        for layer_index, bank_index in enumerate(input_topology)
    )
    converted_config = dataclasses.replace(model.config, expert_bank_for_layer=output_topology)
    converted_blocks = tuple(
        dataclasses.replace(
            block,
            attn=dataclasses.replace(block.attn, cfg=converted_config),
            mlp=dataclasses.replace(block.mlp, cfg=converted_config),
            expert_bank_index=output_topology[layer_index],
        )
        for layer_index, block in enumerate(model.blocks)
    )
    duplicated_bank = _copy_expert_bank(model.expert_banks[shared_bank_index])
    converted_banks = (
        *model.expert_banks[:duplicated_bank_index],
        duplicated_bank,
        *model.expert_banks[duplicated_bank_index:],
    )
    return (
        dataclasses.replace(
            model,
            blocks=converted_blocks,
            expert_banks=converted_banks,
            config=converted_config,
        ),
        shared_bank_index,
        duplicated_bank_index,
    )


def convert_grug_state_for_capacity_oracle_split(
    state: GrugTrainState,
    *,
    source_manifest: MergeCheckpointManifest,
    source_checkpoint: str,
    init_optimizer_state: OptimizerStateInitializer,
    oracle_step: int = 0,
) -> ConvertedMergeCheckpoint:
    """Untie a recovered one-pair bank into identical layer-specific copies.

    This diagnostic changes capacity without changing the model's initial
    function. The source layer keeps its already-permuted router and QB state;
    only its routed expert bank becomes an independent pytree subtree.
    """
    if oracle_step < 0:
        raise ValueError(f"oracle_step must be non-negative, got {oracle_step}")
    if not source_checkpoint:
        raise ValueError("source_checkpoint must identify the recovered tied checkpoint")
    if source_manifest.capacity_oracle is not None:
        raise ValueError("capacity oracle cannot split a checkpoint that is already an oracle")

    input_topology = state.params.config.resolved_expert_bank_for_layer
    if input_topology != source_manifest.target_topology:
        raise ValueError(
            f"checkpoint topology {input_topology} does not match merge manifest target "
            f"{source_manifest.target_topology}"
        )
    representative_layer = source_manifest.spec.representative_layer
    source_layer = source_manifest.spec.source_layer
    converted_params, shared_bank_index, duplicated_bank_index = _split_recovered_shared_bank(
        state.params,
        representative_layer=representative_layer,
        source_layer=source_layer,
    )
    converted_ema_params = (
        None
        if state.ema_params is None
        else _split_recovered_shared_bank(
            state.ema_params,
            representative_layer=representative_layer,
            source_layer=source_layer,
        )[0]
    )
    converted_state = dataclasses.replace(
        state,
        step=_scalar_like(oracle_step, state.step),
        params=converted_params,
        opt_state=init_optimizer_state(converted_params),
        ema_params=converted_ema_params,
    )
    provenance = CapacityOracleProvenance(
        kind=CapacityOracleKind.UNTIED_IDENTICAL_START_DIAGNOSTIC,
        source_checkpoint=source_checkpoint,
        representative_layer=representative_layer,
        source_layer=source_layer,
        input_topology=input_topology,
        source_shared_bank_index=shared_bank_index,
        duplicated_bank_index=duplicated_bank_index,
        source_recovery_step=int(jax.device_get(state.step)),
        output_step=oracle_step,
    )
    manifest = dataclasses.replace(
        source_manifest,
        target_topology=converted_params.config.resolved_expert_bank_for_layer,
        recovery_step=oracle_step,
        ema_converted=converted_ema_params is not None,
        optimizer_state_reset=True,
        recovery_initialization=None,
        recovery_initial_checkpoint=None,
        recovery_stage=None,
        recovery_trainable_scope=None,
        recovery_cross_entropy_weight=None,
        recovery_moe_loss_weight=None,
        recovery_logit_kl_weight=None,
        capacity_oracle=provenance,
    )
    return ConvertedMergeCheckpoint(state=converted_state, manifest=manifest)


def write_merge_checkpoint_manifest(checkpoint_path: str, manifest: MergeCheckpointManifest) -> None:
    """Write the conversion manifest beside a native Levanter checkpoint."""
    path = prefix_join(checkpoint_path, MERGE_CHECKPOINT_MANIFEST_FILENAME)
    StoragePath(path).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))


def read_merge_checkpoint_manifest(checkpoint_path: str) -> MergeCheckpointManifest:
    """Read conversion metadata stored beside a native Levanter checkpoint."""
    path = prefix_join(checkpoint_path, MERGE_CHECKPOINT_MANIFEST_FILENAME)
    return MergeCheckpointManifest.from_dict(json.loads(StoragePath(path).read_text()))
