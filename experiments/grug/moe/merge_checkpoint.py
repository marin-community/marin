# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-state conversion for one-pair Grug expert merges."""

import dataclasses
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.grug.grug_moe import MoEExpertMlp
from rigging.filesystem import StoragePath, prefix_join

from experiments.grug.moe.expert_merge import AssignmentMode, convert_one_expert_pair, permute_pending_qb_beta
from experiments.grug.moe.expert_prefit import PrefitObjective
from experiments.grug.moe.merge_recovery import RecoveryInitialization
from experiments.grug.moe.model import Transformer
from experiments.grug.moe.train import GrugTrainState

MERGE_CHECKPOINT_MANIFEST_FILENAME = "merge_manifest.json"
_MERGE_CHECKPOINT_FORMAT_VERSION = 2

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
    format_version: int = _MERGE_CHECKPOINT_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MergeCheckpointManifest":
        format_version = int(payload["format_version"])
        if format_version != _MERGE_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported merge checkpoint format version {format_version}; "
                f"expected {_MERGE_CHECKPOINT_FORMAT_VERSION}"
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


def write_merge_checkpoint_manifest(checkpoint_path: str, manifest: MergeCheckpointManifest) -> None:
    """Write the conversion manifest beside a native Levanter checkpoint."""
    path = prefix_join(checkpoint_path, MERGE_CHECKPOINT_MANIFEST_FILENAME)
    StoragePath(path).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))


def read_merge_checkpoint_manifest(checkpoint_path: str) -> MergeCheckpointManifest:
    """Read conversion metadata stored beside a native Levanter checkpoint."""
    path = prefix_join(checkpoint_path, MERGE_CHECKPOINT_MANIFEST_FILENAME)
    return MergeCheckpointManifest.from_dict(json.loads(StoragePath(path).read_text()))
