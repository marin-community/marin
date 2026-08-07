# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-state conversion for one-pair June expert merges."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.grug.grug_moe import MoEExpertMlp

from experiments.grug.moe.expert_merge import AssignmentMode
from experiments.june_tpu_67b_a2b.moe.expert_merge import convert_one_expert_pair, permute_pending_qb_beta
from experiments.june_tpu_67b_a2b.moe.model import Transformer
from experiments.june_tpu_67b_a2b.moe.train import GrugTrainState

type OptimizerStateInitializer = Callable[[Transformer], optax.OptState]


@dataclass(frozen=True)
class OnePairMergeCheckpointSpec:
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


@dataclass(frozen=True)
class MergeCheckpointManifest:
    spec: OnePairMergeCheckpointSpec
    source_topology: tuple[int, ...]
    target_topology: tuple[int, ...]
    source_step: int
    recovery_step: int
    ema_converted: bool
    optimizer_state_reset: bool = True
    format_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ConvertedJuneMergeCheckpoint:
    state: GrugTrainState
    manifest: MergeCheckpointManifest


def _scalar_like(value: int, exemplar: jax.Array) -> jax.Array:
    scalar = np.asarray(value, dtype=exemplar.dtype)
    sharding = getattr(exemplar, "sharding", None)
    return jax.device_put(scalar, sharding) if sharding is not None else jnp.asarray(scalar)


def convert_june_state_for_one_pair_merge(
    state: GrugTrainState,
    *,
    spec: OnePairMergeCheckpointSpec,
    init_optimizer_state: OptimizerStateInitializer,
    recovery_step: int = 0,
    shared_bank: MoEExpertMlp | None = None,
) -> ConvertedJuneMergeCheckpoint:
    """Convert June model/QB state and initialize a fresh optimizer for recovery."""
    if recovery_step < 0:
        raise ValueError(f"recovery_step must be non-negative, got {recovery_step}")
    expected_qb_shape = (state.params.config.num_layers, state.params.config.num_experts)
    if state.pending_qb_betas.shape != expected_qb_shape:
        raise ValueError(f"pending_qb_betas must have shape {expected_qb_shape}, got {state.pending_qb_betas.shape}")
    if spec.prefit_applied != (shared_bank is not None):
        raise ValueError("prefit_applied must be true exactly when a prefitted shared bank is supplied")

    source_topology = state.params.config.resolved_expert_bank_for_layer
    permutation = np.asarray(spec.source_to_shared, dtype=np.int32)
    converted_params = convert_one_expert_pair(
        state.params,
        representative_layer=spec.representative_layer,
        source_layer=spec.source_layer,
        source_to_shared=permutation,
        shared_bank=shared_bank,
    )
    if state.ema_params is None:
        converted_ema_params = None
    elif shared_bank is None:
        converted_ema_params = convert_one_expert_pair(
            state.ema_params,
            representative_layer=spec.representative_layer,
            source_layer=spec.source_layer,
            source_to_shared=permutation,
        )
    else:
        converted_ema_params = converted_params

    pending_qb_betas = permute_pending_qb_beta(
        state.pending_qb_betas,
        layer_index=spec.source_layer,
        source_to_shared=permutation,
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
    return ConvertedJuneMergeCheckpoint(state=converted_state, manifest=manifest)


__all__ = [
    "ConvertedJuneMergeCheckpoint",
    "MergeCheckpointManifest",
    "OnePairMergeCheckpointSpec",
    "convert_june_state_for_one_pair_merge",
]
