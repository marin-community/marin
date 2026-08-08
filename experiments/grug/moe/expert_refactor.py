# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Correspondence-free joint refactorization of two routed MoE layers."""

from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh
from jax.tree_util import register_dataclass
from levanter.grug.grug_moe import MoEExpertMlp

from experiments.grug.moe.expert_merge import MoeLayerTrace
from experiments.grug.moe.model import MoEMLP

_DEFAULT_NORMALIZATION_EPSILON = 1e-8
_NUM_AFFECTED_LAYERS = 2


class RefactorSplit(StrEnum):
    TRAIN = "train"
    HELDOUT = "heldout"


@dataclass(frozen=True)
class ExpertRefactorDataset:
    """Cached complete routed-output targets for one affected teacher layer."""

    source_layer: int
    train_inputs: np.ndarray
    train_targets: np.ndarray
    train_target_power: float
    heldout_inputs: np.ndarray
    heldout_targets: np.ndarray
    heldout_target_power: float


@register_dataclass
@dataclass(frozen=True)
class ExpertRefactorBatch:
    """Layer-balanced inputs and complete routed-output targets."""

    inputs: jax.Array
    targets: jax.Array
    layer_indices: jax.Array
    target_power_by_layer: jax.Array


@register_dataclass
@dataclass(frozen=True)
class ExpertRefactorForward:
    """Complete routed predictions and production routing health by layer."""

    predictions: jax.Array
    routing_entropy_by_layer: jax.Array
    routing_counts_by_layer: jax.Array
    capacity_overflow_by_layer: jax.Array


class ExpertRefactorParameters(eqx.Module):
    """The complete trainable surface of a two-layer joint refactorization."""

    bank: MoEExpertMlp
    routers: tuple[MoEMLP, MoEMLP]


@register_dataclass
@dataclass(frozen=True)
class ExpertRefactorState:
    step: jax.Array
    parameters: ExpertRefactorParameters
    opt_state: optax.OptState


def _validate_parameters(parameters: ExpertRefactorParameters) -> None:
    if len(parameters.routers) != _NUM_AFFECTED_LAYERS:
        raise ValueError(f"joint refactorization requires exactly {_NUM_AFFECTED_LAYERS} routers")
    num_experts, hidden_dim, intermediate_dim = parameters.bank.w_gate.shape
    if parameters.bank.w_up.shape != parameters.bank.w_gate.shape:
        raise ValueError("shared expert gate and up projections must have matching shapes")
    if parameters.bank.w_down.shape != (num_experts, intermediate_dim, hidden_dim):
        raise ValueError("shared expert down projection must match the bank's expert and hidden dimensions")
    for router in parameters.routers:
        if router.cfg.num_experts_per_token != 4:
            raise ValueError("joint refactorization requires the production top-4 router configuration")
        if router.router.shape != (hidden_dim, num_experts) or router.router_bias.shape != (num_experts,):
            raise ValueError("each router projection and QB bias must match the shared expert bank")


def _split_target_power(targets: np.ndarray) -> float:
    return float(np.mean(np.sum(np.square(targets.astype(np.float64)), axis=-1)))


def make_expert_refactor_dataset(
    trace: MoeLayerTrace,
    *,
    source_layer: int,
    heldout_fraction: float,
    seed: int,
) -> ExpertRefactorDataset:
    """Deterministically split a trace after discarding all teacher routing fields."""
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError(f"heldout_fraction must lie strictly between zero and one, got {heldout_fraction}")
    inputs = np.asarray(trace.mlp_input)
    targets = np.asarray(trace.routed_output)
    if inputs.ndim != 2 or targets.shape != inputs.shape:
        raise ValueError(
            f"mlp inputs and complete routed outputs must have matching [T, D] shapes, got "
            f"{inputs.shape} and {targets.shape}"
        )
    if inputs.shape[0] < 2:
        raise ValueError("joint refactorization needs at least two trace tokens per layer")

    permutation = np.random.default_rng(seed).permutation(inputs.shape[0])
    heldout_count = min(inputs.shape[0] - 1, max(1, round(inputs.shape[0] * heldout_fraction)))
    heldout_indices = permutation[:heldout_count]
    train_indices = permutation[heldout_count:]
    train_targets = targets[train_indices]
    heldout_targets = targets[heldout_indices]
    return ExpertRefactorDataset(
        source_layer=source_layer,
        train_inputs=inputs[train_indices],
        train_targets=train_targets,
        train_target_power=_split_target_power(train_targets),
        heldout_inputs=inputs[heldout_indices],
        heldout_targets=heldout_targets,
        heldout_target_power=_split_target_power(heldout_targets),
    )


def sample_expert_refactor_batch(
    datasets: tuple[ExpertRefactorDataset, ExpertRefactorDataset],
    *,
    examples_per_layer: int,
    split: RefactorSplit,
    rng: np.random.Generator,
) -> ExpertRefactorBatch:
    """Sample the same number of cached examples from each affected layer."""
    if len(datasets) != _NUM_AFFECTED_LAYERS:
        raise ValueError(f"joint refactorization requires exactly {_NUM_AFFECTED_LAYERS} layer datasets")
    if datasets[0].source_layer == datasets[1].source_layer:
        raise ValueError("joint refactorization source layers must be distinct")
    if examples_per_layer <= 0:
        raise ValueError(f"examples_per_layer must be positive, got {examples_per_layer}")

    input_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    target_power_by_layer: list[float] = []
    for dataset in datasets:
        if split is RefactorSplit.HELDOUT:
            inputs = dataset.heldout_inputs
            targets = dataset.heldout_targets
            target_power = dataset.heldout_target_power
        else:
            inputs = dataset.train_inputs
            targets = dataset.train_targets
            target_power = dataset.train_target_power
        indices = rng.choice(inputs.shape[0], size=examples_per_layer, replace=inputs.shape[0] < examples_per_layer)
        input_rows.append(inputs[indices])
        target_rows.append(targets[indices])
        target_power_by_layer.append(target_power)

    return ExpertRefactorBatch(
        inputs=jnp.asarray(np.stack(input_rows, axis=0)),
        targets=jnp.asarray(np.stack(target_rows, axis=0)),
        layer_indices=jnp.asarray([dataset.source_layer for dataset in datasets], dtype=jnp.int32),
        target_power_by_layer=jnp.asarray(target_power_by_layer, dtype=jnp.float32),
    )


def expert_refactor_forward(
    parameters: ExpertRefactorParameters,
    batch: ExpertRefactorBatch,
) -> ExpertRefactorForward:
    """Evaluate production routing and the shared bank independently per layer."""
    _validate_parameters(parameters)
    if batch.inputs.ndim != 3 or batch.targets.shape != batch.inputs.shape:
        raise ValueError("refactor inputs and targets must have matching [L, N, D] shapes")
    if batch.inputs.shape[0] != _NUM_AFFECTED_LAYERS:
        raise ValueError(f"refactor batches must contain exactly {_NUM_AFFECTED_LAYERS} layers")
    if batch.layer_indices.shape != (_NUM_AFFECTED_LAYERS,):
        raise ValueError(f"refactor layer identities must have shape [{_NUM_AFFECTED_LAYERS}]")
    if batch.target_power_by_layer.shape != (_NUM_AFFECTED_LAYERS,):
        raise ValueError(f"refactor target power must have shape [{_NUM_AFFECTED_LAYERS}]")

    predictions_by_layer = []
    routing_entropy_by_layer = []
    routing_counts_by_layer = []
    capacity_overflow_by_layer = []
    for layer_index, router in enumerate(parameters.routers):
        trace = router.forward_with_trace(batch.inputs[layer_index, :, None, :], parameters.bank)
        predictions_by_layer.append(trace.routed_output[:, 0, :])
        routing_entropy_by_layer.append(trace.router_stats["routing_entropy"])
        routing_counts_by_layer.append(trace.router_stats["routing_counts"])
        capacity_overflow_by_layer.append(trace.router_stats["capacity_overflow"])
    return ExpertRefactorForward(
        predictions=jnp.stack(predictions_by_layer, axis=0),
        routing_entropy_by_layer=jnp.stack(routing_entropy_by_layer),
        routing_counts_by_layer=jnp.stack(routing_counts_by_layer),
        capacity_overflow_by_layer=jnp.stack(capacity_overflow_by_layer),
    )


def expert_refactor_predictions(
    parameters: ExpertRefactorParameters,
    batch: ExpertRefactorBatch,
) -> jax.Array:
    """Return complete routed predictions from both student layers."""
    return expert_refactor_forward(parameters, batch).predictions


def expert_refactor_loss(
    parameters: ExpertRefactorParameters,
    batch: ExpertRefactorBatch,
    *,
    epsilon: float = _DEFAULT_NORMALIZATION_EPSILON,
) -> tuple[jax.Array, jax.Array]:
    """Return layer-balanced normalized complete routed-output loss and NRMSE."""
    predictions = expert_refactor_predictions(parameters, batch)
    squared_error = jnp.sum(jnp.square(predictions.astype(jnp.float32) - batch.targets), axis=-1)
    replicated_sharding = None if get_abstract_mesh().empty else P(None)
    error_by_layer = jnp.einsum("ln->l", squared_error, out_sharding=replicated_sharding)
    error_by_layer = error_by_layer / squared_error.shape[1]
    target_power_by_layer = batch.target_power_by_layer
    if replicated_sharding is not None:
        target_power_by_layer = jax.sharding.reshard(target_power_by_layer, replicated_sharding)
    normalized_mse = error_by_layer / (target_power_by_layer + epsilon)
    return jnp.mean(normalized_mse), jnp.sqrt(normalized_mse)


def initial_expert_refactor_state(
    bank: MoEExpertMlp,
    routers: tuple[MoEMLP, MoEMLP],
    optimizer: optax.GradientTransformation,
) -> ExpertRefactorState:
    """Create optimizer state whose only parameter leaves are one bank and two routers."""
    parameters = ExpertRefactorParameters(bank=bank, routers=routers)
    _validate_parameters(parameters)
    return ExpertRefactorState(
        step=jnp.asarray(0, dtype=jnp.int32),
        parameters=parameters,
        opt_state=optimizer.init(parameters),
    )


def expert_refactor_step(
    state: ExpertRefactorState,
    batch: ExpertRefactorBatch,
    optimizer: optax.GradientTransformation,
    epsilon: float = _DEFAULT_NORMALIZATION_EPSILON,
) -> tuple[ExpertRefactorState, jax.Array]:
    """Update the shared bank and both routers from complete routed-output error."""

    def loss_fn(parameters: ExpertRefactorParameters) -> jax.Array:
        return expert_refactor_loss(parameters, batch, epsilon=epsilon)[0]

    loss, grads = jax.value_and_grad(loss_fn)(state.parameters)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.parameters)
    return (
        ExpertRefactorState(
            step=state.step + jnp.asarray(1, dtype=state.step.dtype),
            parameters=optax.apply_updates(state.parameters, updates),
            opt_state=opt_state,
        ),
        loss,
    )
