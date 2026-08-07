# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Balanced offline distillation for a functionally matched shared expert bank."""

from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh
from jax.tree_util import register_dataclass
from levanter.grug.grug_moe import MoEExpertMlp

from experiments.grug.moe.expert_merge import MoeLayerTrace, eval_expert

_DEFAULT_NORMALIZATION_EPSILON = 1e-8


class PrefitSplit(StrEnum):
    TRAIN = "train"
    HELDOUT = "heldout"


class PrefitObjective(StrEnum):
    PER_EXPERT = "per_expert"
    AGGREGATE_ROUTED = "aggregate_routed"


@dataclass(frozen=True)
class PrefitDataset:
    """Native and spectral examples for one source expert in one source layer."""

    source_layer: int
    source_expert: int
    shared_expert: int
    train_inputs: np.ndarray
    train_targets: np.ndarray
    heldout_inputs: np.ndarray
    heldout_targets: np.ndarray

    def __post_init__(self) -> None:
        for split, inputs, targets in (
            ("train", self.train_inputs, self.train_targets),
            ("heldout", self.heldout_inputs, self.heldout_targets),
        ):
            if inputs.ndim != 2 or targets.shape != inputs.shape:
                raise ValueError(
                    f"{split} inputs and targets must have matching [N, D] shapes, got "
                    f"{inputs.shape} and {targets.shape}"
                )
            if inputs.shape[0] == 0:
                raise ValueError(f"{split} split must contain at least one example")


@register_dataclass
@dataclass(frozen=True)
class PrefitBatch:
    inputs: jax.Array
    targets: jax.Array
    shared_experts: jax.Array
    source_indices: jax.Array
    target_power_by_source: jax.Array


@dataclass(frozen=True)
class AggregatePrefitDataset:
    """Deterministically split routed-MoE trace for one source layer."""

    source_layer: int
    source_to_shared: tuple[int, ...]
    train: MoeLayerTrace
    heldout: MoeLayerTrace


@register_dataclass
@dataclass(frozen=True)
class AggregatePrefitBatch:
    inputs: jax.Array
    targets: jax.Array
    shared_experts: jax.Array
    combine_weights: jax.Array
    layer_indices: jax.Array
    target_power_by_layer: jax.Array


@register_dataclass
@dataclass(frozen=True)
class PrefitState:
    step: jax.Array
    bank: MoEExpertMlp
    opt_state: optax.OptState


@dataclass(frozen=True)
class PrefitConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    steps: int = 2_000
    examples_per_source: int = 2
    heldout_examples_per_source: int = 8
    eval_every: int = 100
    early_stopping_patience: int = 5
    epsilon: float = _DEFAULT_NORMALIZATION_EPSILON
    aggregate_examples_per_layer: int = 256
    aggregate_heldout_examples_per_layer: int = 512
    aggregate_trace_heldout_fraction: float = 0.2


@dataclass(frozen=True)
class PrefitEvaluation:
    step: int
    loss: float
    nrmse_by_source: np.ndarray


@dataclass(frozen=True)
class PrefitResult:
    bank: MoEExpertMlp
    evaluations: tuple[PrefitEvaluation, ...]
    stopped_early: bool


def aggregate_routed_moe_nrmse(
    shared_bank: MoEExpertMlp,
    trace: MoeLayerTrace,
    source_to_shared: tuple[int, ...] | np.ndarray,
    *,
    epsilon: float = _DEFAULT_NORMALIZATION_EPSILON,
) -> jax.Array:
    """Compare a shared bank with one teacher layer under frozen routing.

    ``source_to_shared[i]`` is the shared-bank slot assigned to source expert
    ``i``. The trace's combine weights and teacher routed output are used
    directly, so this excludes the block's always-on dense MLP.
    """
    assignment = np.asarray(source_to_shared, dtype=np.int32)
    num_experts = shared_bank.w_gate.shape[0]
    if assignment.shape != (num_experts,) or not np.array_equal(np.sort(assignment), np.arange(num_experts)):
        raise ValueError(f"source_to_shared must be a bijection over {num_experts} experts")
    if trace.mlp_input.ndim != 2 or trace.routed_output.shape != trace.mlp_input.shape:
        raise ValueError("trace inputs and routed outputs must have matching [T, D] shapes")
    if trace.selected_experts.shape != trace.combine_weights.shape or trace.selected_experts.ndim != 2:
        raise ValueError("trace expert IDs and combine weights must have matching [T, K] shapes")
    if trace.selected_experts.shape[0] != trace.mlp_input.shape[0]:
        raise ValueError("trace routing and input token counts must match")

    selected_experts = np.asarray(trace.selected_experts, dtype=np.int32)
    if np.any(selected_experts < 0) or np.any(selected_experts >= num_experts):
        raise ValueError("trace contains an out-of-range source expert ID")
    mapped_experts = jnp.asarray(assignment[selected_experts])
    shared_output = shared_bank(
        jnp.asarray(trace.mlp_input),
        mapped_experts,
        jnp.asarray(trace.combine_weights),
    )
    if isinstance(shared_output, tuple):
        shared_output = shared_output[0]
    teacher_output = jnp.asarray(trace.routed_output, dtype=jnp.float32)
    error = jnp.sum(jnp.square(shared_output.astype(jnp.float32) - teacher_output))
    teacher_power = jnp.sum(jnp.square(teacher_output))
    return jnp.sqrt(error / (teacher_power + epsilon))


def _slice_trace(trace: MoeLayerTrace, indices: np.ndarray) -> MoeLayerTrace:
    return MoeLayerTrace(
        mlp_input=np.asarray(trace.mlp_input)[indices],
        selected_experts=np.asarray(trace.selected_experts)[indices],
        combine_weights=np.asarray(trace.combine_weights)[indices],
        routed_output=np.asarray(trace.routed_output)[indices],
    )


def make_aggregate_prefit_dataset(
    trace: MoeLayerTrace,
    *,
    source_layer: int,
    source_to_shared: tuple[int, ...],
    heldout_fraction: float,
    seed: int,
) -> AggregatePrefitDataset:
    """Split one cached layer trace into deterministic train and held-out subsets."""
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError(f"heldout_fraction must lie strictly between zero and one, got {heldout_fraction}")
    num_tokens = trace.mlp_input.shape[0]
    if num_tokens < 2:
        raise ValueError("aggregate prefit needs at least two trace tokens per layer")
    assignment = np.asarray(source_to_shared, dtype=np.int32)
    if assignment.ndim != 1 or not np.array_equal(np.sort(assignment), np.arange(assignment.shape[0])):
        raise ValueError("source_to_shared must be a bijection")
    selected_experts = np.asarray(trace.selected_experts)
    if np.any(selected_experts < 0) or np.any(selected_experts >= assignment.shape[0]):
        raise ValueError("trace contains an out-of-range source expert ID")

    permutation = np.random.default_rng(seed).permutation(num_tokens)
    heldout_count = min(num_tokens - 1, max(1, round(num_tokens * heldout_fraction)))
    heldout_indices = permutation[:heldout_count]
    train_indices = permutation[heldout_count:]
    return AggregatePrefitDataset(
        source_layer=source_layer,
        source_to_shared=tuple(int(index) for index in assignment),
        train=_slice_trace(trace, train_indices),
        heldout=_slice_trace(trace, heldout_indices),
    )


def sample_aggregate_prefit_batch(
    datasets: tuple[AggregatePrefitDataset, ...],
    *,
    examples_per_layer: int,
    split: PrefitSplit,
    rng: np.random.Generator,
) -> AggregatePrefitBatch:
    """Sample an equal number of frozen routed examples from every layer."""
    if not datasets:
        raise ValueError("at least one aggregate prefit dataset is required")
    if examples_per_layer <= 0:
        raise ValueError(f"examples_per_layer must be positive, got {examples_per_layer}")
    if len({dataset.source_layer for dataset in datasets}) != len(datasets):
        raise ValueError("aggregate prefit source layers must be distinct")

    input_rows = []
    target_rows = []
    shared_expert_rows = []
    combine_rows = []
    layer_indices = []
    target_power = []
    for layer_index, dataset in enumerate(datasets):
        trace = dataset.heldout if split is PrefitSplit.HELDOUT else dataset.train
        inputs = np.asarray(trace.mlp_input)
        indices = rng.choice(inputs.shape[0], size=examples_per_layer, replace=inputs.shape[0] < examples_per_layer)
        targets = np.asarray(trace.routed_output)
        assignment = np.asarray(dataset.source_to_shared, dtype=np.int32)
        input_rows.append(inputs[indices])
        target_rows.append(targets[indices])
        shared_expert_rows.append(assignment[np.asarray(trace.selected_experts)[indices]])
        combine_rows.append(np.asarray(trace.combine_weights)[indices])
        layer_indices.extend([layer_index] * examples_per_layer)
        target_power.append(float(np.mean(np.sum(np.square(targets.astype(np.float64)), axis=-1))))

    return AggregatePrefitBatch(
        inputs=jnp.asarray(np.concatenate(input_rows)),
        targets=jnp.asarray(np.concatenate(target_rows)),
        shared_experts=jnp.asarray(np.concatenate(shared_expert_rows), dtype=jnp.int32),
        combine_weights=jnp.asarray(np.concatenate(combine_rows)),
        layer_indices=jnp.asarray(layer_indices, dtype=jnp.int32),
        target_power_by_layer=jnp.asarray(target_power, dtype=jnp.float32),
    )


def aggregate_prefit_loss(
    bank: MoEExpertMlp,
    batch: AggregatePrefitBatch,
    *,
    epsilon: float = _DEFAULT_NORMALIZATION_EPSILON,
) -> tuple[jax.Array, jax.Array]:
    """Return layer-balanced normalized aggregate routed-output loss."""
    predictions = bank(batch.inputs, batch.shared_experts, batch.combine_weights)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    squared_error = jnp.sum(jnp.square(predictions.astype(jnp.float32) - batch.targets), axis=-1)
    num_layers = batch.target_power_by_layer.shape[0]
    layer_mask = jax.nn.one_hot(batch.layer_indices, num_layers, dtype=squared_error.dtype)
    replicated_sharding = None if get_abstract_mesh().empty else P(None)
    error_by_layer = jnp.einsum("n,nl->l", squared_error, layer_mask, out_sharding=replicated_sharding)
    count_by_layer = jnp.einsum(
        "n,nl->l",
        jnp.ones_like(squared_error),
        layer_mask,
        out_sharding=replicated_sharding,
    )
    target_power_by_layer = batch.target_power_by_layer
    if replicated_sharding is not None:
        target_power_by_layer = jax.sharding.reshard(target_power_by_layer, replicated_sharding)
    normalized_mse = error_by_layer / count_by_layer / (target_power_by_layer + epsilon)
    return jnp.mean(normalized_mse), jnp.sqrt(normalized_mse)


def make_prefit_dataset(
    source_bank: MoEExpertMlp,
    *,
    source_layer: int,
    source_expert: int,
    shared_expert: int,
    train_inputs: np.ndarray,
    heldout_inputs: np.ndarray,
) -> PrefitDataset:
    """Materialize black-box source targets once for balanced prefit."""
    train_targets = np.asarray(jax.device_get(eval_expert(source_bank, source_expert, train_inputs)))
    heldout_targets = np.asarray(jax.device_get(eval_expert(source_bank, source_expert, heldout_inputs)))
    return PrefitDataset(
        source_layer=source_layer,
        source_expert=source_expert,
        shared_expert=shared_expert,
        train_inputs=np.asarray(train_inputs),
        train_targets=train_targets,
        heldout_inputs=np.asarray(heldout_inputs),
        heldout_targets=heldout_targets,
    )


def _validate_cluster_balance(datasets: tuple[PrefitDataset, ...]) -> None:
    if not datasets:
        raise ValueError("at least one prefit source dataset is required")
    shared_experts = np.asarray([dataset.shared_expert for dataset in datasets])
    if np.any(shared_experts < 0):
        raise ValueError("shared expert IDs must be non-negative")
    counts = np.bincount(shared_experts)
    if np.any(counts == 0) or np.any(counts != counts[0]):
        raise ValueError("every shared expert must receive the same number of source experts")


def sample_prefit_batch(
    datasets: tuple[PrefitDataset, ...],
    *,
    examples_per_source: int,
    split: PrefitSplit,
    rng: np.random.Generator,
) -> PrefitBatch:
    """Sample the same number of examples from every source expert and layer."""
    _validate_cluster_balance(datasets)
    if examples_per_source <= 0:
        raise ValueError(f"examples_per_source must be positive, got {examples_per_source}")

    input_rows = []
    target_rows = []
    shared_experts = []
    source_indices = []
    target_power = []
    for source_index, dataset in enumerate(datasets):
        inputs = dataset.heldout_inputs if split is PrefitSplit.HELDOUT else dataset.train_inputs
        targets = dataset.heldout_targets if split is PrefitSplit.HELDOUT else dataset.train_targets
        indices = rng.choice(inputs.shape[0], size=examples_per_source, replace=inputs.shape[0] < examples_per_source)
        input_rows.append(inputs[indices])
        target_rows.append(targets[indices])
        shared_experts.extend([dataset.shared_expert] * examples_per_source)
        source_indices.extend([source_index] * examples_per_source)
        target_power.append(float(np.mean(np.sum(np.square(targets.astype(np.float64)), axis=-1))))

    return PrefitBatch(
        inputs=jnp.asarray(np.concatenate(input_rows, axis=0)),
        targets=jnp.asarray(np.concatenate(target_rows, axis=0)),
        shared_experts=jnp.asarray(shared_experts, dtype=jnp.int32),
        source_indices=jnp.asarray(source_indices, dtype=jnp.int32),
        target_power_by_source=jnp.asarray(target_power, dtype=jnp.float32),
    )


def prefit_loss(
    bank: MoEExpertMlp,
    batch: PrefitBatch,
    *,
    epsilon: float = _DEFAULT_NORMALIZATION_EPSILON,
) -> tuple[jax.Array, jax.Array]:
    """Return the balanced normalized loss and per-source NRMSE."""
    predictions = bank(
        batch.inputs,
        batch.shared_experts[:, None],
        jnp.ones((batch.inputs.shape[0], 1), dtype=batch.inputs.dtype),
    )
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    squared_error = jnp.sum(jnp.square(predictions - batch.targets), axis=-1)
    num_sources = batch.target_power_by_source.shape[0]
    source_mask = jax.nn.one_hot(batch.source_indices, num_sources, dtype=squared_error.dtype)
    replicated_sharding = None if get_abstract_mesh().empty else P(None)
    error_by_source = jnp.einsum("n,ns->s", squared_error, source_mask, out_sharding=replicated_sharding)
    count_by_source = jnp.einsum(
        "n,ns->s",
        jnp.ones_like(squared_error),
        source_mask,
        out_sharding=replicated_sharding,
    )
    target_power_by_source = batch.target_power_by_source
    if replicated_sharding is not None:
        target_power_by_source = jax.sharding.reshard(target_power_by_source, replicated_sharding)
    normalized_mse = error_by_source / count_by_source / (target_power_by_source + epsilon)
    return jnp.mean(normalized_mse), jnp.sqrt(normalized_mse)


def _prefit_step(
    state: PrefitState,
    batch: PrefitBatch,
    optimizer: optax.GradientTransformation,
    epsilon: float,
) -> tuple[PrefitState, jax.Array]:
    def loss_fn(bank: MoEExpertMlp) -> jax.Array:
        return prefit_loss(bank, batch, epsilon=epsilon)[0]

    loss, grads = jax.value_and_grad(loss_fn)(state.bank)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.bank)
    return (
        PrefitState(
            step=state.step + jnp.array(1, dtype=state.step.dtype),
            bank=optax.apply_updates(state.bank, updates),
            opt_state=opt_state,
        ),
        loss,
    )


def prefit_shared_bank(
    initial_bank: MoEExpertMlp,
    datasets: tuple[PrefitDataset, ...],
    *,
    config: PrefitConfig = PrefitConfig(),
    seed: int = 0,
) -> PrefitResult:
    """Fit a shared bank and stop when held-out functional error stops improving."""
    if config.steps <= 0 or config.eval_every <= 0:
        raise ValueError("steps and eval_every must be positive")
    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    _validate_cluster_balance(datasets)
    shared_experts = {dataset.shared_expert for dataset in datasets}
    expected_experts = set(range(initial_bank.w_gate.shape[0]))
    if shared_experts != expected_experts:
        raise ValueError(
            "prefit datasets must cover every shared expert exactly through balanced clusters; "
            f"got {sorted(shared_experts)}, expected {sorted(expected_experts)}"
        )

    optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    state = PrefitState(
        step=jnp.array(0, dtype=jnp.int32),
        bank=initial_bank,
        opt_state=optimizer.init(initial_bank),
    )
    rng = np.random.default_rng(seed)
    heldout_batch = sample_prefit_batch(
        datasets,
        examples_per_source=config.heldout_examples_per_source,
        split=PrefitSplit.HELDOUT,
        rng=rng,
    )
    best_bank = initial_bank
    best_loss = np.inf
    stale_evaluations = 0
    evaluations = []
    stopped_early = False

    def evaluate(step: int) -> None:
        nonlocal best_bank, best_loss, stale_evaluations
        loss, nrmse = prefit_loss(state.bank, heldout_batch, epsilon=config.epsilon)
        loss_value = float(jax.device_get(loss))
        evaluations.append(
            PrefitEvaluation(
                step=step,
                loss=loss_value,
                nrmse_by_source=np.asarray(jax.device_get(nrmse)),
            )
        )
        if loss_value < best_loss:
            best_loss = loss_value
            best_bank = state.bank
            stale_evaluations = 0
        else:
            stale_evaluations += 1

    evaluate(0)
    step_fn = jax.jit(_prefit_step, static_argnums=(2, 3))
    for step in range(1, config.steps + 1):
        batch = sample_prefit_batch(
            datasets,
            examples_per_source=config.examples_per_source,
            split=PrefitSplit.TRAIN,
            rng=rng,
        )
        state, _ = step_fn(state, batch, optimizer, config.epsilon)
        if step % config.eval_every != 0 and step != config.steps:
            continue
        evaluate(step)
        if stale_evaluations >= config.early_stopping_patience:
            stopped_early = True
            break

    return PrefitResult(bank=best_bank, evaluations=tuple(evaluations), stopped_early=stopped_early)


__all__ = [
    "AggregatePrefitBatch",
    "AggregatePrefitDataset",
    "PrefitBatch",
    "PrefitConfig",
    "PrefitDataset",
    "PrefitEvaluation",
    "PrefitObjective",
    "PrefitResult",
    "PrefitSplit",
    "aggregate_prefit_loss",
    "aggregate_routed_moe_nrmse",
    "make_aggregate_prefit_dataset",
    "make_prefit_dataset",
    "prefit_loss",
    "prefit_shared_bank",
    "sample_aggregate_prefit_batch",
    "sample_prefit_batch",
]
