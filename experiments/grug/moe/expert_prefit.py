# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Balanced offline distillation for a functionally matched shared expert bank."""

from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.tree_util import register_dataclass
from levanter.grug.grug_moe import MoEExpertMlp

from experiments.grug.moe.expert_merge import eval_expert

_DEFAULT_NORMALIZATION_EPSILON = 1e-8


class PrefitSplit(StrEnum):
    TRAIN = "train"
    HELDOUT = "heldout"


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
    error_by_source = jax.ops.segment_sum(squared_error, batch.source_indices, num_segments=num_sources)
    count_by_source = jax.ops.segment_sum(jnp.ones_like(squared_error), batch.source_indices, num_segments=num_sources)
    normalized_mse = error_by_source / count_by_source / (batch.target_power_by_source + epsilon)
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
    "PrefitBatch",
    "PrefitConfig",
    "PrefitDataset",
    "PrefitEvaluation",
    "PrefitResult",
    "PrefitSplit",
    "make_prefit_dataset",
    "prefit_loss",
    "prefit_shared_bank",
    "sample_prefit_batch",
]
