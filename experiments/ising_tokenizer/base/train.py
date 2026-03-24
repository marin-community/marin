# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

import equinox as eqx
import fsspec
import haliax.partitioning
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from experiments.ising_tokenizer.base.data import (
    BklIsingConfig,
    TemperatureConditionedDataset,
    TrajectoryTokenizerConfig,
    decode_event_positions,
    decode_initial_spins,
    decode_wait_times,
)
from experiments.ising_tokenizer.base.model import IsingLmConfig, TemperatureConditionedTransformer

DATA_AXIS_NAME = "data"
T = TypeVar("T")


@dataclass(frozen=True)
class IsingTrainerConfig:
    """Local training knobs for the Ising smoke run."""

    num_train_steps: int = 48
    batch_size: int = 8
    learning_rate: float = 2e-3
    weight_decay: float = 0.0
    seed: int = 0
    eval_every: int = 8


@dataclass(frozen=True)
class IsingRolloutConfig:
    """Small rollout-eval knobs for trajectory-level checks."""

    num_examples_per_temperature: int = 2
    sample_seed: int = 0


@dataclass(frozen=True)
class IsingWandbConfig:
    """Optional W&B logging for local or TPU-backed Ising runs."""

    project: str | None = None
    entity: str | None = "marin-community"
    group: str | None = None
    name: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class IsingRunConfig:
    """Top-level local Ising training config."""

    model: IsingLmConfig
    trainer: IsingTrainerConfig
    rollout: IsingRolloutConfig = IsingRolloutConfig()
    wandb: IsingWandbConfig = IsingWandbConfig()


def build_loss_weights(dataset: TemperatureConditionedDataset) -> np.ndarray:
    """Mask deterministic initial-state position tokens and the terminal EOS token."""

    loss_weight = np.ones(dataset.tokens.shape, dtype=np.float32)
    loss_weight[:, : dataset.initial_state_token_count : 2] = 0.0
    loss_weight[:, dataset.valid_token_count - 1 :] = 0.0
    return loss_weight


def _iterate_batches(
    dataset: TemperatureConditionedDataset,
    loss_weight: np.ndarray,
    *,
    batch_size: int,
    mesh: Mesh,
    rng: np.random.Generator,
):
    indices = np.arange(dataset.tokens.shape[0], dtype=np.int32)
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield (
            _shard_batch_array(jnp.asarray(dataset.tokens[batch_indices]), mesh),
            _shard_batch_array(jnp.asarray(loss_weight[batch_indices]), mesh),
            _shard_batch_array(jnp.asarray(dataset.normalized_temperatures[batch_indices]), mesh),
        )


def _replicated_sharding(mesh: Mesh, ndim: int) -> NamedSharding:
    return NamedSharding(mesh, P(*([None] * ndim)))


def _batch_sharding(mesh: Mesh, shape: tuple[int, ...]) -> NamedSharding:
    if len(shape) == 0 or shape[0] % mesh.size != 0:
        return _replicated_sharding(mesh, len(shape))
    return NamedSharding(mesh, P(DATA_AXIS_NAME, *([None] * (len(shape) - 1))))


def _shard_batch_array(array: jax.Array, mesh: Mesh) -> jax.Array:
    return jax.device_put(array, _batch_sharding(mesh, array.shape))


def _replicate_array_tree(tree: T, mesh: Mesh) -> T:
    def _replicate_leaf(leaf):
        if isinstance(leaf, jax.Array):
            return jax.device_put(leaf, _replicated_sharding(mesh, leaf.ndim))
        return leaf

    return jax.tree.map(_replicate_leaf, tree)


def _create_data_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices()), axis_names=(DATA_AXIS_NAME,))


@eqx.filter_jit
def _train_step(
    model: TemperatureConditionedTransformer,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    tokens: jax.Array,
    loss_weight: jax.Array,
    temperature: jax.Array,
) -> tuple[TemperatureConditionedTransformer, optax.OptState, jax.Array]:
    def loss_fn(current_model: TemperatureConditionedTransformer) -> jax.Array:
        return current_model.next_token_loss(
            tokens,
            loss_weight,
            temperature=temperature,
            reduction="mean",
        )

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    next_model = eqx.apply_updates(model, updates)
    return next_model, opt_state, loss


@eqx.filter_jit
def _batch_loss(
    model: TemperatureConditionedTransformer,
    tokens: jax.Array,
    loss_weight: jax.Array,
    temperature: jax.Array,
) -> jax.Array:
    return model.next_token_loss(tokens, loss_weight, temperature=temperature, reduction="mean")


@eqx.filter_jit
def _next_token_logits_at_position(
    model: TemperatureConditionedTransformer,
    tokens: jax.Array,
    temperature: jax.Array,
    position: jax.Array,
) -> jax.Array:
    logits = model.logits(tokens, temperature)
    return jax.lax.dynamic_index_in_dim(logits, position, axis=1, keepdims=False)


def _flip_is_on_boundary(spins: np.ndarray, position: int) -> bool:
    lattice_size = spins.shape[0]
    row, col = divmod(position, lattice_size)
    spin = spins[row, col]
    return bool(
        spins[(row - 1) % lattice_size, col] != spin
        or spins[(row + 1) % lattice_size, col] != spin
        or spins[row, (col - 1) % lattice_size] != spin
        or spins[row, (col + 1) % lattice_size] != spin
    )


def _rollout_observables(
    tokens: np.ndarray,
    *,
    dynamics_config: BklIsingConfig,
    tokenizer_config: TrajectoryTokenizerConfig,
) -> dict[str, float]:
    initial_spins = decode_initial_spins(tokens, dynamics_config=dynamics_config)
    event_positions = decode_event_positions(tokens, dynamics_config=dynamics_config)
    wait_times = decode_wait_times(
        tokens,
        dynamics_config=dynamics_config,
        tokenizer_config=tokenizer_config,
    )

    spins = initial_spins.copy()
    boundary_count = 0
    for position in event_positions:
        boundary_count += int(_flip_is_on_boundary(spins, int(position)))
        row, col = divmod(int(position), dynamics_config.lattice_size)
        spins[row, col] *= -1

    return {
        "mean_wait_time": float(wait_times.mean()),
        "boundary_flip_fraction": float(boundary_count / max(1, len(event_positions))),
        "final_abs_magnetization": float(abs(spins.mean())),
    }


def _aggregate_observables(observables: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([metrics[key] for metrics in observables])) for key in observables[0]}


def _sample_rollout_tokens(
    model: TemperatureConditionedTransformer,
    reference_tokens: np.ndarray,
    normalized_temperatures: np.ndarray,
    *,
    rollout_config: IsingRolloutConfig,
    dynamics_config: BklIsingConfig,
    tokenizer_config: TrajectoryTokenizerConfig,
    valid_token_count: int,
    mesh: Mesh,
) -> np.ndarray:
    prompt_len = dynamics_config.initial_state_token_count
    eos_token_id = tokenizer_config.eos_token_id(dynamics_config.lattice_size)
    pad_token_id = tokenizer_config.pad_token_id(dynamics_config.lattice_size)
    vocab_size = tokenizer_config.vocab_size(dynamics_config.lattice_size)

    position_mask = np.zeros(vocab_size, dtype=np.bool_)
    position_mask[: dynamics_config.num_sites] = True
    dt_mask = np.zeros(vocab_size, dtype=np.bool_)
    dt_mask[tokenizer_config.dt_token_offset(dynamics_config.lattice_size) : eos_token_id] = True

    sampled_tokens = np.full(reference_tokens.shape, pad_token_id, dtype=np.int32)
    sampled_tokens[:, :prompt_len] = reference_tokens[:, :prompt_len]
    sharded_temperatures = _shard_batch_array(jnp.asarray(normalized_temperatures), mesh)
    key = jax.random.PRNGKey(rollout_config.sample_seed)

    for cursor in range(prompt_len, valid_token_count - 1):
        sharded_tokens = _shard_batch_array(jnp.asarray(sampled_tokens), mesh)
        next_logits = _next_token_logits_at_position(
            model,
            sharded_tokens,
            sharded_temperatures,
            jnp.asarray(cursor - 1, dtype=jnp.int32),
        )
        allowed = position_mask if (cursor - prompt_len) % 2 == 0 else dt_mask
        masked_logits = jnp.where(
            jnp.asarray(allowed)[None, :],
            next_logits,
            jnp.full_like(next_logits, -1e30),
        )
        key, step_key = jax.random.split(key)
        sampled_tokens[:, cursor] = np.asarray(
            jax.device_get(jax.random.categorical(step_key, masked_logits, axis=-1)),
            dtype=np.int32,
        )

    sampled_tokens[:, valid_token_count - 1] = eos_token_id
    return sampled_tokens


def _evaluate_rollouts(
    model: TemperatureConditionedTransformer,
    *,
    validation_dataset: TemperatureConditionedDataset,
    critical_probe_dataset: TemperatureConditionedDataset | None,
    rollout_config: IsingRolloutConfig,
    dynamics_config: BklIsingConfig,
    tokenizer_config: TrajectoryTokenizerConfig,
    mesh: Mesh,
) -> dict[str, object] | None:
    if rollout_config.num_examples_per_temperature <= 0:
        return None

    datasets = [validation_dataset]
    if critical_probe_dataset is not None:
        datasets.append(critical_probe_dataset)

    selected_tokens: list[np.ndarray] = []
    selected_normalized_temperatures: list[np.ndarray] = []
    selected_temperatures: list[np.ndarray] = []

    for dataset in datasets:
        for temperature in sorted({float(value) for value in dataset.temperatures.tolist()}):
            indices = np.flatnonzero(np.isclose(dataset.temperatures, temperature))[
                : rollout_config.num_examples_per_temperature
            ]
            if indices.size == 0:
                continue
            selected_tokens.append(dataset.tokens[indices])
            selected_normalized_temperatures.append(dataset.normalized_temperatures[indices])
            selected_temperatures.append(dataset.temperatures[indices])

    if not selected_tokens:
        return None

    reference_tokens = np.concatenate(selected_tokens, axis=0)
    normalized_temperatures = np.concatenate(selected_normalized_temperatures, axis=0)
    temperatures = np.concatenate(selected_temperatures, axis=0)
    sampled_tokens = _sample_rollout_tokens(
        model,
        reference_tokens,
        normalized_temperatures,
        rollout_config=rollout_config,
        dynamics_config=dynamics_config,
        tokenizer_config=tokenizer_config,
        valid_token_count=validation_dataset.valid_token_count,
        mesh=mesh,
    )

    metrics: dict[str, object] = {
        "num_examples_per_temperature": rollout_config.num_examples_per_temperature,
        "temperatures": {},
    }
    for temperature in sorted({float(value) for value in temperatures.tolist()}):
        mask = np.isclose(temperatures, temperature)
        reference_observables = [
            _rollout_observables(
                tokens,
                dynamics_config=dynamics_config,
                tokenizer_config=tokenizer_config,
            )
            for tokens in reference_tokens[mask]
        ]
        sampled_observables = [
            _rollout_observables(
                tokens,
                dynamics_config=dynamics_config,
                tokenizer_config=tokenizer_config,
            )
            for tokens in sampled_tokens[mask]
        ]
        reference_summary = _aggregate_observables(reference_observables)
        sampled_summary = _aggregate_observables(sampled_observables)
        metrics["temperatures"][f"T{temperature:.3f}"] = {
            "num_examples": int(mask.sum()),
            "reference_mean_wait_time": reference_summary["mean_wait_time"],
            "sampled_mean_wait_time": sampled_summary["mean_wait_time"],
            "wait_time_ratio": float(sampled_summary["mean_wait_time"] / reference_summary["mean_wait_time"]),
            "reference_boundary_flip_fraction": reference_summary["boundary_flip_fraction"],
            "sampled_boundary_flip_fraction": sampled_summary["boundary_flip_fraction"],
            "boundary_flip_fraction_delta": float(
                sampled_summary["boundary_flip_fraction"] - reference_summary["boundary_flip_fraction"]
            ),
            "reference_final_abs_magnetization": reference_summary["final_abs_magnetization"],
            "sampled_final_abs_magnetization": sampled_summary["final_abs_magnetization"],
            "final_abs_magnetization_delta": float(
                sampled_summary["final_abs_magnetization"] - reference_summary["final_abs_magnetization"]
            ),
        }
    return metrics


def _evaluate_dataset(
    model: TemperatureConditionedTransformer,
    dataset: TemperatureConditionedDataset,
    loss_weight: np.ndarray,
    *,
    batch_size: int,
    mesh: Mesh,
) -> float:
    losses: list[float] = []
    for start in range(0, dataset.tokens.shape[0], batch_size):
        batch_slice = slice(start, start + batch_size)
        loss = _batch_loss(
            model,
            _shard_batch_array(jnp.asarray(dataset.tokens[batch_slice]), mesh),
            _shard_batch_array(jnp.asarray(loss_weight[batch_slice]), mesh),
            _shard_batch_array(jnp.asarray(dataset.normalized_temperatures[batch_slice]), mesh),
        )
        losses.append(float(loss))
    return float(np.mean(losses))


def _evaluate_by_temperature(
    model: TemperatureConditionedTransformer,
    dataset: TemperatureConditionedDataset,
    loss_weight: np.ndarray,
    *,
    batch_size: int,
    mesh: Mesh,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for temperature in sorted({float(value) for value in dataset.temperatures.tolist()}):
        mask = np.isclose(dataset.temperatures, temperature)
        subset = TemperatureConditionedDataset(
            name=f"{dataset.name}@{temperature:.3f}",
            tokens=dataset.tokens[mask],
            temperatures=dataset.temperatures[mask],
            normalized_temperatures=dataset.normalized_temperatures[mask],
            initial_abs_magnetization=dataset.initial_abs_magnetization[mask],
            mean_wait_time=dataset.mean_wait_time[mask],
            lattice_size=dataset.lattice_size,
            num_events=dataset.num_events,
            seq_len=dataset.seq_len,
            vocab_size=dataset.vocab_size,
            initial_state_token_count=dataset.initial_state_token_count,
            valid_token_count=dataset.valid_token_count,
        )
        subset_loss_weight = loss_weight[mask]
        metrics[f"{dataset.name}/loss_T{temperature:.3f}"] = _evaluate_dataset(
            model,
            subset,
            subset_loss_weight,
            batch_size=batch_size,
            mesh=mesh,
        )
    return metrics


def run_local_ising_experiment(
    config: IsingRunConfig,
    *,
    dynamics_config: BklIsingConfig,
    tokenizer_config: TrajectoryTokenizerConfig,
    train_dataset: TemperatureConditionedDataset,
    validation_dataset: TemperatureConditionedDataset,
    critical_probe_dataset: TemperatureConditionedDataset | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train and evaluate the local Ising smoke run."""

    wandb_run = None
    if config.wandb.project is not None:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.wandb.name,
            tags=list(config.wandb.tags),
            config={
                "model": asdict(config.model),
                "trainer": asdict(config.trainer),
                "rollout": asdict(config.rollout),
            },
        )

    train_loss_weight = build_loss_weights(train_dataset)
    validation_loss_weight = build_loss_weights(validation_dataset)
    critical_loss_weight = build_loss_weights(critical_probe_dataset) if critical_probe_dataset is not None else None

    rng = np.random.default_rng(config.trainer.seed)
    optimizer = optax.adamw(
        learning_rate=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    mesh = _create_data_mesh()
    with haliax.partitioning.set_mesh(mesh):
        model_key = jax.random.PRNGKey(config.trainer.seed)
        model = _replicate_array_tree(TemperatureConditionedTransformer.init(config.model, key=model_key), mesh)
        opt_state = _replicate_array_tree(optimizer.init(eqx.filter(model, eqx.is_array)), mesh)

        initial_train_loss = _evaluate_dataset(
            model,
            train_dataset,
            train_loss_weight,
            batch_size=config.trainer.batch_size,
            mesh=mesh,
        )
        initial_validation_loss = _evaluate_dataset(
            model,
            validation_dataset,
            validation_loss_weight,
            batch_size=config.trainer.batch_size,
            mesh=mesh,
        )

        train_batches = _iterate_batches(
            train_dataset,
            train_loss_weight,
            batch_size=config.trainer.batch_size,
            mesh=mesh,
            rng=rng,
        )
        for step in range(config.trainer.num_train_steps):
            try:
                tokens, loss_weight, temperature = next(train_batches)
            except StopIteration:
                train_batches = _iterate_batches(
                    train_dataset,
                    train_loss_weight,
                    batch_size=config.trainer.batch_size,
                    mesh=mesh,
                    rng=rng,
                )
                tokens, loss_weight, temperature = next(train_batches)

            model, opt_state, train_loss = _train_step(model, opt_state, optimizer, tokens, loss_weight, temperature)

            if (step + 1) % config.trainer.eval_every == 0 or step + 1 == config.trainer.num_train_steps:
                record: dict[str, float | int] = {
                    "step": step + 1,
                    "train_loss": float(train_loss),
                    "validation_loss": _evaluate_dataset(
                        model,
                        validation_dataset,
                        validation_loss_weight,
                        batch_size=config.trainer.batch_size,
                        mesh=mesh,
                    ),
                }
                if critical_probe_dataset is not None and critical_loss_weight is not None:
                    record["critical_probe_loss"] = _evaluate_dataset(
                        model,
                        critical_probe_dataset,
                        critical_loss_weight,
                        batch_size=config.trainer.batch_size,
                        mesh=mesh,
                    )
                history.append(record)

        validation_by_temperature = _evaluate_by_temperature(
            model,
            validation_dataset,
            validation_loss_weight,
            batch_size=config.trainer.batch_size,
            mesh=mesh,
        )
        final_critical_probe_loss = None
        critical_probe_by_temperature = None
        if critical_probe_dataset is not None and critical_loss_weight is not None:
            final_critical_probe_loss = _evaluate_dataset(
                model,
                critical_probe_dataset,
                critical_loss_weight,
                batch_size=config.trainer.batch_size,
                mesh=mesh,
            )
            critical_probe_by_temperature = _evaluate_by_temperature(
                model,
                critical_probe_dataset,
                critical_loss_weight,
                batch_size=config.trainer.batch_size,
                mesh=mesh,
            )
        rollout_eval = _evaluate_rollouts(
            model,
            validation_dataset=validation_dataset,
            critical_probe_dataset=critical_probe_dataset,
            rollout_config=config.rollout,
            dynamics_config=dynamics_config,
            tokenizer_config=tokenizer_config,
            mesh=mesh,
        )

    summary: dict[str, object] = {
        "config": {
            "model": asdict(config.model),
            "trainer": asdict(config.trainer),
            "rollout": asdict(config.rollout),
        },
        "train_dataset": train_dataset.summary(),
        "validation_dataset": validation_dataset.summary(),
        "initial_train_loss": initial_train_loss,
        "initial_validation_loss": initial_validation_loss,
        "final_train_loss": float(history[-1]["train_loss"]) if history else initial_train_loss,
        "final_validation_loss": float(history[-1]["validation_loss"]) if history else initial_validation_loss,
        "validation_by_temperature": validation_by_temperature,
        "history": history,
    }
    if rollout_eval is not None:
        summary["rollout_eval"] = rollout_eval
    if critical_probe_dataset is not None and critical_loss_weight is not None:
        summary["critical_probe_dataset"] = critical_probe_dataset.summary()
        summary["final_critical_probe_loss"] = final_critical_probe_loss
        summary["critical_probe_by_temperature"] = critical_probe_by_temperature

    if output_dir is not None:
        output_dir_str = str(output_dir)
        if "://" in output_dir_str:
            with fsspec.open(f"{output_dir_str.rstrip('/')}/metrics.json", "w") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        else:
            output_path = Path(output_dir_str)
            output_path.mkdir(parents=True, exist_ok=True)
            with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)

    if wandb_run is not None:
        if history:
            for record in history:
                step = int(record["step"])
                wandb_run.log(
                    {
                        "train/loss": float(record["train_loss"]),
                        "validation/loss": float(record["validation_loss"]),
                        **(
                            {"critical_probe/loss": float(record["critical_probe_loss"])}
                            if "critical_probe_loss" in record
                            else {}
                        ),
                    },
                    step=step,
                )
        wandb_run.summary.update(
            {
                "initial_train_loss": initial_train_loss,
                "initial_validation_loss": initial_validation_loss,
                "final_train_loss": summary["final_train_loss"],
                "final_validation_loss": summary["final_validation_loss"],
                **(
                    {"final_critical_probe_loss": summary["final_critical_probe_loss"]}
                    if "final_critical_probe_loss" in summary
                    else {}
                ),
            }
        )
        wandb_run.summary["train_dataset"] = train_dataset.summary()
        wandb_run.summary["validation_dataset"] = validation_dataset.summary()
        if critical_probe_dataset is not None:
            wandb_run.summary["critical_probe_dataset"] = critical_probe_dataset.summary()
        if rollout_eval is not None:
            wandb_run.summary["rollout_eval"] = rollout_eval
        wandb_run.finish()

    return summary


__all__ = [
    "IsingRolloutConfig",
    "IsingRunConfig",
    "IsingTrainerConfig",
    "IsingWandbConfig",
    "build_loss_weights",
    "run_local_ising_experiment",
]
