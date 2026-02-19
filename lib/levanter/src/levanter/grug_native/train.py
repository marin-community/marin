# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from jax.tree_util import register_dataclass

import levanter.tracker
from levanter.data.mixture import MixtureDataset
from levanter.eval import construct_log_dict
from levanter.grug.model import GrugModelConfig, GrugModelParameters, init_parameters
from levanter.grug.model import loss_fn as grug_loss_fn
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from .checkpoint import (
    maybe_restore_checkpoint,
    save_checkpoint_on_step,
    wait_for_checkpoints,
)
from .config import GrugNativeRunConfig
from .data import build_train_dataset, build_train_loader_for_runtime
from .eval_hooks import build_tagged_evaluator

logger = logging.getLogger(__name__)


@register_dataclass
@dataclass(frozen=True)
class GrugTrainState:
    step: jax.Array
    params: GrugModelParameters
    opt_state: optax.OptState
    training_key: jax.Array
    ema_params: GrugModelParameters


def _make_train_step(
    model_config: GrugModelConfig,
    optimizer: optax.GradientTransformation,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
):
    one = jnp.array(1, dtype=jnp.int32)
    z_loss = z_loss_weight if z_loss_weight > 0 else None

    @jax.jit
    def train_step(state: GrugTrainState, batch):
        def loss_fn(params):
            return grug_loss_fn(
                params,
                batch.tokens,
                batch.loss_weight,
                model_config,
                mask=batch.attn_mask,
                reduction="mean",
                logsumexp_weight=z_loss,
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        if ema_beta is None:
            ema_params = params
        else:
            ema_params = jax.tree_util.tree_map(
                lambda old, new: ema_beta * old + (1.0 - ema_beta) * new, state.ema_params, params
            )

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
        )

        metrics = {
            "train/loss": loss,
        }
        return next_state, metrics

    return train_step


def run_grug_native(config: GrugNativeRunConfig) -> None:
    trainer_runtime = config.trainer.trainer
    trainer_runtime.initialize()
    levanter.tracker.log_configuration(config)

    run_id = trainer_runtime.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    optimizer = config.optimizer.build(trainer_runtime.num_train_steps)
    train_step = _make_train_step(
        config.model,
        optimizer,
        z_loss_weight=config.trainer.z_loss_weight,
        ema_beta=config.trainer.ema_beta,
    )

    seed = trainer_runtime.seed
    data_key, model_key, training_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    with trainer_runtime.use_device_mesh():
        mesh = trainer_runtime.device_mesh
        batch_schedule = trainer_runtime.batch_schedule

        train_dataset: MixtureDataset = build_train_dataset(
            config.data,
            max_seq_len=config.model.max_seq_len,
            batch_schedule=batch_schedule,
            key=data_key,
        )
        train_loader = build_train_loader_for_runtime(
            train_dataset,
            trainer_runtime=trainer_runtime,
            mesh=mesh,
        )

        @jax.jit
        def _init_state(model_rng, train_rng):
            params = init_parameters(config.model, key=model_rng)
            opt_state = optimizer.init(params)
            return GrugTrainState(
                step=jnp.array(0, dtype=jnp.int32),
                params=params,
                opt_state=opt_state,
                training_key=train_rng,
                ema_params=params,
            )

        state = _init_state(model_key, training_key)

        checkpointer = (
            trainer_runtime.checkpointer.create(run_id) if trainer_runtime.checkpointer is not None else None
        )
        checkpoint_path = trainer_runtime.load_checkpoint_path
        if checkpoint_path is None and checkpointer is not None:
            checkpoint_path = trainer_runtime.checkpointer.expanded_path(run_id)
        if checkpoint_path is None:
            if trainer_runtime.load_checkpoint:
                raise FileNotFoundError("load_checkpoint=True but no checkpoint path is configured.")
        elif trainer_runtime.load_checkpoint is not False:
            try:
                state = maybe_restore_checkpoint(
                    state,
                    checkpoint_path=checkpoint_path,
                    axis_mapping=trainer_runtime.parameter_axis_mapping,
                    mesh=mesh,
                    allow_partial=trainer_runtime.allow_partial_checkpoint,
                )
            except FileNotFoundError:
                if trainer_runtime.load_checkpoint is True:
                    raise
                logger.info(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})

        evaluator = build_tagged_evaluator(
            data_config=config.data,
            model_config=config.model,
            max_seq_len=config.model.max_seq_len,
            trainer_runtime=trainer_runtime,
            mesh=mesh,
            max_eval_batches=config.eval.max_eval_batches,
            compute_bpb=config.eval.compute_bpb,
        )

        eval_interval = config.eval.steps_per_eval

        log_every = max(1, config.trainer.log_every)
        last_mixture_stage = -1
        iterator = LoadingTimeTrackerIterator(train_loader.iter_from_step(int(state.step)))

        try:
            while int(state.step) < trainer_runtime.num_train_steps:
                batch = next(iterator)
                step_start = time.perf_counter()
                state, metrics = train_step(state, batch)
                duration = time.perf_counter() - step_start
                step = int(state.step)

                if step % log_every == 0:
                    batch_tokens = int(batch.tokens.shape[0] * batch.tokens.shape[1])
                    inclusive_time = duration + iterator.this_load_time
                    log_data: dict[str, float | int | jax.Array] = {
                        "throughput/step_time": duration,
                        "throughput/loading_time": iterator.this_load_time,
                        "throughput/tokens": batch_tokens,
                        "throughput/tokens_per_second": batch_tokens / max(duration, 1e-9),
                        "throughput/tokens_per_second_with_loading": batch_tokens / max(inclusive_time, 1e-9),
                    }
                    log_data.update(metrics)
                    levanter.tracker.log(log_data, step=step)

                seq_index = batch_schedule.global_data_offset_by_step(step)
                block_id = seq_index // train_dataset.block_size
                stage = train_dataset._get_stage_for_block(block_id)
                if stage != last_mixture_stage:
                    weights = train_dataset.weight_stages[stage][1]
                    mixture_log = {f"mixture/weight/{name}": weight for name, weight in weights.items()}
                    mixture_log["mixture/stage"] = stage
                    levanter.tracker.log(mixture_log, step=step)
                    last_mixture_stage = stage

                if (
                    evaluator is not None
                    and eval_interval is not None
                    and eval_interval > 0
                    and step % eval_interval == 0
                ):
                    if config.eval.eval_current:
                        with levanter.tracker.capture_time() as eval_time:
                            result = evaluator.evaluate(state.params)
                        log_dict = construct_log_dict(evaluator, result, eval_time(), prefix=config.eval.prefix)
                        levanter.tracker.log(log_dict, step=step)
                    if config.eval.eval_ema and config.trainer.ema_beta is not None:
                        with levanter.tracker.capture_time() as eval_time:
                            result = evaluator.evaluate(state.ema_params)
                        ema_prefix = f"{config.eval.prefix}/ema"
                        log_dict = construct_log_dict(evaluator, result, eval_time(), prefix=ema_prefix)
                        levanter.tracker.log(log_dict, step=step)

                save_checkpoint_on_step(checkpointer, state)
        finally:
            save_checkpoint_on_step(checkpointer, state, force=True)
            wait_for_checkpoints(checkpointer)

    try:
        levanter.tracker.current_tracker().finish()
    except Exception:
        logger.exception("Failed to finish tracker cleanly")
