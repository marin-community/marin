# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import functools
import logging
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jmp
import optax
from fray.v1.cluster.device_flops import device_flops_for_jax_device
from jax.tree_util import register_dataclass

import levanter.callbacks as callbacks
import levanter.tracker
from levanter.data.mixture import MixtureDataset
from levanter.eval import construct_log_dict
from levanter.grug.model import GrugModelConfig, GrugModelParameters, init_parameters
from levanter.grug.model import loss_fn as grug_loss_fn
from levanter.utils.flop_utils import lm_flops_per_token
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
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
):
    one = jnp.array(1, dtype=jnp.int32)
    z_loss = z_loss_weight if z_loss_weight > 0 else None

    @functools.partial(jax.jit, donate_argnums=(0,))
    def train_step(state: GrugTrainState, batch):
        def loss_fn(params):
            compute_params = mp.cast_to_compute(params)
            return grug_loss_fn(
                compute_params,
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
        trainer_runtime.mp,
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
            params = trainer_runtime.mp.cast_to_param(init_parameters(config.model, key=model_rng))
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

        flops_per_token = lm_flops_per_token(
            hidden_dim=config.model.hidden_dim,
            intermediate_dim=config.model.intermediate_dim,
            num_layers=config.model.num_layers,
            num_kv_heads=config.model.num_kv_heads,
            num_heads=config.model.num_heads,
            seq_len=config.model.max_seq_len,
            vocab_size=config.model.vocab_size,
            glu=True,
        )
        flops_per_example = 3 * flops_per_token * config.model.max_seq_len
        device_count = jax.device_count()
        device = jax.devices()[0]
        flops_per_device = device_flops_for_jax_device(device.device_kind)
        theoretical_flops = None if flops_per_device is None else flops_per_device * device_count
        levanter.tracker.log_summary(
            {
                "throughput/device_kind": device.device_kind,
                "throughput/flops_per_example": flops_per_example,
            }
        )
        if flops_per_device is not None and theoretical_flops is not None:
            levanter.tracker.log_summary(
                {
                    "throughput/theoretical_flops_per_device": flops_per_device,
                    "throughput/theoretical_flops": theoretical_flops,
                }
            )

        evaluator = build_tagged_evaluator(
            data_config=config.data,
            model_config=config.model,
            max_seq_len=config.model.max_seq_len,
            trainer_runtime=trainer_runtime,
            mesh=mesh,
            max_eval_batches=config.eval.max_eval_batches,
            compute_bpb=config.eval.compute_bpb,
        )

        profiler_path = trainer_runtime.log_dir / run_id / "profiler"
        profiler_start_step = trainer_runtime.profiler_start_step
        profiler_num_steps = trainer_runtime.profiler_num_steps
        profiler_enabled = trainer_runtime.profiler
        if profiler_enabled and profiler_num_steps + profiler_start_step > trainer_runtime.num_train_steps:
            logger.warning(
                f"Adjusting profiler_total_steps from {profiler_num_steps} to"
                f" {trainer_runtime.num_train_steps - profiler_start_step}"
            )
            profiler_num_steps = trainer_runtime.num_train_steps - profiler_start_step
        if profiler_num_steps <= 0:
            profiler_enabled = False
        profiler_ctx: contextlib.AbstractContextManager[None] | None = None

        eval_interval = config.eval.steps_per_eval

        log_every = max(1, config.trainer.log_every)
        last_mixture_stage = -1
        iterator = LoadingTimeTrackerIterator(train_loader.iter_from_step(int(state.step)))

        try:
            while int(state.step) < trainer_runtime.num_train_steps:
                batch = next(iterator)
                step_start = time.perf_counter()
                state, metrics = train_step(state, batch)
                step = int(state.step) - 1
                seq_index = batch_schedule.global_data_offset_by_step(step)

                if profiler_enabled and profiler_ctx is None and step == profiler_start_step - 1:
                    profiler_ctx = callbacks.profile_ctx(
                        str(profiler_path), create_perfetto_link=trainer_runtime.profiler_perfetto_link
                    )
                    profiler_ctx.__enter__()

                if step % log_every == 0:
                    # JAX dispatch is asynchronous; block only when logging to avoid per-step stalls.
                    jax.block_until_ready(metrics["train/loss"])
                    duration = time.perf_counter() - step_start
                    tokens_per_example = int(batch.tokens.shape[1])
                    this_batch_size = int(batch_schedule.batch_size_at_step(step))
                    batch_tokens = this_batch_size * tokens_per_example
                    total_examples = int(batch_schedule.global_data_offset_by_step(step + 1))
                    total_tokens = tokens_per_example * total_examples
                    inclusive_time = duration + iterator.this_load_time
                    total_gflops = flops_per_example * total_examples / 1e9
                    model_flops_instant = flops_per_example * this_batch_size / max(duration, 1e-9)
                    log_data: dict[str, float | int | jax.Array] = {
                        "throughput/duration": duration,
                        "throughput/loading_time": iterator.this_load_time,
                        "throughput/tokens": batch_tokens,
                        "throughput/total_tokens": total_tokens,
                        "throughput/total_gflops": total_gflops,
                        "throughput/examples_per_second": this_batch_size / max(duration, 1e-9),
                        "throughput/gflops_per_second": model_flops_instant / 1e9,
                        "throughput/tokens_per_second": batch_tokens / max(duration, 1e-9),
                        "throughput/tokens_per_second_with_loading": batch_tokens / max(inclusive_time, 1e-9),
                    }
                    if theoretical_flops is not None:
                        log_data["throughput/mfu"] = model_flops_instant / theoretical_flops * 100.0
                    log_data.update(metrics)
                    levanter.tracker.log(log_data, step=step)

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

                if profiler_enabled and profiler_ctx is not None and step == profiler_start_step + profiler_num_steps - 1:
                    profiler_ctx.__exit__(None, None, None)
                    profiler_ctx = None
        finally:
            if profiler_ctx is not None:
                profiler_ctx.__exit__(None, None, None)
            save_checkpoint_on_step(checkpointer, state, force=True)
            wait_for_checkpoints(checkpointer)

    try:
        levanter.tracker.current_tracker().finish()
    except Exception:
        logger.exception("Failed to finish tracker cleanly")
