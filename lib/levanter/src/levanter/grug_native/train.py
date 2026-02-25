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
from jax.tree_util import register_dataclass

import levanter.callbacks as callbacks
import levanter.tracker
from levanter.callbacks.tensorstore_callbacks import (
    build_tensorstore_metrics_logger,
    tensorstore_metrics_interval_from_env,
)
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.data.mixture import MixtureDataset
from levanter.eval import construct_log_dict
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.model import GrugModelConfig, Transformer
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import estimate_jit_flops, parameter_count
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


def grug_loss_fn(
    model: Transformer,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    cfg: GrugModelConfig,
    *,
    mask: GrugAttentionMask | jax.Array | None = None,
    reduction: str = "mean",
    logsumexp_weight: float | None = None,
) -> jax.Array:
    del cfg
    return model.next_token_loss(
        token_ids,
        loss_weight,
        mask=mask,
        reduction=reduction,
        logsumexp_weight=logsumexp_weight,
    )


@register_dataclass
@dataclass(frozen=True)
class GrugTrainState:
    step: jax.Array
    params: Transformer
    opt_state: optax.OptState
    training_key: jax.Array
    ema_params: Transformer


def _make_train_step(
    model_config: GrugModelConfig,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
    watch_config: WatchConfig | None = None,
):
    one = jnp.array(1, dtype=jnp.int32)
    z_loss = z_loss_weight if z_loss_weight > 0 else None
    if watch_config is not None:
        if isinstance(watch_config.watch_targets, str):
            watch_targets = tuple(t.strip() for t in watch_config.watch_targets.split(","))
        else:
            watch_targets = tuple(watch_config.watch_targets)
    else:
        watch_targets = ()

    @functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("compute_watch",))
    def train_step(state: GrugTrainState, batch, *, compute_watch: bool = False):
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

        watch_stats = None
        if watch_config is not None and compute_watch:
            watch_stats = compute_watch_stats(
                watch_targets=watch_targets,
                include_norms=watch_config.include_norms,
                include_per_parameter_norms=watch_config.include_per_parameter_norms,
                include_histogram=watch_config.include_histograms,
                split_scan_layers=watch_config.split_scan_layers,
                params=state.params,
                grads=grads,
                updates=updates,
                opt_state=state.opt_state,
                model_tree_type=type(state.params),
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
        return next_state, metrics, watch_stats

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
        watch_config=trainer_runtime.watch if trainer_runtime.watch.is_enabled else None,
    )
    watch_config = trainer_runtime.watch

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
            params = trainer_runtime.mp.cast_to_param(Transformer.init(config.model, key=model_rng))
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
        z_loss = config.trainer.z_loss_weight if config.trainer.z_loss_weight > 0 else None
        token_ids_spec = jax.ShapeDtypeStruct((1, config.model.max_seq_len), jnp.int32)
        loss_weight_spec = jax.ShapeDtypeStruct((1, config.model.max_seq_len), jnp.float32)

        def _loss_only(params: Transformer, token_ids: jax.Array, loss_weight: jax.Array) -> jax.Array:
            return grug_loss_fn(
                trainer_runtime.mp.cast_to_compute(params),
                token_ids,
                loss_weight,
                config.model,
                mask=GrugAttentionMask.causal(),
                reduction="mean",
                logsumexp_weight=z_loss,
            )

        try:
            forward_loss_flops_per_example_jax = estimate_jit_flops(
                _loss_only, state.params, token_ids_spec, loss_weight_spec
            )
        except Exception:
            logger.exception(
                "Failed to estimate FLOPs with JAX cost_analysis; continuing with analytic estimate only."
            )
            forward_loss_flops_per_example_jax = None
        flops_per_example_jax = (
            None if forward_loss_flops_per_example_jax is None else 3 * forward_loss_flops_per_example_jax
        )
        flops_summary: dict[str, float] = {
            "throughput/flops_per_token_analytic": flops_per_token,
            "throughput/flops_per_example_analytic": flops_per_example,
        }
        if forward_loss_flops_per_example_jax is not None and flops_per_example_jax is not None:
            flops_summary["throughput_jax/flops_per_example_forward"] = forward_loss_flops_per_example_jax
            flops_summary["throughput_jax/flops_per_example_fwd_bwd_est"] = flops_per_example_jax
            flops_summary["throughput_jax/flops_per_token_forward"] = (
                forward_loss_flops_per_example_jax / config.model.max_seq_len
            )
            flops_summary["throughput_jax/flops_per_token_fwd_bwd_est"] = (
                flops_per_example_jax / config.model.max_seq_len
            )
        levanter.tracker.log_summary(flops_summary)

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

        log_every = 1
        last_mixture_stage = -1
        iterator = LoadingTimeTrackerIterator(train_loader.iter_from_step(int(state.step)))
        tensorstore_metrics_every = tensorstore_metrics_interval_from_env()
        tensorstore_metrics_logger = None
        if tensorstore_metrics_every is not None:
            tensorstore_metrics_logger = build_tensorstore_metrics_logger(tensorstore_metrics_every)
        state_callbacks = StateCallbackRunner[GrugTrainState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            eval_model_getter=lambda s: s.ema_params,
            opt_state_getter=lambda s: s.opt_state,
        )
        state_callbacks.add_hook(
            callbacks.log_performance_stats(config.model.max_seq_len, batch_schedule, flops_per_example),
            every=log_every,
        )
        if flops_per_example_jax is not None:
            state_callbacks.add_hook(
                callbacks.log_performance_stats(
                    config.model.max_seq_len,
                    batch_schedule,
                    flops_per_example_jax,
                    prefix="throughput_jax",
                ),
                every=log_every,
            )
        state_callbacks.add_hook(callbacks.pbar_logger(total=trainer_runtime.num_train_steps), every=log_every)
        state_callbacks.add_hook(callbacks.log_step_info(trainer_runtime.num_train_steps), every=log_every)
        if tensorstore_metrics_logger is not None and tensorstore_metrics_every is not None:
            state_callbacks.add_hook(
                lambda info: tensorstore_metrics_logger(info.step), every=tensorstore_metrics_every
            )

        try:
            while int(state.step) < trainer_runtime.num_train_steps:
                batch = next(iterator)
                step_start = time.perf_counter()
                current_step = int(state.step)
                compute_watch = (
                    watch_config.is_enabled and watch_config.interval > 0 and current_step % watch_config.interval == 0
                )
                state, metrics, watch_stats = train_step(state, batch, compute_watch=compute_watch)
                step = int(state.step) - 1
                seq_index = batch_schedule.global_data_offset_by_step(step)

                if profiler_enabled and profiler_ctx is None and step == profiler_start_step - 1:
                    profiler_ctx = callbacks.profile_ctx(
                        str(profiler_path), create_perfetto_link=trainer_runtime.profiler_perfetto_link
                    )
                    profiler_ctx.__enter__()

                # JAX dispatch is asynchronous; block every step so step timing is always accurate.
                jax.block_until_ready(metrics["train/loss"])
                duration = time.perf_counter() - step_start
                hook_start = time.perf_counter()
                state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=step)
                levanter.tracker.log({"throughput/loading_time": iterator.this_load_time}, step=step)

                if watch_stats is not None:
                    levanter.tracker.log(watch_stats, step=step)

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

                if (
                    profiler_enabled
                    and profiler_ctx is not None
                    and step == profiler_start_step + profiler_num_steps - 1
                ):
                    profiler_ctx.__exit__(None, None, None)
                    profiler_ctx = None
        finally:
            if profiler_ctx is not None:
                profiler_ctx.__exit__(None, None, None)
            save_checkpoint_on_step(checkpointer, state, force=True)
            wait_for_checkpoints(checkpointer)

    levanter.tracker.current_tracker().finish()
