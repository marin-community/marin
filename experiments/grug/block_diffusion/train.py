# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Block diffusion grug training loop (learning experiment).

This is a small fork of `experiments/grug/base/train.py` that adds an RNG to the
train state so the objective can sample per-step corruption.

TODO(Learning): once you have the objective working, consider whether you want:
- deterministic corruption keyed only by (step, batch index)
- per-device RNG via fold_in(axis_index)
- corruption in the data pipeline instead of in the model loss
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jmp
import optax
import ray
from jax.tree_util import register_dataclass

import levanter.callbacks as callbacks
import levanter.tracker
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.callbacks.tensorstore_callbacks import (
    build_tensorstore_metrics_logger,
    tensorstore_metrics_interval_from_env,
)
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.checkpoint import load_checkpoint
from levanter.data.text import LmDataConfig
from levanter.eval import cb_tagged_evaluate
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.grug.base import train as base_train
from experiments.grug.base.model import GrugModelConfig, Transformer
from experiments.grug.block_diffusion.objective import (
    BlockDiffusionObjectiveConfig,
    BlockDiffusionTransformer,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrugBlockDiffusionRunConfig:
    model: GrugModelConfig
    objective: BlockDiffusionObjectiveConfig
    data: LmDataConfig
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: base_train.GrugTrainerConfig = field(default_factory=base_train.GrugTrainerConfig)
    eval: base_train.GrugEvalConfig | None = field(default_factory=base_train.GrugEvalConfig)


@register_dataclass
@dataclass(frozen=True)
class GrugBlockDiffusionTrainState:
    step: jax.Array
    rng: jax.Array
    params: BlockDiffusionTransformer
    opt_state: optax.OptState
    ema_params: BlockDiffusionTransformer


def _make_train_step(
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
    def train_step(state: GrugBlockDiffusionTrainState, batch, *, compute_watch: bool = False):
        rng, step_key = jax.random.split(state.rng)

        def loss_fn(params: BlockDiffusionTransformer):
            compute_params = mp.cast_to_compute(params)
            return compute_params.compute_next_token_loss(
                batch.tokens,
                batch.loss_weight,
                mask=batch.attn_mask,
                reduction="mean",
                logsumexp_weight=z_loss,
                key=step_key,
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        if ema_beta is None:
            ema_params = params
        else:
            ema_params = jax.tree_util.tree_map(
                lambda old, new: ema_beta * old + (1.0 - ema_beta) * new,
                state.ema_params,
                params,
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
            rng=rng,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
        )

        return next_state, {"train/loss": loss}, watch_stats

    return train_step


def run_grug_block_diffusion(config: GrugBlockDiffusionRunConfig) -> None:
    trainer = config.trainer.trainer
    if ray.is_initialized():
        trainer = dataclasses.replace(
            trainer,
            ray=dataclasses.replace(trainer.ray, auto_start_cluster=False, start_workers=False),
        )
    trainer.initialize()
    levanter.tracker.log_configuration(config)

    run_id = trainer.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    optimizer = config.optimizer.build(trainer.num_train_steps)
    train_step = _make_train_step(
        optimizer,
        trainer.mp,
        z_loss_weight=config.trainer.z_loss_weight,
        ema_beta=config.trainer.ema_beta,
        watch_config=trainer.watch if trainer.watch.is_enabled else None,
    )
    watch_config = trainer.watch

    data_key, model_key, diffusion_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 3)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    with trainer.use_device_mesh():
        mesh = trainer.device_mesh
        batch_schedule = trainer.batch_schedule

        train_dataset = base_train.build_train_dataset(
            config.data,
            max_seq_len=config.model.max_seq_len,
            batch_schedule=batch_schedule,
            key=data_key,
        )
        train_loader = base_train.build_train_loader(
            train_dataset,
            batch_schedule=batch_schedule,
            mesh=mesh,
            batch_pspec=config.trainer.train_batch_pspec,
        )

        @jax.jit
        def _init_state(model_rng, diffusion_rng):
            base = Transformer.init(config.model, key=model_rng)
            params = BlockDiffusionTransformer(base=base, obj_cfg=config.objective)
            params = trainer.mp.cast_to_param(params)
            return GrugBlockDiffusionTrainState(
                step=jnp.array(0, dtype=jnp.int32),
                rng=diffusion_rng,
                params=params,
                opt_state=optimizer.init(params),
                ema_params=params,
            )

        state = _init_state(model_key, diffusion_key)

        checkpointer = trainer.checkpointer.create(run_id)
        checkpoint_path = trainer.load_checkpoint_path
        if checkpoint_path is None and checkpointer is not None:
            checkpoint_path = trainer.checkpointer.expanded_path(run_id)
        if checkpoint_path is None:
            if trainer.load_checkpoint:
                raise FileNotFoundError("load_checkpoint=True but no checkpoint path is configured.")
        elif trainer.load_checkpoint is not False:
            try:
                state = load_checkpoint(
                    state,
                    checkpoint_path,
                    discover_latest=True,
                    axis_mapping=None,
                    mesh=mesh,
                    allow_partial=trainer.allow_partial_checkpoint,
                )
            except FileNotFoundError:
                if trainer.load_checkpoint is True:
                    raise
                logger.info(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})

        flops_per_example, flops_summary = base_train._compute_flops(model_config=config.model)
        levanter.tracker.log_summary(flops_summary)

        eval_cfg = config.eval
        evaluator = None
        if eval_cfg is not None:
            evaluator = base_train.build_tagged_evaluator(
                data_config=config.data,
                max_seq_len=config.model.max_seq_len,
                mesh=mesh,
                eval_cfg=eval_cfg,
            )

        profiler_cfg = trainer.profiler
        profiler_num_steps = profiler_cfg.resolve_num_profile_steps(num_train_steps=trainer.num_train_steps)
        profiler_enabled = profiler_cfg.is_enabled and profiler_num_steps > 0

        log_every = max(1, config.trainer.log_every)
        iterator = LoadingTimeTrackerIterator(train_loader.iter_from_step(int(state.step)))

        tensorstore_metrics_every = tensorstore_metrics_interval_from_env()
        tensorstore_metrics_logger = None
        if tensorstore_metrics_every is not None:
            tensorstore_metrics_logger = build_tensorstore_metrics_logger(tensorstore_metrics_every)

        state_callbacks = StateCallbackRunner[GrugBlockDiffusionTrainState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            eval_model_getter=lambda s: s.ema_params,
            opt_state_getter=lambda s: s.opt_state,
        )
        state_callbacks.add_hook(
            callbacks.log_performance_stats(config.model.max_seq_len, batch_schedule, flops_per_example),
            every=log_every,
        )
        state_callbacks.add_hook(callbacks.pbar_logger(total=trainer.num_train_steps), every=log_every)
        state_callbacks.add_hook(callbacks.log_step_info(trainer.num_train_steps), every=log_every)
        if profiler_enabled:
            state_callbacks.add_hook(
                callbacks.profile(
                    str(trainer.log_dir / run_id / "profiler"),
                    profiler_cfg.start_step,
                    profiler_num_steps,
                    profiler_cfg.perfetto_link,
                ),
                every=1,
            )
        state_callbacks.add_hook(base_train._make_mixture_stage_callback(train_dataset, batch_schedule), every=1)
        if tensorstore_metrics_logger is not None and tensorstore_metrics_every is not None:
            state_callbacks.add_hook(
                lambda info: tensorstore_metrics_logger(info.step),
                every=tensorstore_metrics_every,
            )
        if evaluator is not None and eval_cfg is not None:
            interval = eval_cfg.steps_per_eval
            eval_ema = eval_cfg.eval_ema and config.trainer.ema_beta is not None
            if interval is not None and interval > 0 and (eval_cfg.eval_current or eval_ema):
                state_callbacks.add_hook(
                    cb_tagged_evaluate(
                        evaluator,
                        prefix=eval_cfg.prefix,
                        eval_current=eval_cfg.eval_current,
                        eval_ema=eval_ema,
                    ),
                    every=interval,
                )

        last_loss: float | jax.Array = 0.0
        last_step_duration = 0.0

        try:
            while int(state.step) < trainer.num_train_steps:
                batch = next(iterator)
                step_start = time.perf_counter()
                current_step = int(state.step)
                compute_watch = (
                    watch_config.is_enabled and watch_config.interval > 0 and current_step % watch_config.interval == 0
                )
                state, metrics, watch_stats = train_step(state, batch, compute_watch=compute_watch)
                step = int(state.step) - 1

                jax.block_until_ready(metrics["train/loss"])
                duration = time.perf_counter() - step_start
                hook_start = time.perf_counter()
                state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                last_loss = metrics["train/loss"]
                last_step_duration = duration
                levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=step)
                levanter.tracker.log({"throughput/loading_time": iterator.this_load_time}, step=step)

                if watch_stats is not None:
                    levanter.tracker.log(watch_stats, step=step)

                if checkpointer is not None:
                    checkpointer.on_step(tree={"train_state": state}, step=int(state.step))
        finally:
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if checkpointer is not None:
                checkpointer.on_step(tree={"train_state": state}, step=int(state.step), force=True)
                checkpointer.wait_until_finished()

    levanter.tracker.current_tracker().finish()


__all__ = [
    "GrugBlockDiffusionRunConfig",
    "GrugBlockDiffusionTrainState",
    "run_grug_block_diffusion",
]
