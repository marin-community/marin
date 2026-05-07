# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train loop for adamh-ref: identical to train.py but uses TransformerAdamHRef for init."""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import jmp
import levanter.callbacks as callbacks
import levanter.tracker
import optax
from jaxtyping import PRNGKeyArray
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.eval import cb_tagged_evaluate
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.nanogpt.model import NanoGPTConfig
from experiments.grug.nanogpt.model import TransformerAdamHRef as Transformer
from experiments.grug.nanogpt.train import (
    GrugEvalConfig,  # noqa: F401 — re-exported for launch_adamh_ref
    GrugRunConfig,
    GrugTrainerConfig,  # noqa: F401 — re-exported for launch_adamh_ref
    GrugTrainState,
    _compute_flops,
    _make_train_step,
    build_tagged_evaluator,
    build_train_dataset,
    build_train_loader,
)

logger = logging.getLogger(__name__)


def initial_state(
    model_config: NanoGPTConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
) -> GrugTrainState:
    params = mp.cast_to_param(Transformer.init(model_config, key=key))
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params),
        ema_params=params if ema_beta is not None else None,
    )


def _run_grug_local(config: GrugRunConfig) -> None:
    trainer = config.trainer.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(config)
    run_id = trainer.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    optimizer = config.optimizer.build(trainer.num_train_steps)
    train_step = _make_train_step(
        optimizer, trainer.mp, z_loss_weight=config.trainer.z_loss_weight, ema_beta=config.trainer.ema_beta
    )

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    with trainer.use_device_mesh():
        mesh = trainer.device_mesh
        batch_schedule = trainer.batch_schedule
        train_dataset = build_train_dataset(
            config.data, max_seq_len=config.model.max_seq_len, batch_schedule=batch_schedule, key=data_key
        )
        train_loader = build_train_loader(
            train_dataset, batch_schedule=batch_schedule, mesh=mesh, batch_pspec=config.trainer.train_batch_pspec
        )

        @jax.jit
        def _init_state(model_rng):
            return initial_state(
                config.model, optimizer=optimizer, mp=trainer.mp, key=model_rng, ema_beta=config.trainer.ema_beta
            )

        state = _init_state(model_key)
        checkpointer = trainer.checkpointer.create(run_id)
        state = restore_grug_state_from_checkpoint(
            state,
            checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
            load_checkpoint_setting=trainer.load_checkpoint,
            mesh=mesh,
            allow_partial=trainer.allow_partial_checkpoint,
        )
        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})
        flops_per_example, flops_summary = _compute_flops(model_config=config.model)
        levanter.tracker.log_summary(flops_summary)

        eval_cfg = config.eval
        evaluator = None
        if eval_cfg is not None:
            evaluator = build_tagged_evaluator(
                data_config=config.data, max_seq_len=config.model.max_seq_len, mesh=mesh, eval_cfg=eval_cfg
            )

        log_every = max(1, config.trainer.log_every)
        iterator = LoadingTimeTrackerIterator(train_loader.iter_from_step(int(state.step)))
        state_callbacks = StateCallbackRunner[GrugTrainState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            eval_model_getter=lambda s: s.ema_params if s.ema_params is not None else s.params,
            opt_state_getter=lambda s: s.opt_state,
        )
        state_callbacks.add_hook(
            callbacks.log_performance_stats(config.model.max_seq_len, batch_schedule, flops_per_example), every=log_every
        )
        state_callbacks.add_hook(callbacks.pbar_logger(total=trainer.num_train_steps), every=log_every)
        state_callbacks.add_hook(callbacks.log_step_info(trainer.num_train_steps), every=log_every)
        if evaluator is not None and eval_cfg is not None:
            interval = eval_cfg.steps_per_eval
            if interval is not None and interval > 0 and eval_cfg.eval_current:
                state_callbacks.add_hook(
                    cb_tagged_evaluate(evaluator, prefix=eval_cfg.prefix, eval_current=True, eval_ema=False),
                    every=interval,
                )

        last_loss: float | jax.Array = 0.0
        last_step_duration = 0.0
        try:
            while int(state.step) < trainer.num_train_steps:
                with jax.profiler.TraceAnnotation("load_batch"):
                    batch = next(iterator)
                step_start = time.perf_counter()
                state, metrics = train_step(state, batch)
                step = int(state.step) - 1
                jax.block_until_ready(metrics["train/loss"])
                duration = time.perf_counter() - step_start
                with jax.profiler.TraceAnnotation("callbacks"):
                    state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                    last_loss = metrics["train/loss"]
                    last_step_duration = duration
                    levanter.tracker.log({"throughput/loading_time": iterator.this_load_time}, step=step)
                if checkpointer is not None:
                    checkpointer.on_step(tree=state, step=int(state.step))
        except BaseException:
            logger.exception("Fatal error in nanogpt adamh-ref training loop")
            raise
        else:
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if checkpointer is not None:
                checkpointer.on_step(tree=state, step=int(state.step), force=True)
                checkpointer.wait_until_finished()

    levanter.tracker.current_tracker().finish()


def run_grug(config: GrugRunConfig) -> None:
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")
    dispatch_grug_training_run(
        run_id=trainer.id, config=config, local_entrypoint=_run_grug_local, resources=config.resources
    )
