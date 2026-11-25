# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June 67B-A2B training loop for the AMD copy of the model.

A copy of ``experiments/june_tpu_67b_a2b/moe/train.py`` on ``experiments.amd.june_67b_a2b.model.Transformer``. It
takes June's run, trainer and eval configs and reuses June's data, FLOP, checkpoint-restore and dispatch helpers.
What differs is the loop: it dispatches step N+1 before it fetches, checks and logs step N (``_train_loop``), and the
step computes the loss's finiteness itself, so the host fetches each step's metrics in one transfer.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import levanter.callbacks as callbacks
import levanter.tracker
import optax
from haliax.partitioning import set_mesh
from jaxtyping import PRNGKeyArray
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.checkpoint import Checkpointer
from levanter.eval import cb_tagged_evaluate
from levanter.grug.sharding import compact_grug_mesh
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.amd.june_67b_a2b.model import Transformer
from experiments.june_tpu_67b_a2b.checkpointing import restore_grug_state_from_checkpoint
from experiments.june_tpu_67b_a2b.dispatch import dispatch_grug_training_run
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.train import (
    _NUMERIC_DIAGNOSTIC_KEYS,
    GrugRunConfig,
    GrugTrainState,
    _compute_flops,
    _make_mixture_stage_callback,
    _tree_all_finite,
    build_tagged_evaluator,
    build_train_dataset,
    build_train_loader,
    exit_if_hung,
    init_weights_only_from_checkpoint,
)

logger = logging.getLogger(__name__)
_LOSS_FINITE_KEY = "train/loss_finite"
# The hyperparameters (learning rate, ...) of the optimizer state a step returns, which the step-info hook logs.
_OPTIMIZER_HYPERPARAMS_KEY = "optimizer_hyperparams"


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array) -> Transformer:
    """Set router biases from QB betas (computed on previous step)."""
    new_biases = -qb_betas
    new_biases = new_biases - jnp.mean(new_biases, axis=-1, keepdims=True)
    return eqx.tree_at(lambda t: t.stacked_blocks.stacked.mlp.router_bias, model, new_biases)


def initial_state(
    model_config: GrugModelConfig,
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
        pending_qb_betas=jnp.zeros((model_config.num_layers, model_config.num_experts)),
    )


def _optimizer_hyperparams(opt_state: optax.OptState) -> dict[str, jax.Array]:
    """The hyperparameters ``log_optimizer_hyperparams`` logs from ``opt_state``: none unless the optimizer injects
    them (``optax.inject_hyperparams``)."""
    if isinstance(opt_state, optax.MultiStepsState):
        opt_state = opt_state.inner_opt_state
    return dict(getattr(opt_state, "hyperparams", {}))


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
    watch_config: WatchConfig | None = None,
    diagnose_numerics: bool = False,
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
        # Apply pending QB betas to router biases inside JIT (avoids eager
        # host-side kernel launches that can cause SPMD sync issues).
        qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)
        qb_ema_params = None if ema_beta is None else _apply_qb_betas(state.ema_params, state.pending_qb_betas)

        def loss_fn(params):
            compute_params = mp.cast_to_compute(params)
            return compute_params.next_token_loss(
                batch.tokens,
                batch.loss_weight,
                mask=batch.attn_mask,
                reduction="mean",
                logsumexp_weight=z_loss,
                return_router_metrics=True,
            )

        (loss, summarized_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(qb_params)
        # The loss's finiteness is checked on the host from this flag, fetched with the other metrics.
        metrics = {"train/loss": loss, _LOSS_FINITE_KEY: jnp.isfinite(loss), **summarized_metrics}
        updates, opt_state = optimizer.update(grads, state.opt_state, qb_params)
        metrics[_OPTIMIZER_HYPERPARAMS_KEY] = _optimizer_hyperparams(opt_state)
        params = optax.apply_updates(qb_params, updates)
        if diagnose_numerics:
            metrics.update(
                {
                    "train/supervised_tokens": jnp.sum(batch.loss_weight),
                    "train/grads_finite": _tree_all_finite(grads),
                    "train/updates_finite": _tree_all_finite(updates),
                    "train/params_finite": _tree_all_finite(params),
                    "train/qb_betas_finite": jnp.all(jnp.isfinite(metrics["qb_beta_per_layer"])),
                }
            )

        if ema_beta is None:
            ema_params = None
        else:
            ema_params = jax.tree_util.tree_map(
                lambda old, new: ema_beta * old + (1.0 - ema_beta) * new, qb_ema_params, params
            )

        watch_stats = None
        if watch_config is not None and compute_watch:
            watch_stats = compute_watch_stats(
                watch_targets=watch_targets,
                include_norms=watch_config.include_norms,
                include_per_parameter_norms=watch_config.include_per_parameter_norms,
                include_histogram=watch_config.include_histograms,
                split_scan_layers=watch_config.split_scan_layers,
                params=qb_params,
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
            pending_qb_betas=metrics["qb_beta_per_layer"],
        )
        return next_state, metrics, watch_stats

    return train_step


def _check_step_numerics(metrics: dict, step: int, diagnose_numerics: bool) -> None:
    """Raise if a step's loss (with ``diagnose_numerics`` also its gradients, updates, parameters or QB betas) is not
    finite. ``metrics`` are the step's host metrics (``_fetch_step_metrics``): the flags were computed in the step."""
    diagnostics = (
        {key: float(metrics[key]) for key in (*_NUMERIC_DIAGNOSTIC_KEYS, "train/supervised_tokens")}
        if diagnose_numerics
        else {}
    )
    if not bool(metrics[_LOSS_FINITE_KEY]):
        raise FloatingPointError(f"Non-finite Grug loss at step {step}: {diagnostics}")
    if diagnose_numerics and not all(bool(metrics[key]) for key in _NUMERIC_DIAGNOSTIC_KEYS):
        raise FloatingPointError(f"Non-finite Grug update at step {step}: {diagnostics}")


def _is_router_metric(key: str) -> bool:
    return (key.startswith("train/router/") or key.startswith("moe_bias/")) and key != "qb_beta_per_layer"


def _fetch_step_metrics(metrics: dict, diagnose_numerics: bool) -> dict:
    """The metrics of a step the loop checks and logs, copied to the host in one transfer, which waits for the step."""
    keys = ["train/loss", _LOSS_FINITE_KEY, _OPTIMIZER_HYPERPARAMS_KEY, "train/cross_entropy_loss"]
    if diagnose_numerics:
        keys += [*_NUMERIC_DIAGNOSTIC_KEYS, "train/supervised_tokens"]
    keys += [key for key in metrics if _is_router_metric(key)]
    return jax.device_get({key: metrics[key] for key in keys})


@dataclass(frozen=True)
class _HostOptimizerState:
    """The part of an optimizer state the step-info hook reads (``log_optimizer_hyperparams``), on the host."""

    hyperparams: dict


@dataclass(frozen=True)
class _HostStepState:
    """What the hooks of a pipelined step see in place of the state it returned, which the next step took: the step
    number and the optimizer hyperparameters, with no model."""

    step: int
    opt_state: _HostOptimizerState | None
    params: None = None
    ema_params: None = None


@dataclass
class _DispatchedStep:
    """A dispatched train step whose metrics the loop has not yet logged."""

    step: int
    """The step's index: the state it returns has ``step + 1``."""
    metrics: dict
    watch_stats: dict | None
    dispatch_time: float
    load_time: float
    may_compile: bool
    """The first step of the process or a gradient-watch step, whose wait has no hang timeout."""
    host_metrics: dict | None = None
    done_time: float | None = None


def _train_loop(
    train_step: Callable,
    state: GrugTrainState,
    iterator: LoadingTimeTrackerIterator,
    *,
    num_train_steps: int,
    state_callbacks: StateCallbackRunner[GrugTrainState | _HostStepState],
    checkpointer: Checkpointer | None,
    watch_interval: int,
    step_timeout: float | None,
    diagnose_numerics: bool,
    hooks_read_state: Callable[[int], bool],
) -> GrugTrainState:
    """Run ``train_step`` on batches from ``iterator`` up to ``num_train_steps``, then the forced hooks and the
    checkpointer's final save on the final state, which it returns.

    Step N's metrics come to the host in one transfer after step N+1 is dispatched, so the host work runs while the
    devices compute: its loss is checked (``_check_step_numerics``), its hooks run, its metrics are logged. The last
    step and the steps for which ``hooks_read_state(step)`` (eval, the profiler's window) finish before the next
    dispatch, and their hooks see the state they returned; the hooks of the others see a ``_HostStepState``, as the
    next step took (donated) their state. A non-finite loss therefore raises one step late, except that with a
    checkpointer each step is checked before its state can be saved. ``throughput/duration`` is the time from the later
    of the step's dispatch and the previous step's completion to its completion. ``watch_interval`` is the gradient
    watch's interval in steps, 0 for none. ``step_timeout`` bounds each wait for a step (``exit_if_hung``) except for
    the first step and watch steps, which can compile."""
    first_step = int(state.step)
    next_step = first_step
    pending: _DispatchedStep | None = None
    last_done = -math.inf
    last_loss = 0.0
    last_duration = 0.0
    last_hook_start: float | None = None

    def fetch(record: _DispatchedStep) -> dict:
        if record.host_metrics is None:
            record.host_metrics = _fetch_step_metrics(record.metrics, diagnose_numerics)
            record.done_time = time.perf_counter()
        return record.host_metrics

    def finish(record: _DispatchedStep, returned_state: GrugTrainState | None) -> None:
        """Check, hook and log a fetched step; ``returned_state`` is the state it returned, None once donated."""
        nonlocal last_done, last_loss, last_duration, last_hook_start
        host = fetch(record)
        assert record.done_time is not None
        duration = record.done_time - max(record.dispatch_time, last_done)
        last_done = record.done_time
        _check_step_numerics(host, record.step + 1, diagnose_numerics)
        hyperparams = host[_OPTIMIZER_HYPERPARAMS_KEY]
        hook_state = (
            returned_state
            if returned_state is not None
            else _HostStepState(record.step + 1, _HostOptimizerState(hyperparams) if hyperparams else None)
        )
        loss = float(host["train/loss"])
        hook_start = time.perf_counter()
        with jax.profiler.TraceAnnotation("callbacks"):
            state_callbacks.run(hook_state, loss=loss, step_duration=duration)
            if last_hook_start is not None:
                # The wall period: unlike the duration, it includes the host work between steps.
                levanter.tracker.log({"throughput/wall_step_s": hook_start - last_hook_start}, step=record.step)
            levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=record.step)
            levanter.tracker.log({"throughput/loading_time": record.load_time}, step=record.step)
            router_metrics = {key: value for key, value in host.items() if _is_router_metric(key)}
            if router_metrics:
                levanter.tracker.log(router_metrics, step=record.step)
            levanter.tracker.log({"train/cross_entropy_loss": host["train/cross_entropy_loss"]}, step=record.step)
            if record.watch_stats is not None:
                levanter.tracker.log(record.watch_stats, step=record.step)
        last_loss = loss
        last_duration = duration
        last_hook_start = hook_start

    try:
        while next_step < num_train_steps:
            with jax.profiler.TraceAnnotation("load_batch"):
                batch = next(iterator)
            step = next_step
            dispatch_time = time.perf_counter()
            compute_watch = watch_interval > 0 and step % watch_interval == 0
            # The first step compiles, so it gets no timeout; neither do watch steps, which can compile a variant.
            may_compile = step == first_step or compute_watch
            waits_for_compile = may_compile or (pending is not None and pending.may_compile)
            with exit_if_hung(None if waits_for_compile else step_timeout):
                state, metrics, watch_stats = train_step(state, batch, compute_watch=compute_watch)
                if pending is not None:
                    fetch(pending)
            next_step = step + 1
            record = _DispatchedStep(step, metrics, watch_stats, dispatch_time, iterator.this_load_time, may_compile)
            if pending is not None:
                finish(pending, None)
                pending = None
            if next_step == num_train_steps or hooks_read_state(step):
                with exit_if_hung(None if may_compile else step_timeout):
                    fetch(record)
                finish(record, state)
            else:
                pending = record
                if checkpointer is not None:
                    # The checkpointer may save the state this step returned, so check the step first; the
                    # checkpointer's save decision waits for the step anyway.
                    with exit_if_hung(None if may_compile else step_timeout):
                        _check_step_numerics(fetch(record), next_step, diagnose_numerics)
            if checkpointer is not None:
                checkpointer.on_step(tree=state, step=next_step)
    except BaseException:
        logger.exception("Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause")
        raise
    # Mirror classic trainer behavior: force callbacks on the last completed step.
    state_callbacks.run(state, loss=last_loss, step_duration=last_duration, force=True)
    if checkpointer is not None:
        # Unless the last step's state is already saved for good (a step-policy save): it would be written again.
        checkpointer.save_final_checkpoint(tree=state, step=next_step)
        checkpointer.wait_until_finished()
    return state


def _run_grug_local(config: GrugRunConfig) -> None:
    """Entry point for the training loop."""
    trainer = config.trainer.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(config)

    run_id = trainer.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    optimizer = config.optimizer.build(trainer.num_train_steps)
    watch_config = trainer.watch
    train_step = _make_train_step(
        optimizer,
        trainer.mp,
        z_loss_weight=config.trainer.z_loss_weight,
        ema_beta=config.trainer.ema_beta,
        watch_config=watch_config if watch_config.is_enabled else None,
        diagnose_numerics=config.trainer.diagnose_numerics,
    )

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
        model_axis_size=config.trainer.model_axis_size,
    )
    with set_mesh(mesh):
        batch_schedule = trainer.batch_schedule

        train_dataset = build_train_dataset(
            config.data,
            max_seq_len=config.model.max_seq_len,
            batch_schedule=batch_schedule,
            key=data_key,
        )
        train_loader = build_train_loader(train_dataset, batch_schedule=batch_schedule, mesh=mesh)

        @jax.jit
        def _init_state(model_rng):
            return initial_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=model_rng,
                ema_beta=config.trainer.ema_beta,
            )

        state = _init_state(model_key)

        checkpointer = trainer.checkpointer.create(run_id) if config.trainer.save_checkpoints else None
        if config.trainer.sft_weights_only_init:
            # SFT/RL: auto-resume from this run's own checkpoints if present (preemption), otherwise load only base
            # weights (+ pending_qb_betas) and keep the fresh optimizer/step (marin #650).
            state = restore_grug_state_from_checkpoint(
                state,
                checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
                load_checkpoint_setting=trainer.load_checkpoint,
                mesh=mesh,
                allow_partial=trainer.allow_partial_checkpoint,
            )
            if int(state.step) == 0 and trainer.initialize_from is not None:
                state = init_weights_only_from_checkpoint(
                    state,
                    trainer.initialize_from,
                    mesh=mesh,
                    load_ema=config.trainer.ema_beta is not None,
                )
        else:
            state = restore_grug_state_from_checkpoint(
                state,
                checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
                load_checkpoint_setting=trainer.load_checkpoint,
                mesh=mesh,
                allow_partial=trainer.allow_partial_checkpoint,
                initialize_from=trainer.initialize_from,
            )

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})

        flops_per_example, flops_summary = _compute_flops(model_config=config.model)
        levanter.tracker.log_summary(flops_summary)

        eval_cfg = config.eval
        evaluator = None
        if eval_cfg is not None:
            evaluator = build_tagged_evaluator(
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

        state_callbacks = StateCallbackRunner[GrugTrainState | _HostStepState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            eval_model_getter=lambda s: s.ema_params if s.ema_params is not None else s.params,
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
                profiler_cfg.build(
                    str(trainer.log_dir / run_id / "profiler"),
                    run_id=run_id,
                    num_steps=profiler_num_steps,
                ),
                every=1,
            )
        state_callbacks.add_hook(_make_mixture_stage_callback(train_dataset, batch_schedule), every=1)
        eval_interval: int | None = None
        if evaluator is not None and eval_cfg is not None:
            interval = eval_cfg.steps_per_eval
            eval_ema = eval_cfg.eval_ema and config.trainer.ema_beta is not None
            if interval is not None and interval > 0 and (eval_cfg.eval_current or eval_ema):
                eval_interval = interval
                state_callbacks.add_hook(
                    cb_tagged_evaluate(
                        evaluator,
                        prefix=eval_cfg.prefix,
                        eval_current=eval_cfg.eval_current,
                        eval_ema=eval_ema,
                    ),
                    every=interval,
                )

        profile_bounds = (
            (profiler_cfg.start_step - 1, profiler_cfg.start_step + profiler_num_steps - 1) if profiler_enabled else ()
        )

        def hooks_read_state(step: int) -> bool:
            # The eval hooks read the model; the profiler's window must start and stop between steps.
            return (eval_interval is not None and (step + 1) % eval_interval == 0) or step in profile_bounds

        _train_loop(
            train_step,
            state,
            iterator,
            num_train_steps=trainer.num_train_steps,
            state_callbacks=state_callbacks,
            checkpointer=checkpointer,
            watch_interval=watch_config.interval if watch_config.is_enabled else 0,
            step_timeout=config.trainer.step_timeout,
            diagnose_numerics=config.trainer.diagnose_numerics,
            hooks_read_state=hooks_read_state,
        )

    levanter.tracker.current_tracker().finish()


def run_grug(config: GrugRunConfig) -> None:
    """Dispatch training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=_run_grug_local,
        resources=config.resources,
    )
