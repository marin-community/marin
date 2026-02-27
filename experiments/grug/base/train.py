# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
from haliax import Axis
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray

import levanter.callbacks as callbacks
import levanter.tracker
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.callbacks.tensorstore_callbacks import (
    build_tensorstore_metrics_logger,
    tensorstore_metrics_interval_from_env,
)
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.checkpoint import load_checkpoint
from levanter.data import AsyncDataset, DataLoader
from levanter.data.mixture import MixtureDataset, rescale_mixture_schedule_for_batch_schedule
from levanter.data.text import GrugLmExample, LmDataConfig
from levanter.data.text.examples import grug_lm_example_from_named
from levanter.eval import TaggedEvaluator, cb_tagged_evaluate
from levanter.models.lm_model import LmExample
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.grug.base.model import GrugModelConfig, Transformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    train_batch_pspec: P = field(default_factory=lambda: P(("data",)))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None
    z_loss_weight: float = 0.0


@dataclass(frozen=True)
class GrugEvalConfig:
    """Perplexity eval settings for grug training."""

    eval_batch_size: int = 512
    eval_batch_pspec: P = field(default_factory=lambda: P(("data",)))
    steps_per_eval: int | None = 1000
    max_eval_batches: int | None = None
    prefix: str = "eval"
    eval_current: bool = True
    eval_ema: bool = True
    compute_bpb: bool = True


@dataclass(frozen=True)
class GrugRunConfig:
    """Top-level config for grug training."""

    model: GrugModelConfig
    data: LmDataConfig
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


def build_train_dataset(
    data_config: LmDataConfig,
    *,
    max_seq_len: int,
    batch_schedule: BatchSchedule,
    key: PRNGKeyArray,
) -> MixtureDataset[GrugLmExample]:
    pos = Axis("position", max_seq_len)
    mix_key, shuffle_key = jax.random.split(key)
    weights = data_config.train_weights
    if isinstance(weights, list):
        weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)

    initial_batch_size = batch_schedule.batch_size_at_step(0)
    datasets = data_config.train_sets(pos, key=shuffle_key, initial_batch_size=initial_batch_size)
    return MixtureDataset(
        datasets=datasets,
        weights=weights,
        stop_strategy=data_config.stop_strategy,
        key=mix_key,
        block_size=data_config.mixture_block_size,
    )


def build_train_loader(
    dataset: AsyncDataset[GrugLmExample],
    *,
    batch_schedule: BatchSchedule,
    mesh: Mesh,
    batch_pspec: P = P(("data",)),
) -> DataLoader[GrugLmExample]:
    axis_resource = batch_pspec[0]
    return DataLoader(
        dataset,
        batch_schedule.schedule,
        mesh=mesh,
        axis_resources={"__BATCH__": axis_resource},
        batch_axis_name="__BATCH__",
        allow_nondivisible_batch_size=False,
    )


def build_tagged_evaluator(
    *,
    data_config: LmDataConfig,
    max_seq_len: int,
    mesh: Mesh,
    eval_cfg: GrugEvalConfig,
) -> TaggedEvaluator[LmExample | GrugLmExample, Transformer] | None:
    pos = Axis("position", max_seq_len)
    tagged_eval_sets = data_config.tagged_eval_sets(pos)
    if len(tagged_eval_sets) == 0:
        logger.warning("No evaluation datasets provided.")
        return None

    max_examples_per_dataset = None
    if eval_cfg.max_eval_batches is not None:
        max_examples_per_dataset = eval_cfg.max_eval_batches * eval_cfg.eval_batch_size

    tokenizer = data_config.the_tokenizer if eval_cfg.compute_bpb else None
    batch_axis_resource = eval_cfg.eval_batch_pspec[0]
    eval_axis_mapping = {"batch": batch_axis_resource}
    eval_batch = Axis("batch", eval_cfg.eval_batch_size)
    eval_array_sharding = NamedSharding(mesh, P(batch_axis_resource, None))

    def eval_loss_fn(model: Transformer, batch: LmExample | GrugLmExample) -> tuple[jax.Array, jax.Array, jax.Array]:
        if isinstance(batch, LmExample):
            batch = grug_lm_example_from_named(batch)
        per_pos_loss = model.compute_next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="none",
            logsumexp_weight=None,
        )
        per_pos_loss = jax.sharding.reshard(per_pos_loss, eval_array_sharding)
        per_pos_weight = jax.sharding.reshard(batch.loss_weight, eval_array_sharding)
        per_pos_token_id = jnp.roll(batch.tokens, -1, axis=-1)
        return per_pos_loss, per_pos_weight, per_pos_token_id

    return TaggedEvaluator(
        EvalBatch=eval_batch,
        tagged_eval_sets=tagged_eval_sets,
        loss_fn=eval_loss_fn,
        tokenizer=tokenizer,
        device_mesh=mesh,
        axis_mapping=eval_axis_mapping,
        max_examples_per_dataset=max_examples_per_dataset,
    )


def _compute_flops(
    *,
    model_config: GrugModelConfig,
) -> tuple[float, dict[str, float]]:
    flops_per_token = lm_flops_per_token(
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        num_heads=model_config.num_heads,
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        glu=False,
    )
    flops_per_example = 3 * flops_per_token * model_config.max_seq_len

    flops_summary: dict[str, float] = {
        "throughput/flops_per_token_analytic": flops_per_token,
        "throughput/flops_per_example_analytic": flops_per_example,
    }

    return flops_per_example, flops_summary


def _make_mixture_stage_callback(train_dataset: MixtureDataset, batch_schedule: BatchSchedule):
    last_mixture_stage = -1

    def log_mixture_stage(step_info):
        nonlocal last_mixture_stage
        seq_index = batch_schedule.global_data_offset_by_step(step_info.step)
        block_id = seq_index // train_dataset.block_size
        stage = train_dataset._get_stage_for_block(block_id)
        if stage == last_mixture_stage:
            return

        weights = train_dataset.weight_stages[stage][1]
        mixture_log = {f"mixture/weight/{name}": weight for name, weight in weights.items()}
        mixture_log["mixture/stage"] = stage
        levanter.tracker.log(mixture_log, step=step_info.step)
        last_mixture_stage = stage

    return log_mixture_stage


@register_dataclass
@dataclass(frozen=True)
class GrugTrainState:
    step: jax.Array
    params: Transformer
    opt_state: optax.OptState
    ema_params: Transformer


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
    def train_step(state: GrugTrainState, batch, *, compute_watch: bool = False):
        def loss_fn(params):
            compute_params = mp.cast_to_compute(params)
            return compute_params.compute_next_token_loss(
                batch.tokens,
                batch.loss_weight,
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
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
        )

        return next_state, {"train/loss": loss}, watch_stats

    return train_step


def run_grug(config: GrugRunConfig) -> None:
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

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    with trainer.use_device_mesh():
        mesh = trainer.device_mesh
        batch_schedule = trainer.batch_schedule

        train_dataset = build_train_dataset(
            config.data,
            max_seq_len=config.model.max_seq_len,
            batch_schedule=batch_schedule,
            key=data_key,
        )
        train_loader = build_train_loader(
            train_dataset,
            batch_schedule=batch_schedule,
            mesh=mesh,
            batch_pspec=config.trainer.train_batch_pspec,
        )

        @jax.jit
        def _init_state(model_rng):
            params = trainer.mp.cast_to_param(Transformer.init(config.model, key=model_rng))
            return GrugTrainState(
                step=jnp.array(0, dtype=jnp.int32),
                params=params,
                opt_state=optimizer.init(params),
                ema_params=params,
            )

        state = _init_state(model_key)

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
        state_callbacks.add_hook(_make_mixture_stage_callback(train_dataset, batch_schedule), every=1)
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
                # grad_watch runs only on its configured interval.
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
            # Mirror classic trainer behavior: force callbacks on the last completed step.
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if checkpointer is not None:
                checkpointer.on_step(tree={"train_state": state}, step=int(state.step), force=True)
                checkpointer.wait_until_finished()

    levanter.tracker.current_tracker().finish()


__all__ = [
    "GrugEvalConfig",
    "GrugRunConfig",
    "GrugTrainState",
    "GrugTrainerConfig",
    "run_grug",
]
