# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training loop for the grug-enhanced nanogpt. Identical to train.py but uses model_grug.Transformer."""

from __future__ import annotations

import dataclasses
import functools
import logging
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jmp
import levanter.callbacks as callbacks
import levanter.tracker
import optax
from fray.cluster import ResourceConfig
from haliax import Axis
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray
from levanter.callbacks.state_adapter import StateCallbackRunner
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

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.nanogpt.model_grug import GrugNanoGPTConfig, Transformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrugTrainerConfig:
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    train_batch_pspec: P = field(default_factory=lambda: P(("data",)))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None
    z_loss_weight: float = 0.0


@dataclass(frozen=True)
class GrugEvalConfig:
    eval_batch_size: int = 512
    eval_batch_pspec: P = field(default_factory=lambda: P(("data",)))
    steps_per_eval: int | None = 125
    max_eval_batches: int | None = 20
    prefix: str = "eval"
    eval_current: bool = True
    eval_ema: bool = False
    compute_bpb: bool = True


@dataclass(frozen=True)
class GrugRunConfig:
    model: GrugNanoGPTConfig
    data: LmDataConfig
    resources: ResourceConfig
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


@register_dataclass
@dataclass(frozen=True)
class GrugTrainState:
    step: jax.Array
    params: Transformer
    opt_state: optax.OptState
    ema_params: Transformer | None


def initial_state(
    model_config: GrugNanoGPTConfig,
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
        return None
    max_examples_per_dataset = None
    if eval_cfg.max_eval_batches is not None:
        max_examples_per_dataset = eval_cfg.max_eval_batches * eval_cfg.eval_batch_size
    tokenizer = data_config.the_tokenizer if eval_cfg.compute_bpb else None
    batch_axis_resource = eval_cfg.eval_batch_pspec[0]
    eval_batch = Axis("batch", eval_cfg.eval_batch_size)
    eval_array_sharding = NamedSharding(mesh, P(batch_axis_resource, None))

    def eval_loss_fn(model: Transformer, batch: LmExample | GrugLmExample) -> tuple[jax.Array, jax.Array, jax.Array]:
        if isinstance(batch, LmExample):
            batch = grug_lm_example_from_named(batch)
        per_pos_loss = model.next_token_loss(
            batch.tokens, batch.loss_weight, mask=batch.attn_mask, reduction="none", logsumexp_weight=None
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
        axis_mapping={"batch": batch_axis_resource},
        max_examples_per_dataset=max_examples_per_dataset,
    )


def _compute_flops(*, model_config: GrugNanoGPTConfig) -> tuple[float, dict[str, float]]:
    flops_per_token = lm_flops_per_token(
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_heads,
        num_heads=model_config.num_heads,
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        glu=True,
    )
    flops_per_example = 3 * flops_per_token * model_config.max_seq_len
    return flops_per_example, {
        "throughput/flops_per_token_analytic": flops_per_token,
        "throughput/flops_per_example_analytic": flops_per_example,
    }


def _make_train_step(
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
            return compute_params.next_token_loss(
                batch.tokens, batch.loss_weight, mask=batch.attn_mask, reduction="mean", logsumexp_weight=z_loss
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        ema_params = None
        if ema_beta is not None and state.ema_params is not None:
            ema_params = jax.tree_util.tree_map(
                lambda old, new: ema_beta * old + (1.0 - ema_beta) * new, state.ema_params, params
            )
        next_state = dataclasses.replace(
            state, step=state.step + one, params=params, opt_state=opt_state, ema_params=ema_params
        )
        return next_state, {"train/loss": loss}

    return train_step


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
            logger.exception("Fatal error in grug nanogpt training loop")
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
