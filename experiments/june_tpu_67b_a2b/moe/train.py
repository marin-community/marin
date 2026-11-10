# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import dataclasses
import faulthandler
import functools
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import levanter.callbacks as callbacks
import levanter.tracker
import optax
from fray.cluster import ResourceConfig
from haliax import Axis
from haliax.partitioning import set_mesh
from jax.sharding import Mesh, NamedSharding, get_abstract_mesh, reshard
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.checkpoint import load_checkpoint
from levanter.data.dataset import AsyncDataset
from levanter.data.loader import DataLoader
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample, grug_lm_example_from_named
from levanter.eval import TaggedEvaluator, cb_tagged_evaluate
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.june_tpu_67b_a2b.checkpointing import restore_grug_state_from_checkpoint
from experiments.june_tpu_67b_a2b.dispatch import dispatch_grug_training_run
from experiments.june_tpu_67b_a2b.moe.model import (
    Block,
    GrugModelConfig,
    Transformer,
    _long_layer_schedule,
    after_scan_reduced_spec,
    qb_betas_spec,
    qb_group_count,
    router_token_validity,
    summarize_loss_metrics,
    unreduced_to_local,
)

# This file intentionally mirrors `experiments/grug/base/train.py` with
# variant-specific model/loss/FLOP wiring, per the grug copy-first workflow in
# `.agents/skills/change-grug/`.

logger = logging.getLogger(__name__)
_NUMERIC_DIAGNOSTIC_KEYS = (
    "train/grads_finite",
    "train/updates_finite",
    "train/params_finite",
    "train/qb_betas_finite",
)


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    z_loss_weight: float = 0.0  # Weight on logsumexp (z-loss) stabilization term.
    diagnose_numerics: bool = False

    # Grug builds its own compact (replica_dcn, data, expert, context, model) mesh instead of using
    # the Trainer's logical axis mapping; `data` absorbs whatever these leave free.
    # Defaults reproduce the historical layout: no expert parallelism and full replication
    # across slices (replica_axis_size=None -> jax.process_count()), i.e. parameters
    # replicated per slice and sharded only over the intra-slice `data` axis. For a model
    # too large to replicate within one slice, set replica_axis_size=1 (FSDP across every
    # slice) and expert_axis_size>1 (expert parallelism over the intra-slice devices).
    expert_axis_size: int = 1
    replica_axis_size: int | None = None
    model_axis_size: int = 1
    save_checkpoints: bool = True
    """False skips the checkpointer entirely, including the forced final save; for benchmarks."""

    step_timeout: float | None = None
    """Seconds after which a train step counts as hung: every thread's traceback is dumped and the process exits
    with status 1. The first step in this process and gradient-watch steps, both of which can compile, have no
    timeout. When one device fails inside a step, for example an allocator OOM, the other devices wait in a
    collective that never times out, and without this the job sits until its scheduler time limit. None disables
    it."""

    carry_compute_params: bool = False
    """Carry a compute-dtype copy of the parameters in the train state, written next to the optimizer update, and
    differentiate with respect to it. Saves re-reading and casting every float32 parameter at the start of each step
    (the update already reads them), for one extra compute-dtype copy of the parameters in memory. The gradients are
    the same: the cast's transpose is the cast of the compute-dtype gradient back to the parameter dtype. Checkpoints
    leave the copy out; it is rebuilt from the parameters on restore."""

    qb_sync_interval: int = 0
    """With per-node QB balancing (``GrugModelConfig.qb_balance="per_node"``): every this many steps, replace each
    node's QB thresholds by the mean over nodes, so the nodes' router biases cannot drift apart. 0 never averages."""

    microbatches: int = 1
    """Split each step's batch into this many microbatches, run forward and backward on one at a time and sum their
    gradients in float32 before the single optimizer update; activation memory is that of one microbatch.
    ``train_batch_size`` stays the batch of one step and must be a multiple of ``microbatches`` times the number of
    batch shards. Each device takes every ``microbatches``-th of its own rows, so the split moves no data. Each
    gradient is reduced across devices once per step, after the last microbatch; until then every device keeps its
    own float32 partial sums, full size for the weights sharded over ``replica_dcn`` or ``data`` (the embedding and
    output projection, FSDP weights) and expert-sharded for the expert weights. The loss,
    the gradients and the routing counts equal those of the whole batch up to rounding, as long as no expert overflows
    its capacity, which applies per microbatch. The next step's QB thresholds are the mean of the microbatches'
    thresholds weighted by their valid tokens, each estimated from its own tokens. 1 runs the whole batch at once.
    The levanter ``TrainerConfig.per_device_parallelism``, if set, must agree: rows per batch shard per microbatch.
    XLA's GPU loop unrolling may run the copies of the microbatch loop body in any order; an optimization barrier at
    the top of each microbatch keeps them in order, so one microbatch's gradients are summed before the next starts."""

    zero1: bool = False
    """ZeRO-1 over the replicas: shard the expert weights' float32 parameters, optimizer state and EMA over
    ``replica_dcn`` as well as ``expert``, so each replica keeps and updates 1/``replica_dcn`` of them. Each expert
    weight's gradient is then reduced once per step into that shard (a reduce-scatter in place of an all-reduce, half
    the bytes), and the updated compute-dtype parameters are all-gathered back to every replica at the end of the step,
    after the update. With ``overlap_last_microbatch`` the per-layer reductions in the last microbatch's backward are
    reduce-scatters into that shard. The other weights are unchanged: the embedding and output projection are already
    sharded over the replicas, and the block weights stay replicated. The numbers are those of the replicated update up
    to the order of the optimizer's norm sums. Needs ``microbatches`` >= 2, ``carry_compute_params`` (the all-gather
    writes the compute copy, which the forward reads), and the expert count divisible by ``expert`` x ``replica_dcn``.
    The checkpoint layout is unchanged, so a checkpoint saved with it on loads with it off and the other way round."""

    overlap_last_microbatch: bool = False
    """With ``microbatches`` > 1: run the last microbatch after the microbatch loop and reduce each layer's expert
    weight gradients (the float32 sum of the earlier microbatches plus the last one's, cast to the compute dtype)
    inside its backward, as soon as that layer's backward is done, so the reductions overlap the backward of the
    layers below it instead of running after the step's last backward. Every weight is still reduced once per step,
    and the gradients are the same as without it. The other weights are reduced after the last microbatch, as
    without it. Peak memory goes up by about one layer's expert gradients. Needs the stacked layer layout
    (``GrugModelConfig.use_array_stacked_blocks``).

    On GPU the overlap needs XLA flags: ``--xla_gpu_all_reduce_combine_threshold_bytes`` above one layer's expert
    gradient bytes (so the per-layer all-reduces of one layer combine into one), stream domains for in-node
    collectives (``--xla_gpu_collective_domain_assignment=scale_up_fabric``, so in-node all-to-alls do not queue
    behind the per-layer all-reduce), room for the latency-hiding scheduler
    (``--xla_gpu_memory_limit_slop_factor=200``; the default keeps it in memory-pressure mode) and a PGLE profile that
    costs the per-layer all-reduce. Without them the per-layer reductions run synchronously and hide nothing."""

    sft_weights_only_init: bool = False
    """SFT/RL init semantics (marin #650). When True and the run has no checkpoint of
    its own to auto-resume from, the trainer loads only the model weights (params +
    ``pending_qb_betas``) from ``TrainerConfig.initialize_from`` and keeps the fresh
    optimizer state and ``step=0`` -- i.e. a fresh LR schedule over the base weights,
    not a full-state resume. False (default) keeps the byte-identical continued-pretrain
    behaviour where ``initialize_from`` loads the whole train state (weights + optimizer +
    step). Own-run checkpoints still take precedence, so preemption resumes normally."""


@dataclass(frozen=True)
class GrugEvalConfig:
    """Perplexity eval settings for grug training."""

    eval_batch_size: int = 512
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
    resources: ResourceConfig
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
    return data_config.train_mixture(Axis("position", max_seq_len), batch_schedule, key=key)


_REPLICA_AXIS = "replica_dcn"
_BATCH_AXES: tuple[str, ...] = (_REPLICA_AXIS, "data", "expert")
_EXPERT_AXIS = "expert"


def build_train_loader(
    dataset: AsyncDataset[GrugLmExample],
    *,
    batch_schedule: BatchSchedule,
    mesh: Mesh,
) -> DataLoader[GrugLmExample]:
    # DataLoader uses this batch axis mapping to shard batches across the distributed mesh.
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, context, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    return DataLoader(
        dataset,
        batch_schedule.schedule,
        mesh=mesh,
        axis_resources={"__BATCH__": _BATCH_AXES},
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
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, context, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    eval_axis_mapping = {"batch": _BATCH_AXES}
    eval_batch = Axis("batch", eval_cfg.eval_batch_size)
    eval_array_sharding = NamedSharding(mesh, P(_BATCH_AXES, None))

    def eval_loss_fn(model: Transformer, batch: LmExample | GrugLmExample) -> tuple[jax.Array, jax.Array, jax.Array]:
        if isinstance(batch, LmExample):
            batch = grug_lm_example_from_named(batch)
        per_pos_loss = model.next_token_loss(
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
    # Hybrid attention: every 4th layer plus the last layer runs full causal
    # attention; the rest use a sliding window (see ``_long_layer_schedule``
    # in model.py). At long context this makes the analytic FLOPs count much
    # smaller than a naive ``all-layers-full-attention`` estimate, because
    # each sliding-window layer's attention span is capped at the window.
    n = model_config.num_layers
    num_full_attention_layers = int(_long_layer_schedule(n).sum())

    flops_per_token = lm_flops_per_token(
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        shared_intermediate_dim=model_config.shared_expert_intermediate_dim,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        num_heads=model_config.num_heads,
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        glu=True,
        num_experts=model_config.num_experts,
        num_shared_experts=1 if model_config.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=model_config.num_experts_per_token,
        sliding_window=model_config.sliding_window,
        num_full_attention_layers=num_full_attention_layers,
    )
    flops_per_example = 3 * flops_per_token * model_config.max_seq_len

    flops_summary: dict[str, float] = {
        "throughput/flops_per_token_analytic": flops_per_token,
        "throughput/flops_per_example_analytic": flops_per_example,
        "throughput/num_full_attention_layers": float(num_full_attention_layers),
        "throughput/num_sliding_attention_layers": float(n - num_full_attention_layers),
        "throughput/sliding_window": float(model_config.sliding_window),
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
    ema_params: Transformer | None
    pending_qb_betas: jax.Array
    compute_params: Transformer | None = None  # mp.cast_to_compute(params), see carry_compute_params


def _checkpointed(state: GrugTrainState) -> GrugTrainState:
    """``state`` as checkpoints hold it: without the compute copy, which is rebuilt from the parameters on restore."""
    return dataclasses.replace(state, compute_params=None)


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array) -> Transformer:
    """Set router biases from QB betas (computed on previous step)."""
    new_biases = -qb_betas
    new_biases = new_biases - jnp.mean(new_biases, axis=-1, keepdims=True)
    if model.stacked_blocks is not None:
        # In the dtype of the model's biases, so that the compute-dtype copy keeps its dtype.
        old_biases = model.stacked_blocks.stacked.mlp.router_bias
        return eqx.tree_at(
            lambda t: t.stacked_blocks.stacked.mlp.router_bias,
            model,
            new_biases.astype(old_biases.dtype),
        )
    assert model.blocks is not None
    new_blocks = list(model.blocks)
    for i, block in enumerate(model.blocks):
        if not isinstance(block, Block) or block.mlp is None:
            continue
        bias = new_biases[i].astype(block.mlp.router_bias.dtype)
        new_mlp = eqx.tree_at(lambda m: m.router_bias, block.mlp, bias)
        new_blocks[i] = eqx.tree_at(lambda b: b.mlp, block, new_mlp)
    return eqx.tree_at(lambda t: t.blocks, model, tuple(new_blocks))


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
    carry_compute_params: bool = False,
    zero1: bool = False,
) -> GrugTrainState:
    """The step-0 train state; with ``zero1`` the expert weights' parameters, optimizer state and EMA are sharded over
    the replicas (``GrugTrainerConfig.zero1``) and the compute copy keeps the model's sharding."""
    if zero1 and not carry_compute_params:
        raise ValueError("zero1 needs carry_compute_params: the end-of-step all-gather writes the compute copy")
    params = mp.cast_to_param(Transformer.init(model_config, key=key))
    if params.blocks is not None:
        num_moe_layers = sum(1 for b in params.blocks if b.mlp is not None)
    else:
        num_moe_layers = model_config.num_layers
    compute_params = mp.cast_to_compute(params) if carry_compute_params else None
    if zero1:
        params = _zero1_sharded(params)
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params),
        ema_params=params if ema_beta is not None else None,
        pending_qb_betas=_initial_qb_betas(model_config, num_moe_layers),
        compute_params=compute_params,
    )


def _zero1_spec(spec: P) -> P:
    """``spec`` with its ``expert`` dimension also split over the replicas, for a weight sharded over ``expert`` and
    not already over the replicas; any other spec unchanged.

    ``replica_dcn`` goes after ``expert``, so it splits each device's expert shard: the replicas that hold the same
    experts (one per node) each keep a part, and the reduce-scatter and all-gather run among them, across nodes only.
    """
    partitions = [entry if isinstance(entry, tuple) else (entry,) for entry in spec]
    if any(_REPLICA_AXIS in axes for axes in partitions):
        return spec
    for i, axes in enumerate(partitions):
        if _EXPERT_AXIS in axes:
            partitions[i] = (*axes, _REPLICA_AXIS)
            return P(*(axes[0] if len(axes) == 1 else axes for axes in partitions))
    return spec


def _zero1_sharded(params: Transformer) -> Transformer:
    def shard(x):
        if not isinstance(x, jax.Array):
            return x
        spec = jax.typeof(x).sharding.spec
        return reshard(x, _zero1_spec(spec))

    return jax.tree.map(shard, params)


def _resharded_like(tree, like):
    """``tree`` resharded leaf by leaf to the shardings of ``like``, a tree of the same structure."""
    return jax.tree.map(
        lambda x, target: reshard(x, jax.typeof(target).sharding) if isinstance(x, jax.Array) else x, tree, like
    )


def rebuilt_compute_params(state: GrugTrainState, mp: jmp.Policy) -> GrugTrainState:
    """``state`` with its compute copy recast from its parameters, in the compute copy's own sharding.

    For a restored state: checkpoints leave the compute copy out, so the caller passes a state whose compute copy (the
    initial one) only gives the sharding. With ``zero1`` this all-gathers the expert weights the parameters shard over
    the replicas.
    """
    if state.compute_params is None:
        return state
    compute_params = jax.jit(lambda p, c: _resharded_like(mp.cast_to_compute(p), c))(state.params, state.compute_params)
    return dataclasses.replace(state, compute_params=compute_params)


def _initial_qb_betas(model_config: GrugModelConfig, num_moe_layers: int) -> jax.Array:
    """Zero QB thresholds: [L, E], or [L, G, E] sharded over the groups with per-node balancing."""
    if model_config.qb_balance == "global":
        return jnp.zeros((num_moe_layers, model_config.num_experts))
    zeros = jnp.zeros((num_moe_layers, qb_group_count(model_config), model_config.num_experts))
    return reshard(zeros, qb_betas_spec(model_config))


def _average_qb_betas_over_groups(betas: jax.Array) -> jax.Array:
    """[L, G, E] per-group QB thresholds -> their mean over the groups, broadcast back to every group."""
    mean = jnp.mean(betas, axis=1, keepdims=True)
    return reshard(jnp.broadcast_to(mean, betas.shape), jax.typeof(betas).sharding.spec)


def init_weights_only_from_checkpoint(
    state: GrugTrainState,
    checkpoint_path: str,
    *,
    mesh: Mesh | None,
    load_ema: bool,
    _load_fn: Callable[..., object] = load_checkpoint,
) -> GrugTrainState:
    """Load only model weights from an external checkpoint, resetting the optimizer.

    This is the SFT/RL init (marin #650): the base checkpoint supplies ``params`` and the
    ``pending_qb_betas`` router-bias state; the optimizer state and ``step`` stay at their
    fresh values in ``state`` so training starts a new LR schedule from step 0 instead of
    resuming the base run's optimizer/step.

    ``load_ema`` mirrors the loaded weights into ``ema_params`` when the run tracks an EMA.
    """
    # Deserialize only the ``params`` subtree and the ``pending_qb_betas`` leaf, keyed by their
    # GrugTrainState field names so they match the on-disk paths. allow_partial lets the base
    # checkpoint's other leaves (opt_state / step / ema_params) go unread, so the base run's
    # optimizer tree is never touched and stays fresh from ``state``.
    exemplar: dict[str, object] = {"params": state.params, "pending_qb_betas": state.pending_qb_betas}
    loaded = cast("dict[str, object]", _load_fn(exemplar, checkpoint_path, mesh=mesh, allow_partial=True))
    updates: dict[str, object] = {"params": loaded["params"], "pending_qb_betas": loaded["pending_qb_betas"]}
    if load_ema and state.ema_params is not None:
        updates["ema_params"] = loaded["params"]
    return dataclasses.replace(state, **updates)


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
    watch_config: WatchConfig | None = None,
    diagnose_numerics: bool = False,
    qb_sync_interval: int = 0,
    microbatches: int = 1,
    zero1: bool = False,
    overlap_last_microbatch: bool = False,
):
    if microbatches < 1:
        raise ValueError(f"microbatches must be at least 1, got {microbatches}")
    if zero1 and microbatches < 2:
        # With one microbatch the gradients are reduced in the backward, so zero1 would only add the all-gather.
        raise ValueError(f"zero1 needs microbatches >= 2, got {microbatches}")
    if overlap_last_microbatch and microbatches < 2:
        raise ValueError(f"overlap_last_microbatch needs at least 2 microbatches, got {microbatches}")
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
        # host-side TPU kernel launches that can cause SPMD sync issues).
        qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)
        if zero1 and state.compute_params is None:
            raise ValueError("zero1 needs carry_compute_params")
        if ema_beta is not None:
            qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas)
        else:
            qb_ema_params = None

        def loss_fn(compute_params):
            return compute_params.next_token_loss(
                batch.tokens,
                batch.loss_weight,
                mask=batch.attn_mask,
                reduction="mean",
                logsumexp_weight=z_loss,
                return_router_metrics=True,
            )

        def loss_terms_fn(compute_params, microbatch, block_grad_offsets=None, block_grad_sinks=None):
            return compute_params.next_token_loss_terms(
                microbatch.tokens,
                microbatch.loss_weight,
                mask=microbatch.attn_mask,
                reduction="sum",
                logsumexp_weight=z_loss,
                block_grad_offsets=block_grad_offsets,
                block_grad_sinks=block_grad_sinks,
            )

        if microbatches > 1:
            if state.compute_params is None:
                qb_compute_params = mp.cast_to_compute(qb_params)
            else:
                qb_compute_params = _apply_qb_betas(state.compute_params, state.pending_qb_betas)
            loss, summarized_metrics, accumulated_grads = _microbatched_loss_and_grads(
                loss_terms_fn,
                qb_compute_params,
                batch,
                microbatches,
                overlap_last_microbatch,
                grad_shardings=qb_params,
            )
            grads = mp.cast_to_param(accumulated_grads)
        elif state.compute_params is None:
            (loss, summarized_metrics), grads = jax.value_and_grad(
                lambda params: loss_fn(mp.cast_to_compute(params)), has_aux=True
            )(qb_params)
        else:
            qb_compute_params = _apply_qb_betas(state.compute_params, state.pending_qb_betas)
            (loss, summarized_metrics), compute_grads = jax.value_and_grad(loss_fn, has_aux=True)(qb_compute_params)
            grads = mp.cast_to_param(compute_grads)
        metrics = {"train/loss": loss, **summarized_metrics}
        updates, opt_state = optimizer.update(grads, state.opt_state, qb_params)
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
            if qb_ema_params is None:
                raise ValueError("ema_params must be initialized when ema_beta is set.")
            ema_params = jax.tree_util.tree_map(
                lambda old, new: ema_beta * old + (1.0 - ema_beta) * new,
                qb_ema_params,
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
                params=qb_params,
                grads=grads,
                updates=updates,
                opt_state=state.opt_state,
                model_tree_type=type(state.params),
            )

        pending_qb_betas = metrics["qb_beta_per_layer"]
        if qb_sync_interval > 0 and pending_qb_betas.ndim == 3:
            pending_qb_betas = jax.lax.cond(
                (state.step + one) % qb_sync_interval == 0,
                _average_qb_betas_over_groups,
                lambda betas: betas,
                pending_qb_betas,
            )

        def next_compute_params(params: Transformer) -> Transformer:
            compute_params = mp.cast_to_compute(params)
            if not zero1:
                return compute_params
            # The all-gather of the updated expert weights, in the compute dtype, back to the model's sharding.
            return _resharded_like(compute_params, state.compute_params)

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
            pending_qb_betas=pending_qb_betas,
            compute_params=None if state.compute_params is None else next_compute_params(params),
        )

        return next_state, metrics, watch_stats

    return train_step


def _split_microbatches(x: jax.Array, microbatches: int) -> jax.Array:
    """[B, ...] batch-sharded -> [K, B/K, ...], microbatch k taking rows k, k + K, k + 2K, ...

    Every device holds a contiguous run of rows, so this takes every K-th of its own rows: no data moves, and each
    microbatch keeps the batch sharding with B/K rows. (A contiguous split, microbatch k = rows k*B/K onwards, would
    all-to-all the batch.) It also keeps every row in its node, which per-node QB balancing relies on.
    """
    x = reshard(x, P(_BATCH_AXES))
    rows = jnp.reshape(x, (x.shape[0] // microbatches, microbatches, *x.shape[1:]), out_sharding=P(_BATCH_AXES))
    return jnp.swapaxes(rows, 0, 1)


def _token_shares(valid_tokens: jax.Array) -> jax.Array:
    """[K, ...] valid-token counts -> each microbatch's share of them along K; equal shares where there are none."""
    total = jnp.sum(valid_tokens, axis=0, keepdims=True)
    return jnp.where(total > 0, valid_tokens / jnp.maximum(total, 1.0), 1.0 / valid_tokens.shape[0])


def _combine_microbatch_router_metrics(
    stacked: dict[str, jax.Array], row_valid_tokens: jax.Array
) -> dict[str, jax.Array]:
    """Per-layer router metrics stacked over K microbatches, [K, L, ...] -> those of the whole batch, [L, ...].

    ``row_valid_tokens[k, i]`` counts the routed (valid) tokens of row i of microbatch k. Counts add up, and the
    routing entropy is recomputed from them; the router z-loss, a mean over valid tokens, is the sum of the
    microbatches' weighted by their shares of the batch's valid tokens (token shares). Both equal the whole batch's.
    The QB thresholds are the token-share-weighted mean of the microbatches' thresholds, each estimated from its own
    tokens: with global balancing that is the same valid-token-weighted mean over shards the whole batch takes, over K
    times as many local estimates; with per-node balancing each group's thresholds take the shares of the group's own
    valid tokens. The load-balancing loss, nonlinear in the counts, is the token-share-weighted mean of the
    microbatches' values, a diagnostic that differs from the whole batch's.
    """
    token_shares = _token_shares(jnp.sum(row_valid_tokens, axis=1))

    def weighted_sum(x: jax.Array) -> jax.Array:
        return jnp.sum(token_shares.reshape(-1, *(1,) * (x.ndim - 1)) * x, axis=0)

    counts = jnp.sum(stacked["routing_counts_per_layer"], axis=0)
    fraction = counts / jnp.maximum(jnp.sum(counts, axis=-1, keepdims=True), 1.0)
    qb_betas = stacked["qb_beta_per_layer"]
    if qb_betas.ndim == 4:
        # Per-node balancing, [K, L, G, E]: group g's rows are the g-th contiguous block of each microbatch's rows,
        # those of its batch shards.
        microbatches, groups = row_valid_tokens.shape[0], qb_betas.shape[2]
        rows = reshard(row_valid_tokens, P(None, None)).reshape(microbatches, groups, -1)
        group_shares = _token_shares(jnp.sum(rows, axis=2))
        qb_beta = jnp.sum(group_shares[:, None, :, None] * qb_betas, axis=0)
    else:
        qb_beta = weighted_sum(qb_betas)
    return {
        "routing_counts_per_layer": counts,
        "routing_entropy_per_layer": -jnp.sum(fraction * jnp.log(fraction + 1e-6), axis=-1),
        "load_balancing_loss_per_layer": weighted_sum(stacked["load_balancing_loss_per_layer"]),
        "router_z_loss_per_layer": weighted_sum(stacked["router_z_loss_per_layer"]),
        "qb_beta_per_layer": reshard(qb_beta, P(*jax.typeof(qb_betas).sharding.spec[1:])),
        "capacity_overflow_per_layer": jnp.sum(stacked["capacity_overflow_per_layer"], axis=0),
        "skipped_assignments_per_layer": jnp.sum(stacked["skipped_assignments_per_layer"], axis=0),
    }


def _microbatched_loss_and_grads(
    loss_terms_fn: Callable[[Transformer, object], tuple[jax.Array, jax.Array, dict[str, jax.Array]]],
    params: Transformer,
    batch,
    microbatches: int,
    overlap_last_microbatch: bool = False,
    *,
    grad_shardings: Transformer,
) -> tuple[jax.Array, dict, Transformer]:
    """Loss, summarized metrics and gradients of the whole ``batch``, one microbatch at a time.

    ``loss_terms_fn(params, microbatch)`` returns the microbatch's summed (``reduction="sum"``) cross-entropy, its
    weighted router aux loss and its raw per-layer router metrics. The loss each microbatch differentiates is its
    share of the whole batch's: its cross-entropy sum over the batch's total loss weight, and its aux loss, a mean
    over valid tokens, weighted by its share of the batch's valid tokens. Weighting the loss rather than the gradient
    keeps the accumulation a plain ``accumulator + gradient``, and dividing by the batch's weight rather than the
    microbatch's keeps a microbatch without loss weight from producing a 0/0 gradient. A batch with no loss weight
    has zero cross-entropy and gradient.

    The gradients are reduced across devices once, after the loop. ``params`` are marked ``reduced`` over the batch
    axes first (``_mark_batch_reduced``), so each microbatch's gradient is a per-device partial sum, ``unreduced``
    over those axes, with no cross-device reduction in the backward. The partials accumulate in float32 on every
    device. ``_finalize_accumulated_grad`` then casts each to the dtype of ``params`` and reduces it once: an
    all-reduce, or a reduce-scatter for a weight sharded over a batch axis. Only the loss, routing statistics and QB
    threshold reductions of the forward still run once per microbatch. The returned gradients have the dtype of
    ``params`` and the sharding of ``grad_shardings``, a tree like ``params``: the parameters the optimizer updates,
    whose sharding over the replicas (``GrugTrainerConfig.zero1``) turns the reduction into a reduce-scatter.

    With ``overlap_last_microbatch`` the loop runs the first ``microbatches - 1`` microbatches and the last one runs
    after it, with the expert weights (``_per_layer_reduced``) left unmarked. Its layer scan gets each layer's slice
    of their accumulators as ``block_grad_offsets``, and its backward reduces that layer's ``accumulator + gradient``,
    cast as ``_finalize_accumulated_grad`` casts it, as soon as the layer is done: the same arithmetic and one reduction
    per weight per step, but overlapping the backward of the layers below it. The other weights are reduced after the
    last microbatch, as without it. An expert weight whose ``grad_shardings`` entry shards more than ``params`` does
    (ZeRO-1) is reduce-scattered there, per layer: its gradient leaves the layer scan through a zero "sink" input with
    that sharding (``block_grad_sinks``), since the weight's own gradient must keep the weight's sharding.
    """
    batch_size = batch.tokens.shape[0]
    batch_shards = _batch_shards(get_abstract_mesh())
    if batch_size % (microbatches * batch_shards) != 0:
        raise ValueError(
            f"batch size {batch_size} must be a multiple of microbatches ({microbatches}) times the batch shards "
            f"({batch_shards})"
        )
    split_batch = jax.tree.map(lambda x: _split_microbatches(x, microbatches), batch)
    total_weight = jnp.sum(batch.loss_weight.astype(jnp.float32))
    loss_denominator = jnp.where(total_weight > 0, total_weight, 1.0)
    row_valid_tokens = jnp.sum(
        _split_microbatches(router_token_validity(batch.tokens, batch.attn_mask), microbatches).astype(jnp.float32),
        axis=2,
    )
    token_shares = _token_shares(jnp.sum(row_valid_tokens, axis=1))

    def loss_and_terms(p, microbatch, token_share):
        cross_entropy_sum, aux_loss, router_metrics = loss_terms_fn(p, microbatch)
        cross_entropy_loss = cross_entropy_sum / loss_denominator
        aux_loss = token_share * aux_loss
        return cross_entropy_loss + aux_loss, (cross_entropy_loss, aux_loss, router_metrics)

    value_and_grad = jax.value_and_grad(loss_and_terms, has_aux=True)
    marked_params = _mark_batch_reduced(params)
    looped = microbatches - 1 if overlap_last_microbatch else microbatches

    def body(carry, xs):
        grad_sum, loss_sum, cross_entropy_sum, aux_sum = carry
        microbatch, token_share = xs
        # When XLA unrolls this loop, the barrier keeps each microbatch after the previous one's gradients are summed,
        # so only one per-microbatch gradient tree is live at a time.
        grad_sum, microbatch, token_share = jax.lax.optimization_barrier((grad_sum, microbatch, token_share))
        (loss, (cross_entropy_loss, aux_loss, router_metrics)), grads = value_and_grad(
            marked_params, microbatch, token_share
        )
        grad_sum = jax.tree.map(lambda acc, g: acc + g.astype(jnp.float32), grad_sum, grads)
        carry = (grad_sum, loss_sum + loss, cross_entropy_sum + cross_entropy_loss, aux_sum + aux_loss)
        return carry, router_metrics

    zero = jnp.zeros((), jnp.float32)
    init = (jax.tree.map(_zero_accumulator, marked_params), zero, zero, zero)
    head = (split_batch, token_shares)
    if overlap_last_microbatch:
        head = jax.tree.map(lambda x: x[:looped], head)
    (grad_sum, loss, cross_entropy_loss, aux_loss), stacked_router_metrics = jax.lax.scan(body, init, head)
    if not overlap_last_microbatch:
        router_metrics = _combine_microbatch_router_metrics(stacked_router_metrics, row_valid_tokens)
        grads = jax.tree.map(_finalize_accumulated_grad, grad_sum, params, grad_shardings)
        return loss, summarize_loss_metrics(router_metrics, cross_entropy_loss, aux_loss), grads

    # The last microbatch, outside the loop: its expert weights stay unmarked, and each layer's gradient plus the
    # earlier microbatches' sum is reduced inside the backward of the layer scan, as soon as that layer is done.
    last_microbatch, last_share = jax.tree.map(lambda x: x[looped], (split_batch, token_shares))
    # Start the last microbatch only after the loop: otherwise XLA may run its forward first and keep its activations
    # live through the whole loop.
    last_microbatch, grad_sum = jax.lax.optimization_barrier((last_microbatch, grad_sum))
    per_layer = _per_layer_reduced(params)
    last_params = jax.tree.map(lambda p, m, pl: p if pl else m, params, marked_params, per_layer)
    offsets = jax.tree.map(
        lambda acc, pl: unreduced_to_local(acc, leading=1) if pl else None,
        grad_sum.stacked_blocks.stacked,
        per_layer.stacked_blocks.stacked,
    )

    def sink(p, target, pl):
        # A zero in the gradient's sharding, for an expert weight whose gradient shards more than the weight.
        spec = jax.typeof(target).sharding.spec
        return jnp.zeros(p.shape, p.dtype, out_sharding=spec) if pl and spec != jax.typeof(p).sharding.spec else None

    sinks = jax.tree.map(
        sink, params.stacked_blocks.stacked, grad_shardings.stacked_blocks.stacked, per_layer.stacked_blocks.stacked
    )
    if all(x is None for x in jax.tree.leaves(sinks, is_leaf=lambda x: x is None)):
        sinks = None

    def last_loss_and_terms(p, block_sinks, microbatch, token_share):
        cross_entropy_sum, aux, router = loss_terms_fn(p, microbatch, offsets, block_sinks)
        cross_entropy = cross_entropy_sum / loss_denominator
        aux = token_share * aux
        return cross_entropy + aux, (cross_entropy, aux, router)

    (last_loss, (last_ce, last_aux, last_router)), (last_grads, sink_grads) = jax.value_and_grad(
        last_loss_and_terms, argnums=(0, 1), has_aux=True
    )(last_params, sinks, last_microbatch, last_share)
    if sinks is not None:
        # The sinks' gradients are the reduce-scattered expert gradients; the weights' own are zero.
        stacked = jax.tree.map(
            lambda g, sink_grad: g if sink_grad is None else sink_grad,
            last_grads.stacked_blocks.stacked,
            sink_grads,
        )
        last_grads = eqx.tree_at(lambda t: t.stacked_blocks.stacked, last_grads, stacked)
    stacked_router_metrics = jax.tree.map(
        lambda h, t: jnp.concatenate([h, t[None]], axis=0), stacked_router_metrics, last_router
    )
    router_metrics = _combine_microbatch_router_metrics(stacked_router_metrics, row_valid_tokens)

    def finalize(acc, g, p, target, pl):
        if pl:
            # Already reduced in the backward, with the accumulator added, into the sharding of ``target``.
            return reshard(g, jax.typeof(target).sharding)
        if not isinstance(acc, jax.Array):
            return acc
        return _finalize_accumulated_grad(acc + g.astype(jnp.float32), p, target)

    grads = jax.tree.map(finalize, grad_sum, last_grads, params, grad_shardings, per_layer)
    loss, cross_entropy_loss, aux_loss = loss + last_loss, cross_entropy_loss + last_ce, aux_loss + last_aux
    return loss, summarize_loss_metrics(router_metrics, cross_entropy_loss, aux_loss), grads


def _per_layer_reduced(params: Transformer) -> Transformer:
    """True for the stacked block weights sharded over ``expert`` (the expert weights): the last microbatch reduces
    their gradients one layer at a time inside the backward. Everything else is reduced after it, as for the loop.

    Not the expert weights the layer scan itself marks ``reduced`` (``replicated_grad_reduction="after_scan"`` with
    a size-1 ``expert`` axis and at most ``replicated_grad_max_bytes`` per layer): they are reduced after the scan
    anyway, so they take the same path as the other weights."""
    if params.stacked_blocks is None:
        raise ValueError("overlap_last_microbatch needs the stacked layer layout (use_array_stacked_blocks)")
    cfg = params.config
    max_bytes = cfg.replicated_grad_max_bytes if cfg.replicated_grad_reduction == "after_scan" else -1

    def per_layer(path, x):
        if not isinstance(x, jax.Array) or _EXPERT_AXIS not in jax.typeof(x).sharding.spec:
            return False
        return after_scan_reduced_spec(path, x, max_bytes) is None

    flags = jax.tree.map(lambda _: False, params)
    stacked = jax.tree_util.tree_map_with_path(per_layer, params.stacked_blocks.stacked)
    return eqx.tree_at(lambda t: t.stacked_blocks.stacked, flags, stacked)


def _mark_batch_reduced(params: Transformer) -> Transformer:
    """Mark every weight ``reduced`` over the batch axes it is not sharded over by ``expert``.

    Its gradient is then ``unreduced`` over them: each device's partial sum over its own tokens, which the caller
    reduces when it chooses. A weight sharded over a batch axis other than ``expert`` (the embedding and output
    projection over ``replica_dcn`` and ``data``, FSDP weights over ``data``) is all-gathered first; its gradient is
    then full size on every device. The expert weights keep their ``expert`` sharding, since only their own shard's
    tokens reach them. ``router_bias`` is left alone: it is added elementwise and has no gradient.
    """

    def mark(path, x):
        if not isinstance(x, jax.Array) or any(getattr(k, "name", None) == "router_bias" for k in path):
            return x
        spec = jax.typeof(x).sharding.spec
        partitions = []
        for entry in spec.partitions:
            axes = entry if isinstance(entry, tuple) else (entry,)
            kept = tuple(a for a in axes if a is not None and (a not in _BATCH_AXES or a == _EXPERT_AXIS))
            partitions.append(kept[0] if len(kept) == 1 else (kept or None))
        kept_axes = {a for entry in partitions for a in (entry if isinstance(entry, tuple) else (entry,))}
        # Every batch axis not kept, size-1 ones included: a dot's unreduced output names its contracting dim's axes.
        reduced = frozenset(a for a in _BATCH_AXES if a not in kept_axes)
        return reshard(x, P(*partitions, reduced=reduced))

    return jax.tree_util.tree_map_with_path(mark, params)


def _zero_accumulator(marked_param):
    """A float32 zero gradient sum with the sharding of ``marked_param``'s gradient: unreduced where it is reduced."""
    if not isinstance(marked_param, jax.Array):
        return marked_param
    spec = jax.typeof(marked_param).sharding.spec.to_ct_spec()
    return jnp.zeros(marked_param.shape, jnp.float32, out_sharding=spec)


def _finalize_accumulated_grad(grad_sum, param, target):
    """Reduce one accumulated gradient across devices into ``param``'s dtype and ``target``'s sharding.

    The float32 sum is cast to ``param``'s dtype (bfloat16 under a bfloat16 compute policy) before the reduction, so
    the step moves as many bytes as one microbatch's gradient did, not twice that."""
    if not isinstance(grad_sum, jax.Array):
        return grad_sum
    return reshard(grad_sum.astype(param.dtype), jax.typeof(target).sharding)


def _batch_shards(mesh) -> int:
    return math.prod(mesh.shape[axis] for axis in _BATCH_AXES)


def _check_per_device_parallelism(
    per_device_parallelism: int, train_batch_size: int | Sequence, *, microbatches: int, batch_shards: int
) -> None:
    """Fail fast when a launcher sets levanter's ``per_device_parallelism`` to a microbatch size the grug loop would
    not use: only ``GrugTrainerConfig.microbatches`` splits the batch. Unset (-1) and batch-size schedules pass."""
    if per_device_parallelism < 0 or not isinstance(train_batch_size, int):
        return
    if per_device_parallelism * batch_shards * microbatches != train_batch_size:
        raise ValueError(
            f"per_device_parallelism ({per_device_parallelism}) x batch shards ({batch_shards}) x microbatches "
            f"({microbatches}) must equal train_batch_size ({train_batch_size}); set GrugTrainerConfig.microbatches "
            "to accumulate gradients"
        )


def _tree_all_finite(tree: object) -> jax.Array:
    finite = jnp.array(True)
    for leaf in jax.tree.leaves(tree):
        if eqx.is_inexact_array(leaf):
            finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite


def _check_step_numerics(metrics: dict, step: int, diagnose_numerics: bool) -> None:
    diagnostics = (
        {key: float(metrics[key]) for key in (*_NUMERIC_DIAGNOSTIC_KEYS, "train/supervised_tokens")}
        if diagnose_numerics
        else {}
    )
    if not bool(jnp.isfinite(metrics["train/loss"])):
        raise FloatingPointError(f"Non-finite Grug loss at step {step}: {diagnostics}")
    if diagnose_numerics and not all(bool(metrics[key]) for key in _NUMERIC_DIAGNOSTIC_KEYS):
        raise FloatingPointError(f"Non-finite Grug update at step {step}: {diagnostics}")


def _eval_hooks(
    evaluator: TaggedEvaluator,
    *,
    prefix: str,
    eval_current: bool,
    eval_ema: bool,
    model_shardings: Transformer | None,
) -> list[Callable[..., None]]:
    """The eval hooks: one evaluating the current and/or the EMA model, or with ``model_shardings`` one per model, each
    gathering its model to ``model_shardings`` first (``_with_model_resharded``)."""
    if model_shardings is None:
        return [cb_tagged_evaluate(evaluator, prefix=prefix, eval_current=eval_current, eval_ema=eval_ema)]
    return [
        _with_model_resharded(
            cb_tagged_evaluate(evaluator, prefix=prefix, eval_current=not ema, eval_ema=ema), model_shardings, ema=ema
        )
        for ema, enabled in ((False, eval_current), (True, eval_ema))
        if enabled
    ]


def _with_model_resharded(hook: Callable[..., None], shardings: Transformer, *, ema: bool) -> Callable[..., None]:
    """``hook`` run with the step's current model (``ema``: its EMA model) resharded to ``shardings``, a tree of
    shardings like the model's, once per step.

    With ``zero1`` the float32 models shard the expert weights over the replicas; evaluating them as they are would
    all-gather every expert weight in every layer of every eval batch. This gathers the one model the hook evaluates
    once, in float32 as the evaluator runs it, and drops it when the hook returns, so one gathered model is live at a
    time. A step already evaluated (the forced call after the last step) gathers nothing."""

    @jax.jit
    def gather(model: Transformer) -> Transformer:
        return jax.tree.map(lambda x, s: reshard(x, s) if isinstance(x, jax.Array) else x, model, shardings)

    field_name = "eval_model" if ema else "model"
    last_step: int | None = None

    def resharded_hook(info: callbacks.StepInfo, force: bool = False) -> None:
        nonlocal last_step
        if info.step < 0 or info.step == last_step:
            return
        view = info.state
        model = gather(getattr(view, field_name))
        hook(dataclasses.replace(info, state=dataclasses.replace(view, **{field_name: model})), force)
        last_step = info.step

    return resharded_hook


@contextlib.contextmanager
def exit_if_hung(timeout: float | None):
    """Dump every thread's traceback and exit the process with status 1 if the block runs longer than ``timeout``."""
    if timeout is None:
        yield
        return
    faulthandler.dump_traceback_later(timeout, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _run_grug_local(config: GrugRunConfig) -> None:
    """Entry point for the grug template training loop."""
    trainer = config.trainer.trainer
    # Read before ``initialize`` fills it in from the batch size and levanter's own mesh, whose batch axis can differ.
    requested_per_device_parallelism = trainer.per_device_parallelism
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
        qb_sync_interval=config.trainer.qb_sync_interval,
        microbatches=config.trainer.microbatches,
        zero1=config.trainer.zero1,
        overlap_last_microbatch=config.trainer.overlap_last_microbatch,
    )

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    # Grug uses raw PartitionSpecs rather than Trainer's logical axis mapping.
    # Keep the mesh compact so the batch pspec derived by `_batch_spec(mesh)` spans slices directly.
    # replica_axis_size=None lets compact_grug_mesh default to jax.process_count() (full
    # cross-slice replication); set it to 1 on GrugTrainerConfig for cross-slice FSDP.
    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
        model_axis_size=config.trainer.model_axis_size,
    )
    _check_per_device_parallelism(
        requested_per_device_parallelism,
        trainer.train_batch_size,
        microbatches=config.trainer.microbatches,
        batch_shards=_batch_shards(mesh),
    )
    with set_mesh(mesh):
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
        )

        @jax.jit
        def _init_state(model_rng):
            return initial_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=model_rng,
                ema_beta=config.trainer.ema_beta,
                carry_compute_params=config.trainer.carry_compute_params,
                zero1=config.trainer.zero1,
            )

        # Checkpoints leave the compute copy out, so the restore exemplar does too: a run can resume from a checkpoint
        # saved with carry_compute_params on or off.
        state = _init_state(model_key)
        initial_compute_params = state.compute_params
        state = _checkpointed(state)

        checkpointer = trainer.checkpointer.create(run_id) if config.trainer.save_checkpoints else None
        if config.trainer.sft_weights_only_init:
            # SFT/RL: auto-resume from this run's own checkpoints if present (preemption),
            # otherwise load only base weights (+ pending_qb_betas) and keep the fresh
            # optimizer/step (marin #650). initialize_from is deliberately withheld here so
            # the restore never does a full-state load; the weights-only init runs below.
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

        # Rebuilt from the (possibly restored) parameters, so it never comes from a checkpoint, in the sharding of the
        # initial copy.
        state = rebuilt_compute_params(dataclasses.replace(state, compute_params=initial_compute_params), trainer.mp)
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

        state_callbacks = StateCallbackRunner[GrugTrainState](
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
        if evaluator is not None and eval_cfg is not None:
            interval = eval_cfg.steps_per_eval
            eval_ema = eval_cfg.eval_ema and config.trainer.ema_beta is not None
            if interval is not None and interval > 0 and (eval_cfg.eval_current or eval_ema):
                model_shardings = None
                if config.trainer.zero1:
                    assert state.compute_params is not None
                    model_shardings = jax.tree.map(lambda x: x.sharding, state.compute_params)
                for eval_hook in _eval_hooks(
                    evaluator,
                    prefix=eval_cfg.prefix,
                    eval_current=eval_cfg.eval_current,
                    eval_ema=eval_ema,
                    model_shardings=model_shardings,
                ):
                    state_callbacks.add_hook(eval_hook, every=interval)

        last_loss: float | jax.Array = 0.0
        last_step_duration = 0.0

        first_step = int(state.step)
        # Main optimization loop.
        try:
            while int(state.step) < trainer.num_train_steps:
                with jax.profiler.TraceAnnotation("load_batch"):
                    batch = next(iterator)
                step_start = time.perf_counter()
                current_step = int(state.step)
                # grad_watch runs only on its configured interval.
                compute_watch = (
                    watch_config.is_enabled and watch_config.interval > 0 and current_step % watch_config.interval == 0
                )
                # The first step compiles, so it gets no timeout; neither do watch steps, which can compile a variant.
                step_timeout = None if current_step == first_step or compute_watch else config.trainer.step_timeout
                with exit_if_hung(step_timeout):
                    state, metrics, watch_stats = train_step(state, batch, compute_watch=compute_watch)
                    step = int(state.step) - 1

                    jax.block_until_ready(metrics["train/loss"])

                _check_step_numerics(metrics, int(state.step), config.trainer.diagnose_numerics)
                duration = time.perf_counter() - step_start
                hook_start = time.perf_counter()
                with jax.profiler.TraceAnnotation("callbacks"):
                    state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                    last_loss = metrics["train/loss"]
                    last_step_duration = duration
                    levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=step)
                    levanter.tracker.log({"throughput/loading_time": iterator.this_load_time}, step=step)
                    router_metrics = {
                        key: value
                        for key, value in metrics.items()
                        if (key.startswith("train/router/") or key.startswith("moe_bias/"))
                        and key not in ("train/router/routing_counts_per_layer", "qb_beta_per_layer")
                    }
                    if router_metrics:
                        levanter.tracker.log(router_metrics, step=step)
                    if "train/cross_entropy_loss" in metrics:
                        levanter.tracker.log(
                            {"train/cross_entropy_loss": metrics["train/cross_entropy_loss"]},
                            step=step,
                        )

                    if watch_stats is not None:
                        levanter.tracker.log(watch_stats, step=step)

                if checkpointer is not None:
                    checkpointer.on_step(tree=_checkpointed(state), step=int(state.step))
        except BaseException:
            logger.exception(
                "Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause"
            )
            raise
        else:
            # Mirror classic trainer behavior: force callbacks on the last completed step.
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if checkpointer is not None:
                checkpointer.on_step(tree=_checkpointed(state), step=int(state.step), force=True)
                checkpointer.wait_until_finished()

    levanter.tracker.current_tracker().finish()


def run_grug(config: GrugRunConfig) -> None:
    """Dispatch grug training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=_run_grug_local,
        resources=config.resources,
    )


__all__ = [
    "GrugEvalConfig",
    "GrugRunConfig",
    "GrugTrainState",
    "GrugTrainerConfig",
    "initial_state",
    "run_grug",
]
