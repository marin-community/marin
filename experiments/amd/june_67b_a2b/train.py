# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June 67B-A2B training loop for the AMD copy of the model.

A copy of ``experiments/june_tpu_67b_a2b/moe/train.py`` on ``experiments.amd.june_67b_a2b.model.Transformer``. It
reuses June's data, eval, FLOP, checkpoint-restore and dispatch helpers. What differs:

- the trainer config adds ``microbatches`` and ``zero1`` to June's fields; with two or more microbatches every gradient
  is reduced across devices once per step, the expert weights' reductions overlap the last microbatch's backward, and
  with ``zero1`` the expert weights' parameters and optimizer state are sharded over the replicas and gathered in the
  next step's first microbatch (``_microbatched_loss_and_grads``);
- the state carries a compute-dtype copy of the parameters, which the step differentiates; checkpoints leave it out
  and a restore rebuilds it;
- the loop dispatches step N+1 before it fetches, checks and logs step N (``_train_loop``), and the step computes the
  loss's finiteness itself, so the host fetches each step's metrics in one transfer.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import levanter.callbacks as callbacks
import levanter.tracker
import optax
from fray.cluster import ResourceConfig
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.checkpoint import Checkpointer
from levanter.data.text.datasets import LmDataConfig
from levanter.eval import cb_tagged_evaluate
from levanter.grug.attention import AttentionMask, token_validity_from_attention_mask
from levanter.grug.sharding import compact_grug_mesh
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.schedule import IntSchedule
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.amd.june_67b_a2b.model import LossTerms, Transformer, summarize_loss_metrics
from experiments.amd.june_67b_a2b.parallel import (
    LayerRouting,
    batch_reduced_spec,
    batch_shards,
    is_zero1_shard,
    local_to_unreduced,
    mark_batch_reduced,
    per_layer_reduced,
    resharded_like,
    summed_in_scan,
    unreduced_to_local,
    zero1_sharded,
)
from experiments.june_tpu_67b_a2b.checkpointing import restore_grug_state_from_checkpoint
from experiments.june_tpu_67b_a2b.dispatch import dispatch_grug_training_run
from experiments.june_tpu_67b_a2b.moe.model import _BATCH_AXES, GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.train import (
    _NUMERIC_DIAGNOSTIC_KEYS,
    GrugEvalConfig,
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
_INPUTS_VALID_KEY = "train/inputs_valid"
# Checked every step: a step whose update is non-finite still has a finite loss.
_ALWAYS_FINITE_KEYS = ("train/params_finite", "train/qb_betas_finite")
# More skipped steps in a row than this means the data, not a damaged buffer, fails the input check.
_MAX_CONSECUTIVE_SKIPPED_STEPS = 3
# The hyperparameters (learning rate, ...) of the optimizer state a step returns, which the step-info hook logs.
_OPTIMIZER_HYPERPARAMS_KEY = "optimizer_hyperparams"


@dataclass(frozen=True)
class GrugTrainerConfig:
    """June's trainer config (``experiments.june_tpu_67b_a2b.moe.train.GrugTrainerConfig``) plus ``microbatches`` and
    ``zero1``."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    z_loss_weight: float = 0.0  # Weight on logsumexp (z-loss) stabilization term.
    diagnose_numerics: bool = False
    # Grug's own compact (replica_dcn, data, expert, context, model) mesh, as in June; `data` absorbs the rest.
    expert_axis_size: int = 1
    replica_axis_size: int | None = None
    model_axis_size: int = 1
    save_checkpoints: bool = True
    """False skips the checkpointer entirely, including the forced final save; for benchmarks."""
    step_timeout: float | None = None
    """Seconds after which a train step counts as hung (June's ``step_timeout``); None disables it."""
    sft_weights_only_init: bool = False
    """June's SFT/RL init (marin #650): without a checkpoint of its own, load only the weights from
    ``TrainerConfig.initialize_from``."""

    microbatches: int = 1
    """Split each step's batch into this many microbatches and run forward and backward on one at a time, so the
    activation memory is that of one microbatch. ``train_batch_size`` stays the batch of one step and must be a
    multiple of ``microbatches`` times the number of batch shards; each device takes every ``microbatches``-th of its
    own rows, so the split moves no data. Each gradient is summed over the microbatches in float32 on every device
    and reduced across devices once per step; the expert weights' reductions run layer by layer in the last
    microbatch's backward, overlapping the layers below. The loss, gradients and routing counts are those of the whole
    batch up to rounding, as long as no expert overflows its capacity, which applies per microbatch. The next step's
    QB thresholds are the token-weighted mean of the microbatches'. 1 runs the whole batch at once."""

    zero1: bool = False
    """ZeRO-1 over the replicas: shard the expert weights' float32 parameters, optimizer state and EMA over
    ``replica_dcn`` as well as ``expert``, so each replica keeps and updates 1/``replica_dcn`` of them. Each expert
    gradient is reduce-scattered into that shard, and the next step's first microbatch all-gathers the updated
    weights layer by layer, cast to the compute dtype, inside its forward layer scan, into the compute copy. Needs
    ``microbatches`` >= 2 and the expert count divisible by ``expert`` x ``replica_dcn``. Checkpoints keep the global
    shapes, so a run can resume with it on or off."""


@dataclass(frozen=True)
class GrugRunConfig:
    """June's run config with this trainer config."""

    model: GrugModelConfig
    data: LmDataConfig
    resources: ResourceConfig
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


@register_dataclass
@dataclass(frozen=True)
class GrugTrainState:
    """June's train state plus ``compute_params``, the compute-dtype copy of the parameters that the step
    differentiates. Checkpoints leave the copy out (``_checkpointed``), so they hold June's state.

    With ``zero1`` the copy's expert weights are those the step that returned it ran with, not the updated ones: the
    next step gathers the updated ones into it before reading them. Anything else that reads the copy first must
    rebuild it (``rebuilt_compute_params``)."""

    step: jax.Array
    params: Transformer
    opt_state: optax.OptState
    ema_params: Transformer | None
    pending_qb_betas: jax.Array
    compute_params: Transformer | None


def _checkpointed(state: GrugTrainState) -> GrugTrainState:
    """``state`` as checkpoints hold it: without the compute copy."""
    return dataclasses.replace(state, compute_params=None)


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array) -> Transformer:
    """Set router biases from QB betas (computed on previous step), in the dtype of the model's biases."""
    new_biases = -qb_betas
    new_biases = new_biases - jnp.mean(new_biases, axis=-1, keepdims=True)
    old_biases = model.stacked_blocks.stacked.mlp.router_bias
    return eqx.tree_at(lambda t: t.stacked_blocks.stacked.mlp.router_bias, model, new_biases.astype(old_biases.dtype))


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
    zero1: bool,
) -> GrugTrainState:
    """The step-0 state; with ``zero1`` the expert weights' parameters, optimizer state and EMA are sharded over the
    replicas, and the compute copy keeps the model's sharding."""
    params = mp.cast_to_param(Transformer.init(model_config, key=key))
    compute_params = mp.cast_to_compute(params)
    if zero1:
        params = zero1_sharded(params)
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params),
        ema_params=params if ema_beta is not None else None,
        pending_qb_betas=jnp.zeros((model_config.num_layers, model_config.num_experts)),
        compute_params=compute_params,
    )


def compute_shardings(state: GrugTrainState) -> Transformer:
    """The shardings of ``state``'s compute copy (the model's), which ``rebuilt_compute_params`` needs once the copy
    is dropped."""
    assert state.compute_params is not None
    return jax.tree.map(lambda x: x.sharding, state.compute_params)


def rebuilt_compute_params(state: GrugTrainState, mp: jmp.Policy, shardings: Transformer) -> GrugTrainState:
    """``state`` with its compute copy cast from its parameters, in ``shardings`` (``compute_shardings``): for a
    restored state, whose checkpoint has no copy. With ``zero1`` this all-gathers the expert weights."""
    compute_params = jax.jit(lambda p: jax.tree.map(reshard, mp.cast_to_compute(p), shardings))(state.params)
    return dataclasses.replace(state, compute_params=compute_params)


def _optimizer_hyperparams(opt_state: optax.OptState) -> dict[str, jax.Array]:
    """The hyperparameters ``log_optimizer_hyperparams`` logs from ``opt_state``: none unless the optimizer injects
    them (``optax.inject_hyperparams``)."""
    if isinstance(opt_state, optax.MultiStepsState):
        opt_state = opt_state.inner_opt_state
    return dict(getattr(opt_state, "hyperparams", {}))


def _batch_inputs_valid(batch, vocab_size: int) -> jax.Array:
    """Whether every array of ``batch`` holds values the data loader produces: token ids in ``[0, vocab_size)``, finite
    loss weights in ``[0, 1]``, and equal query and key segment ids in ``[-1, seq_len)``.

    On MI355X a stray device write occasionally overwrites 16 int32 words of a step's freshly allocated input buffers
    with float32 bit patterns (about once per 9,000 GPU-steps); the damaged words fall outside these ranges."""
    tokens = batch.tokens
    weights = batch.loss_weight
    valid = jnp.all((tokens >= 0) & (tokens < vocab_size))
    valid &= jnp.all(jnp.isfinite(weights) & (weights >= 0) & (weights <= 1))
    mask = batch.attn_mask
    if isinstance(mask, AttentionMask) and mask.segment_ids is not None:
        q_seg, kv_seg = mask.segment_ids
        valid &= jnp.all((q_seg == kv_seg) & (q_seg >= -1) & (q_seg < q_seg.shape[-1]))
    return valid


def _model_mask(mask: AttentionMask | None) -> AttentionMask | None:
    """``mask`` with its query segment ids used for the keys too. The loader's query and key ids are equal, and the
    step compares them in ``_batch_inputs_valid``; the model reads one array, so a damaged copy cannot abort the
    attention kernel's own equality check before the step can be skipped."""
    if not isinstance(mask, AttentionMask) or mask.segment_ids is None:
        return mask
    q_seg, _ = mask.segment_ids
    return dataclasses.replace(mask, segment_ids=(q_seg, q_seg))


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    vocab_size: int,
    z_loss_weight: float,
    ema_beta: float | None,
    microbatches: int,
    zero1: bool,
    watch_config: WatchConfig | None = None,
    diagnose_numerics: bool = False,
):
    if microbatches < 1:
        raise ValueError(f"microbatches must be at least 1, got {microbatches}")
    if zero1 and microbatches < 2:
        # With one microbatch the gradients are reduced in the backward, so ZeRO-1 would only add the gather.
        raise ValueError(f"zero1 needs microbatches >= 2, got {microbatches}")
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
        assert state.compute_params is not None, "the train step needs the compute copy"
        qb_compute_params = _apply_qb_betas(state.compute_params, state.pending_qb_betas)

        def loss_terms(params: Transformer, microbatch, routing: LayerRouting = LayerRouting()) -> LossTerms:
            return params.next_token_loss_terms(
                microbatch.tokens,
                microbatch.loss_weight,
                mask=_model_mask(microbatch.attn_mask),
                reduction="sum" if microbatches > 1 else "mean",
                logsumexp_weight=z_loss,
                routing=routing,
            )

        gathered_blocks = None
        if microbatches == 1:

            def loss_fn(params):
                terms = loss_terms(params, batch)
                loss = terms.cross_entropy_loss + terms.aux_loss
                return loss, summarize_loss_metrics(terms.router_metrics, terms.cross_entropy_loss, terms.aux_loss)

            (loss, summarized_metrics), compute_grads = jax.value_and_grad(loss_fn, has_aux=True)(qb_compute_params)
        else:
            loss, summarized_metrics, compute_grads, gathered_blocks = _microbatched_loss_and_grads(
                loss_terms,
                qb_compute_params,
                batch,
                microbatches,
                grad_shardings=qb_params,
                gather_sources=qb_params if zero1 else None,
            )
        grads = mp.cast_to_param(compute_grads)
        # Read the inputs after the backward, so the check sees any damage that reached the loss or gradients.
        grads, checked_batch = jax.lax.optimization_barrier((grads, batch))
        inputs_valid = _batch_inputs_valid(checked_batch, vocab_size)
        # The loss's finiteness is checked on the host from this flag, fetched with the other metrics.
        metrics = {
            "train/loss": loss,
            _LOSS_FINITE_KEY: jnp.isfinite(loss),
            _INPUTS_VALID_KEY: inputs_valid,
            **summarized_metrics,
        }
        updates, opt_state = optimizer.update(grads, state.opt_state, qb_params)
        params = optax.apply_updates(qb_params, updates)
        if diagnose_numerics:
            metrics.update(
                {
                    "train/supervised_tokens": jnp.sum(batch.loss_weight),
                    "train/grads_finite": _tree_all_finite(grads),
                    "train/updates_finite": _tree_all_finite(updates),
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

        compute_params = mp.cast_to_compute(params)
        if gathered_blocks is not None:
            # The buffers the first microbatch gathered into: the next step gathers the updated expert weights from
            # their ZeRO-1 shard into them again, so none are gathered here.
            stacked = jax.tree.map(
                lambda c, g: c if g is None else g,
                compute_params.stacked_blocks.stacked,
                gathered_blocks,
                is_leaf=lambda x: x is None,
            )
            compute_params = eqx.tree_at(lambda t: t.stacked_blocks.stacked, compute_params, stacked)
        updated = (
            params,
            opt_state,
            ema_params,
            metrics["qb_beta_per_layer"],
            resharded_like(compute_params, state.compute_params),
        )
        kept = (state.params, state.opt_state, state.ema_params, state.pending_qb_betas, state.compute_params)
        # A step whose inputs were damaged leaves the state as it was, apart from the step count: its batch is skipped.
        params, opt_state, ema_params, pending_qb_betas, compute_params = jax.tree.map(
            lambda new, old: jnp.where(inputs_valid, new, old), updated, kept
        )
        metrics[_OPTIMIZER_HYPERPARAMS_KEY] = _optimizer_hyperparams(opt_state)
        metrics["train/params_finite"] = _tree_all_finite(params)
        metrics["train/qb_betas_finite"] = jnp.all(jnp.isfinite(pending_qb_betas))
        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
            pending_qb_betas=pending_qb_betas,
            compute_params=compute_params,
        )
        return next_state, metrics, watch_stats

    return train_step


def _split_microbatches(x: jax.Array, microbatches: int) -> jax.Array:
    """[B, ...] batch-sharded -> [K, B/K, ...], microbatch k taking rows k, k + K, k + 2K, ...

    Every device holds a contiguous run of rows, so this takes every K-th of its own rows: no data moves, and each
    microbatch keeps the batch sharding with B/K rows. (A contiguous split would all-to-all the batch.)"""
    x = reshard(x, P(_BATCH_AXES))
    rows = jnp.reshape(x, (x.shape[0] // microbatches, microbatches, *x.shape[1:]), out_sharding=P(_BATCH_AXES))
    return jnp.swapaxes(rows, 0, 1)


def _routed_tokens(tokens: jax.Array, mask: AttentionMask | jax.Array | None) -> jax.Array:
    """The positions every MoE layer routes and counts as tokens: only the segment IDs of ``mask`` mark padding."""
    segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
    return token_validity_from_attention_mask(
        AttentionMask(is_causal=True, segment_ids=segment_ids),
        batch_size=tokens.shape[0],
        sequence_length=tokens.shape[1],
    )


def _combine_microbatch_router_metrics(stacked: dict[str, jax.Array], token_shares: jax.Array) -> dict[str, jax.Array]:
    """Per-layer router metrics stacked over K microbatches, [K, L, ...] -> those of the whole batch, [L, ...].

    ``token_shares[k]`` is microbatch k's share of the batch's routed (valid) tokens. The counts add up; the router
    z-loss, a mean over valid tokens, is the token-share-weighted sum of the microbatches', as for the whole batch. The
    QB thresholds are the token-share-weighted mean of the microbatches' thresholds, each estimated from its own
    tokens: the valid-token-weighted mean over shards the whole batch takes, over K times as many local estimates."""

    def weighted_sum(x: jax.Array) -> jax.Array:
        return jnp.sum(token_shares.reshape(-1, *(1,) * (x.ndim - 1)) * x, axis=0)

    qb_betas = stacked["qb_beta_per_layer"]
    return {
        "qb_beta_per_layer": reshard(weighted_sum(qb_betas), P(*jax.typeof(qb_betas).sharding.spec[1:])),
        "capacity_overflow_per_layer": jnp.sum(stacked["capacity_overflow_per_layer"], axis=0),
        "skipped_assignments_per_layer": jnp.sum(stacked["skipped_assignments_per_layer"], axis=0),
        "router_z_loss_per_layer": weighted_sum(stacked["router_z_loss_per_layer"]),
    }


def _block_sums(tree, flags: Transformer):
    """The stacked ``Block`` of ``tree``'s entries that ``flags`` flags, None elsewhere."""
    return jax.tree.map(lambda f, x: x if f else None, flags.stacked_blocks.stacked, tree.stacked_blocks.stacked)


def _with_block_sums(tree, sums, flags: Transformer):
    """``tree`` with the stacked block entries that ``flags`` flags replaced by those of ``sums``."""
    stacked = jax.tree.map(
        lambda f, x, s: s if f else x,
        flags.stacked_blocks.stacked,
        tree.stacked_blocks.stacked,
        sums,
        is_leaf=lambda x: x is None,
    )
    return eqx.tree_at(lambda t: t.stacked_blocks.stacked, tree, stacked, is_leaf=lambda x: x is None)


def _local_rows(x: jax.Array) -> jax.Array:
    return unreduced_to_local(x, leading=1)


def _unreduced_rows(x: jax.Array) -> jax.Array:
    return local_to_unreduced(x, leading=1)


def _grads_into_sums(fn, primals: tuple, sums):
    """Differentiate ``fn(*primals, sums) -> ((loss, sums_out), aux)``, whose layer scan adds each layer's gradients
    into ``sums`` in its backward (``LayerRouting.grad_sums``): seeding the cotangent of ``sums_out`` with ``sums``
    starts the backward from them. Returns ``(loss, aux, gradients of primals, new sums)``; the summed weights' own
    gradients are zero."""
    (loss, _), vjp_fn, aux = jax.vjp(fn, *primals, sums, has_aux=True)
    *grads, sums = vjp_fn((jnp.ones_like(loss), sums))
    return loss, aux, tuple(grads), sums


def _zero_accumulator(marked_param):
    """A float32 zero gradient sum with the sharding of ``marked_param``'s gradient: unreduced where it is reduced."""
    spec = jax.typeof(marked_param).sharding.spec.to_ct_spec()
    return jnp.zeros(marked_param.shape, jnp.float32, out_sharding=spec)


def _finalize_accumulated_grad(grad_sum: jax.Array, param: jax.Array, target: jax.Array) -> jax.Array:
    """Reduce one accumulated gradient across devices into ``param``'s dtype and ``target``'s sharding. The float32
    sum is cast first, so the reduction moves as many bytes as one microbatch's gradient would."""
    return reshard(grad_sum.astype(param.dtype), jax.typeof(target).sharding)


def _with_stacked_overlay(tree: Transformer, overlay) -> Transformer:
    """``tree`` with its stacked block leaves replaced by ``overlay``'s, where those are not None."""
    stacked = jax.tree.map(
        lambda x, o: x if o is None else o, tree.stacked_blocks.stacked, overlay, is_leaf=lambda x: x is None
    )
    return eqx.tree_at(lambda t: t.stacked_blocks.stacked, tree, stacked)


def _microbatched_loss_and_grads(
    loss_terms: Callable[..., LossTerms],
    params: Transformer,
    batch,
    microbatches: int,
    *,
    grad_shardings: Transformer,
    gather_sources: Transformer | None,
) -> tuple[jax.Array, dict, Transformer, object]:
    """Loss, summarized metrics and gradients of the whole ``batch``, one microbatch at a time, and the buffers the
    first microbatch gathered into (None without ``gather_sources``).

    ``loss_terms(params, microbatch, routing)`` returns the microbatch's ``LossTerms`` with a summed cross-entropy.
    Each microbatch differentiates its share of the whole batch's loss: its cross-entropy sum over the batch's total
    loss weight, plus its aux loss (a mean over valid tokens) weighted by its share of the batch's valid tokens.
    Weighting the loss rather than the gradient keeps the accumulation a plain ``sum + gradient``, and dividing by the
    batch's weight keeps a microbatch without loss weight from producing a 0/0 gradient.

    The gradients are reduced across devices once per step (``mark_batch_reduced``) and come back in the dtype of
    ``params`` and the sharding of ``grad_shardings``, the parameters the optimizer updates; the expert weights' are
    reduced layer by layer in the last microbatch's backward, as a reduce-scatter where ``grad_shardings`` shards them
    over the replicas (ZeRO-1). With ``gather_sources`` (ZeRO-1: the float32 parameters, the stacked expert weights in
    their ZeRO-1 shard), the first microbatch all-gathers each layer's slice of those weights into ``params`` in its
    forward scan, and the other microbatches read them from there.
    """
    batch_size = batch.tokens.shape[0]
    shards = batch_shards(get_abstract_mesh())
    if batch_size % (microbatches * shards) != 0:
        raise ValueError(
            f"batch size {batch_size} must be a multiple of microbatches ({microbatches}) times the batch shards "
            f"({shards})"
        )
    split_batch = jax.tree.map(lambda x: _split_microbatches(x, microbatches), batch)
    total_weight = jnp.sum(batch.loss_weight.astype(jnp.float32))
    loss_denominator = jnp.where(total_weight > 0, total_weight, 1.0)
    valid = _routed_tokens(batch.tokens, batch.attn_mask)
    valid_tokens = jnp.sum(_split_microbatches(valid, microbatches).astype(jnp.float32), axis=(1, 2))
    total_valid_tokens = jnp.sum(valid_tokens)
    token_shares = jnp.where(
        total_valid_tokens > 0, valid_tokens / jnp.maximum(total_valid_tokens, 1.0), 1.0 / microbatches
    )

    def weighted(terms: LossTerms, token_share):
        cross_entropy = terms.cross_entropy_loss / loss_denominator
        aux = token_share * terms.aux_loss
        return cross_entropy + aux, (cross_entropy, aux, terms.router_metrics)

    marked = mark_batch_reduced(params)
    summed = summed_in_scan(params)

    def body(carry, xs):
        grad_sum, loss_sum, cross_entropy_sum, aux_sum = carry
        microbatch, token_share = xs
        # When XLA unrolls this loop, the barrier keeps each microbatch after the previous one's gradients are summed,
        # so only one per-microbatch gradient tree is live at a time.
        grad_sum, microbatch, token_share = jax.lax.optimization_barrier((grad_sum, microbatch, token_share))

        def fn(p, sums):
            terms = loss_terms(p, microbatch, LayerRouting(grad_sums=sums))
            loss, aux = weighted(terms, token_share)
            return (loss, terms.routing.grad_sums), aux

        loss, (cross_entropy, aux, router_metrics), (grads,), sums = _grads_into_sums(
            fn, (marked,), _block_sums(grad_sum, summed)
        )
        grad_sum = jax.tree.map(lambda acc, g, s: acc if s else acc + g.astype(jnp.float32), grad_sum, grads, summed)
        grad_sum = _with_block_sums(grad_sum, sums, summed)
        carry = (grad_sum, loss_sum + loss, cross_entropy_sum + cross_entropy, aux_sum + aux)
        return carry, router_metrics

    zero = jnp.zeros((), jnp.float32)
    init_sums = jax.tree.map(_zero_accumulator, marked)
    init = (
        _with_block_sums(init_sums, jax.tree.map(_local_rows, _block_sums(init_sums, summed)), summed),
        zero,
        zero,
        zero,
    )
    looped = microbatches - 1
    head = jax.tree.map(lambda x: x[:looped], (split_batch, token_shares))
    first_router_metrics = None
    gathered_blocks = None
    if gather_sources is not None:
        gathered = jax.tree.map(is_zero1_shard, gather_sources.stacked_blocks.stacked)
        if not any(jax.tree.leaves(gathered)):
            raise ValueError("gather_sources needs the expert weights in their ZeRO-1 shard; none are")
        first_microbatch, first_share = jax.tree.map(lambda x: x[0], head)
        head = jax.tree.map(lambda x: x[1:], head)
        buffers = jax.tree.map(lambda p, g: p if g else None, params.stacked_blocks.stacked, gathered)
        # The shards stay unmarked: their slices are marked as they are gathered.
        first_stacked = jax.tree.map(
            lambda m, s, g: s if g else m, marked.stacked_blocks.stacked, gather_sources.stacked_blocks.stacked, gathered
        )
        first_params = eqx.tree_at(lambda t: t.stacked_blocks.stacked, marked, first_stacked)
        gather_sinks = jax.tree.map(
            lambda b: jnp.zeros(b.shape, b.dtype, out_sharding=batch_reduced_spec(jax.typeof(b).sharding.spec)),
            buffers,
        )

        def first_fn(p, sinks):
            # The buffers are written, not differentiated: the gathered weights' gradients leave through the sinks.
            terms = loss_terms(p, first_microbatch, LayerRouting(gather_sinks=sinks, gather_buffers=buffers))
            loss, aux = weighted(terms, first_share)
            return loss, (*aux, terms.routing.gather_buffers)

        (first_loss, (first_cross_entropy, first_aux, first_router_metrics, gathered_blocks)), grads = (
            jax.value_and_grad(first_fn, argnums=(0, 1), has_aux=True)(first_params, gather_sinks)
        )
        first_grads, sink_grads = grads
        first_grads = _with_stacked_overlay(first_grads, sink_grads)
        params = _with_stacked_overlay(params, gathered_blocks)
        marked = mark_batch_reduced(params)
        # Start the float32 sums only once the whole backward is done: converted as each gradient comes out (the
        # output projection's right after the cross-entropy), they would be live through the layer backward.
        first_grads = jax.lax.optimization_barrier(first_grads)
        sums = jax.tree.map(lambda g: g.astype(jnp.float32), first_grads)
        sums = _with_block_sums(sums, jax.tree.map(_local_rows, _block_sums(sums, summed)), summed)
        init = (sums, first_loss, first_cross_entropy, first_aux)
    if jax.tree.leaves(head)[0].shape[0] > 0:
        (grad_sum, loss, cross_entropy_loss, aux_loss), stacked_router_metrics = jax.lax.scan(body, init, head)
        if first_router_metrics is not None:
            stacked_router_metrics = jax.tree.map(
                lambda f, h: jnp.concatenate([f[None], h], axis=0), first_router_metrics, stacked_router_metrics
            )
    else:
        # Only the first microbatch ran before the last one.
        grad_sum, loss, cross_entropy_loss, aux_loss = init
        stacked_router_metrics = jax.tree.map(lambda f: f[None], first_router_metrics)

    # The last microbatch, after the loop: its expert weights stay unmarked, and each layer's gradient plus the
    # earlier microbatches' sum is reduced inside the backward of the layer scan, as soon as that layer is done.
    last_microbatch, last_share = jax.tree.map(lambda x: x[looped], (split_batch, token_shares))
    # Start the last microbatch only after the loop: otherwise XLA may run its forward first and keep its
    # activations live through the loop.
    last_microbatch, grad_sum = jax.lax.optimization_barrier((last_microbatch, grad_sum))
    per_layer = per_layer_reduced(params)
    last_params = jax.tree.map(lambda p, m, pl: p if pl else m, params, marked, per_layer)
    # The per-layer weights' sums, already in the offsets' layout.
    offsets = jax.tree.map(
        lambda acc, pl: acc if pl else None, grad_sum.stacked_blocks.stacked, per_layer.stacked_blocks.stacked
    )

    def sink(p, target, pl):
        # A zero in the gradient's sharding, for an expert weight whose gradient shards more than the weight.
        spec = jax.typeof(target).sharding.spec
        return jnp.zeros(p.shape, p.dtype, out_sharding=spec) if pl and spec != jax.typeof(p).sharding.spec else None

    sinks = jax.tree.map(
        sink, params.stacked_blocks.stacked, grad_shardings.stacked_blocks.stacked, per_layer.stacked_blocks.stacked
    )
    # The other summed weights carry their sums through the last microbatch's scan, which adds its gradients.
    last_summed = jax.tree.map(lambda s, pl: s and not pl, summed, per_layer)

    def last_fn(p, block_sinks, sums):
        terms = loss_terms(
            p, last_microbatch, LayerRouting(grad_offsets=offsets, grad_sinks=block_sinks, grad_sums=sums)
        )
        loss, aux = weighted(terms, last_share)
        return (loss, terms.routing.grad_sums), aux

    last_loss, (last_cross_entropy, last_aux, last_router_metrics), (last_grads, sink_grads), last_sums = (
        _grads_into_sums(last_fn, (last_params, sinks), _block_sums(grad_sum, last_summed))
    )
    # The sinks' gradients are the reduce-scattered expert gradients; the weights' own are zero.
    last_grads = _with_stacked_overlay(last_grads, sink_grads)
    totals = _with_block_sums(
        jax.tree.map(lambda _: None, params), jax.tree.map(_unreduced_rows, last_sums), last_summed
    )
    stacked_router_metrics = jax.tree.map(
        lambda h, t: jnp.concatenate([h, t[None]], axis=0), stacked_router_metrics, last_router_metrics
    )
    router_metrics = _combine_microbatch_router_metrics(stacked_router_metrics, token_shares)

    def finalize(acc, g, p, target, pl, total):
        if pl:
            # Already reduced in the backward, with the sum added, into the sharding of ``target``.
            return reshard(g, jax.typeof(target).sharding)
        if total is not None:
            # Summed in the scan, the last microbatch's gradient included.
            return _finalize_accumulated_grad(total, p, target)
        return _finalize_accumulated_grad(acc + g.astype(jnp.float32), p, target)

    grads = jax.tree.map(
        finalize, grad_sum, last_grads, params, grad_shardings, per_layer, totals, is_leaf=lambda x: x is None
    )
    loss = loss + last_loss
    cross_entropy_loss = cross_entropy_loss + last_cross_entropy
    aux_loss = aux_loss + last_aux
    return loss, summarize_loss_metrics(router_metrics, cross_entropy_loss, aux_loss), grads, gathered_blocks


def _check_per_device_parallelism(
    per_device_parallelism: int, train_batch_size: int | IntSchedule, *, microbatches: int, batch_shards: int
) -> None:
    """Fail fast when a launcher sets Levanter's ``per_device_parallelism`` to a microbatch size this loop would not
    use: only ``GrugTrainerConfig.microbatches`` splits the batch. Every phase of a batch-size schedule must agree.
    Unset (-1) passes."""
    if per_device_parallelism < 0:
        return
    sizes = [train_batch_size] if isinstance(train_batch_size, int) else [phase.value for phase in train_batch_size]
    for size in sizes:
        if per_device_parallelism * batch_shards * microbatches != size:
            raise ValueError(
                f"per_device_parallelism ({per_device_parallelism}) x batch shards ({batch_shards}) x microbatches "
                f"({microbatches}) must equal the train batch size ({size}); set GrugTrainerConfig.microbatches "
                "to accumulate gradients"
            )


def _check_step_numerics(metrics: dict, step: int, diagnose_numerics: bool) -> bool:
    """Raise if the state a step returned has non-finite parameters or QB betas, or if a step that applied its update
    has a non-finite loss (with ``diagnose_numerics`` also non-finite gradients or updates). Return whether the step
    applied its update: False when its inputs failed ``_batch_inputs_valid`` and it kept the previous state.
    ``metrics`` are the step's host metrics (``_fetch_step_metrics``): the flags were computed in the step."""
    diagnostics = (
        {key: float(metrics[key]) for key in (*_NUMERIC_DIAGNOSTIC_KEYS, "train/supervised_tokens")}
        if diagnose_numerics
        else {}
    )
    if not all(bool(metrics[key]) for key in _ALWAYS_FINITE_KEYS):
        raise FloatingPointError(f"Non-finite Grug state at step {step}: {diagnostics}")
    if not bool(metrics[_INPUTS_VALID_KEY]):
        return False
    if not bool(metrics[_LOSS_FINITE_KEY]):
        raise FloatingPointError(f"Non-finite Grug loss at step {step}: {diagnostics}")
    if diagnose_numerics and not all(bool(metrics[key]) for key in _NUMERIC_DIAGNOSTIC_KEYS):
        raise FloatingPointError(f"Non-finite Grug update at step {step}: {diagnostics}")
    return True


def _is_router_metric(key: str) -> bool:
    return (key.startswith("train/router/") or key.startswith("moe_bias/")) and key != "qb_beta_per_layer"


def _fetch_step_metrics(metrics: dict, diagnose_numerics: bool) -> dict:
    """The metrics of a step the loop checks and logs, copied to the host in one transfer, which waits for the step."""
    keys = [
        "train/loss",
        _LOSS_FINITE_KEY,
        _INPUTS_VALID_KEY,
        *_ALWAYS_FINITE_KEYS,
        _OPTIMIZER_HYPERPARAMS_KEY,
        "train/cross_entropy_loss",
    ]
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
    consecutive_skipped = 0

    def fetch(record: _DispatchedStep) -> dict:
        if record.host_metrics is None:
            record.host_metrics = _fetch_step_metrics(record.metrics, diagnose_numerics)
            record.done_time = time.perf_counter()
        return record.host_metrics

    def finish(record: _DispatchedStep, returned_state: GrugTrainState | None) -> None:
        """Check, hook and log a fetched step; ``returned_state`` is the state it returned, None once donated."""
        nonlocal last_done, last_loss, last_duration, last_hook_start, consecutive_skipped
        host = fetch(record)
        assert record.done_time is not None
        duration = record.done_time - max(record.dispatch_time, last_done)
        last_done = record.done_time
        if _check_step_numerics(host, record.step + 1, diagnose_numerics):
            consecutive_skipped = 0
        else:
            consecutive_skipped += 1
            logger.warning(
                "Step %d skipped: its input buffers hold values the data loader cannot produce (loss %s); the state "
                "is unchanged",
                record.step,
                float(host["train/loss"]),
            )
            if consecutive_skipped > _MAX_CONSECUTIVE_SKIPPED_STEPS:
                raise ValueError(
                    f"{consecutive_skipped} consecutive steps up to {record.step} failed the input check; the data, "
                    "not a damaged buffer, is out of range"
                )
        levanter.tracker.log({"train/skipped_step": float(not host[_INPUTS_VALID_KEY])}, step=record.step)
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
                checkpointer.on_step(tree=_checkpointed(state), step=next_step)
    except BaseException:
        logger.exception("Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause")
        raise
    # Mirror classic trainer behavior: force callbacks on the last completed step.
    state_callbacks.run(state, loss=last_loss, step_duration=last_duration, force=True)
    if checkpointer is not None:
        # Unless the last step's state is already saved for good (a step-policy save): it would be written again.
        checkpointer.save_final_checkpoint(tree=_checkpointed(state), step=next_step)
        checkpointer.wait_until_finished()
    return state


def _run_grug_local(config: GrugRunConfig) -> None:
    """Entry point for the training loop."""
    trainer = config.trainer.trainer
    # Read before ``initialize`` fills it in from the batch size and Levanter's own mesh, whose batch axes differ.
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
        vocab_size=config.model.vocab_size,
        z_loss_weight=config.trainer.z_loss_weight,
        ema_beta=config.trainer.ema_beta,
        watch_config=watch_config if watch_config.is_enabled else None,
        diagnose_numerics=config.trainer.diagnose_numerics,
        microbatches=config.trainer.microbatches,
        zero1=config.trainer.zero1,
    )

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
        model_axis_size=config.trainer.model_axis_size,
    )
    _check_per_device_parallelism(
        requested_per_device_parallelism,
        trainer.train_batch_size,
        microbatches=config.trainer.microbatches,
        batch_shards=batch_shards(mesh),
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
                zero1=config.trainer.zero1,
            )

        # Checkpoints leave the compute copy out, so the restore's exemplar does too. Only its shardings are kept, so
        # the initial copy is freed here; it is rebuilt below.
        state = _init_state(model_key)
        shardings = compute_shardings(state)
        state = _checkpointed(state)

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

        state = rebuilt_compute_params(state, trainer.mp, shardings)
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
