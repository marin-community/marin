# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import gc
import logging
import os
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import levanter.callbacks as callbacks
import levanter.tracker
import numpy as np
import optax
from fray.cluster import ResourceConfig
from haliax import Axis
from haliax.partitioning import set_mesh
from jax._src import config as jax_config
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
from levanter.data.dataset import AsyncDataset
from levanter.data.loader import DataLoader
from levanter.data.mixture import MixtureDataset, rescale_mixture_schedule_for_batch_schedule
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample, grug_lm_example_from_named
from levanter.eval import TaggedEvaluator, cb_tagged_evaluate, eval_model
from levanter.grug.grug_moe import MoeImplementation
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.store.jagged_array import set_jagged_array_read_cache_bytes
from levanter.trainer import TrainerConfig
from levanter.training_control import TrainingDashboard
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.grug.checkpointing import (
    restore_grug_state_from_checkpoint,
)
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.fast_track.model import (
    DenseMLP,
    GrugModelConfig,
    Transformer,
)

# This file intentionally mirrors `experiments/grug/base/train.py` with
# variant-specific model/loss/FLOP wiring, per the grug copy-first workflow in
# `.agents/skills/change-grug/`.

logger = logging.getLogger(__name__)

RUNTIME_ENV = {
    "LD_PRELOAD": "libjemalloc.so.2",
    "MALLOC_CONF": "background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0,narenas:2",
    # Per-process CUPTI sessions collide with each other and CoreWeave's DCGM, so PGLE only adds failure modes.
    "JAX_ENABLE_PGLE": "false",
    "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": "192",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
}
XLA_COLLECTIVE_OVERLAP_FLAG = "--xla_gpu_experimental_parallel_collective_overlap_limit"
DEFAULT_COLLECTIVE_OVERLAP_LIMIT = 4
DEFAULT_DROPLESS_MOE_IMPLEMENTATION: MoeImplementation = "sonic_cute"
# Full inline norm watch failed with overlap 4. Overlap 1 completed the selected full-watch gate.
INLINE_WATCH_COLLECTIVE_OVERLAP_LIMIT = 1
# TODO(https://github.com/marin-community/marin/issues/5675): Re-enable XLA GPU
# command buffers after the CUDA graph failure is fixed.
XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="


class WatchMode(StrEnum):
    """Where a watched training step computes gradient and parameter statistics."""

    INLINE = "inline"
    DIAGNOSTIC = "diagnostic"


def restore_template_from(state):
    """ShapeDtypeStructs carrying each leaf's concrete sharding, releasing the leaves.

    `jax.eval_shape` drops the `pinned_host` memory kind that `initial_state` puts on
    offloaded optimizer state, which is most of a large checkpoint. Reading the sharding off a
    built state keeps it.
    """
    template = jax.tree.map(
        lambda leaf: (
            jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding) if isinstance(leaf, jax.Array) else leaf
        ),
        state,
    )
    jax.tree.map(lambda leaf: leaf.delete() if isinstance(leaf, jax.Array) else None, state)
    gc.collect()
    return template


def _apply_runtime_defaults(*, inline_watch_enabled: bool) -> None:
    for name, value in RUNTIME_ENV.items():
        os.environ.setdefault(name, value)
    xla_flags = os.environ.get("XLA_FLAGS", "").split()
    overlap_limit = INLINE_WATCH_COLLECTIVE_OVERLAP_LIMIT if inline_watch_enabled else DEFAULT_COLLECTIVE_OVERLAP_LIMIT
    flag_defaults = (
        f"{XLA_COLLECTIVE_OVERLAP_FLAG}={overlap_limit}",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        # Size the jit_train_step temp arena below the allocator limit, leaving slack for fragmentation.
        "--xla_gpu_memory_limit_slop_factor=85",
        XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG,
    )
    explicit_names = {flag.partition("=")[0] for flag in xla_flags}
    xla_flags.extend(flag for flag in flag_defaults if flag.partition("=")[0] not in explicit_names)
    os.environ["XLA_FLAGS"] = " ".join(xla_flags)


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    data_seed: int | None = None
    log_every: int = 1
    z_loss_weight: float = 1e-4  # Weight on final-logit logsumexp z-loss stabilization term.
    # Inline watch computes statistics on every step and uses the watch interval only for logging.
    # This keeps one training executable resident. A diagnostic watch repeats forward and backward
    # in a separate executable, which costs compute but shortens gradient liveness.
    watch_mode: WatchMode = WatchMode.INLINE
    # A short throughput gate leaves this off. A compute-optimal run needs it: the loop already
    # restores from the latest committed checkpoint, so without a writer an interrupted run
    # restarts at step 0.
    save_checkpoints: bool = False

    # Grug builds its own compact (replica_dcn, data, expert, model) mesh instead of using
    # the Trainer's logical axis mapping; `data` absorbs whatever these two leave free.
    # Defaults reproduce the historical layout: no expert parallelism and full replication
    # across slices (replica_axis_size=None -> jax.process_count()), i.e. parameters
    # replicated per slice and sharded only over the intra-slice `data` axis. For a model
    # too large to replicate within one slice, set replica_axis_size=1 (FSDP across every
    # slice) and expert_axis_size>1 (expert parallelism over the intra-slice devices).
    expert_axis_size: int = 1
    replica_axis_size: int | None = None


@dataclass(frozen=True)
class GrugEvalConfig:
    """Perplexity eval settings for grug training."""

    eval_batch_size: int = 512
    steps_per_eval: int | None = 1000
    compute_bpb: bool = True
    # For expert-parallel runs, evaluate under the dropless local backend on an expert-collapsed
    # mesh, logging an `eval_dropless` macro loss. No-op when the mesh has no expert parallelism.
    dropless_eval: bool = False
    # Local MoE kernel used after collapsing the expert axis. ``sonic`` is the Hopper Triton path;
    # ``sonic_cute`` is the Blackwell QuACK/CUTLASS path.
    dropless_eval_moe_implementation: MoeImplementation = DEFAULT_DROPLESS_MOE_IMPLEMENTATION


@dataclass(frozen=True)
class GrugRunConfig:
    """Top-level config for grug training."""

    model: GrugModelConfig
    data: LmDataConfig
    resources: ResourceConfig
    tensorstore_cache_bytes: int | None = None
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    # Stop after this many steps while `trainer.num_train_steps` still sizes the learning-rate
    # schedule. Warmup and decay are fractions of `num_train_steps`, so training the head of a
    # long schedule requires the two to differ. None runs the whole schedule.
    stop_after_steps: int | None = None
    # GPU processes per task: > 1 runs one JAX process per GPU (multi-controller)
    # via the iris.hooks.multigpu_main supervisor instead of one process per node.
    processes_per_task: int = 1
    # Retry budgets for the training job. The two are separate gates and the job fails when either
    # one trips, thus raise them together. The defaults make a failure terminal, which is what a run
    # that cannot resume wants: a retry would repeat it from step 0. Only a run that both saves and
    # restores checkpoints benefits from a deep budget.
    max_retries_failure: int = 0
    max_task_failures: int = 10


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


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")
_TRAIN_LOADER_BUFFER_SIZE = 512
# On one GB200 tray, four-batch requests delivered the first data in 3.7s and sustained 3.4 batches/s.
_TRAIN_LOADER_FETCH_BATCH_SIZE = 4


def build_train_loader(
    dataset: AsyncDataset[GrugLmExample],
    *,
    batch_schedule: BatchSchedule,
    mesh: Mesh,
) -> DataLoader[GrugLmExample]:
    # DataLoader uses this batch axis mapping to shard batches across the distributed mesh.
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    return DataLoader(
        dataset,
        batch_schedule.schedule,
        max_buffered_batches=_TRAIN_LOADER_BUFFER_SIZE,
        mesh=mesh,
        axis_resources={"__BATCH__": _BATCH_AXES},
        fetch_batch_size=_TRAIN_LOADER_FETCH_BATCH_SIZE,
        batch_axis_name="__BATCH__",
        allow_nondivisible_batch_size=False,
    )


def _reshard_tree_to_mesh(tree, mesh: Mesh):
    """Move each array leaf onto ``mesh``, preserving its PartitionSpec.

    The train and eval meshes name the same axes (only the ``expert``/``data`` sizes differ), so a
    leaf's PartitionSpec is valid on both; ``jax.device_put`` performs the cross-mesh transfer. The
    model's own ``reshard`` calls fix the exact layout inside the forward, so any valid placement on
    the target mesh suffices here. Non-array leaves pass through.
    """

    def move(leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        spec = leaf.sharding.spec if isinstance(leaf.sharding, NamedSharding) else P()
        return jax.device_put(leaf, NamedSharding(mesh, spec))

    return jax.tree.map(move, tree)


def _to_dropless_local(
    model: Transformer, *, implementation: MoeImplementation = DEFAULT_DROPLESS_MOE_IMPLEMENTATION
) -> Transformer:
    """Swap every layer stack's MoE expert backend to the selected dropless local path.

    ``implementation``/``expert_chunks`` are static fields shared across a stacked block, so one
    replacement per stack covers every layer. The forward reads ``self.expert_mlp.implementation``
    (not the model config), so this alone routes the eval dropless. Must run on an expert-collapsed
    mesh: the local backend raises when the mesh expert axis is larger than one.
    """

    def stack_expert_mlps(m: Transformer) -> list:
        return [stack.stacked.mlp.expert_mlp for stack in m.layer_stacks()]

    dropless = [
        dataclasses.replace(expert_mlp, implementation=implementation, expert_chunks=1)
        for expert_mlp in stack_expert_mlps(model)
    ]
    return eqx.tree_at(stack_expert_mlps, model, dropless)


def build_tagged_evaluator(
    *,
    data_config: LmDataConfig,
    max_seq_len: int,
    mesh: Mesh,
    eval_cfg: GrugEvalConfig,
    mp: jmp.Policy,
    model_transform: Callable[[Transformer], Transformer] | None = None,
) -> TaggedEvaluator[LmExample | GrugLmExample, Transformer] | None:
    pos = Axis("position", max_seq_len)
    tagged_eval_sets = data_config.tagged_eval_sets(pos)
    if len(tagged_eval_sets) == 0:
        logger.warning("No evaluation datasets provided.")
        return None

    max_examples_per_dataset = None
    tokenizer = data_config.the_tokenizer if eval_cfg.compute_bpb else None
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    eval_axis_mapping = {"batch": _BATCH_AXES}
    eval_batch = Axis("batch", eval_cfg.eval_batch_size)
    eval_array_sharding = NamedSharding(mesh, P(_BATCH_AXES, None))

    def eval_loss_fn(model: Transformer, batch: LmExample | GrugLmExample) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Evaluate at the compute dtype, as the train step does at `mp.cast_to_compute(params)`.
        # Parameters are stored float32, and `gpu_fa4_cute` accepts only bf16/fp16, so without this
        # every eval raises `TypeError: ... supports only bf16/fp16, got float32` on Blackwell. The
        # reference attention path takes float32, which hid this on H100.
        model = mp.cast_to_compute(model)
        if model_transform is not None:
            model = model_transform(model)
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
        per_pos_token_id = jnp.pad(batch.tokens[:, 1:], ((0, 0), (0, 1)))
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


def _forced_final_global_flops(cfg: GrugModelConfig) -> float:
    """FLOPs/token for the forced-final-global layer that `lm_flops_per_token` misses.

    The model makes the last layer full-attention even when `num_layers` is not a multiple of
    `global_every` (see `_long_layer_schedule`), but the shared util counts only
    `num_layers // global_every` global layers. Return the local->global delta for that one layer
    (full-`seq_len` attention + global-KV projection, minus the sliding-window local layer it was
    counted as), or 0 when the depth is already a multiple of `global_every`.
    """
    sw, ge, seq = cfg.sliding_window, cfg.global_every, cfg.max_seq_len
    if not (sw is not None and ge and 0 < sw < seq and cfg.num_layers % ge != 0):
        return 0.0
    n, hd = cfg.num_heads, cfg.inferred_head_dim
    local_kv = cfg.local_kv_heads if cfg.local_kv_heads is not None else cfg.num_kv_heads
    global_kv = cfg.global_kv_heads if cfg.global_kv_heads is not None else cfg.num_kv_heads

    def qkv(kv: int) -> float:  # matches `_qkv_proj` in lm_flops_per_token
        return 2 * cfg.hidden_dim * (n * hd + 2 * kv * hd)

    def attn(span: int) -> float:  # matches `_attn_per_token` in lm_flops_per_token
        return ((2 * seq * span * n * hd) + (3 * seq * span * n) + (2 * seq * span * hd * n)) / seq

    return (qkv(global_kv) + attn(seq)) - (qkv(local_kv) + attn(sw))


def _compute_flops(
    *,
    model_config: GrugModelConfig,
) -> tuple[float, dict[str, float]]:
    # The dense variant runs a single GLU MLP of width `intermediate_dim` per layer -- no routed or
    # shared experts and no latent. Pricing it with the MoE terms (top-k + shared experts) overcounts
    # its FLOPs ~2x, which would inflate both its reported MFU and its scaling-law compute.
    if model_config.dense_mlp:
        flops_per_token = lm_flops_per_token(
            hidden_dim=model_config.hidden_dim,
            intermediate_dim=model_config.intermediate_dim,
            shared_intermediate_dim=0,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            num_heads=model_config.num_heads,
            seq_len=model_config.max_seq_len,
            vocab_size=model_config.vocab_size,
            glu=True,
            num_experts=1,
            num_shared_experts=0,
            num_experts_per_tok=1,
            sliding_window=model_config.sliding_window,
            global_every=model_config.global_every,
            local_kv_heads=model_config.local_kv_heads,
            global_kv_heads=model_config.global_kv_heads,
        )
    else:
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
            num_shared_experts=(
                model_config.num_shared_experts if model_config.shared_expert_intermediate_dim > 0 else 0
            ),
            num_experts_per_tok=model_config.num_experts_per_token,
            sliding_window=model_config.sliding_window,
            global_every=model_config.global_every,
            local_kv_heads=model_config.local_kv_heads,
            global_kv_heads=model_config.global_kv_heads,
        )
        # `lm_flops_per_token` prices every matmul at `hidden_dim`. Under LatentMoE the routed experts
        # live at `latent_dim` instead, and two projections are added per layer, so correct both terms
        # or MFU is overstated by roughly the compression ratio.
        if model_config.latent_dim is not None:
            latent, hidden = model_config.latent_dim, model_config.hidden_dim
            # Matches the routed term in `lm_flops_per_token`: 2 * 3 * width * intermediate * top_k.
            routed_delta = 2 * 3 * model_config.intermediate_dim * model_config.num_experts_per_token * (latent - hidden)
            # W_down (hidden -> latent) and W_up (latent -> hidden), once per token each.
            projection = 2 * 2 * hidden * latent
            flops_per_token += model_config.num_layers * (routed_delta + projection)

    # The last layer is forced global even when depth is not a multiple of `global_every`; add the one
    # global layer the shared util misses (applies to both dense and MoE).
    flops_per_token += _forced_final_global_flops(model_config)

    flops_per_example = 3 * flops_per_token * model_config.max_seq_len

    flops_summary: dict[str, float] = {
        "throughput/flops_per_token_analytic": flops_per_token,
        "throughput/flops_per_example_analytic": flops_per_example,
    }

    return flops_per_example, flops_summary


def log_device_memory(step_info) -> None:
    """Log this process's local-device HBM peak, live bytes, and allocator limit in GiB.

    The EP runs had no peak-HBM telemetry, which makes a whole class of result unreadable: XLA's
    ``HloRematerialization`` engages only when peak crosses the allocator limit, so a config change
    that moves peak across that boundary produces an MFU step change that has nothing to do with the
    change itself. Issue #8054 traced its own +9.08% headline to exactly this -- 3.69 GiB of peak
    took it under the limit and switched remat off -- and the win fell to +3.31% once one process
    per GPU put 8.79 GiB back. Any ablation that moves activation memory needs this logged, or its
    rungs cannot be told apart from allocator-limit crossings.

    """
    stats = jax.local_devices()[0].memory_stats()
    levanter.tracker.log(
        {
            "memory/peak_gib": stats["peak_bytes_in_use"] / 1024**3,
            "memory/in_use_gib": stats["bytes_in_use"] / 1024**3,
            "memory/limit_gib": stats["bytes_limit"] / 1024**3,
        },
        step=step_info.step,
    )


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
    master_params: Transformer | None
    ema_params: Transformer | None  # EMA of params for eval/checkpoint; None unless ema_beta is set.
    opt_state: optax.OptState
    pending_qb_betas: jax.Array


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array) -> Transformer:
    """Set router biases from QB betas (computed on previous step)."""
    if isinstance(model.stacked_blocks.stacked.mlp, DenseMLP):
        # Dense blocks have no router (QB routing dropped), so there is no router_bias to set.
        return model
    new_bias = -qb_betas
    new_bias = new_bias - jnp.mean(new_bias, axis=-1, keepdims=True)
    # qb_betas are in layer order; each stack takes the rows of its own layers.
    per_stack = [new_bias[np.asarray(indices)] for indices in model.stack_layer_indices()]
    return eqx.tree_at(lambda t: [stack.stacked.mlp.router_bias for stack in t.layer_stacks()], model, per_stack)


def _next_qb_betas(state: GrugTrainState, new_betas: jax.Array) -> jax.Array:
    """This step's QB betas, or the held ones once ``qb_freeze_step`` is reached."""
    freeze_step = state.params.config.qb_freeze_step
    if freeze_step is None:
        return new_betas
    return jnp.where(state.step + 1 >= freeze_step, state.pending_qb_betas, new_betas)


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None = None,
) -> GrugTrainState:
    initialized_params = Transformer.init(model_config, key=key)
    num_moe_layers = model_config.num_layers
    params = mp.cast_to_param(initialized_params)
    master_params = None
    opt_state = optimizer.init(params)
    return GrugTrainState(
        step=jax.sharding.reshard(jnp.array(0, dtype=jnp.int32), P()),
        params=params,
        master_params=master_params,
        ema_params=params if ema_beta is not None else None,
        opt_state=opt_state,
        pending_qb_betas=jnp.zeros((num_moe_layers, model_config.num_experts)),
    )


def _drop_metrics(
    dropped_assignments: jax.Array,
    sender_dropped_assignments: jax.Array,
    receiver_dropped_assignments: jax.Array,
    *,
    batch_size: int,
    sequence_length: int,
    top_k: int,
    num_layers: int,
) -> dict[str, int | float]:
    # Per-layer int32 counts summed over layers in int64 on the host: the global totals exceed int32 at
    # large batch (jax_enable_x64 is off, so an in-device sum would overflow), and float32 would round them.
    def _sum_int64(per_layer: jax.Array) -> int:
        return int(np.asarray(per_layer).astype(np.int64).sum())

    dropped_assignments_host = _sum_int64(dropped_assignments)
    sender_dropped_assignments_host = _sum_int64(sender_dropped_assignments)
    receiver_dropped_assignments_host = _sum_int64(receiver_dropped_assignments)
    if dropped_assignments_host != sender_dropped_assignments_host + receiver_dropped_assignments_host:
        raise ValueError("total dropped assignments must equal sender plus receiver dropped assignments")
    total_assignments = batch_size * sequence_length * top_k * num_layers
    receiver_assignments = total_assignments - sender_dropped_assignments_host
    return {
        "moe/dropped_assignments": dropped_assignments_host,
        "moe/drop_fraction": dropped_assignments_host / total_assignments,
        "moe/sender_dropped_assignments": sender_dropped_assignments_host,
        "moe/sender_drop_fraction": sender_dropped_assignments_host / total_assignments,
        "moe/receiver_dropped_assignments": receiver_dropped_assignments_host,
        "moe/receiver_drop_fraction": receiver_dropped_assignments_host / total_assignments,
        "moe/receiver_drop_fraction_of_received": receiver_dropped_assignments_host / max(receiver_assignments, 1),
    }


def _aux_loss_weight(model_config, step: jax.Array) -> jax.Array | None:
    """The early auxiliary LM loss weight at ``step``: linear from ``aux_lm_weight`` to 0 at ``aux_lm_steps``."""
    if model_config.aux_lm_layer is None:
        return None
    frac = jnp.clip(1.0 - step.astype(jnp.float32) / model_config.aux_lm_steps, 0.0, 1.0)
    return model_config.aux_lm_weight * frac


def _loss_and_grads(params, batch, mp: jmp.Policy, z_loss: float | None, step: jax.Array | None = None):
    aux_weight = None if step is None else _aux_loss_weight(params.config, step)
    grow_step = params.config.loop_grow_step
    loop_active = None if step is None or grow_step is None else step >= grow_step

    def loss_fn(model):
        compute_params = mp.cast_to_compute(model)
        return compute_params.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="mean",
            logsumexp_weight=z_loss,
            return_router_metrics=True,
            aux_loss_weight=aux_weight,
            loop_active=loop_active,
        )

    return jax.value_and_grad(loss_fn, has_aux=True)(params)


def _compute_diagnostic_watch_stats(params, batch, mp: jmp.Policy, z_loss: float | None, watch_config: WatchConfig):
    (_, _), grads = _loss_and_grads(params, batch, mp, z_loss)
    return compute_watch_stats(
        watch_targets=watch_config.watch_targets,
        include_norms=watch_config.include_norms,
        include_per_parameter_norms=watch_config.include_per_parameter_norms,
        include_histogram=watch_config.include_histograms,
        split_scan_layers=watch_config.split_scan_layers,
        params=params,
        grads=grads,
        model_tree_type=type(params),
    )


def _make_diagnostic_watch_step(mp: jmp.Policy, *, z_loss_weight: float, watch_config: WatchConfig):
    watch_targets = (
        tuple(t.strip() for t in watch_config.watch_targets.split(","))
        if isinstance(watch_config.watch_targets, str)
        else tuple(watch_config.watch_targets)
    )
    unsupported_targets = set(watch_targets) - {"grads", "params"}
    if unsupported_targets:
        raise ValueError(f"diagnostic watch does not support targets {sorted(unsupported_targets)}")
    diagnostic_watch_config = replace(watch_config, watch_targets=list(watch_targets))
    z_loss = z_loss_weight if z_loss_weight > 0 else None

    @jax.jit
    def diagnostic_watch_step(params: Transformer, batch, pending_qb_betas: jax.Array):
        params = _apply_qb_betas(params, pending_qb_betas)
        return _compute_diagnostic_watch_stats(params, batch, mp, z_loss, diagnostic_watch_config)

    return diagnostic_watch_step


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None = None,
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

    @functools.partial(jax.jit, donate_argnums=(0,))
    def train_step(state: GrugTrainState, batch):
        # Apply pending QB betas to router biases inside JIT (avoids eager
        # host-side kernel launches that can cause SPMD sync issues).
        qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)

        (loss, summarized_metrics), grads = _loss_and_grads(qb_params, batch, mp, z_loss, state.step)
        metrics = {"train/loss": loss, **summarized_metrics}
        opt_state_in = state.opt_state
        if os.environ.get("GRUG_SKIP_OPTIMIZER"):
            # MFU-baseline mode: run fwd+bwd (grads above) but skip the optimizer update (Muon NS).
            # Params/master/opt_state pass through unchanged, so loss does not descend -- this isolates
            # the optimizer's share of step time (MFU_skip vs MFU_full = the Muon overhead).
            updates = jax.tree_util.tree_map(jnp.zeros_like, qb_params)
            opt_state = opt_state_in
            params = qb_params
            master_params = state.master_params
        else:
            updates, opt_state = optimizer.update(grads, opt_state_in, qb_params)
            params = optax.apply_updates(qb_params, updates)
            master_params = None

        if ema_beta is None:
            ema_params = None
        else:
            # EMA tracks the QB-biased params, so re-apply the pending betas before blending.
            qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas)
            ema_params = jax.tree_util.tree_map(
                lambda old, new: ema_beta * old + (1.0 - ema_beta) * new, qb_ema_params, params
            )

        watch_stats = None
        if watch_config is not None:
            watch_stats = compute_watch_stats(
                watch_targets=watch_targets,
                include_norms=watch_config.include_norms,
                include_per_parameter_norms=watch_config.include_per_parameter_norms,
                include_histogram=watch_config.include_histograms,
                split_scan_layers=watch_config.split_scan_layers,
                params=qb_params,
                grads=grads,
                updates=updates,
                opt_state=opt_state_in,
                model_tree_type=type(state.params),
            )

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            master_params=master_params,
            ema_params=ema_params,
            opt_state=opt_state,
            # Dense blocks have no router, so the forward emits no qb_beta_per_layer; keep the
            # (zeros) pending betas -- _apply_qb_betas is already a no-op for dense.
            pending_qb_betas=_next_qb_betas(state, metrics.get("qb_beta_per_layer", state.pending_qb_betas)),
        )

        return next_state, metrics, watch_stats

    return train_step


def _run_grug_local(config: GrugRunConfig) -> None:
    """Entry point for the grug template training loop."""
    if config.tensorstore_cache_bytes is not None:
        set_jagged_array_read_cache_bytes(config.tensorstore_cache_bytes)

    trainer = config.trainer.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(config)

    run_id = trainer.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    optimizer = config.optimizer.build(trainer.num_train_steps)
    watch_config = trainer.watch
    diagnostic_watch_step = None
    inline_watch_config = watch_config if watch_config.is_enabled else None
    if watch_config.is_enabled and config.trainer.watch_mode == WatchMode.DIAGNOSTIC:
        diagnostic_watch_step = _make_diagnostic_watch_step(
            trainer.mp,
            z_loss_weight=config.trainer.z_loss_weight,
            watch_config=watch_config,
        )
        inline_watch_config = None
    train_step = _make_train_step(
        optimizer,
        trainer.mp,
        z_loss_weight=config.trainer.z_loss_weight,
        watch_config=inline_watch_config,
    )

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    # Grug uses raw PartitionSpecs rather than Trainer's logical axis mapping.
    # Keep the mesh compact so the batch pspec derived by `_batch_spec()` spans slices directly.
    # replica_axis_size=None lets compact_grug_mesh default to jax.process_count() (full
    # cross-slice replication); set it to 1 on GrugTrainerConfig for cross-slice FSDP.
    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
    )
    # Armed before the state is built or restored. The watchdog's step and process deadlines only
    # arm once a step reports progress, so its startup deadline is the only thing bounding a stall
    # in initialization, checkpoint restore, cache construction or compilation.
    progress_watchdog = trainer.progress_watchdog.create(process_index=jax.process_index())

    checkpointer = trainer.checkpointer.create(run_id) if config.trainer.save_checkpoints else None
    dashboard = (
        TrainingDashboard(config, checkpointer.request_checkpoint, run_id) if checkpointer is not None else nullcontext()
    )
    with set_mesh(mesh), dashboard:
        batch_schedule = trainer.batch_schedule

        @jax.jit
        def _init_state(model_rng):
            return initial_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=model_rng,
            )

        state = _init_state(model_key)
        released_initial_state = trainer.load_checkpoint is not False and not trainer.allow_partial_checkpoint
        if released_initial_state:
            state = restore_template_from(state)

        state = restore_grug_state_from_checkpoint(
            state,
            checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
            load_checkpoint_setting=trainer.load_checkpoint,
            mesh=mesh,
            allow_partial=trainer.allow_partial_checkpoint,
        )
        if released_initial_state and any(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(state)):
            state = _init_state(model_key)

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})

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

        flops_per_example, flops_summary = _compute_flops(model_config=config.model)
        levanter.tracker.log_summary(flops_summary)

        eval_cfg = config.eval
        evaluator = None
        dropless_evaluator = None
        dropless_eval_mesh = None
        if eval_cfg is not None:
            dropless = eval_cfg.dropless_eval and mesh.shape["expert"] > 1
            # EP runs score dropless (accurate) under the normal `eval` prefix; skip the lossy
            # train-mesh evaluator so it doesn't double-log capacity-dropped numbers under the same tag.
            train_mesh_eval = not dropless
            if train_mesh_eval:
                evaluator = build_tagged_evaluator(
                    data_config=config.data,
                    max_seq_len=config.model.max_seq_len,
                    mesh=mesh,
                    eval_cfg=eval_cfg,
                    mp=trainer.mp,
                )
            # Expert-parallel runs drop tokens over capacity; a second evaluator scores the same
            # weights dropless under the local backend on an expert-collapsed mesh (expert folded
            # into `data`), which the local backend requires. FSDP runs already have expert=1.
            if dropless:
                dropless_eval_mesh = compact_grug_mesh(
                    expert_axis_size=1,
                    replica_axis_size=mesh.shape["replica_dcn"],
                    model_axis_size=mesh.shape["model"],
                )
                # Build under the eval mesh so every constant the evaluator captures at construction
                # (e.g. `log2e`, the byte-per-token table, output shardings) is bound to the eval mesh
                # rather than the ambient train mesh; otherwise those leak a train-mesh aval into the
                # eval jit and fail the explicit-mesh check.
                with set_mesh(dropless_eval_mesh):
                    dropless_evaluator = build_tagged_evaluator(
                        data_config=config.data,
                        max_seq_len=config.model.max_seq_len,
                        mesh=dropless_eval_mesh,
                        eval_cfg=eval_cfg,
                        mp=trainer.mp,
                        model_transform=functools.partial(
                            _to_dropless_local,
                            implementation=eval_cfg.dropless_eval_moe_implementation,
                        ),
                    )

        # `trainer.num_train_steps` sizes the schedule; this bounds the run. Progress and the loop
        # both use it so a head-of-schedule run reports against the steps it will actually take.
        requested_stop_step = trainer.num_train_steps if config.stop_after_steps is None else config.stop_after_steps
        stop_step = min(requested_stop_step, trainer.num_train_steps)

        profiler_cfg = trainer.profiler
        profiler_num_steps = profiler_cfg.resolve_num_profile_steps(num_train_steps=stop_step)
        profiler_enabled = profiler_cfg.is_enabled and profiler_num_steps > 0

        log_every = max(1, config.trainer.log_every)
        batch_source = train_loader.iter_from_step(int(state.step))
        iterator = LoadingTimeTrackerIterator(batch_source)

        state_callbacks = StateCallbackRunner[GrugTrainState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            eval_model_getter=lambda s: s.params,
            opt_state_getter=lambda s: s.opt_state,
        )
        if progress_watchdog is not None:
            state_callbacks.add_hook(progress_watchdog, every=1)
        state_callbacks.add_hook(
            callbacks.log_performance_stats(config.model.max_seq_len, batch_schedule, flops_per_example),
            every=log_every,
        )
        state_callbacks.add_hook(callbacks.pbar_logger(total=stop_step), every=log_every)
        state_callbacks.add_hook(callbacks.log_step_info(stop_step), every=log_every)
        if profiler_enabled:
            state_callbacks.add_hook(
                profiler_cfg.build(
                    str(trainer.log_dir / run_id / "profiler"),
                    run_id=run_id,
                    num_steps=profiler_num_steps,
                ),
                every=1,
            )
        if train_dataset is not None:
            state_callbacks.add_hook(_make_mixture_stage_callback(train_dataset, batch_schedule), every=1)
        state_callbacks.add_hook(log_device_memory, every=1)
        if eval_cfg is not None:
            interval = eval_cfg.steps_per_eval
            eval_hooks: list[Callable[..., None]] = []
            if evaluator is not None:
                eval_hooks.append(
                    cb_tagged_evaluate(
                        evaluator,
                        prefix="eval",
                        eval_current=True,
                        eval_ema=False,
                    )
                )
            if dropless_evaluator is not None and dropless_eval_mesh is not None:
                # The training loop runs under `set_mesh(mesh)` (expert-parallel). The dropless
                # evaluator runs under the expert-collapsed mesh, so the model params -- sharded on
                # the train mesh -- must be resharded onto the eval mesh before its eval jit (JAX
                # does not auto-reshard across explicit meshes), then the local backend sees
                # expert=1. PGLE is disabled for the eval module as in `cb_tagged_evaluate`.
                # Log the dropless (accurate) eval under the normal prefix -- it is the primary eval
                # for EP runs (the lossy train-mesh evaluator is skipped above when dropless).
                dropless_prefix = "eval"
                # The forced end-of-run callback pass revisits the last step; skip it when the
                # periodic cadence already scored that step, as `cb_tagged_evaluate` does.
                last_dropless_eval_step: int | None = None

                def dropless_eval_hook(
                    step, *args, _mesh=dropless_eval_mesh, _ev=dropless_evaluator, _prefix=dropless_prefix, **kwargs
                ):
                    nonlocal last_dropless_eval_step
                    step_count = int(step.step)
                    if step_count < 0 or step_count == last_dropless_eval_step:
                        return
                    last_dropless_eval_step = step_count
                    # `model` must stay a local. The eval mesh has expert=1, so a leaf sharded on
                    # the expert axis lands replicated, and the copy is much larger than the
                    # train-mesh params. The train step needs almost the whole device budget for
                    # its temporary buffer, thus this copy must die before the next step.
                    with set_mesh(_mesh):
                        model = _reshard_tree_to_mesh(step.model, _mesh)
                        with jax_config.enable_pgle(False):
                            log_dict = eval_model(_ev, model, prefix=_prefix)
                        levanter.tracker.log(log_dict, step=step_count)

                eval_hooks.append(dropless_eval_hook)

            if interval is not None and interval > 0:
                for hook in eval_hooks:
                    state_callbacks.add_hook(hook, every=interval)

        last_loss: float | jax.Array = 0.0
        last_step_duration = 0.0

        # Main optimization loop.
        try:
            while int(state.step) < stop_step:
                with jax.profiler.TraceAnnotation("load_batch"):
                    batch = next(iterator)
                current_step = int(state.step)
                watch_due = (
                    watch_config.is_enabled and watch_config.interval > 0 and current_step % watch_config.interval == 0
                )
                if watch_due and diagnostic_watch_step is not None:
                    watch_stats = diagnostic_watch_step(state.params, batch, state.pending_qb_betas)
                    jax.block_until_ready(watch_stats)
                else:
                    watch_stats = None
                step_start = time.perf_counter()
                state_callbacks.emit_event(callbacks.ProgressEvent.TRAIN_STEP_STARTED)
                state, metrics, inline_watch_stats = train_step(state, batch)
                if inline_watch_stats is not None and watch_due:
                    watch_stats = inline_watch_stats
                step = int(state.step) - 1

                jax.block_until_ready(metrics["train/loss"])
                state_callbacks.emit_event(callbacks.ProgressEvent.TRAIN_STEP_FINISHED)

                if not jnp.isfinite(metrics["train/loss"]):
                    raise RuntimeError(f"Non-finite loss ({float(metrics['train/loss'])}) at step {int(state.step)}.")
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
                        if key.startswith(("train/router/", "moe_bias/", "train/attn_res/"))
                        and key not in ("train/router/routing_counts_per_layer", "qb_beta_per_layer")
                    }
                    if router_metrics:
                        levanter.tracker.log(router_metrics, step=step)
                    if "train/cross_entropy_loss" in metrics:
                        levanter.tracker.log(
                            {"train/cross_entropy_loss": metrics["train/cross_entropy_loss"]},
                            step=step,
                        )
                    if "moe/dropped_assignments" in metrics:
                        drop_metrics = _drop_metrics(
                            metrics["moe/dropped_assignments"],
                            metrics["moe/sender_dropped_assignments"],
                            metrics["moe/receiver_dropped_assignments"],
                            batch_size=batch.tokens.shape[0],
                            sequence_length=batch.tokens.shape[1],
                            top_k=config.model.num_experts_per_token,
                            num_layers=config.model.num_layers,
                        )
                        levanter.tracker.log(drop_metrics, step=step)

                    if watch_stats is not None:
                        levanter.tracker.log(watch_stats, step=step)

                if checkpointer is not None:
                    with callbacks.progress_event_scope(
                        state_callbacks.emit_event,
                        callbacks.ProgressEvent.CHECKPOINT_STARTED,
                        callbacks.ProgressEvent.CHECKPOINT_FINISHED,
                    ):
                        checkpointer.on_step(tree=state, step=int(state.step))

        except BaseException:
            logger.exception(
                "Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause"
            )
            raise
        else:
            # Mirror classic trainer behavior: force callbacks on the last completed step.
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if checkpointer is not None:
                with callbacks.progress_event_scope(
                    state_callbacks.emit_event,
                    callbacks.ProgressEvent.CHECKPOINT_STARTED,
                    callbacks.ProgressEvent.CHECKPOINT_FINISHED,
                ):
                    checkpointer.on_step(tree=state, step=int(state.step), force=True)
                    checkpointer.wait_until_finished()
        finally:
            state_callbacks.emit_event(callbacks.ProgressEvent.TRAINING_FINISHED)

    levanter.tracker.current_tracker().finish()


def run_grug(config: GrugRunConfig) -> None:
    """Dispatch grug training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    # Dispatch snapshots os.environ for the child task, so apply the runtime defaults first.
    inline_watch_enabled = trainer.watch.is_enabled and config.trainer.watch_mode == WatchMode.INLINE
    _apply_runtime_defaults(inline_watch_enabled=inline_watch_enabled)
    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=_run_grug_local,
        resources=config.resources,
        processes_per_task=config.processes_per_task,
        max_retries_failure=config.max_retries_failure,
        max_task_failures=config.max_task_failures,
    )


__all__ = [
    "GrugEvalConfig",
    "GrugRunConfig",
    "GrugTrainState",
    "GrugTrainerConfig",
    "initial_state",
    "run_grug",
]
