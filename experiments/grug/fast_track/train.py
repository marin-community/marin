# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import dataclasses
import functools
import gc
import glob
import logging
import os
import re
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from enum import StrEnum

import equinox as eqx
import fsspec
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
from levanter.grug._moe.ep_ragged_all_to_all import RAGGED_NCCL_SEND_RECV_XLA_FLAGS, RAGGED_REQUIRED_XLA_FLAGS
from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import MoeImplementation
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.pipeline import reshape_batch_into_microbatches
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
from experiments.grug.fast_track.byte_targets import token_byte_table
from experiments.grug.fast_track.grad_capture import CaptureWriter, capture_matrices, capture_steps
from experiments.grug.fast_track.host_stall import HostStallSampler
from experiments.grug.fast_track.model import (
    FINAL_HIDDEN_KEY,
    NEWTON_GRAM_KEY,
    DenseMLP,
    GrugModelConfig,
    HeadReplay,
    MtpMode,
    Transformer,
    ngram_stat_table_add,
    tie_routers,
    write_ngram_stats,
)
from experiments.grug.fast_track.optimizer import magma_metrics, optimizer_diagnostics

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
# XLA's default command-buffer (CUDA graph) set with collectives left eager: the hang in
# https://github.com/marin-community/marin/issues/5675 bisects to the COLLECTIVES capture set. At d512
# (lc1-cmdbuf-full, 2817 steps) this is +1.8% ex/s at the same loss; the step is host-launch bound.
XLA_GPU_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN"
RAGGED_MOE_IMPLEMENTATION = "ragged_all_to_all"
# As in moe_hero_ep: the ragged dispatch and combine form one long dependent chain, so admitting several
# concurrent collectives only contends for the SMs the transport itself needs.
RAGGED_COLLECTIVE_OVERLAP_LIMIT = 1


class RaggedTransport(StrEnum):
    """XLA GPU kernel behind ``ragged_all_to_all`` (all three ship in the stock x86_64 PJRT plugin).

    ``DEVICE`` and ``ONE_SHOT`` write straight into peer GPUs' buffers, which XLA only allows when one process
    owns every GPU of the collective (``processes_per_task=1``); ``NCCL`` also runs with one process per GPU.
    """

    DEVICE = "device"
    """Device-initiated NCCL GIN + LSA kernel on NCCL symmetric buffers: the GB200 hero's transport."""
    ONE_SHOT = "one_shot"
    """XLA's stock default: the host-launched one-shot copy kernel over peer pointers."""
    NCCL = "nccl"
    """NCCL send/recv per (peer, expert) update."""


RAGGED_TRANSPORT_XLA_FLAGS: dict[RaggedTransport, tuple[str, ...]] = {
    RaggedTransport.DEVICE: RAGGED_REQUIRED_XLA_FLAGS,
    RaggedTransport.ONE_SHOT: (
        "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=false",
        "--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=true",
    ),
    RaggedTransport.NCCL: RAGGED_NCCL_SEND_RECV_XLA_FLAGS,
}
_RAGGED_TRANSPORT_FLAG_NAMES = frozenset(
    flag.partition("=")[0] for flags in RAGGED_TRANSPORT_XLA_FLAGS.values() for flag in flags
)


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


def _apply_runtime_defaults(*, inline_watch_enabled: bool, ragged_transport: RaggedTransport | None) -> None:
    """Set the runtime env and XLA flag defaults; ``ragged_transport`` is None unless the MoE is ragged."""
    for name, value in RUNTIME_ENV.items():
        os.environ.setdefault(name, value)
    xla_flags = os.environ.get("XLA_FLAGS", "").split()
    if ragged_transport is not None:
        overlap_limit = RAGGED_COLLECTIVE_OVERLAP_LIMIT
    elif inline_watch_enabled:
        overlap_limit = INLINE_WATCH_COLLECTIVE_OVERLAP_LIMIT
    else:
        overlap_limit = DEFAULT_COLLECTIVE_OVERLAP_LIMIT
    flag_defaults = (
        f"{XLA_COLLECTIVE_OVERLAP_FLAG}={overlap_limit}",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        # Size the jit_train_step temp arena below the allocator limit, leaving slack for fragmentation.
        "--xla_gpu_memory_limit_slop_factor=85",
        XLA_GPU_COMMAND_BUFFER_FLAG,
    )
    explicit_names = {flag.partition("=")[0] for flag in xla_flags}
    xla_flags.extend(flag for flag in flag_defaults if flag.partition("=")[0] not in explicit_names)
    if ragged_transport is not None:
        # The transport is selected by ``ragged_transport`` alone: a stray kernel flag in XLA_FLAGS would mix
        # kernels, so drop it rather than rely on which occurrence XLA's parser keeps.
        xla_flags = [f for f in xla_flags if f.partition("=")[0] not in _RAGGED_TRANSPORT_FLAG_NAMES]
        xla_flags.extend(RAGGED_TRANSPORT_XLA_FLAGS[ragged_transport])
    os.environ["XLA_FLAGS"] = " ".join(xla_flags)


@contextmanager
def _pgle_disabled():
    """Turn PGLE off process-wide for the duration. ``jax_config.enable_pgle(False)`` is thread-local, so the
    data loader's background thread would keep profiling its batch jit under PGLE and abort mid-eval."""
    prev = jax.config.jax_enable_pgle
    jax.config.update("jax_enable_pgle", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_pgle", prev)


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
    # Write the compiled (optimized) train-step HLO text here after the first step, for profile attribution.
    hlo_dump_path: str | None = None
    # Dump XLA's buffer assignment and memory-usage report for the train step and upload them here (process 0),
    # also when the step fails, e.g. with an out-of-memory error: attributes the temp buffer to HLO values.
    xla_memory_report_path: str | None = None
    # Weight EMA over the last ``ema_last_steps`` steps (the whole run when None): until then the EMA
    # tracks the params, afterwards ``ema <- ema_beta * ema + (1 - ema_beta) * params``. Evals from the
    # EMA start on score the EMA. None: no EMA.
    ema_beta: float | None = None
    ema_last_steps: int | None = None
    # After training, also evaluate ``a * ema + (1 - a) * params`` for each ``a`` here (logged under
    # ``eval_blend<a>/``), to sweep how much of the EMA to keep.
    ema_blend_sweep: tuple[float, ...] = ()
    # Per-group blend probe: with every group at ``ema_group_base_blend``, also evaluate each group of
    # ``EMA_BLEND_GROUPS`` at 0 (raw final weights) and at 1 (pure EMA), logged as ``eval_blend_<group><a>/``.
    # Head replay: every ``head_replay_period`` steps, store the batch's final hidden states and labels in one
    # of ``head_replay_slots`` slots; each step also trains the lm_head on the oldest slot at
    # ``head_replay_scale`` x its CE (exact head gradient at the current weights; the stored hidden is a
    # stale view of that data). 0 slots: off.
    head_replay_slots: int = 0
    head_replay_period: int = 100
    head_replay_scale: float = 0.1
    ema_group_sweep: bool = False
    ema_group_base_blend: float = 0.5
    # Before step 0, fill the model's n-gram statistic table (``ngram_stat_rows``) from this many batches of the
    # training stream taken *after* the run's last step, i.e. tokens the run never trains on. Untimed: it runs
    # before the loop. 0: the table starts empty and fills online from the batches the run trains on.
    ngram_stat_prefill_batches: int = 0
    # With ``lm_head_unigram_bias``: before step 0, set the lm_head bias to the log unigram frequencies of the
    # first this many training batches (untimed, before the loop).
    lm_head_unigram_batches: int = 64
    # Expert-specialization dump: once this many steps have completed (0: before the first step; a step past
    # the end: after the last one), route ``routing_dump_batches`` train-size batches of fixed held-out
    # sequences (the validation sets round-robin, see ``_routing_dump_sequences``) and write per-layer
    # (token, expert) assignment counts to ``<routing_dump_path>/routing_step<N>.npz`` (``_routing_dumper``).
    routing_dump_steps: tuple[int, ...] = ()
    routing_dump_batches: int = 8
    routing_dump_path: str | None = None
    # Optimizer diagnostics (``grad_capture.py``): for ``grad_capture_len`` steps from each start, write the raw
    # gradient and applied update of the captured matrices to ``<grad_capture_path>/grad_capture_step<N>.npz``.
    grad_capture_starts: tuple[int, ...] = ()
    grad_capture_len: int = 48
    grad_capture_path: str | None = None
    # Split each train batch into this many equal microbatches, run forward and backward on one at a time and
    # average their gradients: the same update with 1/k of the activation memory, at some speed cost. MoE
    # capacity and routing statistics then apply per microbatch. 1: off.
    grad_accum_microbatches: int = 1

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
    # XLA kernel behind the ragged all-to-all; read only when `model.moe_implementation` is ragged.
    ragged_transport: RaggedTransport = RaggedTransport.DEVICE
    # Retry budgets for the training job. The two are separate gates and the job fails when either
    # one trips, thus raise them together. The defaults make a failure terminal, which is what a run
    # that cannot resume wants: a retry would repeat it from step 0. Only a run that both saves and
    # restores checkpoints benefits from a deep budget.
    max_retries_failure: int = 0
    max_task_failures: int = 10

    def __post_init__(self) -> None:
        if (
            self.model.moe_implementation == RAGGED_MOE_IMPLEMENTATION
            and self.ragged_transport is not RaggedTransport.NCCL
            and self.processes_per_task > 1
        ):
            raise ValueError(
                f"ragged_transport={self.ragged_transport.value} writes into peer GPUs' buffers, which needs one "
                f"process owning every GPU (processes_per_task=1), got processes_per_task={self.processes_per_task}"
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
        # Every expert bank of every stack (``expert_mlp_b`` with heterogeneous experts).
        banks = []
        for stack in m.layer_stacks():
            banks.append(stack.stacked.mlp.expert_mlp)
            if stack.stacked.mlp.expert_mlp_b is not None:
                banks.append(stack.stacked.mlp.expert_mlp_b)
        return banks

    dropless = [
        dataclasses.replace(expert_mlp, implementation=implementation, expert_chunks=1, fp8_dispatch=False)
        for expert_mlp in stack_expert_mlps(model)
    ]
    return eqx.tree_at(stack_expert_mlps, model, dropless)


def _cast_to_compute(mp: jmp.Policy, model: Transformer) -> Transformer:
    """``mp.cast_to_compute``, except the n-gram statistic table stays float32: it is only gathered from, and a bf16
    copy would round its counts and cost a table-sized cast every step. The lm_head bias also stays float32 (the
    loss splits it into two compute-dtype parts)."""
    compute = mp.cast_to_compute(model)
    if model.lm_head_bias is not None:
        compute = eqx.tree_at(lambda m: m.lm_head_bias, compute, model.lm_head_bias)
    if model.ngram_stat_table is None:
        return compute
    return eqx.tree_at(lambda m: m.ngram_stat_table, compute, model.ngram_stat_table)


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
        model = _cast_to_compute(mp, model)
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
        # read `expert_in_dim` and write `expert_out_dim` instead, plus the latent projections, so correct
        # both terms or MFU is overstated by roughly the compression ratio.
        if model_config.latent_dim is not None or model_config.latent_out_dim is not None:
            hidden = model_config.hidden_dim
            read, write = model_config.expert_in_dim, model_config.expert_out_dim
            # Matches the routed term in `lm_flops_per_token`: 2 * (gate + up + down) * intermediate * top_k.
            routed_delta = (
                2
                * model_config.intermediate_dim
                * model_config.num_experts_per_token
                * (2 * (read - hidden) + (write - hidden))
            )
            # W_latent_down (hidden -> read) and W_latent_up (write -> hidden), once per token each.
            down = read if model_config.latent_dim is not None else 0
            up = write if model_config.has_latent_up else 0
            projection = 2 * hidden * (down + up)
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
class NewtonMuonState:
    """``newton_muon`` preconditioner state, per layer in layer order (``GrugModelConfig.newton_muon``)."""

    second_moment: jax.Array  # [L, n, n] EMA of the expert-input Z Z^T / N
    inverse: jax.Array  # [L, n, n] (second_moment + damping I)^{-1}, refreshed every newton_muon_every steps
    eigenvalues: jax.Array  # [L, n] ascending eigenvalues of second_moment at the last refresh
    damping: jax.Array  # [L] gamma * tr(second_moment) / n at the last refresh


@register_dataclass
@dataclass(frozen=True)
class GrugTrainState:
    step: jax.Array
    params: Transformer
    master_params: Transformer | None
    ema_params: Transformer | None  # EMA of params for eval/checkpoint; None unless ema_beta is set.
    opt_state: optax.OptState
    pending_qb_betas: jax.Array
    replay_hidden: jax.Array | None = None  # [slots, B, S, D] stored final hidden states (head replay)
    replay_labels: jax.Array | None = None  # [slots, B, S]
    replay_weight: jax.Array | None = None  # [slots, B, S]
    newton_muon: NewtonMuonState | None = None


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
    """This step's QB betas (damped toward the held ones by ``qb_bias_damping``), or the held ones once
    ``qb_freeze_step`` is reached. The bias is ``-beta`` mean-centered, a linear map, so damping the betas is
    damping the bias: ``b <- (1 - gamma) b + gamma (-beta)``."""
    config = state.params.config
    if config.qb_bias_damping is not None:
        gamma = config.qb_bias_damping
        new_betas = (1.0 - gamma) * state.pending_qb_betas + gamma * new_betas
    freeze_step = config.qb_freeze_step
    if freeze_step is None:
        return new_betas
    return jnp.where(state.step + 1 >= freeze_step, state.pending_qb_betas, new_betas)


def _initial_newton_muon(config: GrugModelConfig) -> NewtonMuonState | None:
    if not config.newton_muon:
        return None
    n = config.latent_dim if config.latent_dim is not None else config.hidden_dim
    layers = config.num_layers

    def replicated(x):
        return jax.sharding.reshard(x, P(*(None,) * x.ndim))

    return NewtonMuonState(
        second_moment=replicated(jnp.zeros((layers, n, n), jnp.float32)),
        inverse=replicated(jnp.broadcast_to(jnp.eye(n, dtype=jnp.float32), (layers, n, n))),
        eigenvalues=replicated(jnp.zeros((layers, n), jnp.float32)),
        damping=replicated(jnp.zeros((layers,), jnp.float32)),
    )


def _refresh_newton_muon(
    config: GrugModelConfig, state: NewtonMuonState, gram: jax.Array, step: jax.Array
) -> NewtonMuonState:
    """Every ``newton_muon_every`` steps, fold this batch's ``Z Z^T / N`` into the EMA and re-invert.

    The first step sets the EMA to the batch moment (the paper starts from ``1e-3 I``, which at its
    ``k``-step refresh makes the first ``k`` steps plain Muon with a rescaled gradient in the momentum).
    The damped inverse is taken through ``eigh``, which also gives the spectrum for logging.
    """

    def refresh(prev: NewtonMuonState) -> NewtonMuonState:
        beta = config.newton_muon_beta
        second_moment = jnp.where(step == 0, gram, beta * prev.second_moment + (1.0 - beta) * gram)
        n = second_moment.shape[-1]
        damping = config.newton_muon_eps * jnp.trace(second_moment, axis1=-2, axis2=-1) / n
        eigenvalues, eigenvectors = jnp.linalg.eigh(second_moment)
        scaled = eigenvectors / (jnp.maximum(eigenvalues, 0.0) + damping[:, None])[:, None, :]
        inverse = jnp.einsum("lij,lkj->lik", scaled, eigenvectors)
        return NewtonMuonState(second_moment, inverse, eigenvalues, damping)

    return jax.lax.cond(step % config.newton_muon_every == 0, refresh, lambda prev: prev, state)


def _newton_precondition(grads: Transformer, inverse: jax.Array) -> Transformer:
    """Right-precondition the routed-expert gate/up gradients: ``G <- G K^{-1}`` (arXiv 2604.01472).

    The stacks are ``[L, E, n_in, n_out]`` (``x @ W``), so the paper's ``G K^{-1}`` on an ``[out, in]``
    matrix is ``K^{-1} G`` here, contracting the input axis, which is gathered for the product.
    """

    def precondition(grad: jax.Array, layer_inverse: jax.Array) -> jax.Array:
        spec = tuple(jax.typeof(grad).sharding.spec)
        spec = P(*spec, *(None,) * (grad.ndim - len(spec)))
        gathered = P(spec[0], spec[1], None, spec[3])
        product = jnp.einsum(
            "lnm,lemi->leni",
            layer_inverse,
            jax.sharding.reshard(grad, gathered).astype(jnp.float32),
            out_sharding=gathered,
        )
        return jax.sharding.reshard(product.astype(grad.dtype), spec)

    sites, replacements = [], []
    for k, indices in enumerate(grads.stack_layer_indices()):
        layer_inverse = inverse[np.asarray(indices)]
        mlp = grads.layer_stacks()[k].stacked.mlp
        for bank in ("expert_mlp", "expert_mlp_b"):
            for leaf in ("w_gate", "w_up"):
                grad = None if getattr(mlp, bank) is None else getattr(getattr(mlp, bank), leaf)
                if grad is not None:
                    sites.append((k, bank, leaf))
                    replacements.append(precondition(grad, layer_inverse))

    def expert_leaves(model: Transformer) -> list[jax.Array]:
        return [getattr(getattr(model.layer_stacks()[k].stacked.mlp, b), leaf) for k, b, leaf in sites]

    return eqx.tree_at(expert_leaves, grads, replacements)


def _newton_muon_metrics(state: NewtonMuonState) -> dict[str, jax.Array]:
    """Per-layer spectrum of the expert-input second moment ``K``: raw and damped condition numbers."""
    lo = jnp.maximum(state.eigenvalues[:, 0], 0.0)
    hi = state.eigenvalues[:, -1]
    cond = hi / jnp.maximum(lo, 1e-30)
    damped_cond = (hi + state.damping) / (lo + state.damping)
    metrics = {
        "train/newton_muon/log10_cond_mean": jnp.mean(jnp.log10(cond)),
        "train/newton_muon/damped_cond_mean": jnp.mean(damped_cond),
    }
    for i in range(cond.shape[0]):
        metrics[f"train/newton_muon/log10_cond_L{i}"] = jnp.log10(cond[i])
        metrics[f"train/newton_muon/damped_cond_L{i}"] = damped_cond[i]
        metrics[f"train/newton_muon/eig_max_L{i}"] = hi[i]
        metrics[f"train/newton_muon/eig_min_L{i}"] = lo[i]
    return metrics


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None = None,
    head_replay_shape: tuple[int, int, int, int] | None = None,
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
        pending_qb_betas=jnp.zeros((num_moe_layers, model_config.num_experts + model_config.num_null_experts)),
        **_empty_head_replay(head_replay_shape, mp),
        newton_muon=_initial_newton_muon(model_config),
    )


def _empty_head_replay(shape: tuple[int, int, int, int] | None, mp: jmp.Policy) -> dict[str, jax.Array | None]:
    """Zeroed ``[slots, B, S, D]`` head-replay buffers, batch-sharded (or Nones when off)."""
    if shape is None:
        return {"replay_hidden": None, "replay_labels": None, "replay_weight": None}
    slots, b, s, d = shape
    spec3, spec4 = P(None, _BATCH_AXES, None), P(None, _BATCH_AXES, None, None)
    return {
        "replay_hidden": jax.sharding.reshard(jnp.zeros((slots, b, s, d), mp.compute_dtype), spec4),
        "replay_labels": jax.sharding.reshard(jnp.zeros((slots, b, s), jnp.int32), spec3),
        # Ones, not zeros: an unfilled slot must give a finite CE (its replay scale is 0 until filled),
        # while an all-zero weight makes the weighted-mean CE 0/0 = NaN in the loss and the gradients.
        "replay_weight": jax.sharding.reshard(jnp.ones((slots, b, s), jnp.float32), spec3),
    }


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


def _store_head_replay(state: GrugTrainState, final_hidden: jax.Array | None, batch, period: int) -> dict:
    """Every ``period`` steps, write this batch's final hidden states and labels into the next slot."""
    if state.replay_hidden is None or final_hidden is None:
        return {}
    assert state.replay_labels is not None and state.replay_weight is not None
    slots = state.replay_hidden.shape[0]
    slot = (state.step // period) % slots
    write = state.step % period == 0
    labels = jnp.pad(batch.tokens[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)

    def put(buf, new):
        updated = jax.lax.dynamic_update_index_in_dim(buf, new.astype(buf.dtype), slot, axis=0)
        return jnp.where(write, updated, buf)

    return {
        "replay_hidden": put(state.replay_hidden, final_hidden),
        "replay_labels": put(state.replay_labels, labels),
        "replay_weight": put(state.replay_weight, batch.loss_weight.astype(jnp.float32)),
    }


def _aux_loss_weight(model_config, step: jax.Array) -> jax.Array | None:
    """The early auxiliary LM loss weight at ``step``: linear from ``aux_lm_weight`` to 0 at ``aux_lm_steps``."""
    if model_config.aux_lm_layer is None:
        return None
    frac = jnp.clip(1.0 - step.astype(jnp.float32) / model_config.aux_lm_steps, 0.0, 1.0)
    return model_config.aux_lm_weight * frac


def _loss_and_grads(
    params,
    batch,
    mp: jmp.Policy,
    z_loss: float | None,
    step: jax.Array | None = None,
    loop_active: bool | None = None,
    head_replay: HeadReplay | None = None,
    byte_table: jax.Array | None = None,
    byte_weight: jax.Array | None = None,
    router_tie_active: bool | None = None,
):
    """``loop_active`` is a static pass selector for looped growth (see ``GrugModelConfig.loop_grow_step``);
    ``router_tie_active`` statically applies the router ties (see ``GrugModelConfig.router_embed_tie_release_step``)."""
    aux_weight = None if step is None else _aux_loss_weight(params.config, step)
    route_key = None
    cfg = params.config
    mtp_subsample = cfg.mtp_mode != MtpMode.OFF and cfg.mtp_position_frac < 1.0
    if step is not None and (cfg.moe_gumbel_tau > 0 or cfg.erc_loss_weight > 0 or mtp_subsample):
        route_key = jax.random.fold_in(jax.random.PRNGKey(ROUTE_NOISE_SEED), step)

    def loss_fn(model):
        compute_params = _cast_to_compute(mp, model)
        return compute_params.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="mean",
            logsumexp_weight=z_loss,
            return_router_metrics=True,
            aux_loss_weight=aux_weight,
            loop_active=loop_active,
            train_terms=True,
            route_key=route_key,
            head_replay=head_replay,
            byte_table=byte_table,
            byte_aux_weight=byte_weight,
            router_tie_active=router_tie_active,
        )

    return jax.value_and_grad(loss_fn, has_aux=True)(params)


def _accumulated_loss_and_grads(
    num_microbatches: int, params, batch, mp: jmp.Policy, z_loss, step, loop_active, router_tie_active
):
    """``_loss_and_grads`` over ``num_microbatches`` equal slices of ``batch``, one at a time: the loss, gradients
    and float metrics are averaged, integer metrics (counts) summed. Every microbatch runs in one ``lax.scan``
    body from a zero accumulator, so the forward and backward compile once and one microbatch's activations are
    live at a time."""
    micro = reshape_batch_into_microbatches(batch, num_microbatches)

    def one(microbatch):
        return _loss_and_grads(params, microbatch, mp, z_loss, step, loop_active, router_tie_active=router_tie_active)

    shapes = jax.eval_shape(one, jax.tree.map(lambda x: x[0], micro))
    zeros = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype, out_sharding=getattr(s.sharding, "spec", None)), shapes)
    total, _ = jax.lax.scan(lambda acc, mb: (jax.tree.map(jnp.add, acc, one(mb)), None), zeros, micro)
    (loss, metrics), grads = total

    def mean(x):
        return x / num_microbatches if jnp.issubdtype(x.dtype, jnp.floating) else x

    return (loss / num_microbatches, jax.tree.map(mean, metrics)), jax.tree.map(mean, grads)


def _compute_diagnostic_watch_stats(
    params, batch, mp: jmp.Policy, z_loss: float | None, watch_config: WatchConfig, router_tie_active: bool
):
    (_, _), grads = _loss_and_grads(params, batch, mp, z_loss, router_tie_active=router_tie_active)
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


def _make_grad_capture_step(mp: jmp.Policy, *, z_loss_weight: float):
    """``capture(params, batch, pending_qb_betas, step, ...)`` recomputes the train step's gradient on the same batch
    and returns the captured matrices' gradients and current values (see ``grad_capture.py``)."""
    z_loss = z_loss_weight if z_loss_weight > 0 else None

    @functools.partial(jax.jit, static_argnames=("loop_active", "router_tie_active"))
    def capture(params: Transformer, batch, pending_qb_betas, step, loop_active, router_tie_active):
        params = _apply_qb_betas(params, pending_qb_betas)
        (_, _), grads = _loss_and_grads(
            params, batch, mp, z_loss, step, loop_active, router_tie_active=router_tie_active
        )
        return capture_matrices(grads), capture_matrices(params)

    return capture


_captured_params = jax.jit(capture_matrices)


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

    @functools.partial(jax.jit, static_argnames=("router_tie_active",))
    def diagnostic_watch_step(params: Transformer, batch, pending_qb_betas: jax.Array, router_tie_active: bool):
        params = _apply_qb_betas(params, pending_qb_betas)
        return _compute_diagnostic_watch_stats(params, batch, mp, z_loss, diagnostic_watch_config, router_tie_active)

    return diagnostic_watch_step


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None = None,
    ema_start_step: int = 0,
    head_replay_period: int = 100,
    head_replay_scale: float = 0.1,
    watch_config: WatchConfig | None = None,
    byte_table: jax.Array | None = None,
    byte_aux_steps: int = 0,
    grad_accum_microbatches: int = 1,
):
    """``grad_accum_microbatches`` > 1 averages gradients over that many microbatches (``_accumulated_loss_and_grads``).
    ``byte_table`` (with ``byte_aux_steps``) turns on the byte-level auxiliary loss, its weight decaying
    linearly from the model's ``byte_aux_weight`` to 0 at ``byte_aux_steps``."""
    one = jnp.array(1, dtype=jnp.int32)
    z_loss = z_loss_weight if z_loss_weight > 0 else None
    if watch_config is not None:
        if isinstance(watch_config.watch_targets, str):
            watch_targets = tuple(t.strip() for t in watch_config.watch_targets.split(","))
        else:
            watch_targets = tuple(watch_config.watch_targets)
    else:
        watch_targets = ()

    @functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("loop_active", "router_tie_active"))
    def train_step(state: GrugTrainState, batch, loop_active: bool | None = None, router_tie_active: bool | None = None):
        # Apply pending QB betas to router biases inside JIT (avoids eager
        # host-side kernel launches that can cause SPMD sync issues).
        qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)

        head_replay = None
        if state.replay_hidden is not None:
            slots = state.replay_hidden.shape[0]
            wave = state.step // head_replay_period
            oldest = (wave + 1) % slots
            filled = state.step >= slots * head_replay_period
            head_replay = HeadReplay(
                hidden=state.replay_hidden[oldest],
                labels=state.replay_labels[oldest],
                weight=state.replay_weight[oldest],
                scale=jnp.where(filled, head_replay_scale, 0.0).astype(jnp.float32),
            )
        byte_weight = None
        if byte_table is not None:
            progress = state.step.astype(jnp.float32) / max(byte_aux_steps, 1)
            byte_weight = qb_params.config.byte_aux_weight * jnp.clip(1.0 - progress, 0.0, 1.0)
        if grad_accum_microbatches > 1:
            if head_replay is not None or byte_table is not None or state.newton_muon is not None:
                raise ValueError("grad_accum_microbatches needs no head replay, byte aux loss or Newton-Muon")
            (loss, summarized_metrics), grads = _accumulated_loss_and_grads(
                grad_accum_microbatches, qb_params, batch, mp, z_loss, state.step, loop_active, router_tie_active
            )
        else:
            (loss, summarized_metrics), grads = _loss_and_grads(
                qb_params,
                batch,
                mp,
                z_loss,
                state.step,
                loop_active,
                head_replay,
                byte_table,
                byte_weight,
                router_tie_active,
            )
        final_hidden = summarized_metrics.pop(FINAL_HIDDEN_KEY, None)
        newton_muon = state.newton_muon
        opt_grads = grads
        if newton_muon is not None:
            newton_muon = _refresh_newton_muon(
                qb_params.config, newton_muon, summarized_metrics.pop(NEWTON_GRAM_KEY), state.step
            )
            opt_grads = _newton_precondition(grads, newton_muon.inverse)
            summarized_metrics.update(_newton_muon_metrics(newton_muon))
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
            updates, opt_state = optimizer.update(opt_grads, opt_state_in, qb_params)
            metrics.update(optimizer_diagnostics(opt_state))
            metrics.update(magma_metrics(opt_state))
            params = optax.apply_updates(qb_params, updates)
            master_params = None
        if params.ngram_stat_table is not None:
            # Write after read: this batch's targets enter the statistic table only once its step is done.
            params = write_ngram_stats(params, batch.tokens, batch.loss_weight, _segment_ids(batch))

        if ema_beta is None:
            ema_params = None
        else:
            # EMA tracks the QB-biased params, so re-apply the pending betas before blending.
            qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas)
            ema_active = state.step >= ema_start_step
            ema_params = jax.tree_util.tree_map(
                lambda old, new: jnp.where(ema_active, ema_beta * old + (1.0 - ema_beta) * new, new),
                qb_ema_params,
                params,
            )
            if params.ngram_stat_table is not None:
                # The statistic table is a running sum, not a trained weight: the EMA keeps the live table.
                ema_params = eqx.tree_at(lambda m: m.ngram_stat_table, ema_params, params.ngram_stat_table)

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
            **_store_head_replay(state, final_hidden, batch, head_replay_period),
            # Dense blocks have no router, so the forward emits no qb_beta_per_layer; keep the
            # (zeros) pending betas -- _apply_qb_betas is already a no-op for dense.
            pending_qb_betas=_next_qb_betas(state, metrics.get("qb_beta_per_layer", state.pending_qb_betas)),
            newton_muon=newton_muon,
        )

        return next_state, metrics, watch_stats

    return train_step


def _segment_ids(batch) -> jax.Array | None:
    segment_ids = batch.attn_mask.segment_ids
    return None if segment_ids is None else segment_ids[0]


def _prefill_ngram_stats(state: GrugTrainState, train_loader, *, start_step: int, num_batches: int) -> GrugTrainState:
    """Write ``num_batches`` training batches from ``start_step`` on into the n-gram statistic table (see
    ``GrugTrainerConfig.ngram_stat_prefill_batches``).

    Only the table goes through the jitted write (donated), and each write is waited on before the next is
    dispatched: without the wait the host runs hundreds of writes ahead of the GPU, each holding its temporaries,
    and exhausts HBM (the first d512 pre-fill runs died this way)."""
    cfg = state.params.config
    code = state.params.ngram_stat_code

    @functools.partial(jax.jit, donate_argnums=(0,))
    def write(table: jax.Array, batch) -> jax.Array:
        return ngram_stat_table_add(cfg, table, code, batch.tokens, batch.loss_weight, _segment_ids(batch))

    @jax.jit
    def filled_fraction(table: jax.Array) -> jax.Array:
        return jnp.mean((table[:, -1] > 0).astype(jnp.float32))

    started = time.time()
    table = state.params.ngram_stat_table
    batches = train_loader.iter_from_step(start_step)
    for i in range(num_batches):
        table = jax.block_until_ready(write(table, next(batches)))
        if (i + 1) % 500 == 0:
            logger.info("ngram stat prefill: %d / %d batches (%.0fs)", i + 1, num_batches, time.time() - started)
    filled = float(filled_fraction(table))
    logger.info(
        "ngram stat prefill done: %d batches in %.0fs, %.1f%% rows filled",
        num_batches,
        time.time() - started,
        100 * filled,
    )
    levanter.tracker.log_summary({"ngram_stat/prefill_batches": num_batches, "ngram_stat/prefill_rows_filled": filled})
    params = eqx.tree_at(lambda m: m.ngram_stat_table, state.params, table)
    # The EMA carries the live table (see the train step). It gets its own copy: the train step donates the state, and
    # one buffer cannot be donated twice.
    ema_params = (
        None
        if state.ema_params is None
        else eqx.tree_at(lambda m: m.ngram_stat_table, state.ema_params, jnp.copy(table))
    )
    return dataclasses.replace(state, params=params, ema_params=ema_params)


_materialize_router_ties = eqx.filter_jit(tie_routers)


def _router_ties_active(config: GrugModelConfig, step: int) -> bool:
    """Whether the train step at ``step`` runs the tied router (before ``router_embed_tie_release_step``)."""
    release = config.router_embed_tie_release_step
    return release is None or step < release


def _router_tie_view(model: Transformer, step: int) -> Transformer:
    """The model an untied forward (evals, routing dumps) sees at ``step``: before the tie release, the ties
    materialized into the router columns (the same function as the tied forward)."""
    release = model.config.router_embed_tie_release_step
    if release is not None and step < release:
        return _materialize_router_ties(model)
    return model


def _release_router_ties(state: GrugTrainState) -> GrugTrainState:
    """The ``router_embed_tie_release_step`` rewrite: every param copy's tied router columns take their tied value
    ``alpha * mean(token_embed[V...])`` (the EMA its own), so the untied program continues the tied function. The
    optimizer state is kept: the tied columns got zero gradient, so their Adam moments are zero and restart."""

    def release(model: Transformer | None) -> Transformer | None:
        return None if model is None else _materialize_router_ties(model)

    return dataclasses.replace(
        state,
        params=release(state.params),
        master_params=release(state.master_params),
        ema_params=release(state.ema_params),
    )


def _init_unigram_bias(state: GrugTrainState, train_loader, *, num_batches: int) -> GrugTrainState:
    """Set ``lm_head_bias`` to ``log((count + 1) / (total + V))``, the add-one log unigram frequency of the
    loss-weighted tokens of the first ``num_batches`` training batches (the EMA copy too). Adam's moments are
    zeros either way, so the optimizer state needs no change."""
    vocab = state.params.config.vocab_size

    @functools.partial(jax.jit, donate_argnums=(0,))
    def add(counts: jax.Array, tokens: jax.Array, loss_weight: jax.Array) -> jax.Array:
        weight = loss_weight.astype(jnp.float32).reshape(-1)
        return counts + jnp.zeros((vocab,), jnp.float32).at[tokens.reshape(-1)].add(weight, out_sharding=P(None))

    started = time.time()
    counts = jnp.zeros((vocab,), jnp.float32)
    batches = train_loader.iter_from_step(0)
    for _ in range(num_batches):
        batch = next(batches)
        counts = jax.block_until_ready(add(counts, batch.tokens, batch.loss_weight))
    bias = jnp.log((counts + 1.0) / (jnp.sum(counts) + vocab))
    bias = jax.device_put(bias, state.params.lm_head_bias.sharding)
    logger.info("unigram lm_head bias from %d batches in %.0fs", num_batches, time.time() - started)
    params = eqx.tree_at(lambda m: m.lm_head_bias, state.params, bias)
    ema_params = (
        None if state.ema_params is None else eqx.tree_at(lambda m: m.lm_head_bias, state.ema_params, jnp.copy(bias))
    )
    return dataclasses.replace(state, params=params, ema_params=ema_params)


ROUTING_DUMP_FILE = "routing_step{step}.npz"


def _routing_dump_sequences(
    data_config: LmDataConfig, *, seq_len: int, num_sequences: int
) -> tuple[np.ndarray, np.ndarray]:
    """``num_sequences`` fixed held-out sequences and their segment ids (``-1`` marks padding): every
    validation set from its start, round-robin over the sets in name order, so each dump sees the same
    tokens. Raises when the validation sets hold fewer sequences."""
    sets = data_config.validation_sets(Axis("position", seq_len))
    if not sets:
        raise ValueError("routing_dump_steps needs validation sets")
    names = sorted(sets)
    datasets = {name: sets[name].as_sync_dataset() for name in names}
    lengths = {name: len(datasets[name]) for name in names}
    picks: list[tuple[str, int]] = []
    for index in range(max(lengths.values())):
        picks += [(name, index) for name in names if index < lengths[name]]
    if len(picks) < num_sequences:
        raise ValueError(f"routing dump needs {num_sequences} sequences, the validation sets hold {len(picks)}")
    picks = picks[:num_sequences]
    fetched = {}
    for name in names:
        indices = [index for n, index in picks if n == name]
        if indices:
            examples = [grug_lm_example_from_named(ex) for ex in datasets[name].get_batch(indices)]
            fetched.update({(name, index): ex for index, ex in zip(indices, examples, strict=True)})
    tokens, segments = [], []
    for pick in picks:
        example = fetched[pick]
        tokens.append(np.asarray(example.tokens, np.int32))
        seg = example.attn_mask.segment_ids
        segments.append(np.zeros(seq_len, np.int32) if seg is None else np.asarray(seg[0], np.int32))
    return np.stack(tokens), np.stack(segments)


def _routing_counts_step(mp: jmp.Policy):
    """Jitted ``(params, qb_betas, tokens, segment_ids, counts) -> counts``: adds one batch's routed
    assignments (every top-K slot, weight 1, and a combine-weight-weighted copy) to the ``[L, V, E]``
    counts by current token and by previous token (same document only)."""

    @jax.jit
    def add(params: Transformer, qb_betas: jax.Array, tokens: jax.Array, segment_ids: jax.Array, counts):
        model = _cast_to_compute(mp, _apply_qb_betas(params, qb_betas))
        selected, weights = model.routing_assignments(tokens, AttentionMask.causal().with_segment_ids(segment_ids))
        k = selected.shape[-1]
        valid = segment_ids >= 0
        prev_tokens = jnp.pad(tokens[:, :-1], ((0, 0), (1, 0)))
        prev_valid = valid & (jnp.pad(segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=-1) == segment_ids)
        out_spec = P(None, None, None)

        def slots(x):
            return jnp.broadcast_to(x[..., None], (*x.shape, k))

        cur, prev, cur_ok, prev_ok = slots(tokens), slots(prev_tokens), slots(valid), slots(prev_valid)
        current, previous, weighted = counts
        for layer in range(selected.shape[0]):
            sel = selected[layer]
            current = current.at[layer, cur, sel].add(cur_ok.astype(jnp.int32), out_sharding=out_spec)
            previous = previous.at[layer, prev, sel].add(prev_ok.astype(jnp.int32), out_sharding=out_spec)
            w = jnp.where(cur_ok, weights[layer].astype(jnp.float32), 0.0)
            weighted = weighted.at[layer, cur, sel].add(w, out_sharding=out_spec)
        return current, previous, weighted

    return add


def write_routing_dump(
    path: str,
    counts: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: GrugModelConfig,
    layer_kinds: np.ndarray,
    step: int,
    num_tokens: int,
) -> None:
    """Write one routing dump (``analyze_routing.py`` reads it) to an fsspec ``path``."""
    current, previous, weighted = counts
    with fsspec.open(path, "wb") as f:
        np.savez_compressed(
            f,
            counts=current,
            counts_prev=previous,
            counts_weighted=weighted,
            num_experts=cfg.num_experts,
            num_null_experts=cfg.num_null_experts,
            top_k=cfg.num_experts_per_token,
            layer_kinds=layer_kinds,
            step=step,
            num_tokens=num_tokens,
        )


def _routing_dumper(config: GrugRunConfig, model: Transformer, mesh: Mesh, batch_size: int) -> Callable[..., None]:
    """Build the ``routing_dump_steps`` dump: ``dump(state)`` routes the fixed held-out batches and writes
    ``routing_step<N>.npz`` (``counts``, ``counts_prev``, ``counts_weighted`` as ``[L, V, E]``, plus the
    routing metadata) under ``routing_dump_path``, once per step; process 0 writes."""
    trainer_cfg = config.trainer
    if trainer_cfg.routing_dump_path is None:
        raise ValueError("routing_dump_steps needs routing_dump_path")
    cfg = config.model
    num_sequences = trainer_cfg.routing_dump_batches * batch_size
    # The dataset's per-example jit pins its output to a CPU device, which conflicts with the train loop's GPU
    # context mesh; JAX's mesh context is thread-local, so fetch on a worker thread (as the data loader does).
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        tokens, segments = pool.submit(
            _routing_dump_sequences, config.data, seq_len=cfg.max_seq_len, num_sequences=num_sequences
        ).result()
    sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
    add = _routing_counts_step(trainer_cfg.trainer.mp)
    shape = (cfg.num_layers, cfg.vocab_size, cfg.num_experts + cfg.num_null_experts)
    replicated = NamedSharding(mesh, P(None, None, None))
    layer_kinds = np.asarray(model.layer_kinds())

    def global_batch(host: np.ndarray) -> jax.Array:
        return jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])

    dumped: set[int] = set()

    def dump(state: GrugTrainState) -> None:
        step = int(state.step)
        if step in dumped:
            return
        dumped.add(step)
        started = time.time()
        counts = (
            jax.device_put(np.zeros(shape, np.int32), replicated),
            jax.device_put(np.zeros(shape, np.int32), replicated),
            jax.device_put(np.zeros(shape, np.float32), replicated),
        )
        params = _router_tie_view(state.params, step)
        for start in range(0, num_sequences, batch_size):
            batch = slice(start, start + batch_size)
            counts = add(
                params, state.pending_qb_betas, global_batch(tokens[batch]), global_batch(segments[batch]), counts
            )
        current, previous, weighted = (np.asarray(c) for c in jax.block_until_ready(counts))
        path = f"{trainer_cfg.routing_dump_path.rstrip('/')}/{ROUTING_DUMP_FILE.format(step=step)}"
        if jax.process_index() == 0:
            write_routing_dump(path, (current, previous, weighted), cfg, layer_kinds, step, int(np.sum(segments >= 0)))
        logger.info(
            "routing dump at step %d: %d sequences in %.1fs -> %s", step, num_sequences, time.time() - started, path
        )

    return dump


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
    byte_table = None
    if config.model.byte_aux_bytes:
        byte_table = jnp.asarray(
            token_byte_table(config.data.the_tokenizer, config.model.vocab_size, config.model.byte_aux_bytes)
        )
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
    ema_start_step = (
        0 if config.trainer.ema_last_steps is None else max(0, trainer.num_train_steps - config.trainer.ema_last_steps)
    )
    train_step = _make_train_step(
        optimizer,
        trainer.mp,
        z_loss_weight=config.trainer.z_loss_weight,
        ema_beta=config.trainer.ema_beta,
        ema_start_step=ema_start_step,
        head_replay_period=config.trainer.head_replay_period,
        head_replay_scale=config.trainer.head_replay_scale,
        watch_config=inline_watch_config,
        byte_table=byte_table,
        byte_aux_steps=int(config.model.byte_aux_decay_frac * trainer.num_train_steps),
        grad_accum_microbatches=config.trainer.grad_accum_microbatches,
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
                ema_beta=config.trainer.ema_beta,
                head_replay_shape=(
                    (
                        config.trainer.head_replay_slots,
                        trainer.train_batch_size,
                        config.model.max_seq_len,
                        config.model.hidden_dim,
                    )
                    if config.trainer.head_replay_slots > 0
                    else None
                ),
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
        if config.trainer.ngram_stat_prefill_batches and int(state.step) == 0:
            if state.params.ngram_stat_table is None:
                raise ValueError("ngram_stat_prefill_batches needs a model with ngram_stat_rows > 0")
            state = _prefill_ngram_stats(
                state,
                train_loader,
                start_step=trainer.num_train_steps,
                num_batches=config.trainer.ngram_stat_prefill_batches,
            )
        if state.params.lm_head_bias is not None and int(state.step) == 0:
            state = _init_unigram_bias(state, train_loader, num_batches=config.trainer.lm_head_unigram_batches)
        dump_routing = None
        pending_dumps = set(config.trainer.routing_dump_steps)
        if pending_dumps:
            dump_routing = _routing_dumper(config, state.params, mesh, batch_schedule.batch_size_at_step(0))
            if 0 in pending_dumps and int(state.step) == 0:
                dump_routing(state)
            # A resumed run skips the dumps it already passed.
            pending_dumps = {step for step in pending_dumps if step > int(state.step)}
        grad_capture_due = capture_steps(config.trainer.grad_capture_starts, config.trainer.grad_capture_len)
        grad_capture_step = capture_writer = None
        if grad_capture_due:
            if config.trainer.grad_capture_path is None:
                raise ValueError("grad_capture_starts needs grad_capture_path")
            grad_capture_step = _make_grad_capture_step(trainer.mp, z_loss_weight=config.trainer.z_loss_weight)
            capture_writer = CaptureWriter(config.trainer.grad_capture_path)
        batch_source = train_loader.iter_from_step(int(state.step))
        iterator = LoadingTimeTrackerIterator(batch_source)

        state_callbacks = StateCallbackRunner[GrugTrainState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            # From the EMA start on, evals score the EMA weights.
            eval_model_getter=lambda s: _router_tie_view(
                s.ema_params if s.ema_params is not None and int(s.step) > ema_start_step else s.params, int(s.step)
            ),
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
                        model = _reshard_tree_to_mesh(step.eval_model, _mesh)
                        with _pgle_disabled():
                            log_dict = eval_model(_ev, model, prefix=_prefix)
                        levanter.tracker.log(log_dict, step=step_count)

                eval_hooks.append(dropless_eval_hook)

            if interval is not None and interval > 0:
                for hook in eval_hooks:
                    state_callbacks.add_hook(hook, every=interval)

        last_loss: float | jax.Array = 0.0
        last_step_duration = 0.0
        host_gap_start: float | None = None
        gc.callbacks.append(_warn_on_long_gc)
        # Automatic GC pauses each rank at a different step (gen-2 over the model/optimizer pytrees and
        # compiled executables: 0.4-0.6 s at d512), and every collective waits for the paused rank. So
        # freeze everything alive at loop start, disable automatic collection, and collect on every rank
        # at the same step every GC_EVERY_STEPS, after the step completes.
        gc.collect()
        gc.freeze()
        gc.disable()
        hlo_written = False

        # Main optimization loop.
        try:
            with HostStallSampler(HOST_STALL_SAMPLE_THRESHOLD) as stall_sampler:
                while int(state.step) < stop_step:
                    with jax.profiler.TraceAnnotation("load_batch"):
                        batch = next(iterator)
                    current_step = int(state.step)
                    watch_due = (
                        watch_config.is_enabled
                        and watch_config.interval > 0
                        and current_step % watch_config.interval == 0
                    )
                    if watch_due and diagnostic_watch_step is not None:
                        watch_stats = diagnostic_watch_step(
                            state.params,
                            batch,
                            state.pending_qb_betas,
                            router_tie_active=_router_ties_active(config.model, current_step),
                        )
                        jax.block_until_ready(watch_stats)
                    else:
                        watch_stats = None
                    step_start = time.perf_counter()
                    if host_gap_start is not None and step_start - host_gap_start > HOST_GAP_WARN:
                        logger.warning(
                            "host gap %.3f s before step %d on process %d (load %.3f s)",
                            step_start - host_gap_start,
                            current_step,
                            jax.process_index(),
                            iterator.this_load_time,
                        )
                    state_callbacks.emit_event(callbacks.ProgressEvent.TRAIN_STEP_STARTED)
                    grow_step = config.model.loop_grow_step
                    # Looped growth compiles one program per pass count (a traced switch would keep both alive).
                    loop_active = None if grow_step is None else int(state.step) >= grow_step
                    # The router tie release likewise switches programs once (plus a host-side param rewrite below).
                    router_tie_active = _router_ties_active(config.model, current_step)
                    captured_before = None
                    if grad_capture_step is not None and current_step in grad_capture_due:
                        # Before the step: the train step donates the state.
                        captured_before = jax.device_get(
                            grad_capture_step(
                                state.params,
                                batch,
                                state.pending_qb_betas,
                                state.step,
                                loop_active=loop_active,
                                router_tie_active=router_tie_active,
                            )
                        )
                    state, metrics, inline_watch_stats = train_step(
                        state, batch, loop_active=loop_active, router_tie_active=router_tie_active
                    )
                    if inline_watch_stats is not None and watch_due:
                        watch_stats = inline_watch_stats
                    step = int(state.step) - 1
                    if captured_before is not None:
                        captured_grads, params_before = captured_before
                        params_after = jax.device_get(_captured_params(state.params))
                        if jax.process_index() == 0:
                            capture_writer.submit(
                                current_step,
                                captured_grads,
                                {name: params_after[name] - params_before[name] for name in params_before},
                                params_before if current_step - 1 not in grad_capture_due else None,
                            )

                    jax.block_until_ready(metrics["train/loss"])
                    ready_time = time.perf_counter()
                    stall_sampler.arm()
                    if config.trainer.hlo_dump_path is not None and not hlo_written and jax.process_index() == 0:
                        _write_train_step_hlo(
                            train_step, state, batch, loop_active, router_tie_active, config.trainer.hlo_dump_path
                        )
                    hlo_written = True
                    state_callbacks.emit_event(callbacks.ProgressEvent.TRAIN_STEP_FINISHED)

                    if not jnp.isfinite(metrics["train/loss"]):
                        raise RuntimeError(
                            f"Non-finite loss ({float(metrics['train/loss'])}) at step {int(state.step)}."
                        )
                    duration = time.perf_counter() - step_start
                    if int(state.step) == config.model.router_embed_tie_release_step:
                        # Before the callbacks, so the checkpoint and evals at this step already see the release.
                        state = _release_router_ties(state)
                        logger.info("router ties released at step %d", int(state.step))
                    hook_start = time.perf_counter()
                    with jax.profiler.TraceAnnotation("callbacks"):
                        state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                        last_loss = metrics["train/loss"]
                        last_step_duration = duration
                        levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=step)
                        levanter.tracker.log({"throughput/loading_time": iterator.this_load_time}, step=step)
                        # One batched device-to-host copy: handing the tracker device scalars makes it fetch each
                        # one separately, which cost 0.1-0.7 s of host time per logging step (and more with one
                        # process driving every GPU).
                        router_metrics = jax.device_get(
                            {
                                key: value
                                for key, value in metrics.items()
                                if key.startswith(
                                    (
                                        "train/router/",
                                        "moe_bias/",
                                        "train/attn_res/",
                                        "train/aux/",
                                        "train/newton_muon/",
                                        "train/optim/",
                                    )
                                )
                                and key not in ("train/router/routing_counts_per_layer", "qb_beta_per_layer")
                            }
                        )
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
                            levanter.tracker.log(jax.device_get(drop_metrics), step=step)

                        if watch_stats is not None:
                            levanter.tracker.log(jax.device_get(watch_stats), step=step)

                    if dump_routing is not None and int(state.step) in pending_dumps:
                        dump_routing(state)
                        pending_dumps.discard(int(state.step))
                    if checkpointer is not None:
                        with callbacks.progress_event_scope(
                            state_callbacks.emit_event,
                            callbacks.ProgressEvent.CHECKPOINT_STARTED,
                            callbacks.ProgressEvent.CHECKPOINT_FINISHED,
                        ):
                            checkpointer.on_step(tree=state, step=int(state.step))
                    if current_step % GC_EVERY_STEPS == 0:
                        gc.collect()
                        gc.freeze()
                    host_gap_start = time.perf_counter()
                    stall_sampler.disarm(current_step)
                    if host_gap_start - ready_time > HOST_GAP_WARN:
                        logger.warning(
                            "post-step host work %.3f s after step %d on process %d",
                            host_gap_start - ready_time,
                            current_step,
                            jax.process_index(),
                        )

        except BaseException:
            logger.exception(
                "Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause"
            )
            raise
        else:
            # Mirror classic trainer behavior: force callbacks on the last completed step.
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if dump_routing is not None and pending_dumps:
                # Steps past the end of the run dump the final weights.
                dump_routing(state)
            if capture_writer is not None:
                capture_writer.close()
            blends: list[tuple[str, Callable[[str], float]]] = [
                (f"eval_blend{a:g}", lambda _path, a=a: a) for a in config.trainer.ema_blend_sweep
            ]
            if config.trainer.ema_group_sweep:
                base = config.trainer.ema_group_base_blend
                for group in (*EMA_BLEND_GROUPS, "other"):
                    for a in (0.0, 1.0):
                        blends.append(
                            (
                                f"eval_blend_{group}{a:g}",
                                lambda path, a=a, group=group: a if _ema_blend_group(path) == group else base,
                            )
                        )
            if blends:
                if state.ema_params is None or dropless_evaluator is None or dropless_eval_mesh is None:
                    raise ValueError("EMA blend evals need ema_beta and the dropless evaluator")
                for prefix, alpha_for in blends:
                    blended = jax.tree_util.tree_map_with_path(
                        lambda path, e, p, alpha_for=alpha_for: (a := alpha_for(jax.tree_util.keystr(path))) * e
                        + (1.0 - a) * p,
                        state.ema_params,
                        state.params,
                    )
                    with set_mesh(dropless_eval_mesh):
                        blended = _reshard_tree_to_mesh(blended, dropless_eval_mesh)
                        with _pgle_disabled():
                            blend_log = eval_model(dropless_evaluator, blended, prefix=prefix)
                    levanter.tracker.log(blend_log, step=int(state.step))
                    del blended
            if checkpointer is not None:
                with callbacks.progress_event_scope(
                    state_callbacks.emit_event,
                    callbacks.ProgressEvent.CHECKPOINT_STARTED,
                    callbacks.ProgressEvent.CHECKPOINT_FINISHED,
                ):
                    checkpointer.on_step(tree=state, step=int(state.step), force=True)
                    checkpointer.wait_until_finished()
        finally:
            gc.enable()
            state_callbacks.emit_event(callbacks.ProgressEvent.TRAINING_FINISHED)

    levanter.tracker.current_tracker().finish()


# Every rank logs host-side stalls longer than this: a late rank stalls every other rank's collectives.
HOST_GAP_WARN = 0.1
# Post-step host work above this is logged with the Python stacks that ran during it (see `host_stall`).
HOST_STALL_SAMPLE_THRESHOLD = 0.3
# Parameter groups of the per-group EMA blend probe (regex over the pytree key path); "other" is the rest.
EMA_BLEND_GROUPS: dict[str, str] = {
    "lmhead": r"output_proj",
    "embed": r"token_embed",
    "attn": r"\.attn\b|\.attn\.",
    "moe": r"\.mlp\b|\.shared\b",
}


def _ema_blend_group(path: str) -> str:
    """The ``EMA_BLEND_GROUPS`` group of a key path, or ``other``."""
    return next((group for group, pattern in EMA_BLEND_GROUPS.items() if re.search(pattern, path)), "other")


# Seed of the training-only router noise (``moe_gumbel_tau``); folded with the step.
ROUTE_NOISE_SEED = 7
GC_EVERY_STEPS = 50
GC_PAUSE_WARN = 0.05
_gc_start: dict[str, float] = {}


def _warn_on_long_gc(phase: str, info: dict) -> None:
    if phase == "start":
        _gc_start["t"] = time.perf_counter()
        return
    pause = time.perf_counter() - _gc_start.get("t", time.perf_counter())
    if pause > GC_PAUSE_WARN:
        logger.warning("gc gen%d pause %.3f s on process %d", info.get("generation", -1), pause, jax.process_index())


def _write_train_step_hlo(
    train_step, state, batch, loop_active: bool | None, router_tie_active: bool, path: str
) -> None:
    """Write the optimized HLO of ``train_step`` (with op metadata) to ``path``; recompiles once."""
    text = (
        train_step.lower(state, batch, loop_active=loop_active, router_tie_active=router_tie_active).compile().as_text()
    )
    with fsspec.open(path, "w") as f:
        f.write(text)
    logger.info("Wrote train-step HLO (%d chars) to %s", len(text), path)


_XLA_DUMP_DIR = "/tmp/grug_xla_dump"
_XLA_MEMORY_REPORT_SUFFIXES = ("buffer-assignment.txt", "memory-usage-report.txt")


def _xla_memory_report_flags() -> str:
    return f"--xla_dump_to={_XLA_DUMP_DIR} --xla_dump_hlo_module_re=jit_train_step --xla_dump_hlo_as_text"


def _upload_xla_memory_reports(dest: str) -> None:
    reports = [
        path
        for path in glob.glob(f"{_XLA_DUMP_DIR}/*jit_train_step*")
        if path.endswith(_XLA_MEMORY_REPORT_SUFFIXES) and "after_optimizations" in path
    ]
    for path in reports:
        with open(path, "rb") as src, fsspec.open(f"{dest.rstrip('/')}/{os.path.basename(path)}", "wb") as dst:
            dst.write(src.read())
    logger.info("uploaded %d XLA memory reports to %s", len(reports), dest)


def _run_grug_local_with_memory_report(config: GrugRunConfig) -> None:
    try:
        _run_grug_local(config)
    finally:
        if jax.process_index() == 0:
            _upload_xla_memory_reports(config.trainer.xla_memory_report_path)


def run_grug(config: GrugRunConfig) -> None:
    """Dispatch grug training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    # Dispatch snapshots os.environ for the child task, so apply the runtime defaults first.
    inline_watch_enabled = trainer.watch.is_enabled and config.trainer.watch_mode == WatchMode.INLINE
    ragged = config.model.moe_implementation == RAGGED_MOE_IMPLEMENTATION
    _apply_runtime_defaults(
        inline_watch_enabled=inline_watch_enabled, ragged_transport=config.ragged_transport if ragged else None
    )
    local_entrypoint = _run_grug_local
    if config.trainer.xla_memory_report_path is not None:
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {_xla_memory_report_flags()}".strip()
        local_entrypoint = _run_grug_local_with_memory_report
    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=local_entrypoint,
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
    "RaggedTransport",
    "initial_state",
    "run_grug",
]
