# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import dataclasses
import faulthandler
import functools
import gc
import glob
import hashlib
import io
import json
import logging
import math
import os
import subprocess
import sys
import threading
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
from jax.experimental import multihost_utils
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
from experiments.grug.fast_track.host_stall import HostStallSampler
from experiments.grug.fast_track.library_kernels import load_library_kernels
from experiments.grug.fast_track.model import (
    DenseMLP,
    GrugModelConfig,
    LayerSkip,
    Transformer,
    forward_flops_per_token,
)

# This file intentionally mirrors `experiments/grug/base/train.py` with
# variant-specific model/loss/FLOP wiring, per the grug copy-first workflow in
# `.agents/skills/change-grug/`.

logger = logging.getLogger(__name__)

_HOST_MEMORY_LIMIT_ENV = "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"
RUNTIME_ENV = {
    "LD_PRELOAD": "libjemalloc.so.2",
    "MALLOC_CONF": "background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0,narenas:2",
    # Per-process CUPTI sessions collide with each other and CoreWeave's DCGM, so PGLE only adds failure modes.
    "JAX_ENABLE_PGLE": "false",
    "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": "192",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
}
XLA_COLLECTIVE_OVERLAP_FLAG = "--xla_gpu_experimental_parallel_collective_overlap_limit"
XLA_FUSION_AUTOTUNER_FLAG = "--xla_gpu_experimental_enable_fusion_autotuner"
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

    ``ONE_SHOT`` writes straight into peer GPUs' buffers over peer pointers, which XLA only allows when one process
    owns every GPU of the collective (``processes_per_task=1``). ``DEVICE`` goes through NCCL symmetric buffers and,
    like ``NCCL``, runs with one process per GPU (the GB200 hero runs it with 4 processes per node).
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


def _apply_runtime_defaults(
    *,
    inline_watch_enabled: bool,
    ragged_transport: RaggedTransport | None,
    collective_overlap_limit: int | None,
    processes_per_task: int,
) -> None:
    """Set the runtime env and XLA flag defaults; ``ragged_transport`` is None unless the MoE is ragged, and
    ``collective_overlap_limit`` (None: by mode) overrides the concurrent-collective limit."""
    for name, value in RUNTIME_ENV.items():
        os.environ.setdefault(name, value)
    xla_flags = os.environ.get("XLA_FLAGS", "").split()
    if collective_overlap_limit is not None:
        overlap_limit = collective_overlap_limit
    elif ragged_transport is not None:
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
    if processes_per_task > 1:
        # XLA's experimental block-level fusion autotuner hangs new multi-process GB200 compiles: with sharded
        # autotuning every rank waits on the exchange, and with autotuning off its untuned default kernels hang
        # the first step (lc1-gb200-kma48nogate{,2,3}). GEMM autotuning is unaffected.
        flag_defaults += (f"{XLA_FUSION_AUTOTUNER_FLAG}=false",)
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


TRAIN_LOADER_BUFFER_SIZE = 512
# Models with at most this many experts log each expert's QB bias (``moe_bias/qb_beta_L*_E*``).
_LOGGED_QB_BETA_MAX_EXPERTS = 16


class MasterParamMode(StrEnum):
    """Where the authoritative weights live.

    DEVICE keeps them as the device params themselves, with no separate master copy; FP32_PINNED_HOST keeps a
    pinned-host fp32 master while the device params are its cast to the param dtype (as in grug/moe_hero_ep).
    """

    DEVICE = "device"
    FP32_PINNED_HOST = "fp32_pinned_host"


_FP32_POLICY = jmp.get_policy("params=float32,compute=float32,output=float32")


def _tree_to_memory_kind(tree, memory_kind: str):
    """Move named-sharded arrays to a JAX memory kind."""

    def _move(leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        sharding = jax.typeof(leaf).sharding
        mesh = getattr(sharding, "mesh", None)
        if mesh is None or len(getattr(mesh, "axis_names", ())) == 0:
            if jax.sharding.get_abstract_mesh().empty:
                return leaf
            # Scalar optimizer metadata has no operand to inherit the active mesh from; bind it before changing
            # the memory kind so the initial and updated states have the same JIT input signature.
            leaf = jax.sharding.reshard(leaf, P())
            sharding = jax.typeof(leaf).sharding
        return jax.device_put(leaf, sharding.with_memory_kind(memory_kind))

    return jax.tree.map(_move, tree)


@dataclass(frozen=True)
class LayerSkipRule:
    """Tokens that skip ``layers`` (no attention or MLP output there; their K/V still computed) in training and
    eval. A token matches by document position (``positions = (lo, hi)``, ``hi < 0`` for open-ended) or by id
    (``token_ids``); a matching token skips with probability ``rate``, drawn by a fixed hash of (position,
    token id, seed) so train and eval pick the same tokens."""

    layers: tuple[int, ...]
    rate: float = 1.0
    positions: tuple[int, int] | None = None
    token_ids: tuple[int, ...] = ()
    seed: int = 0

    def __post_init__(self):
        if (self.positions is None) == (not self.token_ids):
            raise ValueError("LayerSkipRule needs exactly one of positions or token_ids")
        if not 0.0 < self.rate <= 1.0:
            raise ValueError(f"LayerSkipRule rate must be in (0, 1], got {self.rate}")


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
    # After the last step, run ``TOKEN_DUMP_BATCHES`` train-size batches of held-out sequences and write each token's
    # next-token loss here (``write_token_dump``).
    token_dump_path: str | None = None
    # The token dump also records each token's loss with every token skipping these layers (``LayerSkip``).
    token_dump_skip_layers: tuple[int, ...] = ()
    # XLA GPU autotune results shared across runs of the same program (``_use_xla_autotune_cache``): loaded at start
    # when present, written by process 0 after the first step otherwise. None: autotune every compile.
    xla_autotune_cache_path: str | None = None
    # Seed ``xla_autotune_cache_path`` from one process: run under this mock GPU topology ("slices x hosts x
    # devices"; "1x64x1" for a GB200 rack, which runs one process per GPU), so the one process compiles the full
    # mesh's train step and autotunes it alone (no cross-process exchange); compile ahead of time, publish the
    # results and stop before running a step. None: a normal run.
    xla_autotune_seed_topology: str | None = None
    # Before the first step, run each cuBLAS call of the compiled train step once alone on each local device
    # (``library_kernels``), so no kernel library loads mid-step (a first-step deadlock on GB200).
    preload_library_kernels: bool = False
    # Exit the process (``FIRST_STEP_HANG_EXIT_CODE``) when the first step, compiled and preloaded, has not finished
    # this many seconds after its dispatch, so the scheduler restarts a gang that deadlocked at startup (a lazy
    # kernel load inside the step on GB200). None: wait forever.
    first_step_timeout: float | None = None
    # Train (and eval) with these tokens skipping some layers (``LayerSkipRule``); None: every token runs every layer.
    layer_skip: LayerSkipRule | None = None
    # When the extra_mlp_layers MLPs switch on, as a fraction of training (0: from the start).
    extra_mlp_start_frac: float = 0.5
    # Train steps dispatched before the previous step's host work: 1 overlaps that work (logging, callbacks) with
    # the devices; 0 finishes every step before dispatching the next. 1 crashes the devices around step 515
    # (CUDA_ERROR_ILLEGAL_ADDRESS, with or without PGLE; batch 160), so it is opt-in until that is understood.
    pipeline_depth: int = 0
    # Batches the train loader's background thread keeps on the devices ahead of the step.
    loader_buffer_batches: int = TRAIN_LOADER_BUFFER_SIZE
    # Dump XLA's buffer assignment and memory-usage report for the train step and upload them here (process 0),
    # also when the step fails, e.g. with an out-of-memory error: attributes the temp buffer to HLO values.
    xla_memory_report_path: str | None = None
    # Weight EMA over the last ``ema_last_steps`` steps (the whole run when None): until then the EMA
    # tracks the params, afterwards ``ema <- ema_beta * ema + (1 - ema_beta) * params``. Evals from the
    # EMA start on score the EMA. None: no EMA.
    ema_beta: float | None = None
    ema_last_steps: int | None = None
    # Written by process 0 once the final eval has run; a restarted job that finds it exits instead of retraining
    # (a preemption during the post-training blend evals otherwise reran the whole run).
    completion_marker_path: str | None = None
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
    # Hero-scale memory: optimizer state on pinned host memory between steps, and where the master weights live.
    offload_opt_state: bool = False
    master_param_mode: MasterParamMode = MasterParamMode.DEVICE


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
    # XLA's concurrent-collective limit; None takes the mode default (ragged 1, inline watch 1, otherwise 4).
    collective_overlap_limit: int | None = None
    # Retry budgets for the training job. The two are separate gates and the job fails when either
    # one trips, thus raise them together. The defaults make a failure terminal, which is what a run
    # that cannot resume wants: a retry would repeat it from step 0. Only a run that both saves and
    # restores checkpoints benefits from a deep budget.
    max_retries_failure: int = 0
    max_task_failures: int = 10

    def __post_init__(self) -> None:
        if (
            self.model.moe_implementation == RAGGED_MOE_IMPLEMENTATION
            and self.ragged_transport is RaggedTransport.ONE_SHOT
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
# On one GB200 tray, four-batch requests delivered the first data in 3.7s and sustained 3.4 batches/s.
_TRAIN_LOADER_FETCH_BATCH_SIZE = 4


def build_train_loader(
    dataset: AsyncDataset[GrugLmExample],
    *,
    batch_schedule: BatchSchedule,
    mesh: Mesh,
    max_buffered_batches: int = TRAIN_LOADER_BUFFER_SIZE,
) -> DataLoader[GrugLmExample]:
    # DataLoader uses this batch axis mapping to shard batches across the distributed mesh.
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    return DataLoader(
        dataset,
        batch_schedule.schedule,
        max_buffered_batches=max_buffered_batches,
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


def dropless_moe_implementation(cfg: GrugModelConfig) -> MoeImplementation:
    """The dropless MoE kernel for evals and probes: sonic needs equal expert read and write widths; scatter also
    handles ``latent_out_dim`` and the full-width tail layers of ``latent_out_full_layers``."""
    if cfg.expert_in_dim == cfg.expert_out_dim and not cfg.latent_out_full_layers:
        return "sonic"
    return "scatter"


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
        return banks

    dropless = [
        dataclasses.replace(expert_mlp, implementation=implementation, expert_chunks=1, fp8_dispatch=False)
        for expert_mlp in stack_expert_mlps(model)
    ]
    return eqx.tree_at(stack_expert_mlps, model, dropless)


def _layer_stacks(model: Transformer):
    return (
        model.stacked_blocks,
        model.kda_blocks,
        model.stacked_blocks_rest,
        model.kda_blocks_rest,
        model.stacked_blocks_tail,
        model.kda_blocks_tail,
    )


def _cast_to_compute(mp: jmp.Policy, model: Transformer) -> Transformer:
    """The compute-dtype model. Under ``scan_layers`` the layer stacks stay in the param dtype: the scan casts one
    unit's weights at a time, so the step never holds a compute copy (or its gradient) of every layer's weights."""
    compute = mp.cast_to_compute(model)
    if not model.config.scan_layers:
        return compute
    return eqx.tree_at(_layer_stacks, compute, replace=_layer_stacks(model), is_leaf=lambda x: x is None)


def build_tagged_evaluator(
    *,
    data_config: LmDataConfig,
    max_seq_len: int,
    mesh: Mesh,
    eval_cfg: GrugEvalConfig,
    mp: jmp.Policy,
    model_transform: Callable[[Transformer], Transformer] | None = None,
    layer_skip: LayerSkipRule | None = None,
) -> TaggedEvaluator[LmExample | GrugLmExample, Transformer] | None:
    pos = Axis("position", max_seq_len)
    tagged_eval_sets = data_config.tagged_eval_sets(pos)
    if len(tagged_eval_sets) == 0:
        logger.warning("No evaluation datasets provided.")
        return None
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
            layer_skip=_layer_skip_mask(layer_skip, batch.tokens, batch.attn_mask),
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
        max_examples_per_dataset=None,
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


def _compute_match_flops(
    *,
    model_config: GrugModelConfig,
) -> float:
    """Training FLOPs/example under the generic GQA/GLU formula that ``_BASELINE_FLOPS_PER_EXAMPLE`` in
    ``launch.py`` was recorded with; ``--match compute`` budgets stay in these units. MFU and the logged
    throughput use the architecture-accurate ``forward_flops_per_token`` instead."""
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

    return 3 * flops_per_token * model_config.max_seq_len


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
    # A stack with fewer experts than the widest layer (final_num_experts) takes only its own columns.
    per_stack = [
        new_bias[np.asarray(indices)][:, : stack.stacked.mlp.router_bias.shape[-1]]
        for stack, indices in zip(model.layer_stacks(), model.stack_layer_indices(), strict=True)
    ]
    return eqx.tree_at(lambda t: [stack.stacked.mlp.router_bias for stack in t.layer_stacks()], model, per_stack)


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None = None,
    offload_opt_state: bool = False,
    master_param_mode: MasterParamMode = MasterParamMode.DEVICE,
) -> GrugTrainState:
    initialized = Transformer.init(model_config, key=key)
    if master_param_mode == MasterParamMode.FP32_PINNED_HOST:
        master_params = _FP32_POLICY.cast_to_param(initialized)
        params = mp.cast_to_param(master_params)
        opt_state = optimizer.init(master_params)
        master_params = _tree_to_memory_kind(master_params, "pinned_host")
    else:
        params = mp.cast_to_param(initialized)
        master_params = None
        opt_state = optimizer.init(params)
    if offload_opt_state:
        opt_state = _tree_to_memory_kind(opt_state, "pinned_host")
    return GrugTrainState(
        step=jax.sharding.reshard(jnp.array(0, dtype=jnp.int32), P()),
        params=params,
        master_params=master_params,
        ema_params=params if ema_beta is not None else None,
        opt_state=opt_state,
        pending_qb_betas=jnp.zeros((model_config.num_layers, model_config.qb_num_experts)),
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


def _doc_positions(tokens: jax.Array, mask) -> jax.Array:
    """Each token's position within its document (segment), ``[B, S]``."""
    seg = None if not isinstance(mask, AttentionMask) or mask.segment_ids is None else mask.segment_ids[0]
    idx = jnp.broadcast_to(jnp.arange(tokens.shape[1], dtype=jnp.int32), tokens.shape)
    if seg is None:
        return idx
    starts = jnp.concatenate([jnp.ones_like(seg[:, :1], bool), seg[:, 1:] != seg[:, :-1]], axis=1)
    return idx - jax.lax.cummax(jnp.where(starts, idx, 0), axis=1)


def _hash_uniform(pos: jax.Array, tokens: jax.Array, seed: int) -> jax.Array:
    """A fixed pseudo-random uniform in [0, 1) per (position, token id, seed)."""
    h = pos.astype(jnp.uint32) * jnp.uint32(2654435761) + tokens.astype(jnp.uint32) * jnp.uint32(2246822519)
    h = h + jnp.uint32((seed * 3266489917) % (1 << 32))
    h = h ^ (h >> 15)
    h = h * jnp.uint32(2246822519)
    h = h ^ (h >> 13)
    return h.astype(jnp.float32) / jnp.float32(1 << 32)


def _layer_skip_mask(rule: LayerSkipRule | None, tokens: jax.Array, mask) -> LayerSkip | None:
    if rule is None:
        return None
    pos = _doc_positions(tokens, mask)
    if rule.positions is not None:
        lo, hi = rule.positions
        match = (pos >= lo) & ((pos < hi) if hi >= 0 else True)
    else:
        table = np.zeros(int(max(rule.token_ids)) + 1, bool)
        table[np.asarray(rule.token_ids)] = True
        # The replicated table gathered at the batch-sharded ids keeps the ids' sharding.
        lookup = jnp.asarray(table).at[jnp.minimum(tokens, len(table) - 1)].get(out_sharding=jax.typeof(tokens).sharding)
        match = lookup & (tokens < len(table))
    skip = match & (_hash_uniform(pos, tokens, rule.seed) < rule.rate)
    return LayerSkip(rule.layers, ~skip)


def _loss_and_grads(
    params,
    batch,
    mp: jmp.Policy,
    z_loss: float | None,
    layer_skip: LayerSkipRule | None = None,
    extra_mlp_active: jax.Array | bool = True,
):
    def loss_fn(model):
        compute_params = _cast_to_compute(mp, model)
        skip = _layer_skip_mask(layer_skip, batch.tokens, batch.attn_mask)
        loss, metrics = compute_params.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="mean",
            logsumexp_weight=z_loss,
            return_router_metrics=True,
            train_terms=True,
            layer_skip=skip,
            extra_mlp_active=extra_mlp_active,
        )
        if skip is not None:
            metrics = {**metrics, "train/aux/layer_skip_frac": jnp.mean(1.0 - skip.keep.astype(jnp.float32))}
        return loss, metrics

    return jax.value_and_grad(loss_fn, has_aux=True)(params)


def _accumulated_loss_and_grads(
    num_microbatches: int,
    params,
    batch,
    mp: jmp.Policy,
    z_loss,
    layer_skip: LayerSkipRule | None = None,
    extra_mlp_active: jax.Array | bool = True,
):
    """``_loss_and_grads`` over ``num_microbatches`` equal slices of ``batch``, one at a time: the loss, gradients
    and float metrics are averaged, integer metrics (counts) summed. Every microbatch runs in one ``lax.scan``
    body from a zero accumulator, so the forward and backward compile once and one microbatch's activations are
    live at a time."""
    micro = reshape_batch_into_microbatches(batch, num_microbatches)

    def one(microbatch):
        return _loss_and_grads(params, microbatch, mp, z_loss, layer_skip, extra_mlp_active)

    shapes = jax.eval_shape(one, jax.tree.map(lambda x: x[0], micro))
    zeros = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype, out_sharding=getattr(s.sharding, "spec", None)), shapes)
    total, _ = jax.lax.scan(lambda acc, mb: (jax.tree.map(jnp.add, acc, one(mb)), None), zeros, micro)
    (loss, metrics), grads = total

    def mean(x):
        return x / num_microbatches if jnp.issubdtype(x.dtype, jnp.floating) else x

    return (loss / num_microbatches, jax.tree.map(mean, metrics)), jax.tree.map(mean, grads)


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
    ema_start_step: int = 0,
    watch_config: WatchConfig | None = None,
    grad_accum_microbatches: int = 1,
    offload_opt_state: bool = False,
    master_param_mode: MasterParamMode = MasterParamMode.DEVICE,
    layer_skip: LayerSkipRule | None = None,
    extra_mlp_start_step: int = 0,
):
    """``extra_mlp_start_step``: the first step the ``extra_mlp_layers`` MLPs are on.
    ``grad_accum_microbatches`` > 1 averages gradients over that many microbatches
    (``_accumulated_loss_and_grads``). ``offload_opt_state`` keeps the optimizer state on pinned host memory
    between steps; ``master_param_mode`` picks where the authoritative weights live (``MasterParamMode``)."""
    one = jnp.array(1, dtype=jnp.int32)
    z_loss = z_loss_weight if z_loss_weight > 0 else None
    if watch_config is not None:
        if isinstance(watch_config.watch_targets, str):
            watch_targets = tuple(t.strip() for t in watch_config.watch_targets.split(","))
        else:
            watch_targets = tuple(watch_config.watch_targets)
    else:
        watch_targets = ()

    def train_step(state: GrugTrainState, batch):
        # Apply pending QB betas to router biases inside JIT (avoids eager
        # host-side kernel launches that can cause SPMD sync issues).
        with jax.named_scope("qb_apply"):
            qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)
        extra_active = state.step >= extra_mlp_start_step

        if grad_accum_microbatches > 1:
            (loss, summarized_metrics), grads = _accumulated_loss_and_grads(
                grad_accum_microbatches, qb_params, batch, mp, z_loss, layer_skip, extra_active
            )
        else:
            (loss, summarized_metrics), grads = _loss_and_grads(qb_params, batch, mp, z_loss, layer_skip, extra_active)
        metrics = {"train/loss": loss, **summarized_metrics}
        opt_state_in = _tree_to_memory_kind(state.opt_state, "device") if offload_opt_state else state.opt_state
        if os.environ.get("GRUG_SKIP_OPTIMIZER"):
            # MFU-baseline mode: run fwd+bwd (grads above) but skip the optimizer update (Muon NS).
            # Params/master/opt_state pass through unchanged, so loss does not descend -- this isolates
            # the optimizer's share of step time (MFU_skip vs MFU_full = the Muon overhead).
            updates = jax.tree_util.tree_map(jnp.zeros_like, qb_params)
            opt_state = opt_state_in
            params = qb_params
            master_params = state.master_params
        elif master_param_mode == MasterParamMode.FP32_PINNED_HOST:
            if state.master_params is None:
                raise ValueError("master_params must be initialized for an fp32 pinned-host master")
            master_in = _apply_qb_betas(_tree_to_memory_kind(state.master_params, "device"), state.pending_qb_betas)
            with jax.named_scope("opt_update"):
                updates, opt_state = optimizer.update(_FP32_POLICY.cast_to_param(grads), opt_state_in, master_in)
            with jax.named_scope("apply_updates"):
                master_params = optax.apply_updates(master_in, updates)
            params = mp.cast_to_param(master_params)
            master_params = _tree_to_memory_kind(master_params, "pinned_host")
        else:
            with jax.named_scope("opt_update"):
                updates, opt_state = optimizer.update(grads, opt_state_in, qb_params)
            with jax.named_scope("apply_updates"):
                params = optax.apply_updates(qb_params, updates)
            master_params = None
        if offload_opt_state:
            opt_state = _tree_to_memory_kind(opt_state, "pinned_host")

        if ema_beta is None:
            ema_params = None
        else:
            # EMA tracks the QB-biased params, so re-apply the pending betas before blending.
            with jax.named_scope("ema"):
                qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas)
                ema_active = state.step >= ema_start_step
                ema_params = jax.tree_util.tree_map(
                    lambda old, new: jnp.where(ema_active, ema_beta * old + (1.0 - ema_beta) * new, new),
                    qb_ema_params,
                    params,
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
            pending_qb_betas=metrics.get("qb_beta_per_layer", state.pending_qb_betas),
        )
        if next_state.pending_qb_betas.shape[-1] <= _LOGGED_QB_BETA_MAX_EXPERTS:
            # Few experts (the specialization study models): log every expert's QB balancing bias.
            for layer in range(next_state.pending_qb_betas.shape[0]):
                for expert in range(next_state.pending_qb_betas.shape[1]):
                    metrics[f"moe_bias/qb_beta_L{layer}_E{expert}"] = next_state.pending_qb_betas[layer, expert]

        return next_state, metrics, watch_stats

    return train_step


def _flat_train_step(step_fn, treedef):
    """``step_fn`` (``_make_train_step``'s step) jitted over the train state's flat leaves, donated.

    Passing the ``GrugTrainState`` itself makes every call flatten the whole equinox tree in Python (one callback per
    module node: 25-40 ms per step at d512, with the GPUs idle). A list of arrays takes JAX's C++ path; the tree is
    rebuilt inside the trace, and ``treedef`` must stay the step's output structure."""

    @functools.partial(jax.jit, donate_argnums=(0,))
    def train_step(leaves, batch):
        state = jax.tree.unflatten(treedef, leaves)
        next_state, metrics, watch_stats = step_fn(state, batch)
        next_leaves, next_treedef = jax.tree.flatten(next_state)
        if next_treedef != treedef:
            raise ValueError("the train step changed the state's tree structure")
        return next_leaves, metrics, watch_stats

    return train_step


def _segment_ids(batch) -> jax.Array | None:
    segment_ids = batch.attn_mask.segment_ids
    return None if segment_ids is None else segment_ids[0]


_PY_SPY_TIMEOUT = 60


def _stall_diagnostic(timeout) -> None:
    """Before the progress watchdog kills a stalled process: every Python thread's stack, then a native (C++) dump of
    this process from py-spy when the container allows it, so a hang in compilation or a collective is visible."""
    logger.critical("stall diagnostic: %s; Python thread stacks follow", timeout)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    try:
        dump = subprocess.run(
            ["py-spy", "dump", "--native", "--pid", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=_PY_SPY_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.critical("py-spy dump unavailable: %s", e)
        return
    logger.critical("py-spy native dump (exit %d):\n%s%s", dump.returncode, dump.stdout, dump.stderr)


def _run_grug_local(config: GrugRunConfig) -> None:
    """Entry point for the grug template training loop."""
    # The GPU client reads the pinned-host limit when it is created, which happens in this process after the
    # dispatcher: an environment variable forwarded from the coordinator does not reach it on GB200 (probe
    # lc1-hostlim2 saw it unset and capped at 64 GB), so set it here before any device is touched.
    os.environ.setdefault(_HOST_MEMORY_LIMIT_ENV, RUNTIME_ENV[_HOST_MEMORY_LIMIT_ENV])
    seed_topology = config.trainer.xla_autotune_seed_topology
    if seed_topology is not None:
        if config.trainer.xla_autotune_cache_path is None:
            raise ValueError("xla_autotune_seed_topology needs xla_autotune_cache_path")
        _seed_xla_autotune_cache(config, seed_topology)
        return
    autotune_cache_loaded = (
        _use_xla_autotune_cache(config.trainer.xla_autotune_cache_path)
        if config.trainer.xla_autotune_cache_path is not None
        else None
    )
    marker = config.trainer.completion_marker_path
    if marker is not None and fsspec.core.url_to_fs(marker)[0].exists(marker):
        logger.info("run already completed (%s exists); not retraining", marker)
        return
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
    train_step_fn, diagnostic_watch_step = _build_train_step(config, optimizer)
    ema_start_step = _ema_start_step(config)

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
    progress_watchdog = trainer.progress_watchdog.create(process_index=jax.process_index(), diagnostic=_stall_diagnostic)

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
                offload_opt_state=config.trainer.offload_opt_state,
                master_param_mode=config.trainer.master_param_mode,
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
            max_buffered_batches=config.trainer.loader_buffer_batches,
        )

        flops_per_token = forward_flops_per_token(state.params)
        flops_per_example = 3 * flops_per_token * config.model.max_seq_len
        levanter.tracker.log_summary(
            {
                "throughput/flops_per_token_analytic": flops_per_token,
                "throughput/flops_per_example_analytic": flops_per_example,
                "throughput/compute_match_flops_per_example": _compute_match_flops(model_config=config.model),
            }
        )

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
                    layer_skip=config.trainer.layer_skip,
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
                        layer_skip=config.trainer.layer_skip,
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
            # From the EMA start on, evals score the EMA weights.
            # Lagged steps (``_LaggedStateView``) carry no params; only the per-step hooks see them.
            eval_model_getter=lambda s: (
                None
                if s.params is None
                else (s.ema_params if s.ema_params is not None and int(s.step) > ema_start_step else s.params)
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
                    # One directory per process: processes on one host would otherwise write the same trace file.
                    str(trainer.log_dir / run_id / "profiler" / f"process{jax.process_index()}"),
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
                # The forced end-of-run callback pass revisits the last step; skip it when the
                # periodic cadence already scored that step, as `cb_tagged_evaluate` does.
                last_dropless_eval_step: int | None = None

                def dropless_eval_hook(
                    step, *args, _mesh=dropless_eval_mesh, _ev=dropless_evaluator, _prefix=("eval"), **kwargs
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

        last_loss = 0.0
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
        autotune_published = False

        # The step runs on the state's flat leaves (``_flat_train_step``); ``state`` is rebuilt from them each step.
        state_leaves, state_treedef = jax.tree.flatten(state)
        train_step = _flat_train_step(train_step_fn, state_treedef)
        # The train step's optimized HLO, set when the first step compiles it ahead of time.
        train_step_hlo: str | None = None
        first_step_watchdog: threading.Timer | None = None

        eval_interval = eval_cfg.steps_per_eval if eval_cfg is not None and eval_cfg.steps_per_eval else None
        last_ready_time: float | None = None

        def finish_step(rec: "_PendingStep", step_state: GrugTrainState | None) -> None:
            """Wait for ``rec``'s step and do its host work. With ``step_state`` (the state after the step, not yet
            donated) the state-dependent work runs too; without it the step was finished lagged, behind the next
            dispatch, and the callbacks see only its step number and loss."""
            nonlocal state, state_leaves, last_loss, last_step_duration, host_gap_start, hlo_written, last_ready_time
            nonlocal autotune_published, first_step_watchdog
            metrics, step = rec.metrics, rec.step
            # A host float, and no device math on the metrics below: a device op would queue behind the step
            # dispatched after this one and wait for it, undoing the overlap.
            loss = float(jax.device_get(metrics["train/loss"]))
            ready_time = time.perf_counter()
            stall_sampler.arm()
            # The devices' time for this step: from its dispatch, or from the previous step's completion when it was
            # dispatched before that (the pipelined case).
            duration = ready_time - (rec.step_start if last_ready_time is None else max(last_ready_time, rec.step_start))
            last_ready_time = ready_time
            if step_state is not None and config.trainer.hlo_dump_path is not None and not hlo_written:
                if jax.process_index() == 0:
                    assert train_step_hlo is not None
                    _write_train_step_hlo(train_step_hlo, config.trainer.hlo_dump_path)
                hlo_written = True
            state_callbacks.emit_event(callbacks.ProgressEvent.TRAIN_STEP_FINISHED)
            if first_step_watchdog is not None:
                first_step_watchdog.cancel()
                first_step_watchdog = None
            if autotune_cache_loaded is False and not autotune_published:
                # The first finished step: train_step compiled, so its autotune results are on disk.
                _publish_xla_autotune_cache(config.trainer.xla_autotune_cache_path)
                autotune_published = True
            if not math.isfinite(loss):
                raise RuntimeError(f"Non-finite loss ({loss}) at step {step + 1}.")
            hook_start = time.perf_counter()
            with jax.profiler.TraceAnnotation("callbacks"):
                view = step_state if step_state is not None else _LaggedStateView(step=step + 1)
                state_callbacks.run(view, loss=loss, step_duration=duration)
                last_loss = loss
                last_step_duration = duration
                levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=step)
                levanter.tracker.log({"throughput/dispatch_time": rec.dispatch_time}, step=step)
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
                                "train/optim/",
                            )
                        )
                        and key not in ("train/router/routing_counts_per_layer", "qb_beta_per_layer")
                    }
                )
                if router_metrics:
                    levanter.tracker.log(router_metrics, step=step)
                if "train/cross_entropy_loss" in metrics:
                    levanter.tracker.log({"train/cross_entropy_loss": metrics["train/cross_entropy_loss"]}, step=step)
                if "moe/dropped_assignments" in metrics:
                    drop_metrics = _drop_metrics(
                        metrics["moe/dropped_assignments"],
                        metrics["moe/sender_dropped_assignments"],
                        metrics["moe/receiver_dropped_assignments"],
                        batch_size=rec.batch.tokens.shape[0],
                        sequence_length=rec.batch.tokens.shape[1],
                        top_k=config.model.num_experts_per_token,
                        num_layers=config.model.num_layers,
                    )
                    levanter.tracker.log(jax.device_get(drop_metrics), step=step)
                if rec.watch_stats is not None:
                    levanter.tracker.log(jax.device_get(rec.watch_stats), step=step)
            if step_state is not None:
                if checkpointer is not None:
                    with callbacks.progress_event_scope(
                        state_callbacks.emit_event,
                        callbacks.ProgressEvent.CHECKPOINT_STARTED,
                        callbacks.ProgressEvent.CHECKPOINT_FINISHED,
                    ):
                        checkpointer.on_step(tree=step_state, step=step + 1)
            if step % GC_EVERY_STEPS == 0:
                gc.collect()
                gc.freeze()
            host_gap_start = time.perf_counter()
            stall_sampler.disarm(step)
            if host_gap_start - ready_time > HOST_GAP_WARN:
                logger.warning(
                    "post-step host work %.3f s after step %d on process %d",
                    host_gap_start - ready_time,
                    step,
                    jax.process_index(),
                )

        # Main optimization loop.
        try:
            with HostStallSampler(HOST_STALL_SAMPLE_THRESHOLD) as stall_sampler:
                # Pipelined loop: step n+1 is dispatched before step n's loss is waited on, so the host's dispatch
                # and logging overlap the devices' work. A step whose follow-up needs its state (an eval, a
                # checkpoint poll, the router-tie release, the HLO dump, the last step) is finished before the
                # next dispatch, since that dispatch donates the state.
                host_step = int(state.step)
                pending: _PendingStep | None = None
                if config.trainer.pipeline_depth not in (0, 1):
                    raise ValueError(f"pipeline_depth must be 0 or 1, got {config.trainer.pipeline_depth}")
                while host_step < stop_step:
                    with jax.profiler.TraceAnnotation("load_batch"):
                        batch = next(iterator)
                    if pending is not None and pending.needs_state:
                        finish_step(pending, state)
                        pending = None
                    current_step = host_step
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
                    if train_step_hlo is None:
                        # Compile ahead of time and run the executable directly: its HLO is then free to read.
                        compiled_step = train_step.lower(state_leaves, batch).compile()
                        train_step_hlo = compiled_step.as_text()
                        # Every rank must compile the same program; a mismatch deadlocks its collectives.
                        logger.info(
                            "train step HLO sha256 %s (%d chars) on process %d, device memory limit %s",
                            hashlib.sha256(train_step_hlo.encode()).hexdigest()[:16],
                            len(train_step_hlo),
                            jax.process_index(),
                            (jax.local_devices()[0].memory_stats() or {}).get("bytes_limit"),
                        )
                        if config.trainer.preload_library_kernels:
                            load_library_kernels(train_step_hlo, jax.local_devices())
                        train_step = compiled_step
                        if config.trainer.first_step_timeout is not None:
                            first_step_watchdog = threading.Timer(
                                config.trainer.first_step_timeout,
                                _exit_on_first_step_hang,
                                (config.trainer.first_step_timeout,),
                            )
                            first_step_watchdog.daemon = True
                            first_step_watchdog.start()
                    dispatch_start = time.perf_counter()
                    state_leaves, metrics, inline_watch_stats = train_step(state_leaves, batch)
                    state = jax.tree.unflatten(state_treedef, state_leaves)
                    # Host time to hand the step to the devices (rebuilding the state tree, launch).
                    dispatch_time = time.perf_counter() - dispatch_start
                    if inline_watch_stats is not None and watch_due:
                        watch_stats = inline_watch_stats
                    host_step += 1
                    if pending is not None:
                        # The previous step, finished while this one runs on the devices.
                        finish_step(pending, None)
                    pending = _PendingStep(
                        step=current_step,
                        metrics=metrics,
                        batch=batch,
                        watch_stats=watch_stats,
                        dispatch_time=dispatch_time,
                        step_start=step_start,
                        needs_state=(
                            host_step >= stop_step
                            or config.trainer.pipeline_depth == 0
                            or (eval_interval is not None and host_step % eval_interval == 0)
                            or host_step % CHECKPOINT_POLL_STEPS == 0
                            or (config.trainer.hlo_dump_path is not None and not hlo_written)
                        ),
                    )
                if pending is not None:
                    finish_step(pending, state)
                    pending = None

        except BaseException:
            logger.exception(
                "Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause"
            )
            raise
        else:
            # Mirror classic trainer behavior: force callbacks on the last completed step.
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if config.trainer.token_dump_path is not None:
                write_token_dump(
                    config, state, mesh, batch_schedule.batch_size_at_step(0), config.trainer.token_dump_path
                )
            if marker is not None and jax.process_index() == 0:
                with fsspec.open(marker, "w") as f:
                    json.dump({"step": int(state.step), "final_loss": float(last_loss)}, f)
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


TOKEN_DUMP_BATCHES = 4


def _held_out_sequences(data_config: LmDataConfig, *, seq_len: int, num_sequences: int) -> tuple[np.ndarray, np.ndarray]:
    """``num_sequences`` fixed held-out sequences and their segment ids: every validation set from its start,
    round-robin over the sets in name order."""
    sets = data_config.validation_sets(Axis("position", seq_len))
    if not sets:
        raise ValueError("the token dump needs validation sets")
    names = sorted(sets)
    datasets = {name: sets[name].as_sync_dataset() for name in names}
    lengths = {name: len(datasets[name]) for name in names}
    picks = [(name, index) for index in range(max(lengths.values())) for name in names if index < lengths[name]]
    if len(picks) < num_sequences:
        raise ValueError(f"the token dump needs {num_sequences} sequences, the validation sets hold {len(picks)}")
    picks = picks[:num_sequences]
    fetched = {}
    for name in names:
        indices = [index for n, index in picks if n == name]
        if indices:
            examples = [grug_lm_example_from_named(ex) for ex in datasets[name].get_batch(indices)]
            fetched.update({(name, index): ex for index, ex in zip(indices, examples, strict=True)})
    tokens = np.stack([np.asarray(fetched[pick].tokens, np.int32) for pick in picks])
    segments = np.stack(
        [
            np.zeros(seq_len, np.int32) if (seg := fetched[pick].attn_mask.segment_ids) is None else np.asarray(seg[0])
            for pick in picks
        ]
    ).astype(np.int32)
    return tokens, segments


def _token_dump_step(
    mp: jmp.Policy,
    num_real_experts: int,
    null_layers: bool,
    skip_layers: tuple[int, ...],
    mid_head: bool,
    final_conf: bool,
    skip_probe: bool,
):
    """Jitted ``(params, qb_betas, tokens, segment_ids) -> dict`` of per-token ``[B, S]`` arrays (``[L, B, S]`` for
    ``null_slots``): ``next_token_loss``, with ``null_layers`` each token's ``null_slots`` per layer, with
    ``skip_layers`` ``skipped_loss``, the loss with every token skipping those layers, and the ``Transformer``
    diagnostics: ``mid_head_diagnostics`` with ``mid_head``, ``final_conf_diagnostics`` with ``final_conf``, and
    ``skip_probe_diagnostics`` with ``skip_probe`` (over ``skip_layers``)."""

    @jax.jit
    def run(params: Transformer, qb_betas: jax.Array, tokens: jax.Array, segment_ids: jax.Array):
        model = _cast_to_compute(mp, _apply_qb_betas(params, qb_betas))
        mask = AttentionMask.causal().with_segment_ids(segment_ids)
        weight = jnp.ones(tokens.shape, jnp.float32)
        out = {"next_token_loss": model.next_token_loss(tokens, weight, mask=mask, reduction="none")}
        if null_layers:
            selected, _ = model.routing_assignments(tokens, mask)
            out["null_slots"] = jnp.sum(selected >= num_real_experts, axis=-1).astype(jnp.uint8)
        if skip_layers:
            skip = LayerSkip(skip_layers, jnp.zeros(tokens.shape, jnp.bool_))
            out["skipped_loss"] = model.next_token_loss(tokens, weight, mask=mask, reduction="none", layer_skip=skip)
        if mid_head:
            out.update(model.mid_head_diagnostics(tokens, mask))
        if final_conf:
            out.update(model.final_conf_diagnostics(tokens, mask))
        if skip_probe:
            out.update(model.skip_probe_diagnostics(tokens, mask, skip_layers))
        return out

    return run


def write_token_dump(config: GrugRunConfig, state: GrugTrainState, mesh: Mesh, batch_size: int, path: str) -> None:
    """Run held-out batches and write ``tokens``, ``segment_ids``, ``next_token_loss`` ``[N, S]`` (the loss of
    predicting token s+1 at s), and per ``_token_dump_step`` ``null_slots`` ``[L, N, S]`` and ``skipped_loss``."""
    cfg = config.model
    num_sequences = TOKEN_DUMP_BATCHES * batch_size
    # The dataset's per-example jit pins its output to a CPU device; JAX's mesh context is thread-local, so fetch
    # on a worker thread (as the data loader does).
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        tokens, segments = pool.submit(
            _held_out_sequences, config.data, seq_len=cfg.max_seq_len, num_sequences=num_sequences
        ).result()
    sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
    skip_layers = config.trainer.token_dump_skip_layers
    step = _token_dump_step(
        config.trainer.trainer.mp,
        cfg.num_experts,
        bool(cfg.moe_null_layers),
        skip_layers,
        cfg.mid_head_layer is not None,
        bool(cfg.final_conf_layers),
        cfg.skip_probe_layer is not None,
    )
    parts: list[dict[str, np.ndarray]] = []
    for start in range(0, num_sequences, batch_size):
        batch = [
            jax.make_array_from_callback(x.shape, sharding, lambda index, x=x: x[index])
            for x in (tokens[start : start + batch_size], segments[start : start + batch_size])
        ]
        # Gather the batch-sharded outputs onto every process (a multi-process run cannot device_get them).
        outputs = multihost_utils.process_allgather(step(state.params, state.pending_qb_betas, *batch), tiled=True)
        parts.append({k: np.asarray(v) for k, v in outputs.items()})
    if jax.process_index() != 0:
        return
    buffer = io.BytesIO()
    arrays = {k: np.concatenate([p[k] for p in parts], axis=1 if k == "null_slots" else 0) for k in parts[0]}
    for k, v in arrays.items():
        if v.dtype == np.float32:
            arrays[k] = v.astype(np.float16)
    np.savez(
        buffer,
        tokens=tokens,
        segment_ids=segments,
        null_layers=np.asarray(cfg.moe_null_layers, np.int32),
        slots_per_token=np.asarray(2 * cfg.num_experts_per_token),
        skip_layers=np.asarray(skip_layers, np.int32),
        step=np.asarray(int(state.step)),
        **arrays,
    )
    with fsspec.open(path, "wb") as f:
        f.write(buffer.getvalue())
    logger.info("Wrote the token dump to %s", path)


def _autotune_dump_file() -> str:
    # One file per process: the processes of a multi-GPU task share /tmp.
    return f"/tmp/xla_autotune_results_{os.getpid()}.textproto"


def _use_xla_autotune_cache(path: str) -> bool:
    """Point XLA at the shared autotune results: load ``path`` when it exists, and dump this process's results.

    Multi-process GB200 compiles hang in sharded autotuning's cross-process key-value exchange (every rank waits in
    ``ConfigAssigner::AssignConfigs`` for keys that never arrive), and loading results alone does not skip it. With a
    cache, every rank takes its configs from the same file with sharding off, so nothing is exchanged and all ranks
    pick identical kernels. Must run before the backend is created; returns whether a cache was loaded."""
    fs = fsspec.core.url_to_fs(path)[0]
    flags = [f"--xla_gpu_dump_autotune_results_to={_autotune_dump_file()}"]
    loaded = fs.exists(path)
    if loaded:
        local = f"/tmp/xla_autotune_cached_{os.getpid()}.textproto"
        fs.get(path, local)
        flags += [
            f"--xla_gpu_load_autotune_results_from={local}",
            "--xla_gpu_require_complete_aot_autotune_results=true",
            "--xla_gpu_shard_autotuning=false",
        ]
    os.environ["XLA_FLAGS"] = " ".join([os.environ.get("XLA_FLAGS", ""), *flags]).strip()
    logger.info("XLA autotune cache %s: %s", "loaded" if loaded else "empty, will publish", path)
    return loaded


def _ema_start_step(config: GrugRunConfig) -> int:
    num_steps, last = config.trainer.trainer.num_train_steps, config.trainer.ema_last_steps
    return 0 if last is None else max(0, num_steps - last)


def _build_train_step(config: GrugRunConfig, optimizer):
    """The jitted train step and the optional diagnostic watch step, shared by training and the autotune seed so
    both compile the same program."""
    trainer = config.trainer.trainer
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
    train_step_fn = _make_train_step(
        optimizer,
        trainer.mp,
        z_loss_weight=config.trainer.z_loss_weight,
        ema_beta=config.trainer.ema_beta,
        ema_start_step=_ema_start_step(config),
        watch_config=inline_watch_config,
        grad_accum_microbatches=config.trainer.grad_accum_microbatches,
        offload_opt_state=config.trainer.offload_opt_state,
        master_param_mode=config.trainer.master_param_mode,
        layer_skip=config.trainer.layer_skip,
        # Late by default, once the entropy head has learned something.
        extra_mlp_start_step=int(trainer.num_train_steps * config.trainer.extra_mlp_start_frac),
    )
    return train_step_fn, diagnostic_watch_step


def _seed_xla_autotune_cache(config: GrugRunConfig, topology: str) -> None:
    """Seed ``xla_autotune_cache_path`` from this one process standing in for ``topology``'s full mesh (mock NCCL):
    build the state and one real batch, compile the train step ahead of time with sharded autotuning off (so the
    process autotunes every fusion itself) and publish the results. Skips the trainer's host-level setup (tracker,
    distributed syncs, data loader), which waits for processes that do not exist. Must run before the backend is
    created."""
    assert config.trainer.xla_autotune_cache_path is not None
    # The mock compile's executables use no-op NCCL: they must never reach the persistent compilation cache, where a
    # real run's ranks would load them and deadlock in collectives that do nothing.
    jax.config.update("jax_enable_compilation_cache", False)
    jax.config.update("jax_mock_gpu_topology", topology)
    flags = [f"--xla_gpu_dump_autotune_results_to={_autotune_dump_file()}", "--xla_gpu_shard_autotuning=false"]
    os.environ["XLA_FLAGS"] = " ".join([os.environ.get("XLA_FLAGS", ""), *flags]).strip()
    logger.info("Seeding the XLA autotune cache under mock topology %s", topology)
    trainer = config.trainer.trainer
    optimizer = config.optimizer.build(trainer.num_train_steps)
    train_step_fn, _ = _build_train_step(config, optimizer)
    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
    )
    with set_mesh(mesh):
        state = jax.jit(
            lambda key: initial_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=key,
                ema_beta=config.trainer.ema_beta,
                offload_opt_state=config.trainer.offload_opt_state,
                master_param_mode=config.trainer.master_param_mode,
            )
        )(jax.random.PRNGKey(trainer.seed))
        dataset = build_train_dataset(
            config.data,
            max_seq_len=config.model.max_seq_len,
            batch_schedule=trainer.batch_schedule,
            key=jax.random.PRNGKey(0),
        ).as_sync_dataset()
        batch_size = trainer.batch_schedule.batch_size_at_step(0)
        host = jax.tree.map(
            lambda *xs: np.stack([np.asarray(x) for x in xs]), *dataset.get_batch(list(range(batch_size)))
        )
        batch = jax.tree.map(
            lambda x: jax.make_array_from_callback(
                x.shape, NamedSharding(mesh, P(_BATCH_AXES, *([None] * (x.ndim - 1)))), lambda index, x=x: x[index]
            ),
            host,
        )
        state_leaves, state_treedef = jax.tree.flatten(state)
        compile_start = time.perf_counter()
        _flat_train_step(train_step_fn, state_treedef).lower(state_leaves, batch).compile()
        logger.info("Seed compile of the train step took %.0f s", time.perf_counter() - compile_start)
    _publish_xla_autotune_cache(config.trainer.xla_autotune_cache_path)


def _publish_xla_autotune_cache(path: str) -> None:
    """Process 0 uploads its autotune results (written during the compile) to the shared cache."""
    dump = _autotune_dump_file()
    if jax.process_index() != 0 or not os.path.exists(dump):
        return
    with open(dump, "rb") as src, fsspec.open(path, "wb") as dst:
        dst.write(src.read())
    logger.info("Published the XLA autotune results to %s", path)


# Every rank logs host-side stalls longer than this: a late rank stalls every other rank's collectives.
HOST_GAP_WARN = 0.1
# Post-step host work above this is logged with the Python stacks that ran during it (see `host_stall`).
HOST_STALL_SAMPLE_THRESHOLD = 0.3


GC_EVERY_STEPS = 50
# The pipelined loop finishes a step with its state (for the checkpointer's time-based saves) at least this often.
CHECKPOINT_POLL_STEPS = 25


@dataclasses.dataclass(frozen=True)
class _LaggedStateView:
    """What the per-step callbacks see for a step finished behind the next dispatch: its step number only (the
    state was donated). Hooks that need the model (evals) run on steps finished with their state."""

    step: int
    params: None = None
    ema_params: None = None
    opt_state: None = None


@dataclasses.dataclass
class _PendingStep:
    """A dispatched step whose host work (wait, logging, callbacks) is still to do."""

    step: int
    metrics: dict
    batch: object
    watch_stats: object
    dispatch_time: float
    step_start: float
    needs_state: bool


GC_PAUSE_WARN = 0.05
_gc_start: dict[str, float] = {}


def _warn_on_long_gc(phase: str, info: dict) -> None:
    if phase == "start":
        _gc_start["t"] = time.perf_counter()
        return
    pause = time.perf_counter() - _gc_start.get("t", time.perf_counter())
    if pause > GC_PAUSE_WARN:
        logger.warning("gc gen%d pause %.3f s on process %d", info.get("generation", -1), pause, jax.process_index())


FIRST_STEP_HANG_EXIT_CODE = 75


def _exit_on_first_step_hang(timeout: float) -> None:
    logger.error("First train step not finished %.0f s after dispatch: exiting so the job restarts", timeout)
    faulthandler.dump_traceback(all_threads=True)
    os._exit(FIRST_STEP_HANG_EXIT_CODE)


def _write_train_step_hlo(text: str, path: str) -> None:
    """Write the train step's optimized HLO (with op metadata) to ``path``."""
    with fsspec.open(path, "w") as f:
        f.write(text)
    logger.info("Wrote train-step HLO (%d chars) to %s", len(text), path)


_XLA_DUMP_DIR = "/tmp/grug_xla_dump"
_XLA_MEMORY_REPORT_SUFFIXES = ("buffer-assignment.txt", "memory-usage-report.txt")
_XLA_MEMORY_REPORT_POLL_SECONDS = 5


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


def _upload_xla_memory_reports_when_written(dest: str, stop: threading.Event) -> None:
    """Upload the reports as soon as XLA writes them: a step that OOMs takes the peer processes down, and the
    coordination service then kills this one before any end-of-run upload."""
    while not stop.wait(_XLA_MEMORY_REPORT_POLL_SECONDS):
        if any(
            path.endswith(_XLA_MEMORY_REPORT_SUFFIXES) and "after_optimizations" in path
            for path in glob.glob(f"{_XLA_DUMP_DIR}/*jit_train_step*")
        ):
            # The dump files are written back to back; let the second one land. The dump means the backend
            # is up, so process_index no longer forces an early init.
            time.sleep(_XLA_MEMORY_REPORT_POLL_SECONDS)
            if jax.process_index() == 0:
                _upload_xla_memory_reports(dest)
            return


def _run_grug_local_with_memory_report(config: GrugRunConfig) -> None:
    dest = config.trainer.xla_memory_report_path
    stop = threading.Event()
    threading.Thread(target=_upload_xla_memory_reports_when_written, args=(dest, stop), daemon=True).start()
    try:
        _run_grug_local(config)
    finally:
        stop.set()
        if jax.process_index() == 0:
            _upload_xla_memory_reports(dest)


def run_grug(config: GrugRunConfig) -> None:
    """Dispatch grug training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    # Dispatch snapshots os.environ for the child task, so apply the runtime defaults first.
    inline_watch_enabled = trainer.watch.is_enabled and config.trainer.watch_mode == WatchMode.INLINE
    ragged = config.model.moe_implementation == RAGGED_MOE_IMPLEMENTATION
    _apply_runtime_defaults(
        inline_watch_enabled=inline_watch_enabled,
        ragged_transport=config.ragged_transport if ragged else None,
        collective_overlap_limit=config.collective_overlap_limit,
        processes_per_task=config.processes_per_task,
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
