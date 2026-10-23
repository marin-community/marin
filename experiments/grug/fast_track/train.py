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
from contextlib import nullcontext
from dataclasses import dataclass, field
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
from jax.sharding import Mesh, NamedSharding, reshard
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
from levanter.grug._moe.ep_ragged_all_to_all import RAGGED_NCCL_SEND_RECV_XLA_FLAGS
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
from experiments.grug.fast_track.model import (
    FINAL_NORM_INPUT_RMS_KEY,
    DenseMLP,
    GradProbeSite,
    GrugModelConfig,
    LayerSkip,
    Transformer,
    forward_flops_per_token,
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
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
}
XLA_COLLECTIVE_OVERLAP_FLAG = "--xla_gpu_experimental_parallel_collective_overlap_limit"
XLA_FUSION_AUTOTUNER_FLAG = "--xla_gpu_experimental_enable_fusion_autotuner"
DEFAULT_COLLECTIVE_OVERLAP_LIMIT = 4
# The dropless local MoE kernel of the evals: ``sonic`` is the Hopper Triton path.
DEFAULT_DROPLESS_MOE_IMPLEMENTATION: MoeImplementation = "sonic"
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
    """XLA GPU kernel behind ``ragged_all_to_all`` (both ship in the stock x86_64 PJRT plugin).

    ``ONE_SHOT`` writes straight into peer GPUs' buffers over peer pointers, which XLA only allows when one process
    owns every GPU of the collective (``processes_per_task=1``); ``NCCL`` runs with one process per GPU.
    """

    ONE_SHOT = "one_shot"
    """XLA's stock default: the host-launched one-shot copy kernel over peer pointers."""
    NCCL = "nccl"
    """NCCL send/recv per (peer, expert) update."""


RAGGED_TRANSPORT_XLA_FLAGS: dict[RaggedTransport, tuple[str, ...]] = {
    RaggedTransport.ONE_SHOT: (
        "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=false",
        "--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=true",
    ),
    RaggedTransport.NCCL: RAGGED_NCCL_SEND_RECV_XLA_FLAGS,
}
_RAGGED_TRANSPORT_FLAG_NAMES = frozenset(
    flag.partition("=")[0] for flags in RAGGED_TRANSPORT_XLA_FLAGS.values() for flag in flags
)


def restore_template_from(state):
    """ShapeDtypeStructs carrying each leaf's concrete sharding, releasing the leaves (the restore target)."""
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
    processes_per_task: int,
) -> None:
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


TRAIN_LOADER_BUFFER_SIZE = 512
# Models with at most this many experts log each expert's QB bias (``moe_bias/qb_beta_L*_E*``).
_LOGGED_QB_BETA_MAX_EXPERTS = 16


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
    log_every: int = 1
    z_loss_weight: float = 1e-4  # Weight on final-logit logsumexp z-loss stabilization term.
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
    # With ``lm_head_grad_dump_every``: write the train batch's per-token lm_head input-gradient norms and token ids
    # to ``<lm_head_grad_dump_dir>/step-<step>.npz`` on the ``lm_head_grad_dump_due`` steps.
    lm_head_grad_dump_dir: str | None = None
    # Train (and eval) with these tokens skipping some layers (``LayerSkipRule``); None: every token runs every layer.
    layer_skip: LayerSkipRule | None = None
    # When the extra_mlp_layers MLPs switch on, as a fraction of training (0: from the start).
    extra_mlp_start_frac: float = 0.5
    # Batches the train loader's background thread keeps on the devices ahead of the step.
    loader_buffer_batches: int = TRAIN_LOADER_BUFFER_SIZE
    # Dump XLA's buffer assignment and memory-usage report for the train step and upload them here (process 0),
    # also when the step fails, e.g. with an out-of-memory error: attributes the temp buffer to HLO values.
    xla_memory_report_path: str | None = None
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
    ragged_transport: RaggedTransport = RaggedTransport.ONE_SHOT
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
        datasets={name: dataset.map(_with_segment_ids) for name, dataset in datasets.items()},
        weights=weights,
        stop_strategy=data_config.stop_strategy,
        key=mix_key,
        block_size=data_config.mixture_block_size,
    )


def _with_segment_ids(example: GrugLmExample) -> GrugLmExample:
    """``example`` with segment ids: zeros (one segment, the same attention and loss) when its source gave none.

    Every process must build batches of one pytree structure. A mixture block whose examples all come from a
    source without segment ids otherwise yields a batch with one leaf fewer on that process, and its train step
    is a different program (lc1-gb200-k10b/k10c: 259 vs 260 inputs), which deadlocks or fails the shared compile.
    """
    if example.attn_mask.segment_ids is not None:
        return example
    zeros = jnp.zeros(example.tokens.shape, jnp.int32)
    return dataclasses.replace(example, attn_mask=example.attn_mask.with_segment_ids(zeros))


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
        dataclasses.replace(expert_mlp, implementation=implementation, expert_chunks=1)
        for expert_mlp in stack_expert_mlps(model)
    ]
    return eqx.tree_at(stack_expert_mlps, model, dropless)


def _cast_to_compute(mp: jmp.Policy, model: Transformer) -> Transformer:
    return mp.cast_to_compute(model)


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
        # read and write the latent instead, plus the latent projections, so correct both terms or MFU is
        # overstated by roughly the compression ratio.
        if model_config.latent_dim is not None:
            hidden, latent = model_config.hidden_dim, model_config.latent_dim
            # Matches the routed term in `lm_flops_per_token`: 2 * (gate + up + down) * intermediate * top_k.
            routed_delta = 2 * model_config.intermediate_dim * model_config.num_experts_per_token * 3 * (latent - hidden)
            # W_latent_down (hidden -> latent) and W_latent_up (latent -> hidden), once per token each.
            projection = 2 * hidden * 2 * latent
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
) -> GrugTrainState:
    params = mp.cast_to_param(Transformer.init(model_config, key=key))
    opt_state = optimizer.init(params)
    return GrugTrainState(
        step=jax.sharding.reshard(jnp.array(0, dtype=jnp.int32), P()),
        params=params,
        opt_state=opt_state,
        pending_qb_betas=jnp.zeros((model_config.num_layers, model_config.num_experts)),
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


LM_HEAD_GRAD_PREFIX = "train/lm_head_grad/"
LM_HEAD_GRAD_PRE_PREFIX = "train/lm_head_grad_pre/"
FINAL_NORM_INPUT_RMS_PREFIX = "train/final_norm_input_rms/"
# The step's per-token norms ``[B, S]`` (float32, loss-token units), kept on device unless a dump fetches them.
LM_HEAD_GRAD_NORM_KEY = "_lm_head_grad_norm"
LM_HEAD_GRAD_PRE_NORM_KEY = "_lm_head_grad_norm_pre"
FINAL_NORM_INPUT_RMS_QUANTILES = (0.01, 0.5, 0.99)
# Dump field -> metrics key of the per-token arrays written when present.
LM_HEAD_GRAD_DUMP_FIELDS = {
    "norm": LM_HEAD_GRAD_NORM_KEY,
    "norm_pre": LM_HEAD_GRAD_PRE_NORM_KEY,
    "rms_pre": FINAL_NORM_INPUT_RMS_KEY,
}
LM_HEAD_GRAD_QUANTILES = (0.5, 0.9, 0.99, 0.999)
LM_HEAD_GRAD_MEDIAN_MULTIPLES = (2, 5, 10)
# Half-open ranges of each token's position within its document.
LM_HEAD_GRAD_POSITION_BUCKETS = ((0, 1), (1, 16), (16, 256), (256, 1024), (1024, 4096))
# Loss spikes come early: dump every LM_HEAD_GRAD_EARLY_DUMP_EVERY steps up to this step, then every
# ``lm_head_grad_dump_every``.
LM_HEAD_GRAD_EARLY_DUMP_STEPS = 200
LM_HEAD_GRAD_EARLY_DUMP_EVERY = 10


def lm_head_grad_dump_due(step: int, every: int) -> bool:
    """Whether the 0-based train ``step`` writes an lm_head gradient dump (``every`` = 0: never)."""
    if every <= 0:
        return False
    early = step <= LM_HEAD_GRAD_EARLY_DUMP_STEPS and step % LM_HEAD_GRAD_EARLY_DUMP_EVERY == 0
    return early or step % every == 0


def _grad_norm_stats(norms: jax.Array, batch, clip: float | None) -> dict[str, jax.Array]:
    """``_lm_head_grad_metrics``' stats of one probe site's per-token norms (loss-token units)."""
    weight = batch.loss_weight.astype(jnp.float32)
    loss_tokens = jnp.sum(weight)
    valid = weight > 0
    count = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
    valid_norms = jnp.where(valid, norms, 0.0)
    everywhere = reshard(jnp.where(valid, norms, jnp.nan), P(None, None))
    quantiles = jnp.nanquantile(everywhere, jnp.asarray(LM_HEAD_GRAD_QUANTILES))
    median = quantiles[0]
    out = {
        "loss_tokens": loss_tokens,
        "mean": jnp.sum(valid_norms) / count,
        "rms": jnp.sqrt(jnp.sum(jnp.square(valid_norms)) / count),
        "max": jnp.max(valid_norms),
        "masked_max": jnp.max(jnp.where(valid, 0.0, norms)),
        **{f"p{100 * q:g}": quantiles[i] for i, q in enumerate(LM_HEAD_GRAD_QUANTILES)},
        **{
            f"frac_above_{k}x_median": jnp.sum((valid & (norms > k * median)).astype(jnp.float32)) / count
            for k in LM_HEAD_GRAD_MEDIAN_MULTIPLES
        },
    }
    if clip is not None:
        out["frac_clipped"] = jnp.sum((valid & (norms > clip)).astype(jnp.float32)) / count
    positions = _doc_positions(batch.tokens, batch.attn_mask)
    for lo, hi in LM_HEAD_GRAD_POSITION_BUCKETS:
        in_bucket = valid & (positions >= lo) & (positions < hi)
        name = f"{lo}" if hi == lo + 1 else f"{lo}-{hi - 1}"
        out[f"pos_mean/{name}"] = jnp.sum(jnp.where(in_bucket, norms, 0.0)) / jnp.maximum(
            jnp.sum(in_bucket.astype(jnp.float32)), 1.0
        )
    return out


def _lm_head_grad_metrics(
    raw_norms: dict[GradProbeSite, jax.Array], rms_pre: jax.Array | None, batch, clip: float | None
) -> dict[str, jax.Array]:
    """Stats of the per-token gradient norms of each probe site over the batch's loss tokens, the per-token norms
    themselves (for dumps) and, at the pre-norm site, stats of each token's RMS entering ``final_norm``.

    The loss is a mean over the loss-weighted tokens, so ``raw_norms`` carry a factor 1 / sum(loss_weight); the
    norms here are scaled by that sum: each is the gradient norm of the token's own loss term (its cross-entropy
    plus its z-loss share), comparable across batch sizes. Tokens with zero loss weight are excluded from the stats
    (``masked_max`` checks they are 0). With ``clip`` (``lm_head_grad_clip``), the post-norm ``frac_clipped`` is
    the fraction of loss tokens whose pre-clip norm exceeds it."""
    loss_tokens = jnp.sum(batch.loss_weight.astype(jnp.float32))
    out: dict[str, jax.Array] = {}
    sites = {
        GradProbeSite.POST_NORM: (LM_HEAD_GRAD_PREFIX, LM_HEAD_GRAD_NORM_KEY, clip),
        GradProbeSite.PRE_NORM: (LM_HEAD_GRAD_PRE_PREFIX, LM_HEAD_GRAD_PRE_NORM_KEY, None),
    }
    for site, raw in raw_norms.items():
        prefix, key, site_clip = sites[site]
        norms = raw * loss_tokens
        stats = _grad_norm_stats(norms, batch, site_clip)
        out.update({f"{prefix}{k}": jax.lax.stop_gradient(v) for k, v in stats.items()})
        out[key] = norms
    if rms_pre is not None:
        valid = batch.loss_weight > 0
        everywhere = reshard(jnp.where(valid, rms_pre, jnp.nan), P(None, None))
        quantiles = jnp.nanquantile(everywhere, jnp.asarray(FINAL_NORM_INPUT_RMS_QUANTILES))
        out[f"{FINAL_NORM_INPUT_RMS_PREFIX}min"] = jnp.min(jnp.where(valid, rms_pre, jnp.inf))
        for i, q in enumerate(FINAL_NORM_INPUT_RMS_QUANTILES):
            out[f"{FINAL_NORM_INPUT_RMS_PREFIX}p{100 * q:g}"] = quantiles[i]
    return out


TOKEN_EMBED_STATS_PREFIX = "train/embed/"
TOKEN_EMBED_ROW_QUANTILES = (0.01, 0.5)


def _token_embed_stats(before: jax.Array, after: jax.Array) -> dict[str, jax.Array]:
    """``token_embed`` after the step: Frobenius norm, row-norm min / quantiles / max, and ``||dE|| / ||E||``."""
    after32, before32 = after.astype(jnp.float32), before.astype(jnp.float32)
    rows = reshard(jnp.sqrt(jnp.sum(jnp.square(after32), axis=-1)), P(None))
    quantiles = jnp.quantile(rows, jnp.asarray(TOKEN_EMBED_ROW_QUANTILES))
    out = {
        "frob_norm": jnp.sqrt(jnp.sum(jnp.square(after32))),
        "row_norm_min": jnp.min(rows),
        **{f"row_norm_p{100 * q:g}": quantiles[i] for i, q in enumerate(TOKEN_EMBED_ROW_QUANTILES)},
        "row_norm_max": jnp.max(rows),
        "rel_update": jnp.sqrt(jnp.sum(jnp.square(after32 - before32))) / jnp.sqrt(jnp.sum(jnp.square(before32))),
    }
    return {f"{TOKEN_EMBED_STATS_PREFIX}{k}": v for k, v in out.items()}


def _write_lm_head_grad_dump(dump_dir: str, step: int, metrics: dict, batch) -> None:
    """Write ``step``'s per-token norms (``_lm_head_grad_metrics`` units: ``norm`` post-norm, ``norm_pre`` pre-norm),
    ``rms_pre`` (each token's RMS entering ``final_norm``), tokens, loss weights and segment ids."""
    per_token = {field: metrics[key] for field, key in LM_HEAD_GRAD_DUMP_FIELDS.items() if key in metrics}
    arrays = {**per_token, "tokens": batch.tokens, "loss_weight": batch.loss_weight}
    segment_ids = _segment_ids(batch)
    if segment_ids is not None:
        arrays["segment_ids"] = segment_ids
    host = multihost_utils.process_allgather(arrays, tiled=True)
    if jax.process_index() != 0:
        return
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        step=np.asarray(step),
        **{field: np.asarray(host[field], np.float32) for field in per_token},
        tokens=np.asarray(host["tokens"], np.int32),
        loss_weight=np.asarray(host["loss_weight"], np.float16),
        **({"segment_ids": np.asarray(host["segment_ids"], np.int32)} if "segment_ids" in host else {}),
    )
    path = f"{dump_dir}/step-{step:06d}.npz"
    with fsspec.open(path, "wb") as f:
        f.write(buffer.getvalue())
    logger.info("Wrote the lm_head gradient dump to %s", path)


def _loss_and_grads(
    params,
    batch,
    mp: jmp.Policy,
    z_loss: float | None,
    layer_skip: LayerSkipRule | None = None,
    extra_mlp_active: jax.Array | bool = True,
    lm_head_grad_probe_site: GradProbeSite | None = None,
    lm_head_grad_clip: float | None = None,
):
    """With ``lm_head_grad_probe_site`` the metrics also hold ``_lm_head_grad_metrics``: the per-token norms come
    back as the gradients of zeros probes (``model.lm_head_grad_probe``) differentiated next to the params."""

    def loss_fn(model, probes):
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
            head_grad_probe=probes.get(GradProbeSite.POST_NORM),
            pre_norm_grad_probe=probes.get(GradProbeSite.PRE_NORM),
        )
        if skip is not None:
            metrics = {**metrics, "train/aux/layer_skip_frac": jnp.mean(1.0 - skip.keep.astype(jnp.float32))}
        return loss, metrics

    if lm_head_grad_probe_site is None:
        return jax.value_and_grad(lambda model: loss_fn(model, {}), has_aux=True)(params)
    if lm_head_grad_probe_site == GradProbeSite.BOTH:
        sites = (GradProbeSite.POST_NORM, GradProbeSite.PRE_NORM)
    else:
        sites = (lm_head_grad_probe_site,)
    probes = {site: jnp.zeros_like(batch.tokens, dtype=jnp.float32) for site in sites}
    (loss, metrics), (grads, raw_norms) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(params, probes)
    rms_pre = metrics.pop(FINAL_NORM_INPUT_RMS_KEY, None)
    probe_metrics = _lm_head_grad_metrics(raw_norms, rms_pre, batch, lm_head_grad_clip)
    if rms_pre is not None:
        probe_metrics[FINAL_NORM_INPUT_RMS_KEY] = rms_pre
    return (loss, {**metrics, **probe_metrics}), grads


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


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    watch_config: WatchConfig | None = None,
    grad_accum_microbatches: int = 1,
    layer_skip: LayerSkipRule | None = None,
    extra_mlp_start_step: int = 0,
    lm_head_grad_probe_site: GradProbeSite | None = None,
    lm_head_grad_clip: float | None = None,
    token_embed_stats: bool = False,
):
    """``extra_mlp_start_step``: the first step the ``extra_mlp_layers`` MLPs are on.
    ``lm_head_grad_probe_site`` adds the per-token lm_head input-gradient stats (``_lm_head_grad_metrics``),
    ``token_embed_stats`` the ``_token_embed_stats``.
    ``grad_accum_microbatches`` > 1 averages gradients over that many microbatches
    (``_accumulated_loss_and_grads``)."""
    one = jnp.array(1, dtype=jnp.int32)
    z_loss = z_loss_weight if z_loss_weight > 0 else None
    if lm_head_grad_probe_site is not None and grad_accum_microbatches > 1:
        raise ValueError("lm_head_grad_probe does not support gradient accumulation")
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
            (loss, summarized_metrics), grads = _loss_and_grads(
                qb_params, batch, mp, z_loss, layer_skip, extra_active, lm_head_grad_probe_site, lm_head_grad_clip
            )
        metrics = {"train/loss": loss, **summarized_metrics}
        with jax.named_scope("opt_update"):
            updates, opt_state = optimizer.update(grads, state.opt_state, qb_params)
        with jax.named_scope("apply_updates"):
            params = optax.apply_updates(qb_params, updates)
        if token_embed_stats:
            metrics.update(_token_embed_stats(qb_params.token_embed, params.token_embed))

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
                opt_state=state.opt_state,
                model_tree_type=type(state.params),
            )

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
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
    train_step_fn = _build_train_step(config, optimizer)

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)

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

        # The step runs on the state's flat leaves (``_flat_train_step``); ``state`` is rebuilt from them each step.
        state_leaves, state_treedef = jax.tree.flatten(state)
        train_step = _flat_train_step(train_step_fn, state_treedef)
        # The train step's optimized HLO, set when the first step compiles it ahead of time.
        train_step_hlo: str | None = None

        def finish_step(rec: "_PendingStep", step_state: GrugTrainState) -> None:
            """Wait for ``rec``'s step and do its host work; ``step_state`` is the state after the step."""
            nonlocal last_loss, last_step_duration, host_gap_start, hlo_written
            metrics, step = rec.metrics, rec.step
            # A host float, and no device math on the metrics below.
            loss = float(jax.device_get(metrics["train/loss"]))
            ready_time = time.perf_counter()
            stall_sampler.arm()
            duration = ready_time - rec.step_start
            if config.trainer.hlo_dump_path is not None and not hlo_written:
                if jax.process_index() == 0:
                    assert train_step_hlo is not None
                    _write_train_step_hlo(train_step_hlo, config.trainer.hlo_dump_path)
                hlo_written = True
            state_callbacks.emit_event(callbacks.ProgressEvent.TRAIN_STEP_FINISHED)
            if not math.isfinite(loss):
                raise RuntimeError(f"Non-finite loss ({loss}) at step {step + 1}.")
            hook_start = time.perf_counter()
            with jax.profiler.TraceAnnotation("callbacks"):
                state_callbacks.run(step_state, loss=loss, step_duration=duration)
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
                                LM_HEAD_GRAD_PREFIX,
                                LM_HEAD_GRAD_PRE_PREFIX,
                                FINAL_NORM_INPUT_RMS_PREFIX,
                                TOKEN_EMBED_STATS_PREFIX,
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
                dump_dir = config.trainer.lm_head_grad_dump_dir
                if dump_dir is not None and lm_head_grad_dump_due(step, config.model.lm_head_grad_dump_every):
                    _write_lm_head_grad_dump(dump_dir, step, metrics, rec.batch)
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

        # Main optimization loop. Each step's host work runs after the next batch is loaded and before the next
        # step is dispatched (the dispatch donates the state).
        try:
            with HostStallSampler(HOST_STALL_SAMPLE_THRESHOLD) as stall_sampler:
                host_step = int(state.step)
                pending: _PendingStep | None = None
                while host_step < stop_step:
                    with jax.profiler.TraceAnnotation("load_batch"):
                        batch = next(iterator)
                    if pending is not None:
                        finish_step(pending, state)
                        pending = None
                    current_step = host_step
                    watch_due = (
                        watch_config.is_enabled
                        and watch_config.interval > 0
                        and current_step % watch_config.interval == 0
                    )
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
                        logger.info(
                            "train step HLO sha256 %s (%d chars) on process %d, device memory limit %s",
                            hashlib.sha256(train_step_hlo.encode()).hexdigest()[:16],
                            len(train_step_hlo),
                            jax.process_index(),
                            (jax.local_devices()[0].memory_stats() or {}).get("bytes_limit"),
                        )
                        train_step = compiled_step
                    dispatch_start = time.perf_counter()
                    state_leaves, metrics, inline_watch_stats = train_step(state_leaves, batch)
                    state = jax.tree.unflatten(state_treedef, state_leaves)
                    # Host time to hand the step to the devices (rebuilding the state tree, launch).
                    dispatch_time = time.perf_counter() - dispatch_start
                    watch_stats = inline_watch_stats if watch_due else None
                    host_step += 1
                    pending = _PendingStep(
                        step=current_step,
                        metrics=metrics,
                        batch=batch,
                        watch_stats=watch_stats,
                        dispatch_time=dispatch_time,
                        step_start=step_start,
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


def _token_dump_step(mp: jmp.Policy, skip_layers: tuple[int, ...]):
    """Jitted ``(params, qb_betas, tokens, segment_ids) -> dict`` of per-token ``[B, S]`` arrays: ``next_token_loss``,
    and with ``skip_layers`` ``skipped_loss``, the loss with every token skipping those layers."""

    @jax.jit
    def run(params: Transformer, qb_betas: jax.Array, tokens: jax.Array, segment_ids: jax.Array):
        model = _cast_to_compute(mp, _apply_qb_betas(params, qb_betas))
        mask = AttentionMask.causal().with_segment_ids(segment_ids)
        weight = jnp.ones(tokens.shape, jnp.float32)
        out = {"next_token_loss": model.next_token_loss(tokens, weight, mask=mask, reduction="none")}
        if skip_layers:
            skip = LayerSkip(skip_layers, jnp.zeros(tokens.shape, jnp.bool_))
            out["skipped_loss"] = model.next_token_loss(tokens, weight, mask=mask, reduction="none", layer_skip=skip)
        return out

    return run


def write_token_dump(config: GrugRunConfig, state: GrugTrainState, mesh: Mesh, batch_size: int, path: str) -> None:
    """Run held-out batches and write ``tokens``, ``segment_ids``, ``next_token_loss`` ``[N, S]`` (the loss of
    predicting token s+1 at s), and with ``token_dump_skip_layers`` ``skipped_loss``."""
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
    step = _token_dump_step(config.trainer.trainer.mp, skip_layers)
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
    arrays = {k: np.concatenate([p[k] for p in parts]).astype(np.float16) for k in parts[0]}
    np.savez(
        buffer,
        tokens=tokens,
        segment_ids=segments,
        skip_layers=np.asarray(skip_layers, np.int32),
        step=np.asarray(int(state.step)),
        **arrays,
    )
    with fsspec.open(path, "wb") as f:
        f.write(buffer.getvalue())
    logger.info("Wrote the token dump to %s", path)


def _build_train_step(config: GrugRunConfig, optimizer):
    """The train step (``_make_train_step``) of ``config``."""
    trainer = config.trainer.trainer
    return _make_train_step(
        optimizer,
        trainer.mp,
        z_loss_weight=config.trainer.z_loss_weight,
        watch_config=trainer.watch if trainer.watch.is_enabled else None,
        grad_accum_microbatches=config.trainer.grad_accum_microbatches,
        layer_skip=config.trainer.layer_skip,
        # Late by default, once the entropy head has learned something.
        extra_mlp_start_step=int(trainer.num_train_steps * config.trainer.extra_mlp_start_frac),
        lm_head_grad_probe_site=config.model.lm_head_grad_probe_site if config.model.lm_head_grad_probe else None,
        lm_head_grad_clip=config.model.lm_head_grad_clip,
        token_embed_stats=config.model.token_embed_stats,
    )


# Every rank logs host-side stalls longer than this: a late rank stalls every other rank's collectives.
HOST_GAP_WARN = 0.1
# Post-step host work above this is logged with the Python stacks that ran during it (see `host_stall`).
HOST_STALL_SAMPLE_THRESHOLD = 0.3


GC_EVERY_STEPS = 50


@dataclasses.dataclass
class _PendingStep:
    """A dispatched step whose host work (wait, logging, callbacks) is still to do."""

    step: int
    metrics: dict
    batch: object
    watch_stats: object
    dispatch_time: float
    step_start: float


GC_PAUSE_WARN = 0.05
_gc_start: dict[str, float] = {}


def _warn_on_long_gc(phase: str, info: dict) -> None:
    if phase == "start":
        _gc_start["t"] = time.perf_counter()
        return
    pause = time.perf_counter() - _gc_start.get("t", time.perf_counter())
    if pause > GC_PAUSE_WARN:
        logger.warning("gc gen%d pause %.3f s on process %d", info.get("generation", -1), pause, jax.process_index())


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
    ragged = config.model.moe_implementation == RAGGED_MOE_IMPLEMENTATION
    _apply_runtime_defaults(
        inline_watch_enabled=trainer.watch.is_enabled,
        ragged_transport=config.ragged_transport if ragged else None,
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
