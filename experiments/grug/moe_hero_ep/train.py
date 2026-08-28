# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import gc
import importlib.metadata
import itertools
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
from fray.types import GpuConfig
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
from levanter.grug._moe.ep_ragged_all_to_all import RAGGED_REQUIRED_XLA_FLAGS
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

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.sharding_dump import dump_grug_state_sharding_run_artifact

# This file intentionally mirrors `experiments/grug/base/train.py` with
# variant-specific model/loss/FLOP wiring, per the grug copy-first workflow in
# `.agents/skills/change-grug/`.

logger = logging.getLogger(__name__)

HERO_EP_RUNTIME_ENV = {
    "LD_PRELOAD": "libjemalloc.so.2",
    "MALLOC_CONF": "background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0,narenas:2",
    "JAX_ENABLE_PGLE": "false",
    "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": "192",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
    # `cuda_async` reads this fraction as the mempool release threshold, and PJRT sizes the
    # collective pool at the remaining `(1 - fraction) x 184.3 GiB`. 0.75 sets a 138.2 GiB
    # threshold and leaves ~46 GiB for collectives, above the ~28.5 GiB NCCL, cuBLAS, and the
    # CUDA context hold outside the allocator. A higher fraction strands memory the step never
    # uses, because the startup preallocation probe commits the whole threshold at once: 0.83
    # pins 153.0 GiB in the pool and leaves under 3 GiB free on the device, so the arena
    # allocation below has no room to remap into.
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
}
XLA_COLLECTIVE_OVERLAP_FLAG = "--xla_gpu_experimental_parallel_collective_overlap_limit"
DEFAULT_COLLECTIVE_OVERLAP_LIMIT = 4
DEFAULT_DROPLESS_MOE_IMPLEMENTATION: MoeImplementation = "sonic_cute"
# Full inline norm watch failed with overlap 4. Overlap 1 completed the selected full-watch gate.
INLINE_WATCH_COLLECTIVE_OVERLAP_LIMIT = 1
# The ragged transport wants the opposite scheduling posture from the fixed and pooled ones. Its
# dispatch and combine form one long dependent chain, so admitting several concurrent collectives
# only contends for the SMs the transport itself needs. The latency-hiding scheduler stays off for
# a memory reason, not a scheduling one: its longer buffer live ranges push the step past the HBM
# budget and NCCL's first-step allocations fail. With the layer activations offloaded to pinned
# host the freed ~36 GiB fits those live ranges and the scheduler measures ~0.9 MFU faster
# (#8317); a follow-up lands that offload and turns the scheduler on.
RAGGED_COLLECTIVE_OVERLAP_LIMIT = 1
RAGGED_MOE_IMPLEMENTATION = "ragged_all_to_all"
# TODO(https://github.com/marin-community/marin/issues/5675): Re-enable XLA GPU
# command buffers after the CUDA graph failure is fixed.
XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
_RAGGED_REQUIRED_XLA_FLAG_NAMES = frozenset(flag.partition("=")[0] for flag in RAGGED_REQUIRED_XLA_FLAGS)
RAGGED_ACCELERATOR = "GB200"
PJRT_DISTRIBUTION = "jax-cuda13-pjrt"
_FP32_POLICY = jmp.get_policy("params=float32,compute=float32,output=float32")


class WatchMode(StrEnum):
    """Where a watched training step computes gradient and parameter statistics."""

    INLINE = "inline"
    DIAGNOSTIC = "diagnostic"


class MasterParamMode(StrEnum):
    """Storage mode for optimizer master parameters."""

    DISABLED = "disabled"
    FP32_PINNED_HOST = "fp32_pinned_host"


class TrainingDataMode(StrEnum):
    """Source of training batches."""

    MIXTURE = "mixture"
    SYNTHETIC = "synthetic"


def restore_template_from(state):
    """ShapeDtypeStructs carrying each leaf's concrete sharding, releasing the leaves.

    `jax.eval_shape` drops the `pinned_host` memory kind that `initial_state` puts on
    offloaded optimizer state, which is most of a hero checkpoint. Reading the sharding off a
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


def _apply_hero_ep_runtime_defaults(
    *, inline_watch_enabled: bool, moe_implementation: MoeImplementation | None, processes_per_task: int = 1
) -> None:
    env_defaults = dict(HERO_EP_RUNTIME_ENV)
    if processes_per_task > 1:
        # With one process per GPU, the per-process CUPTI sessions collide with each
        # other and with CoreWeave's DCGM, so PGLE cannot profile and its recompile
        # machinery only adds failure modes. Default it off; an explicit env wins.
        env_defaults["JAX_ENABLE_PGLE"] = "false"
    for name, value in env_defaults.items():
        os.environ.setdefault(name, value)
    xla_flags = os.environ.get("XLA_FLAGS", "").split()
    ragged = moe_implementation == RAGGED_MOE_IMPLEMENTATION
    if ragged:
        overlap_limit = RAGGED_COLLECTIVE_OVERLAP_LIMIT
    elif inline_watch_enabled:
        overlap_limit = INLINE_WATCH_COLLECTIVE_OVERLAP_LIMIT
    else:
        overlap_limit = DEFAULT_COLLECTIVE_OVERLAP_LIMIT
    flag_defaults = (
        f"{XLA_COLLECTIVE_OVERLAP_FLAG}={overlap_limit}",
        f"--xla_gpu_enable_latency_hiding_scheduler={'false' if ragged else 'true'}",
        # The scheduler sizes the single `jit_train_step` temp arena against this percentage of
        # its memory budget, roughly `133.6 GiB x percentage`. The pool holds 138.2 GiB and
        # persistent state occupies 18.1 GiB of it, so an arena above 120.2 GiB cannot be served
        # from pool free space and forces a fresh mapping against the ~17 GiB of physical memory
        # outside the pool. The default 95 asks for 125.7 GiB and fails that way. 85 sizes the
        # arena at 113.6 GiB, leaving enough slack for per-node variation in fragmentation. A
        # lower percentage costs throughput, because a smaller arena makes `HloRematerialization`
        # recompute more of the step.
        "--xla_gpu_memory_limit_slop_factor=85",
        XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG,
    )
    explicit_names = {flag.partition("=")[0] for flag in xla_flags}
    xla_flags.extend(flag for flag in flag_defaults if flag.partition("=")[0] not in explicit_names)
    if ragged:
        # Unlike the defaults above, these are not overridable. Selecting the host-launched
        # one-shot kernel needs both flags cleared together plus a splits-per-peer count this
        # branch no longer carries, so honoring a partial override would run a configuration
        # nothing here measures. Drop any conflicting entry rather than relying on which
        # occurrence XLA's parser keeps.
        xla_flags = [f for f in xla_flags if f.partition("=")[0] not in _RAGGED_REQUIRED_XLA_FLAG_NAMES]
        xla_flags.extend(RAGGED_REQUIRED_XLA_FLAGS)
    os.environ["XLA_FLAGS"] = " ".join(xla_flags)


def require_ragged_capable_fleet(moe_implementation: MoeImplementation | None, resources: ResourceConfig) -> None:
    """Reject a ragged launch on a fleet that cannot run it, before the fleet is allocated.

    The transport needs GB200 for its SM100 expert MLP and for Marin's patched PJRT build, which
    is published for aarch64 alone (lib/marin/pyproject.toml). Any other fleet would sync the
    stock plugin and fail ``verify_ragged_pjrt`` only after allocation.
    """
    if moe_implementation != RAGGED_MOE_IMPLEMENTATION:
        return
    device = resources.device
    if not isinstance(device, GpuConfig) or device.variant != RAGGED_ACCELERATOR:
        raise ValueError(
            f"{RAGGED_MOE_IMPLEMENTATION} needs {RAGGED_ACCELERATOR} for its SM100 expert MLP and "
            f"the aarch64-only patched PJRT wheel, got {device}."
        )


def verify_ragged_pjrt() -> None:
    """Raise unless this process runs Marin's patched GPU PJRT plugin.

    The fork's ragged all-to-all delta lives in the plugin binary and nothing else observes it:
    ``jax.__version__`` reports the stock generation, and the stock plugin runs the same flags
    correctly at a materially lower throughput. Without this check the difference between the
    patched and the stock runtime is a number on a dashboard rather than a failure. The patched
    wheel installs through the gpu extra's aarch64 source in lib/marin/pyproject.toml.
    """
    try:
        installed = importlib.metadata.version(PJRT_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as missing:
        raise RuntimeError(
            f"{PJRT_DISTRIBUTION} is not installed, so this process has no GPU PJRT plugin at all."
        ) from missing
    expected_prefix = f"{jax.__version__}+marin."
    if not installed.startswith(expected_prefix):
        raise RuntimeError(
            f"{RAGGED_MOE_IMPLEMENTATION} needs Marin's patched {PJRT_DISTRIBUTION} "
            f"({expected_prefix}*), found {installed}."
        )


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    z_loss_weight: float = 1e-4  # Weight on final-logit logsumexp z-loss stabilization term.
    # Keep disabled except on model sizes where Grace-Blackwell host offload has been measured.
    # The d6144 EP64 runs used it; d5120 required a 135 GiB pinned-host arena and regressed.
    offload_opt_state: bool = False
    master_param_mode: MasterParamMode = MasterParamMode.DISABLED
    training_data_mode: TrainingDataMode = TrainingDataMode.MIXTURE
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
    sharding_dump_path: str | None = None


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
    # For expert-parallel runs, also evaluate under the dropless local backend on an
    # expert-collapsed mesh, logging a separate `eval_dropless` macro loss alongside the
    # as-trained (with-drop) eval. No-op when the mesh has no expert parallelism.
    dropless_eval: bool = False
    # Local MoE kernel used after collapsing the expert axis. ``sonic`` is the Hopper Triton path;
    # ``sonic_cute`` is the Blackwell QuACK/CUTLASS path.
    dropless_eval_moe_implementation: MoeImplementation = DEFAULT_DROPLESS_MOE_IMPLEMENTATION
    # Run the evals once after the first optimization step, for a baseline at the start of the loss
    # curve. The periodic cadence first fires at `steps_per_eval`, thus it leaves that start bare.
    eval_at_first_step: bool = False


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


def _make_synthetic_batch(
    *,
    batch_size: int,
    max_seq_len: int,
    vocab_size: int,
    seed: int,
    mesh: Mesh,
) -> GrugLmExample:
    """Build one deterministic batch directly on the global batch sharding."""
    sharding = NamedSharding(mesh, P(_BATCH_AXES, None))

    def tokens_for_slice(index):
        batch_slice, position_slice = index
        batch_start, batch_stop, batch_stride = batch_slice.indices(batch_size)
        position_start, position_stop, position_stride = position_slice.indices(max_seq_len)
        if batch_stride != 1 or position_stride != 1:
            raise ValueError("synthetic batch sharding requires contiguous slices")
        batch_indices = np.arange(batch_start, batch_stop, dtype=np.int64)[:, None]
        position_indices = np.arange(position_start, position_stop, dtype=np.int64)[None, :]
        return ((batch_indices * max_seq_len + position_indices + seed) % vocab_size).astype(np.int32)

    def loss_weight_for_slice(index):
        batch_slice, position_slice = index
        batch_start, batch_stop, batch_stride = batch_slice.indices(batch_size)
        position_start, position_stop, position_stride = position_slice.indices(max_seq_len)
        if batch_stride != 1 or position_stride != 1:
            raise ValueError("synthetic batch sharding requires contiguous slices")
        loss_weight = np.ones((batch_stop - batch_start, position_stop - position_start), dtype=np.float32)
        if position_start <= max_seq_len - 1 < position_stop:
            loss_weight[:, max_seq_len - 1 - position_start] = 0
        return loss_weight

    shape = (batch_size, max_seq_len)
    tokens = jax.make_array_from_callback(shape, sharding, tokens_for_slice)
    loss_weight = jax.make_array_from_callback(shape, sharding, loss_weight_for_slice)
    return GrugLmExample(tokens=tokens, loss_weight=loss_weight)


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
    """Swap the scanned block's MoE expert backend to the selected dropless local path.

    ``implementation``/``expert_chunks`` are static fields shared across the whole stacked block,
    so one replacement covers every layer. The forward reads ``self.expert_mlp.implementation``
    (not the model config), so this alone routes the eval dropless. Must run on an expert-collapsed
    mesh: the local backend raises when the mesh expert axis is larger than one.
    """
    expert_mlp = model.stacked_blocks.stacked.mlp.expert_mlp
    dropless = dataclasses.replace(expert_mlp, implementation=implementation, expert_chunks=1)
    return eqx.tree_at(lambda m: m.stacked_blocks.stacked.mlp.expert_mlp, model, dropless)


def _first_step_only(hook: Callable[..., None]) -> Callable[..., None]:
    """Wrap ``hook`` so that it runs one time only, after the first optimization step.

    ``StateCallbackRunner`` dispatches on ``next_step % every``, thus ``every=1`` is the only
    interval that covers the first step. The gate makes that registration one-shot. A resumed run
    starts above step 1 and never fires it.
    """

    # `LambdaCallback` reads the signature to decide whether to pass `force`, and the `**kwargs`
    # below would otherwise advertise a `force` parameter that the wrapped hook can lack.
    @functools.wraps(hook)
    def gated(step, *args, **kwargs):
        if step.next_step != 1:
            return
        hook(step, *args, **kwargs)

    return gated


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
    if eval_cfg.max_eval_batches is not None:
        max_examples_per_dataset = eval_cfg.max_eval_batches * eval_cfg.eval_batch_size

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


def _compute_flops(
    *,
    model_config: GrugModelConfig,
) -> tuple[float, dict[str, float]]:
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
        num_shared_experts=model_config.num_shared_experts if model_config.shared_expert_intermediate_dim > 0 else 0,
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

    flops_per_example = 3 * flops_per_token * model_config.max_seq_len

    flops_summary: dict[str, float] = {
        "throughput/flops_per_token_analytic": flops_per_token,
        "throughput/flops_per_example_analytic": flops_per_example,
    }

    return flops_per_example, flops_summary


def log_device_memory(step_info) -> None:
    """Log this process's local-device HBM peak, live bytes, and allocator limit in GiB.

    The EP hero had no peak-HBM telemetry, which makes a whole class of result unreadable: XLA's
    ``HloRematerialization`` engages only when peak crosses the allocator limit, so a config change
    that moves peak across that boundary produces an MFU step change that has nothing to do with the
    change itself. Issue #8054 traced its own +9.08% headline to exactly this -- 3.69 GiB of peak
    took it under the limit and switched remat off -- and the win fell to +3.31% once one process
    per GPU put 8.79 GiB back. Any ablation that moves activation memory needs this logged, or its
    rungs cannot be told apart from allocator-limit crossings.

    Ported from ``experiments/grug/moe_hero_fsdp/train.py``.
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
    opt_state: optax.OptState
    ema_params: Transformer | None
    pending_qb_betas: jax.Array


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array) -> Transformer:
    """Set router biases from QB betas (computed on previous step)."""
    new_bias = -qb_betas
    new_bias = new_bias - jnp.mean(new_bias, axis=-1, keepdims=True)
    return eqx.tree_at(lambda t: t.stacked_blocks.stacked.mlp.router_bias, model, new_bias)


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
            # Scalar optimizer metadata has no operand from which to inherit the active mesh.
            # Bind it before changing memory kind so the initial and updated states have the
            # same JIT input signature.
            leaf = jax.sharding.reshard(leaf, P())
            sharding = jax.typeof(leaf).sharding
        return jax.device_put(leaf, sharding.with_memory_kind(memory_kind))

    return jax.tree.map(_move, tree)


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
    offload_opt_state: bool = False,
    master_param_mode: MasterParamMode = MasterParamMode.DISABLED,
) -> GrugTrainState:
    initialized_params = Transformer.init(model_config, key=key)
    num_moe_layers = model_config.num_layers
    if master_param_mode == MasterParamMode.FP32_PINNED_HOST:
        master_params = _FP32_POLICY.cast_to_param(initialized_params)
        params = mp.cast_to_param(master_params)
        opt_state = optimizer.init(master_params)
        master_params = _tree_to_memory_kind(master_params, "pinned_host")
    else:
        params = mp.cast_to_param(initialized_params)
        master_params = None
        opt_state = optimizer.init(params)
    if offload_opt_state:
        opt_state = _tree_to_memory_kind(opt_state, "pinned_host")
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        master_params=master_params,
        opt_state=opt_state,
        ema_params=params if ema_beta is not None else None,
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


def _loss_and_grads(params, batch, mp: jmp.Policy, z_loss: float | None):
    def loss_fn(model):
        compute_params = mp.cast_to_compute(model)
        return compute_params.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="mean",
            logsumexp_weight=z_loss,
            return_router_metrics=True,
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
    ema_beta: float | None,
    watch_config: WatchConfig | None = None,
    offload_opt_state: bool = False,
    master_param_mode: MasterParamMode = MasterParamMode.DISABLED,
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
        # host-side TPU kernel launches that can cause SPMD sync issues).
        qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)
        if ema_beta is not None:
            qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas)
        else:
            qb_ema_params = None

        (loss, summarized_metrics), grads = _loss_and_grads(qb_params, batch, mp, z_loss)
        metrics = {"train/loss": loss, **summarized_metrics}
        opt_state_in = _tree_to_memory_kind(state.opt_state, "device") if offload_opt_state else state.opt_state
        if master_param_mode == MasterParamMode.FP32_PINNED_HOST:
            if state.master_params is None:
                raise ValueError("master_params must be initialized for an FP32 pinned-host master.")
            master_params_in = _tree_to_memory_kind(state.master_params, "device")
            master_params_in = _apply_qb_betas(master_params_in, state.pending_qb_betas)
            master_grads = _FP32_POLICY.cast_to_param(grads)
            updates, opt_state = optimizer.update(master_grads, opt_state_in, master_params_in)
            master_params = optax.apply_updates(master_params_in, updates)
            params = mp.cast_to_param(master_params)
            master_params = _tree_to_memory_kind(master_params, "pinned_host")
        else:
            updates, opt_state = optimizer.update(grads, opt_state_in, qb_params)
            params = optax.apply_updates(qb_params, updates)
            master_params = None

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

        if offload_opt_state:
            opt_state = _tree_to_memory_kind(opt_state, "pinned_host")

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            master_params=master_params,
            opt_state=opt_state,
            ema_params=ema_params,
            pending_qb_betas=metrics["qb_beta_per_layer"],
        )

        return next_state, metrics, watch_stats

    return train_step


def _run_grug_local(config: GrugRunConfig) -> None:
    """Entry point for the grug template training loop."""
    if config.model.moe_implementation == RAGGED_MOE_IMPLEMENTATION:
        verify_ragged_pjrt()
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
        ema_beta=config.trainer.ema_beta,
        watch_config=inline_watch_config,
        offload_opt_state=config.trainer.offload_opt_state,
        master_param_mode=config.trainer.master_param_mode,
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
        dump_grug_state_sharding_run_artifact(
            state,
            log_dir=trainer.log_dir,
            run_id=run_id,
            path_override=config.trainer.sharding_dump_path,
        )

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})

        train_dataset = None
        train_loader = None
        if config.trainer.training_data_mode == TrainingDataMode.MIXTURE:
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
            if eval_cfg.dropless_eval and mesh.shape["expert"] > 1:
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
        if config.trainer.training_data_mode == TrainingDataMode.SYNTHETIC:
            synthetic_batch = _make_synthetic_batch(
                batch_size=batch_schedule.batch_size_at_step(int(state.step)),
                max_seq_len=config.model.max_seq_len,
                vocab_size=config.model.vocab_size,
                seed=trainer.seed + 1 if config.trainer.data_seed is None else config.trainer.data_seed,
                mesh=mesh,
            )
            batch_source = itertools.repeat(synthetic_batch)
        else:
            assert train_loader is not None
            batch_source = train_loader.iter_from_step(int(state.step))
        iterator = LoadingTimeTrackerIterator(batch_source)

        state_callbacks = StateCallbackRunner[GrugTrainState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            eval_model_getter=lambda s: s.ema_params if s.ema_params is not None else s.params,
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
        if evaluator is not None and eval_cfg is not None:
            interval = eval_cfg.steps_per_eval
            eval_ema = eval_cfg.eval_ema and config.trainer.ema_beta is not None
            if eval_cfg.eval_current or eval_ema:
                tagged_eval_hook = cb_tagged_evaluate(
                    evaluator,
                    prefix=eval_cfg.prefix,
                    eval_current=eval_cfg.eval_current,
                    eval_ema=eval_ema,
                )
                eval_hooks: list[Callable[..., None]] = [tagged_eval_hook]
                if dropless_evaluator is not None and dropless_eval_mesh is not None:
                    # The training loop runs under `set_mesh(mesh)` (expert-parallel). The dropless
                    # evaluator runs under the expert-collapsed mesh, so the model params -- sharded on
                    # the train mesh -- must be resharded onto the eval mesh before its eval jit (JAX
                    # does not auto-reshard across explicit meshes), then the local backend sees
                    # expert=1. PGLE is disabled for the eval module as in `cb_tagged_evaluate`.
                    dropless_prefix = f"{eval_cfg.prefix}_dropless"

                    def dropless_eval_hook(
                        step, *args, _mesh=dropless_eval_mesh, _ev=dropless_evaluator, _prefix=dropless_prefix, **kwargs
                    ):
                        step_count = int(step.step)
                        if step_count < 0:
                            return
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

                # Baseline point at the start of the loss curve. The periodic cadence first fires at
                # `steps_per_eval` (step 3000 on the hero), thus a fresh run gets no early point.
                # These run after the first optimization step, not before it: the first train step
                # then allocates against a clean pool, and the eval-to-train handoff gets a gate at
                # step 2 instead of first at step 3000. `every=1` is the only interval that covers
                # the first step, and `_first_step_only` makes the hook fire once. A resumed run
                # starts above step 1, thus it never fires. The hooks log at `StepInfo.step`, which
                # is 0 there, so the point lands at step 0 on the curve.
                if eval_cfg.eval_at_first_step:
                    for hook in eval_hooks:
                        state_callbacks.add_hook(_first_step_only(hook), every=1)

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

    # Dispatch snapshots os.environ for the child task, so apply the hero defaults first.
    inline_watch_enabled = trainer.watch.is_enabled and config.trainer.watch_mode == WatchMode.INLINE
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=inline_watch_enabled,
        processes_per_task=config.processes_per_task,
        moe_implementation=config.model.moe_implementation,
    )
    require_ragged_capable_fleet(config.model.moe_implementation, config.resources)
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
    "MasterParamMode",
    "initial_state",
    "require_ragged_capable_fleet",
    "run_grug",
]
