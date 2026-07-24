# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import functools
import importlib
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, cast

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
from haliax.quantization import OverwriteWithGradient, partition_for_grad_overwrite
from haliax.quantization import apply_updates as apply_quantized_updates
from jax import core
from jax.interpreters import ad as jax_ad
from jax.sharding import AxisType, Mesh, NamedSharding
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
from levanter.eval import TaggedEvaluator, cb_tagged_evaluate
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.trainer import TrainerConfig
from levanter.trainer_state import init_optimizer_for_trainables
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.model import (
    GRUG_MOE_NCCL_EP_CAPACITY_FACTOR,
    GrugModelConfig,
    Transformer,
    TransformerPipelineStage,
)
from experiments.grug.sharding_dump import dump_grug_state_sharding_run_artifact

# This file intentionally mirrors `experiments/grug/base/train.py` with
# variant-specific model/loss/FLOP wiring, per the grug copy-first workflow in
# `.agents/skills/change-grug/`.

try:
    import jaxpp.api as jaxpp
    import jaxpp.core as jaxpp_core
    from jaxpp.experimental import mpmd as jaxpp_explicit_mpmd
except ModuleNotFoundError:
    jaxpp = None
    jaxpp_core = None
    jaxpp_explicit_mpmd = None

logger = logging.getLogger(__name__)

JaxPPSchedule = Literal[
    "gpipe",
    "std_1f1b",
    "eager_1f1b",
    "zero_bubble",
    "interleaved_gpipe",
    "interleaved_1f1b",
    "dualpipe_v",
    "kimi_k2",
]
JaxPPImplementation = Literal["auto", "explicit_mpmd"]
JaxPPExplicitMpmdScheduleMode = Literal["default", "transfer_priority", "input_gradient_first"]
ExplicitMpmdPipelineWireFormat = Literal["bf16", "fp8"]
Fp8PipelineWireDtype = Literal["e4m3", "e5m2"]
SonicFsdpMaterialization = Literal["per_task", "staged_per_step"]
_RESEARCH_FP8_EXPERT_GEMM_SCHEDULES = ("gpipe", "interleaved_gpipe", "std_1f1b")


def _fp8_pipeline_wire_dtype(dtype: Fp8PipelineWireDtype) -> jnp.dtype:
    if dtype == "e4m3":
        return jnp.dtype(jnp.float8_e4m3fn)
    if dtype == "e5m2":
        return jnp.dtype(jnp.float8_e5m2)
    raise ValueError(f"unknown FP8 pipeline wire dtype: {dtype}")


def pack_fp8_pipeline_wire(value: jax.Array, dtype: Fp8PipelineWireDtype) -> jax.Array:
    """Pack per-token FP8 values and their FP32 scales into one uint8 tensor."""
    if value.ndim < 1 or value.shape[-1] == 0:
        raise ValueError(f"pipeline wire values require a non-empty hidden axis, got shape {value.shape}")
    fp8_dtype = _fp8_pipeline_wire_dtype(dtype)
    value_f32 = value.astype(jnp.float32)
    amax = jnp.max(jnp.abs(value_f32), axis=-1)
    fp8_max = jnp.asarray(jnp.finfo(fp8_dtype).max, dtype=jnp.float32)
    dequant_scale = jnp.where(amax > 0, amax / fp8_max, jnp.ones_like(amax))
    quantized = (value_f32 / dequant_scale[..., None]).astype(fp8_dtype)
    value_bytes = jax.lax.bitcast_convert_type(quantized, jnp.uint8)
    scale_bytes = jax.lax.bitcast_convert_type(dequant_scale, jnp.uint8)
    return jnp.concatenate((value_bytes, scale_bytes), axis=-1)


def unpack_fp8_pipeline_wire(packed: jax.Array, dtype: Fp8PipelineWireDtype) -> jax.Array:
    """Unpack a single-tensor FP8 pipeline payload to BF16 activations."""
    if packed.dtype != jnp.uint8:
        raise TypeError(f"packed pipeline wire values must have dtype uint8, got {packed.dtype}")
    if packed.ndim < 1 or packed.shape[-1] <= 4:
        raise ValueError(f"packed pipeline wire values require FP8 data plus four scale bytes, got {packed.shape}")
    fp8_dtype = _fp8_pipeline_wire_dtype(dtype)
    value_bytes = packed[..., :-4]
    scale_bytes = packed[..., -4:]
    quantized = jax.lax.bitcast_convert_type(value_bytes, fp8_dtype)
    dequant_scale = jax.lax.bitcast_convert_type(scale_bytes, jnp.float32)
    return (quantized.astype(jnp.float32) * dequant_scale[..., None]).astype(jnp.bfloat16)


@dataclass(frozen=True)
class GrugJaxPPConfig:
    """Experimental JaxPP pipeline-parallel training settings."""

    stages: int
    microbatches: int
    schedule: JaxPPSchedule = "std_1f1b"
    implementation: JaxPPImplementation = "auto"
    mpmd_dim: int | None = None
    stage_axis_name: str = "pipeline"
    stage_layer_counts: tuple[int, ...] | None = None
    explicit_mpmd_schedule_mode: JaxPPExplicitMpmdScheduleMode = "default"
    explicit_mpmd_pipeline_wire_format: ExplicitMpmdPipelineWireFormat = "bf16"
    sonic_fsdp_materialization: SonicFsdpMaterialization = "per_task"

    def __post_init__(self) -> None:
        if self.explicit_mpmd_schedule_mode not in ("default", "transfer_priority", "input_gradient_first"):
            raise ValueError(f"unknown explicit MPMD schedule mode: {self.explicit_mpmd_schedule_mode}")
        if self.sonic_fsdp_materialization not in ("per_task", "staged_per_step"):
            raise ValueError(f"unknown Sonic FSDP materialization mode: {self.sonic_fsdp_materialization}")
        if self.explicit_mpmd_pipeline_wire_format not in ("bf16", "fp8"):
            raise ValueError(f"unknown explicit MPMD pipeline wire format: {self.explicit_mpmd_pipeline_wire_format}")
        if self.stages <= 0:
            raise ValueError(f"stages must be positive, got {self.stages}")
        if self.microbatches <= 0:
            raise ValueError(f"microbatches must be positive, got {self.microbatches}")
        if self.mpmd_dim is not None and self.mpmd_dim <= 0:
            raise ValueError(f"mpmd_dim must be positive when set, got {self.mpmd_dim}")
        if self.stage_layer_counts is not None:
            if len(self.stage_layer_counts) != self.stages:
                raise ValueError(
                    "stage_layer_counts must have one entry per pipeline stage; "
                    f"got {len(self.stage_layer_counts)} counts for {self.stages} stages"
                )
            if any(layer_count <= 0 for layer_count in self.stage_layer_counts):
                raise ValueError(f"stage_layer_counts must be positive, got {self.stage_layer_counts}")
        if self.implementation == "explicit_mpmd":
            if self.stages < 2:
                raise ValueError("explicit_mpmd requires at least 2 pipeline stages")
            explicit_microbatch_schedules = ("gpipe", "std_1f1b", "interleaved_gpipe")
            if self.microbatches != 1 and self.schedule not in explicit_microbatch_schedules:
                raise ValueError(
                    "explicit_mpmd currently supports microbatches > 1 only for "
                    f"schedule in {explicit_microbatch_schedules}"
                )
            mpmd_dim = _pipeline_mpmd_dim(self)
            if self.schedule == "interleaved_gpipe" and self.stages % mpmd_dim != 0:
                raise ValueError(
                    "explicit interleaved_gpipe requires stages to be divisible by mpmd_dim; "
                    f"got stages={self.stages}, mpmd_dim={mpmd_dim}"
                )
            if self.schedule != "interleaved_gpipe" and mpmd_dim != self.stages:
                raise ValueError("explicit_mpmd requires PP_MPMD_DIM to match PP_STAGES")
        if self.explicit_mpmd_schedule_mode == "transfer_priority":
            if self.implementation != "explicit_mpmd" or self.schedule != "std_1f1b":
                raise ValueError(
                    "transfer_priority explicit MPMD schedule mode requires "
                    "implementation='explicit_mpmd' and schedule='std_1f1b'"
                )
            if self.microbatches == 1:
                raise ValueError("transfer_priority explicit MPMD schedule mode requires microbatches > 1")
        if self.explicit_mpmd_schedule_mode == "input_gradient_first":
            if self.implementation != "explicit_mpmd" or self.schedule != "std_1f1b":
                raise ValueError(
                    "input_gradient_first explicit MPMD schedule mode requires "
                    "implementation='explicit_mpmd' and schedule='std_1f1b'"
                )
            if _pipeline_mpmd_dim(self) != self.stages:
                raise ValueError("input_gradient_first requires one pipeline stage per MPMD rank")
            if self.microbatches < self.stages:
                raise ValueError(
                    "input_gradient_first requires microbatches >= stages; "
                    f"got microbatches={self.microbatches}, stages={self.stages}"
                )
        if self.explicit_mpmd_pipeline_wire_format == "fp8":
            if self.implementation != "explicit_mpmd" or self.schedule != "std_1f1b":
                raise ValueError(
                    "FP8 explicit MPMD pipeline wire format requires "
                    "implementation='explicit_mpmd' and schedule='std_1f1b'"
                )
            if self.microbatches == 1:
                raise ValueError("FP8 explicit MPMD pipeline wire format requires microbatches > 1")
        if self.sonic_fsdp_materialization == "staged_per_step":
            if self.implementation != "explicit_mpmd" or self.schedule != "std_1f1b":
                raise ValueError(
                    "staged_per_step Sonic FSDP materialization requires "
                    "implementation='explicit_mpmd' and schedule='std_1f1b'"
                )
            if self.microbatches == 1:
                raise ValueError("staged_per_step Sonic FSDP materialization requires microbatches > 1")
        if self.schedule in ("zero_bubble", "dualpipe_v") and self.microbatches < self.stages:
            raise ValueError(
                f"{self.schedule} requires microbatches >= stages; got "
                f"microbatches={self.microbatches}, stages={self.stages}"
            )
        if not self.stage_axis_name:
            raise ValueError("stage_axis_name must be non-empty")


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    z_loss_weight: float = 1e-4  # Weight on final-logit logsumexp z-loss stabilization term.

    # Grug builds its own compact (replica_dcn, data, expert, model) mesh instead of using
    # the Trainer's logical axis mapping; `data` absorbs whatever these two leave free.
    # Defaults reproduce the historical layout: no expert parallelism and full replication
    # across slices (replica_axis_size=None -> jax.process_count()), i.e. parameters
    # replicated per slice and sharded only over the intra-slice `data` axis. For a model
    # too large to replicate within one slice, set replica_axis_size=1 (FSDP across every
    # slice) and expert_axis_size>1 (expert parallelism over the intra-slice devices).
    expert_axis_size: int = 1
    replica_axis_size: int | None = None
    pipeline: GrugJaxPPConfig | None = None
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


@dataclass(frozen=True)
class GrugRunConfig:
    """Top-level config for grug training."""

    model: GrugModelConfig
    data: LmDataConfig
    resources: ResourceConfig
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    # GPU processes per task: > 1 runs one JAX process per GPU (multi-controller)
    # via the iris.hooks.multigpu_main supervisor instead of one process per node.
    processes_per_task: int = 1
    post_setup_scripts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.model.moe_implementation == "nccl_ep":
            pipeline = self.trainer.pipeline
            if pipeline is None or pipeline.implementation != "explicit_mpmd":
                raise ValueError("NCCL_EP requires the explicit MPMD JaxPP pipeline")
            if self.processes_per_task != self.trainer.expert_axis_size:
                raise ValueError(
                    "NCCL_EP requires one process per expert rank; "
                    f"got processes_per_task={self.processes_per_task} and "
                    f"expert_axis_size={self.trainer.expert_axis_size}"
                )
            if self.processes_per_task <= 1:
                raise ValueError("NCCL_EP requires more than one process per task")
        if self.model.research_fp8_expert_gemm is None:
            return
        pipeline = self.trainer.pipeline
        if pipeline is None or pipeline.implementation != "explicit_mpmd":
            raise ValueError("research FP8 expert GEMMs require the explicit MPMD JaxPP pipeline")
        if pipeline.schedule not in _RESEARCH_FP8_EXPERT_GEMM_SCHEDULES:
            raise ValueError(
                "research FP8 expert GEMMs require gpipe, interleaved_gpipe, or std_1f1b explicit scheduling"
            )
        if self.trainer.expert_axis_size <= 1:
            raise ValueError("research FP8 expert GEMMs require expert_axis_size greater than 1")


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
_GRUG_MESH_AXIS_NAMES: tuple[str, ...] = ("replica_dcn", "data", "expert", "model")


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
        mesh=mesh,
        axis_resources={"__BATCH__": _BATCH_AXES},
        batch_axis_name="__BATCH__",
        allow_nondivisible_batch_size=False,
    )


def _require_jaxpp():
    if jaxpp is None:
        raise ModuleNotFoundError(
            "jaxpp is required when GrugTrainerConfig.pipeline is set. "
            "Install NVIDIA/jaxpp in the training environment."
        )
    return jaxpp


def _require_jaxpp_explicit_mpmd():
    _require_jaxpp()
    if jaxpp_explicit_mpmd is None:
        raise ModuleNotFoundError(
            "jaxpp.experimental.mpmd is required when GrugTrainerConfig.pipeline.implementation='explicit_mpmd'."
        )
    return jaxpp_explicit_mpmd


def _load_nccl_ep_modules() -> tuple[Any, Any]:
    """Import NCCL_EP before Trainer initialization registers the CUDA client."""
    try:
        ep = importlib.import_module("transformer_engine.jax.ep")
        sharding = importlib.import_module("transformer_engine.jax.sharding")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Transformer Engine with NCCL_EP is required when moe_implementation='nccl_ep'."
        ) from error
    return ep, sharding


def _install_jaxpp_const_sharding_patch() -> None:
    """Preserve ClosedJaxpr const shardings for automatic JaxPP schedule probes."""
    _require_jaxpp()
    if jaxpp_core is None:
        raise ModuleNotFoundError("jaxpp.core is required for GRUG_JAXPP_PATCH_CONST_SHARDINGS")
    original_extract_params = jaxpp_core.extract_params
    if getattr(original_extract_params, "_grug_const_sharding_patch", False):
        return

    def extract_params_with_const_shardings(params, n_consts, replicated_sharding):
        (
            donated_invars,
            flat_in_shardings,
            flat_out_shardings,
            flat_in_layouts,
            flat_out_layouts,
        ) = original_extract_params(params, n_consts, replicated_sharding)

        consts = tuple(params["jaxpr"].consts[:n_consts])
        if not consts:
            return (
                donated_invars,
                flat_in_shardings,
                flat_out_shardings,
                flat_in_layouts,
                flat_out_layouts,
            )

        def const_sharding(value):
            if not _is_shardable_array(value):
                return replicated_sharding
            value_sharding = value.sharding
            if isinstance(value_sharding, NamedSharding):
                sharding = NamedSharding(replicated_sharding.mesh, value_sharding.spec)
                memory_kind = getattr(value_sharding, "memory_kind", None)
                if memory_kind is not None:
                    sharding = sharding.with_memory_kind(memory_kind)
                return sharding
            return NamedSharding(replicated_sharding.mesh, P(*([None] * value.ndim)))

        const_shardings = tuple(const_sharding(const) for const in consts)
        return (
            donated_invars,
            const_shardings + tuple(flat_in_shardings[n_consts:]),
            flat_out_shardings,
            flat_in_layouts,
            flat_out_layouts,
        )

    extract_params_with_const_shardings._grug_const_sharding_patch = True
    jaxpp_core.extract_params = extract_params_with_const_shardings


def _pipeline_schedule(pipeline: GrugJaxPPConfig):
    pp = _require_jaxpp()
    schedules = __import__("jaxpp.schedules", fromlist=["schedules"])
    mpmd_dim = _pipeline_mpmd_dim(pipeline)
    if pipeline.schedule == "gpipe":
        return schedules.GPipe(num_stages=pipeline.stages, mpmd_dim=mpmd_dim)
    if pipeline.schedule == "std_1f1b":
        return pp.Std1F1B(num_stages=pipeline.stages)
    if pipeline.schedule == "eager_1f1b":
        return pp.Eager1F1B(num_stages=pipeline.stages)
    if pipeline.schedule == "zero_bubble":
        return pp.ZeroBubble(num_stages=pipeline.stages)
    if pipeline.schedule == "interleaved_gpipe":
        return schedules.InterleavedGPipe(num_stages=pipeline.stages, mpmd_dim=mpmd_dim)
    if pipeline.schedule == "interleaved_1f1b":
        return pp.Interleaved1F1B(num_stages=pipeline.stages, mpmd_dim=mpmd_dim)
    if pipeline.schedule == "dualpipe_v":
        return pp.DualPipeV(num_stages=pipeline.stages, mpmd_dim=mpmd_dim)
    if pipeline.schedule == "kimi_k2":
        return pp.KimiK2(num_stages=pipeline.stages, mpmd_dim=mpmd_dim)
    raise ValueError(f"Unknown JaxPP schedule: {pipeline.schedule}")


def jaxpp_setup_scripts(*, revision: str = "7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9") -> tuple[str, ...]:
    """Install JaxPP into an Iris worker venv after the normal GPU sync."""
    package = f"jaxpp @ git+https://github.com/NVIDIA/jaxpp.git@{revision}"
    jax_tvm_ffi_revision = "e238a28483123efc8f56b9de358c2fb8b8de77e5"
    return (
        "\n".join(
            [
                "set -euxo pipefail",
                'cd "$IRIS_WORKDIR"',
                "echo 'installing JaxPP runtime deps'",
                "uv pip install --link-mode symlink cupy-cuda13x",
                "rm -rf /tmp/jax-tvm-ffi",
                "git clone --quiet --filter=blob:none https://github.com/NVIDIA/jax-tvm-ffi.git /tmp/jax-tvm-ffi",
                f"git -C /tmp/jax-tvm-ffi checkout --quiet {jax_tvm_ffi_revision}",
                'git -C /tmp/jax-tvm-ffi apply "$IRIS_WORKDIR/experiments/grug/moe/jax_tvm_ffi_multidevice.patch"',
                "uv pip install --link-mode symlink --force-reinstall --no-deps /tmp/jax-tvm-ffi",
                "uv pip install --link-mode symlink --no-deps " + repr(package),
                'bash experiments/grug/moe/patch_cutlass_dsl_mlir_type_guard.sh "$IRIS_VENV/bin/python"',
            ]
        )
        + "\n",
    )


def _pipeline_mpmd_dim(pipeline: GrugJaxPPConfig) -> int:
    return pipeline.mpmd_dim or pipeline.stages


def _pipeline_stage_mpmd_indices(pipeline: GrugJaxPPConfig) -> tuple[int, ...]:
    schedule = _pipeline_schedule(pipeline)
    return tuple(int(schedule.get_mpmd_idx(stage_index)) for stage_index in range(pipeline.stages))


def _interleaved_gpipe_task_order(pipeline: GrugJaxPPConfig) -> tuple[tuple[str, int, int], ...]:
    schedule = _pipeline_schedule(pipeline)
    rank_tasks = schedule.tasks(pipeline.microbatches)
    queue_heads = [0] * len(rank_tasks)
    completed_forwards: set[tuple[int, int]] = set()
    completed_backwards: set[tuple[int, int]] = set()
    task_order = []

    def task_ready(direction: str, stage_index: int, microbatch_index: int) -> bool:
        key = (stage_index, microbatch_index)
        if direction == "fwd":
            return stage_index == 0 or (stage_index - 1, microbatch_index) in completed_forwards
        if key not in completed_forwards:
            return False
        return stage_index == pipeline.stages - 1 or (stage_index + 1, microbatch_index) in completed_backwards

    while any(head < len(tasks) for head, tasks in zip(queue_heads, rank_tasks, strict=True)):
        made_progress = False
        for mpmd_index, tasks in enumerate(rank_tasks):
            head = queue_heads[mpmd_index]
            if head == len(tasks):
                continue
            task = tasks[head]
            if task is None or not hasattr(task, "stage_id"):
                raise ValueError("explicit interleaved_gpipe requires unfused forward/backward schedule tasks")
            direction = task.fwd_or_bwd.name.lower()
            if direction not in ("fwd", "bwd"):
                raise ValueError(f"explicit interleaved_gpipe does not support task type {task.fwd_or_bwd}")
            stage_index = int(task.stage_id)
            microbatch_index = int(task.mubatch_idx)
            if int(schedule.get_mpmd_idx(stage_index)) != mpmd_index:
                raise ValueError(f"schedule placed logical stage {stage_index} on unexpected MPMD rank {mpmd_index}")
            if not task_ready(direction, stage_index, microbatch_index):
                continue

            task_order.append((direction, stage_index, microbatch_index))
            queue_heads[mpmd_index] += 1
            if direction == "fwd":
                completed_forwards.add((stage_index, microbatch_index))
            else:
                completed_backwards.add((stage_index, microbatch_index))
            made_progress = True

        if not made_progress:
            blocked = [tasks[head] for head, tasks in zip(queue_heads, rank_tasks, strict=True) if head < len(tasks)]
            raise ValueError(f"interleaved_gpipe schedule has no dependency-ready queue head: {blocked}")

    return tuple(task_order)


def _reshape_batch_for_pipeline(batch: GrugLmExample, microbatches: int) -> GrugLmExample:
    def reshape_leaf(x):
        if not isinstance(x, jax.Array | core.Tracer):
            return x
        if x.ndim == 0:
            return x
        if x.shape[0] % microbatches != 0:
            raise ValueError(f"Batch axis size {x.shape[0]} must be divisible by microbatches={microbatches}")
        microbatch_size = x.shape[0] // microbatches
        return x.reshape(
            (microbatches, microbatch_size, *x.shape[1:]),
            out_sharding=P(None, _BATCH_AXES, *([None] * (x.ndim - 1))),
        )

    return jax.tree.map(reshape_leaf, batch)


def _select_pipeline_microbatch(batch: GrugLmExample, microbatch_index: int, microbatches: int) -> GrugLmExample:
    def select_leaf(x):
        if isinstance(x, jax.Array) and x.ndim > 0 and x.shape[0] == microbatches:
            return x[microbatch_index]
        return x

    return jax.tree.map(select_leaf, batch)


def _compact_or_pipeline_grug_mesh(
    *,
    expert_axis_size: int,
    replica_axis_size: int | None,
    pipeline: GrugJaxPPConfig | None,
) -> Mesh:
    if pipeline is None:
        return compact_grug_mesh(expert_axis_size=expert_axis_size, replica_axis_size=replica_axis_size)

    mpmd_dim = _pipeline_mpmd_dim(pipeline)
    if replica_axis_size is None:
        replica_axis_size = max(1, jax.process_count() // mpmd_dim)

    global_device_count = jax.device_count()
    fixed_axes = mpmd_dim * replica_axis_size * expert_axis_size
    if global_device_count % fixed_axes != 0:
        raise ValueError(
            f"global_device_count ({global_device_count}) must be divisible by pipeline mpmd_dim ({mpmd_dim}) * "
            f"replica_axis_size ({replica_axis_size}) * expert_axis_size ({expert_axis_size})"
        )

    data_axis_size = global_device_count // fixed_axes
    shape = (mpmd_dim, replica_axis_size, data_axis_size, expert_axis_size, 1)
    axis_names = (pipeline.stage_axis_name, *_GRUG_MESH_AXIS_NAMES)
    devices = np.array(jax.devices(), dtype=object).reshape(shape)
    mesh = Mesh(devices, axis_names, axis_types=tuple(AxisType.Explicit for _ in axis_names))

    if mesh.is_multi_process:
        local_pipeline_indices = {np.argwhere(devices == device)[0][0] for device in jax.local_devices()}
        if len(local_pipeline_indices) != 1:
            raise ValueError(
                "Each JAX process must own devices from exactly one JaxPP pipeline stage; "
                f"local process spans stages {sorted(local_pipeline_indices)}"
            )

    return mesh


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
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, model); length-1 axes
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
    ema_params: Transformer | None
    pending_qb_betas: jax.Array | None


@register_dataclass
@dataclass(frozen=True)
class GrugPipelineTrainState:
    step: jax.Array
    params: tuple[TransformerPipelineStage, ...]
    opt_state: tuple[optax.OptState, ...]
    ema_params: tuple[TransformerPipelineStage, ...] | None
    pending_qb_betas: tuple[jax.Array, ...]


def _is_overwrite(value) -> bool:
    return isinstance(value, OverwriteWithGradient)


def _cast_preserving_overwrites(tree, cast_fn):
    overwrites, ordinary = partition_for_grad_overwrite(tree)
    return eqx.combine(overwrites, cast_fn(ordinary), is_leaf=_is_overwrite)


def _accumulate_microbatch_tree(accumulated, value):
    """Add ordinary leaves and max delayed-scaling state across microbatches."""
    if accumulated is None:
        return value
    accumulated_overwrites, _ = partition_for_grad_overwrite(accumulated)
    overwrites, ordinary = partition_for_grad_overwrite(value)
    max_overwrites = jax.tree.map(jnp.maximum, accumulated_overwrites, overwrites)
    return apply_quantized_updates(accumulated, ordinary, max_overwrites)


def _average_microbatch_tree(tree, microbatches: int):
    """Average ordinary leaves without scaling overwrite-state leaves."""
    overwrites, ordinary = partition_for_grad_overwrite(tree)
    scale = jnp.asarray(1.0 / microbatches, dtype=jnp.float32)
    averaged = jax.tree.map(lambda value: value * scale, ordinary)
    return eqx.combine(overwrites, averaged, is_leaf=_is_overwrite)


def _ema_update(old, new, beta: float):
    new_overwrites, new_ordinary = partition_for_grad_overwrite(new)
    _, old_ordinary = partition_for_grad_overwrite(old)
    ema_ordinary = jax.tree.map(
        lambda old_value, new_value: beta * old_value + (1.0 - beta) * new_value, old_ordinary, new_ordinary
    )
    return eqx.combine(new_overwrites, ema_ordinary, is_leaf=_is_overwrite)


def _split_state_for_pipeline(
    state: GrugTrainState,
    *,
    pipeline: GrugJaxPPConfig,
    optimizer: optax.GradientTransformation,
) -> GrugPipelineTrainState:
    params = state.params.split_for_pipeline(pipeline.stages, pipeline.stage_layer_counts)
    ema_params = (
        None
        if state.ema_params is None
        else state.ema_params.split_for_pipeline(pipeline.stages, pipeline.stage_layer_counts)
    )
    if state.pending_qb_betas is None:
        raise ValueError("explicit pipeline state splitting requires pending_qb_betas")
    pending_qb_betas = tuple(state.pending_qb_betas[stage.start_layer : stage.end_layer] for stage in params)
    return GrugPipelineTrainState(
        step=state.step,
        params=params,
        opt_state=tuple(init_optimizer_for_trainables(optimizer, stage_params) for stage_params in params),
        ema_params=ema_params,
        pending_qb_betas=pending_qb_betas,
    )


def _merge_pipeline_state(state: GrugPipelineTrainState) -> GrugTrainState:
    return GrugTrainState(
        step=state.step,
        params=Transformer.merge_pipeline_stages(state.params),
        opt_state=state.opt_state,
        ema_params=None if state.ema_params is None else Transformer.merge_pipeline_stages(state.ema_params),
        pending_qb_betas=jnp.concatenate(state.pending_qb_betas, axis=0),
    )


def _array_sharding(value, mesh: Mesh) -> NamedSharding:
    value_sharding = value.sharding
    if hasattr(value_sharding, "spec"):
        sharding = NamedSharding(mesh, value_sharding.spec)
        memory_kind = getattr(value_sharding, "memory_kind", None)
        if memory_kind is not None:
            sharding = sharding.with_memory_kind(memory_kind)
        return sharding
    return NamedSharding(mesh, P(*([None] * value.ndim)))


def _is_shardable_array(value) -> bool:
    return hasattr(value, "sharding") and hasattr(value, "ndim")


def _tree_named_shardings_on_stage(mpmd_mesh, stage_index: int, tree):
    stage_mesh = mpmd_mesh.unstack[stage_index]

    def leaf_sharding(value):
        if _is_shardable_array(value):
            return _array_sharding(value, stage_mesh)
        return None

    return jax.tree.map(leaf_sharding, tree)


def _tree_named_shardings_on_mesh(mesh: Mesh, tree):
    def leaf_sharding(value):
        if _is_shardable_array(value):
            return _array_sharding(value, mesh)
        return None

    return jax.tree.map(leaf_sharding, tree)


def _localize_automatic_jaxpp_input_shardings(compiled, mpmd_mesh):
    """Bind automatic JaxPP runtime inputs to this process's pipeline-stage mesh."""
    if not mpmd_mesh.jax_mesh.is_multi_process:
        return compiled

    stage_mesh = mpmd_mesh.lowering_mesh()

    def localize(sharding: NamedSharding) -> NamedSharding:
        localized = NamedSharding(stage_mesh, sharding.spec)
        if sharding.memory_kind is not None:
            localized = localized.with_memory_kind(sharding.memory_kind)
        return localized

    return dataclasses.replace(
        compiled,
        in_info=dataclasses.replace(
            compiled.in_info,
            in_shardings=tuple(localize(sharding) for sharding in compiled.in_info.in_shardings),
        ),
    )


def _mpmd_sharding_like(value, mpmd_mesh, stage_index: int):
    if isinstance(value.sharding, NamedSharding):
        return _require_jaxpp().MpmdSharding(
            mpmd_mesh,
            mesh_ids={stage_index},
            spec=value.sharding.spec,
            memory_kind=value.sharding.memory_kind,
        )
    return _require_jaxpp().MpmdSharding(
        mpmd_mesh,
        mesh_ids={stage_index},
        spec=P(*([None] * value.ndim)),
    )


def _tree_mpmd_shardings_on_stage(mpmd_mesh, stage_index: int, tree):
    def leaf_sharding(value):
        if _is_shardable_array(value):
            return _mpmd_sharding_like(value, mpmd_mesh, stage_index)
        return None

    return jax.tree.map(leaf_sharding, tree)


def _pipeline_state_shardings(
    mpmd_mesh,
    state: GrugPipelineTrainState,
    stage_mpmd_indices: tuple[int, ...],
) -> GrugPipelineTrainState:
    return dataclasses.replace(
        state,
        step=_require_jaxpp().MpmdSharding(mpmd_mesh, mesh_ids={0}, spec=P()),
        params=tuple(
            _tree_mpmd_shardings_on_stage(mpmd_mesh, mpmd_index, params)
            for mpmd_index, params in zip(stage_mpmd_indices, state.params, strict=True)
        ),
        opt_state=tuple(
            _tree_mpmd_shardings_on_stage(mpmd_mesh, mpmd_index, opt_state)
            for mpmd_index, opt_state in zip(stage_mpmd_indices, state.opt_state, strict=True)
        ),
        pending_qb_betas=tuple(
            _tree_mpmd_shardings_on_stage(mpmd_mesh, mpmd_index, qb_betas)
            for mpmd_index, qb_betas in zip(stage_mpmd_indices, state.pending_qb_betas, strict=True)
        ),
    )


def _pipeline_state_named_shardings(
    mpmd_mesh,
    state: GrugPipelineTrainState,
    stage_mpmd_indices: tuple[int, ...],
) -> GrugPipelineTrainState:
    return dataclasses.replace(
        state,
        step=NamedSharding(mpmd_mesh.unstack[0], P()),
        params=tuple(
            _tree_named_shardings_on_stage(mpmd_mesh, mpmd_index, params)
            for mpmd_index, params in zip(stage_mpmd_indices, state.params, strict=True)
        ),
        opt_state=tuple(
            _tree_named_shardings_on_stage(mpmd_mesh, mpmd_index, opt_state)
            for mpmd_index, opt_state in zip(stage_mpmd_indices, state.opt_state, strict=True)
        ),
        pending_qb_betas=tuple(
            _tree_named_shardings_on_stage(mpmd_mesh, mpmd_index, qb_betas)
            for mpmd_index, qb_betas in zip(stage_mpmd_indices, state.pending_qb_betas, strict=True)
        ),
    )


def _reshard_to_mpmd(mpmd_mesh, tree, target_shardings):
    threshold_raw = os.environ.get("GRUG_JAXPP_RESHARD_THRESHOLD_BYTES")
    threshold = int(threshold_raw) if threshold_raw else None
    return _require_jaxpp().spmd_to_mpmd_reshard(mpmd_mesh, tree, target_shardings, threshold=threshold)


def _split_state_for_explicit_mpmd(
    state: GrugTrainState,
    *,
    pipeline: GrugJaxPPConfig,
    optimizer: optax.GradientTransformation,
    mpmd_mesh,
) -> GrugPipelineTrainState:
    stage_state = _split_state_for_pipeline(state, pipeline=pipeline, optimizer=optimizer)
    if stage_state.ema_params is not None:
        raise ValueError("explicit_mpmd does not yet support EMA")
    stage_mpmd_indices = _pipeline_stage_mpmd_indices(pipeline)
    return _reshard_to_mpmd(
        mpmd_mesh,
        stage_state,
        _pipeline_state_shardings(mpmd_mesh, stage_state, stage_mpmd_indices),
    )


def _put_batch_on_stage(mpmd_mesh, stage_index: int, batch: GrugLmExample) -> GrugLmExample:
    return _reshard_to_mpmd(mpmd_mesh, batch, _tree_mpmd_shardings_on_stage(mpmd_mesh, stage_index, batch))


def _copy_shardable_tree(tree):
    def copy_leaf(value):
        if isinstance(value, jax.Array):
            return jnp.array(value, copy=True)
        return value

    return jax.tree.map(copy_leaf, tree)


def _shape_parameter_count(tree) -> int:
    _, tree = partition_for_grad_overwrite(tree)
    total = 0
    for leaf in jax.tree.leaves(tree):
        shape = getattr(leaf, "shape", None)
        if shape is not None:
            total += int(np.prod(shape, dtype=np.int64))
    return total


def _process_has_sharding(sharding: NamedSharding) -> bool:
    process_index = jax.process_index()
    return any(device.process_index == process_index for device in sharding.mesh.devices.flat)


def _empty_sharded_array(shape: tuple[int, ...], dtype: jnp.dtype, sharding: NamedSharding) -> jax.Array:
    return jax.make_array_from_single_device_arrays(
        shape=shape,
        sharding=sharding,
        arrays=[],
        dtype=dtype,
    )


def _stage_local_scalar(value: jax.Array, sharding: NamedSharding) -> jax.Array:
    dtype = value.dtype
    if not _process_has_sharding(sharding):
        return _empty_sharded_array((), dtype, sharding)
    return jax.device_put(np.array(0, dtype=np.dtype(dtype)), sharding)


def _localize_stage_optimizer_state(mpmd_mesh, stage_index: int, opt_state: optax.OptState) -> optax.OptState:
    stage_mesh = mpmd_mesh.unstack[stage_index]

    def localize_leaf(value):
        if _is_shardable_array(value) and value.shape == ():
            return _stage_local_scalar(value, NamedSharding(stage_mesh, P()))
        return value

    return jax.tree.map(localize_leaf, opt_state)


@dataclass(frozen=True)
class _LocalLoweredExplicitMpmdStep:
    lowered: Any

    def __call__(
        self,
        state: GrugPipelineTrainState,
        batches: tuple[GrugLmExample, ...],
    ) -> tuple[GrugPipelineTrainState, dict[str, Any], None]:
        flat_args, args_tree = jax.tree_util.tree_flatten((state, batches))
        in_tree = jax.tree_util.tree_structure(self.lowered.in_shardings)
        if args_tree != in_tree:
            raise ValueError("lowered explicit MPMD train step received an unexpected input tree")

        local_jaxpr = self.lowered._local_jaxpr
        local_outs = self.lowered.eval_local(*(flat_args[idx] for idx in local_jaxpr.global_invar_indices))
        local_outs_by_idx = dict(zip(local_jaxpr.global_outvar_indices, local_outs, strict=True))

        local_loss = jax.device_put(
            jnp.zeros((), dtype=jnp.float32),
            NamedSharding(self.lowered.mpmd_mesh.unstack[self.lowered.mpmd_mesh.my_mpmd_axis_index], P()),
        )
        fallback = (state, {"train/loss": local_loss, "qb_beta_per_layer": state.pending_qb_betas}, None)
        flat_fallback, fallback_tree = jax.tree_util.tree_flatten(fallback)
        flat_out_shape, out_tree = jax.tree_util.tree_flatten(self.lowered.out_shape)
        if fallback_tree != out_tree:
            raise ValueError("explicit MPMD lowered output fallback does not match the traced output tree")

        flat_outs = [
            local_outs_by_idx[idx] if idx in local_outs_by_idx else flat_fallback[idx]
            for idx in range(len(flat_out_shape))
        ]
        return jax.tree_util.tree_unflatten(out_tree, flat_outs)


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array | None) -> Transformer:
    """Set router biases from QB betas (computed on previous step)."""
    if qb_betas is None:
        return model
    new_blocks = list(model.blocks)
    moe_idx = 0
    for i, block in enumerate(model.blocks):
        if block.mlp is None:
            continue
        new_bias = -qb_betas[moe_idx]
        new_bias = new_bias - jnp.mean(new_bias)
        new_mlp = eqx.tree_at(lambda m: m.router_bias, block.mlp, new_bias)
        new_blocks[i] = eqx.tree_at(lambda b: b.mlp, block, new_mlp)
        moe_idx += 1
    return eqx.tree_at(lambda t: t.blocks, model, tuple(new_blocks))


def _apply_stage_qb_betas(stage: TransformerPipelineStage, qb_betas: jax.Array) -> TransformerPipelineStage:
    new_blocks = list(stage.blocks)
    for local_index, block in enumerate(stage.blocks):
        if block.mlp is None:
            continue
        new_bias = -qb_betas[local_index]
        new_bias = new_bias - jnp.mean(new_bias)
        new_mlp = eqx.tree_at(lambda m: m.router_bias, block.mlp, new_bias)
        new_blocks[local_index] = eqx.tree_at(lambda b: b.mlp, block, new_mlp)
    return eqx.tree_at(lambda t: t.blocks, stage, tuple(new_blocks))


StageBackwardResiduals = tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]


def _compute_stage(
    params: TransformerPipelineStage,
    qb_betas: jax.Array,
    mp: jmp.Policy,
) -> TransformerPipelineStage:
    return _cast_preserving_overwrites(_apply_stage_qb_betas(params, qb_betas), mp.cast_to_compute)


def _stack_stage_router_metrics(router_stats: tuple[dict[str, jax.Array], ...]) -> dict[str, jax.Array]:
    if not router_stats:
        raise ValueError("pipeline stages must own at least one transformer block")
    return {
        "routing_entropy_per_layer": jnp.stack([stats["routing_entropy"] for stats in router_stats], axis=0),
        "routing_counts_per_layer": jnp.stack([stats["routing_counts"] for stats in router_stats], axis=0),
        "load_balancing_loss_per_layer": jnp.stack([stats["load_balancing_loss"] for stats in router_stats], axis=0),
        "router_z_loss_per_layer": jnp.stack([stats["router_z_loss"] for stats in router_stats], axis=0),
        "qb_beta_per_layer": jnp.stack([stats["qb_beta"] for stats in router_stats], axis=0),
        "capacity_overflow_per_layer": jnp.stack([stats["capacity_overflow"] for stats in router_stats], axis=0),
    }


def _stage_input_gradient_backward(
    params: TransformerPipelineStage,
    qb_betas: jax.Array,
    hidden: jax.Array,
    batch: GrugLmExample,
    d_hidden: jax.Array,
    mp: jmp.Policy,
) -> tuple[jax.Array, StageBackwardResiduals]:
    compute_params = _compute_stage(params, qb_betas, mp)
    block_inputs = []
    stage_hidden = hidden
    for local_index in range(len(compute_params.blocks)):
        block_inputs.append(stage_hidden)
        stage_hidden, _ = compute_params.run_block(local_index, stage_hidden, batch.attn_mask)

    output_cotangents = [jnp.zeros_like(hidden) for _ in compute_params.blocks]
    arriving_cotangent = d_hidden
    for local_index in reversed(range(len(compute_params.blocks))):
        block_input = block_inputs[local_index]
        output_cotangents[local_index] = arriving_cotangent

        def activation_projection(
            stage_input,
            block_index=local_index,
            output_cotangent=arriving_cotangent,
        ):
            block_output, _ = compute_params.run_block(block_index, stage_input, batch.attn_mask)
            return jnp.sum(block_output.astype(jnp.float32) * output_cotangent.astype(jnp.float32))

        arriving_cotangent = jax.grad(activation_projection)(block_input)

    return arriving_cotangent, (tuple(block_inputs), tuple(output_cotangents))


def _stage_weight_backward(
    params: TransformerPipelineStage,
    qb_betas: jax.Array,
    residuals: StageBackwardResiduals,
    batch: GrugLmExample,
    mp: jmp.Policy,
):
    block_inputs, output_cotangents = residuals

    def independent_block_projections(stage_params):
        compute_params = _compute_stage(stage_params, qb_betas, mp)
        projection = jnp.zeros((), dtype=jnp.float32)
        for local_index, (block_input, output_cotangent) in enumerate(zip(block_inputs, output_cotangents, strict=True)):
            block_output, _ = compute_params.run_block(
                local_index,
                jax.lax.stop_gradient(block_input),
                batch.attn_mask,
            )
            projection = projection + jnp.sum(
                block_output.astype(jnp.float32) * jax.lax.stop_gradient(output_cotangent).astype(jnp.float32)
            )
        return projection

    return jax.grad(independent_block_projections)(params)


def _last_stage_input_gradient_backward(
    params: TransformerPipelineStage,
    qb_betas: jax.Array,
    hidden: jax.Array,
    batch: GrugLmExample,
    mp: jmp.Policy,
    *,
    logsumexp_weight: float | None,
) -> tuple[jax.Array, jax.Array, jax.Array, StageBackwardResiduals]:
    compute_params = _compute_stage(params, qb_betas, mp)
    block_inputs = []
    router_stats = []
    stage_hidden = hidden
    for local_index in range(len(compute_params.blocks)):
        block_inputs.append(stage_hidden)
        stage_hidden, block_router_stats = compute_params.run_block(local_index, stage_hidden, batch.attn_mask)
        router_stats.append(block_router_stats)
    router_metrics = _stack_stage_router_metrics(tuple(router_stats))
    stopped_router_metrics = jax.tree.map(jax.lax.stop_gradient, router_metrics)

    def head_loss(block_output):
        final_hidden = compute_params.finalize_hidden(block_output)
        loss, metrics = compute_params.hidden_next_token_loss(
            final_hidden,
            batch.tokens,
            batch.loss_weight,
            stopped_router_metrics,
            reduction="mean",
            logsumexp_weight=logsumexp_weight,
            return_router_metrics=True,
        )
        return loss, metrics["qb_beta_per_layer"]

    (loss, qb_betas_next), arriving_cotangent = jax.value_and_grad(head_loss, has_aux=True)(stage_hidden)
    output_cotangents = [jnp.zeros_like(hidden) for _ in compute_params.blocks]
    router_z_loss_scale = compute_params.config.router_z_loss_coef / len(compute_params.blocks)
    for local_index in reversed(range(len(compute_params.blocks))):
        block_input = block_inputs[local_index]
        output_cotangents[local_index] = arriving_cotangent

        def activation_projection(
            stage_input,
            block_index=local_index,
            output_cotangent=arriving_cotangent,
        ):
            block_output, block_router_stats = compute_params.run_block(block_index, stage_input, batch.attn_mask)
            output_projection = jnp.sum(block_output.astype(jnp.float32) * output_cotangent.astype(jnp.float32))
            return output_projection + router_z_loss_scale * block_router_stats["router_z_loss"]

        arriving_cotangent = jax.grad(activation_projection)(block_input)

    return loss, qb_betas_next, arriving_cotangent, (tuple(block_inputs), tuple(output_cotangents))


def _last_stage_weight_backward(
    params: TransformerPipelineStage,
    qb_betas: jax.Array,
    residuals: StageBackwardResiduals,
    batch: GrugLmExample,
    mp: jmp.Policy,
    *,
    logsumexp_weight: float | None,
):
    block_inputs, output_cotangents = residuals

    def independent_projections(stage_params):
        compute_params = _compute_stage(stage_params, qb_betas, mp)
        projection = jnp.zeros((), dtype=jnp.float32)
        block_outputs = []
        router_stats = []
        router_z_loss_scale = compute_params.config.router_z_loss_coef / len(compute_params.blocks)
        for local_index, (block_input, output_cotangent) in enumerate(zip(block_inputs, output_cotangents, strict=True)):
            block_output, block_router_stats = compute_params.run_block(
                local_index,
                jax.lax.stop_gradient(block_input),
                batch.attn_mask,
            )
            block_outputs.append(block_output)
            router_stats.append(block_router_stats)
            projection = projection + jnp.sum(
                block_output.astype(jnp.float32) * jax.lax.stop_gradient(output_cotangent).astype(jnp.float32)
            )
            projection = projection + router_z_loss_scale * block_router_stats["router_z_loss"]

        router_metrics = jax.tree.map(
            jax.lax.stop_gradient,
            _stack_stage_router_metrics(tuple(router_stats)),
        )
        final_hidden = compute_params.finalize_hidden(jax.lax.stop_gradient(block_outputs[-1]))
        head_loss = cast(
            jax.Array,
            compute_params.hidden_next_token_loss(
                final_hidden,
                batch.tokens,
                batch.loss_weight,
                router_metrics,
                reduction="mean",
                logsumexp_weight=logsumexp_weight,
            ),
        )
        return projection + head_loss

    return jax.grad(independent_projections)(params)


def _make_explicit_mpmd_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    pipeline: GrugJaxPPConfig,
    mpmd_mesh,
    sample_state: GrugPipelineTrainState,
    sample_batches,
):
    mpmd = _require_jaxpp_explicit_mpmd()
    z_loss = z_loss_weight if z_loss_weight > 0 else None
    num_stages = len(sample_state.params)
    if num_stages < 2:
        raise ValueError("explicit MPMD requires at least 2 pipeline stages")
    stage_mpmd_indices = _pipeline_stage_mpmd_indices(pipeline)
    if len(stage_mpmd_indices) != num_stages:
        raise ValueError(f"expected {num_stages} schedule stage placements, got {len(stage_mpmd_indices)}")
    if pipeline.microbatches == 1:
        if len(sample_batches) != num_stages:
            raise ValueError(f"expected {num_stages} stage-local batches, got {len(sample_batches)}")
    else:
        if len(sample_batches) != pipeline.microbatches:
            raise ValueError(f"expected {pipeline.microbatches} microbatch groups, got {len(sample_batches)}")
        for microbatch_index, microbatch_batches in enumerate(sample_batches):
            if len(microbatch_batches) != num_stages:
                raise ValueError(
                    f"expected {num_stages} stage-local batches for microbatch {microbatch_index}, "
                    f"got {len(microbatch_batches)}"
                )
    explicit_microbatch_schedules = ("gpipe", "std_1f1b", "interleaved_gpipe")
    if pipeline.microbatches != 1 and pipeline.schedule not in explicit_microbatch_schedules:
        raise ValueError(f"explicit MPMD microbatching supports only schedule in {explicit_microbatch_schedules}")
    activation_pspec = P(_BATCH_AXES, None, None)
    qb_pspec = P(None, None)

    activation_shardings = tuple(
        NamedSharding(mpmd_mesh.unstack[mpmd_index], activation_pspec) for mpmd_index in stage_mpmd_indices
    )
    backward_residual_shardings = tuple(
        (
            tuple(activation_shardings[stage_index] for _ in stage.blocks),
            tuple(activation_shardings[stage_index] for _ in stage.blocks),
        )
        for stage_index, stage in enumerate(sample_state.params)
    )
    input_gradient_first_schedules = None
    if pipeline.explicit_mpmd_schedule_mode == "input_gradient_first":
        planner_tasks = _require_jaxpp().ZeroBubble(num_stages=num_stages).tasks(pipeline.microbatches)
        input_gradient_first_schedules = tuple(
            tuple((task.fwd_or_bwd.name, task.mubatch_idx) for task in stage_tasks) for stage_tasks in planner_tasks
        )
    qb_shardings = tuple(NamedSharding(mpmd_mesh.unstack[mpmd_index], qb_pspec) for mpmd_index in stage_mpmd_indices)
    stage0_loss_sharding = NamedSharding(mpmd_mesh.unstack[0], P())
    last_loss_sharding = NamedSharding(mpmd_mesh.unstack[stage_mpmd_indices[-1]], P())
    stage0_step_sharding = NamedSharding(mpmd_mesh.unstack[0], P())
    stage_token_shardings = tuple(NamedSharding(mpmd_mesh.unstack[mpmd_index], P()) for mpmd_index in stage_mpmd_indices)

    param_shardings = tuple(
        _tree_named_shardings_on_stage(mpmd_mesh, mpmd_index, params)
        for mpmd_index, params in zip(stage_mpmd_indices, sample_state.params, strict=True)
    )
    opt_state_shardings = tuple(
        _tree_named_shardings_on_stage(mpmd_mesh, mpmd_index, opt_state)
        for mpmd_index, opt_state in zip(stage_mpmd_indices, sample_state.opt_state, strict=True)
    )

    compute_param_shardings = []
    stages_with_sonic_weights = []
    for mpmd_index, stage_params, stage_shardings in zip(
        stage_mpmd_indices, sample_state.params, param_shardings, strict=True
    ):
        stage_mesh = mpmd_mesh.unstack[mpmd_index]
        has_sonic_weights = False
        for block_index, block in enumerate(stage_params.blocks):
            if block.mlp is None or block.mlp.expert_mlp.implementation != "sonic":
                continue
            has_sonic_weights = True
            replicated_expert_sharding = NamedSharding(stage_mesh, P())
            for weight_name in ("w_gate", "w_up", "w_down"):
                stage_shardings = eqx.tree_at(
                    lambda tree, block_index=block_index, weight_name=weight_name: getattr(
                        tree.blocks[block_index].mlp.expert_mlp, weight_name
                    ),
                    stage_shardings,
                    replicated_expert_sharding,
                )
        compute_param_shardings.append(stage_shardings)
        stages_with_sonic_weights.append(has_sonic_weights)
    compute_param_shardings = tuple(compute_param_shardings)
    stages_with_sonic_weights = tuple(stages_with_sonic_weights)
    if pipeline.sonic_fsdp_materialization == "staged_per_step" and not all(stages_with_sonic_weights):
        missing_stages = tuple(index for index, has_weights in enumerate(stages_with_sonic_weights) if not has_weights)
        raise ValueError(
            "staged_per_step Sonic FSDP materialization requires Sonic expert weights on every stage; "
            f"missing stages: {missing_stages}"
        )

    def stage_batch_shardings(stage_batches):
        return tuple(
            _tree_named_shardings_on_stage(mpmd_mesh, mpmd_index, sample_batch)
            for mpmd_index, sample_batch in zip(stage_mpmd_indices, stage_batches, strict=True)
        )

    if pipeline.microbatches == 1:
        batch_in_shardings = stage_batch_shardings(sample_batches)
    else:
        batch_in_shardings = tuple(stage_batch_shardings(microbatch_batches) for microbatch_batches in sample_batches)

    in_shardings = (
        _pipeline_state_named_shardings(mpmd_mesh, sample_state, stage_mpmd_indices),
        batch_in_shardings,
    )

    def apply_stage_updates(params, updates):
        def apply_one(param, update):
            if param is None:
                return None
            return (param + update).astype(param.dtype)

        return jax.tree.map(apply_one, params, updates, is_leaf=lambda x: x is None)

    def stage0_forward(params: TransformerPipelineStage, qb_betas: jax.Array, batch: GrugLmExample):
        compute_params = _compute_stage(params, qb_betas, mp)
        hidden = compute_params.embed_tokens(batch.tokens)
        hidden, router_metrics = compute_params.block_range(hidden, mask=batch.attn_mask)
        return hidden, router_metrics["qb_beta_per_layer"]

    def stage_forward(params: TransformerPipelineStage, qb_betas: jax.Array, hidden, batch: GrugLmExample):
        compute_params = _compute_stage(params, qb_betas, mp)
        hidden, router_metrics = compute_params.block_range(hidden, mask=batch.attn_mask)
        return hidden, router_metrics["qb_beta_per_layer"]

    def stage0_backward(params: TransformerPipelineStage, qb_betas: jax.Array, batch: GrugLmExample, d_hidden):
        def activation_projection(stage_params):
            hidden, _ = stage0_forward(stage_params, qb_betas, batch)
            return jnp.sum(hidden.astype(jnp.float32) * d_hidden.astype(jnp.float32))

        return jax.grad(activation_projection)(params)

    def stage_backward(
        params: TransformerPipelineStage,
        qb_betas: jax.Array,
        hidden,
        batch: GrugLmExample,
        d_hidden,
    ):
        def activation_projection(stage_params, stage_hidden):
            stage_hidden, _ = stage_forward(stage_params, qb_betas, stage_hidden, batch)
            return jnp.sum(stage_hidden.astype(jnp.float32) * d_hidden.astype(jnp.float32))

        return jax.grad(activation_projection, argnums=(0, 1))(params, hidden)

    def stage_input_gradient_backward(params, qb_betas, hidden, batch, d_hidden):
        return _stage_input_gradient_backward(params, qb_betas, hidden, batch, d_hidden, mp)

    def stage_weight_backward(params, qb_betas, residuals, batch):
        return _stage_weight_backward(params, qb_betas, residuals, batch, mp)

    def last_stage_loss(
        params: TransformerPipelineStage,
        qb_betas: jax.Array,
        hidden,
        batch: GrugLmExample,
    ):
        compute_params = _compute_stage(params, qb_betas, mp)
        hidden, router_metrics = compute_params.block_range(hidden, mask=batch.attn_mask)
        hidden = compute_params.finalize_hidden(hidden)
        loss, metrics = compute_params.hidden_next_token_loss(
            hidden,
            batch.tokens,
            batch.loss_weight,
            router_metrics,
            reduction="mean",
            logsumexp_weight=z_loss,
            return_router_metrics=True,
        )
        return loss, metrics["qb_beta_per_layer"]

    def last_stage_loss_and_grads(
        params: TransformerPipelineStage,
        qb_betas: jax.Array,
        hidden,
        batch: GrugLmExample,
    ):
        def loss_fn(stage_params, stage_hidden):
            return last_stage_loss(stage_params, qb_betas, stage_hidden, batch)

        (loss, qb_betas_next), (grads, d_hidden) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            params, hidden
        )
        return loss, qb_betas_next, grads, d_hidden

    def last_stage_backward(
        params: TransformerPipelineStage,
        qb_betas: jax.Array,
        hidden,
        batch: GrugLmExample,
    ):
        def loss_fn(stage_params, stage_hidden):
            loss, _ = last_stage_loss(stage_params, qb_betas, stage_hidden, batch)
            return loss

        return jax.grad(loss_fn, argnums=(0, 1))(params, hidden)

    def last_stage_input_gradient_backward(params, qb_betas, hidden, batch):
        return _last_stage_input_gradient_backward(
            params,
            qb_betas,
            hidden,
            batch,
            mp,
            logsumexp_weight=z_loss,
        )

    def last_stage_weight_backward(params, qb_betas, residuals, batch):
        return _last_stage_weight_backward(
            params,
            qb_betas,
            residuals,
            batch,
            mp,
            logsumexp_weight=z_loss,
        )

    def update_stage(params: TransformerPipelineStage, opt_state: optax.OptState, grads):
        overwrites, ordinary_grads = partition_for_grad_overwrite(grads)
        _, ordinary_params = partition_for_grad_overwrite(params)
        updates, opt_state = optimizer.update(ordinary_grads, opt_state, ordinary_params)
        updated_params = apply_stage_updates(ordinary_params, updates)
        return eqx.combine(overwrites, updated_params, is_leaf=_is_overwrite), opt_state

    def keep_step(step: jax.Array):
        return step

    def materialize_compute_params(params: TransformerPipelineStage):
        return params

    def sonic_materialization_completion_token(
        params: TransformerPipelineStage,
        incoming_token: jax.Array,
    ) -> jax.Array:
        completion_token = incoming_token.astype(jnp.float32)
        for block in params.blocks:
            if block.mlp is None or block.mlp.expert_mlp.implementation != "sonic":
                continue
            expert_mlp = block.mlp.expert_mlp
            completion_token = completion_token + jnp.sum(expert_mlp.w_gate[0, :, 0], dtype=jnp.float32)
            completion_token = completion_token + jnp.sum(expert_mlp.w_up[0, :, 0], dtype=jnp.float32)
            completion_token = completion_token + jnp.sum(expert_mlp.w_down[0, 0, :], dtype=jnp.float32)
        return completion_token

    def add_trees(left, right):
        return _accumulate_microbatch_tree(left, right)

    def average_tree(tree):
        return _average_microbatch_tree(tree, pipeline.microbatches)

    def average_loss(loss):
        return loss * jnp.asarray(1.0 / pipeline.microbatches, dtype=loss.dtype)

    def accumulate_or_set(accumulated, value, *, name: str, out_shardings):
        if accumulated is None:
            return value
        return mpmd.task(add_trees, name=name, out_shardings=out_shardings)(accumulated, value)

    def transfer_between_stages(value, source_stage_index: int, target_stage_index: int, out_shardings):
        if stage_mpmd_indices[source_stage_index] == stage_mpmd_indices[target_stage_index]:
            return value
        return mpmd.transfer(value, out_shardings=out_shardings).done()

    def send_between_stages(value, source_stage_index: int, target_stage_index: int, out_shardings):
        if stage_mpmd_indices[source_stage_index] == stage_mpmd_indices[target_stage_index]:
            return value
        return mpmd.transfer(value, out_shardings=out_shardings)

    def receive_between_stages(value_or_future, source_stage_index: int, target_stage_index: int):
        if stage_mpmd_indices[source_stage_index] == stage_mpmd_indices[target_stage_index]:
            return value_or_future
        return value_or_future.done()

    @mpmd.mpmd(mpmd_mesh, in_shardings=in_shardings, infer_donation=True)
    def explicit_pipeline_step(
        state: GrugPipelineTrainState,
        batches: tuple[GrugLmExample, ...],
    ) -> tuple[GrugPipelineTrainState, dict[str, Any], None]:
        params = list(state.params)
        opt_state = list(state.opt_state)
        qb_betas = state.pending_qb_betas
        qb_betas_next = [None] * num_stages
        grads = [None] * num_stages

        hidden, qb_betas_next[0] = mpmd.task(
            stage0_forward,
            name="grug_stage0_forward",
            out_shardings=(activation_shardings[0], qb_shardings[0]),
        )(params[0], qb_betas[0], batches[0])

        stage_inputs = [hidden]
        for stage_index in range(1, num_stages - 1):
            hidden = transfer_between_stages(
                hidden,
                stage_index - 1,
                stage_index,
                activation_shardings[stage_index],
            )
            stage_inputs.append(hidden)
            hidden, qb_betas_next[stage_index] = mpmd.task(
                stage_forward,
                name=f"grug_stage{stage_index}_forward",
                out_shardings=(activation_shardings[stage_index], qb_shardings[stage_index]),
            )(params[stage_index], qb_betas[stage_index], hidden, batches[stage_index])

        hidden = transfer_between_stages(
            hidden,
            num_stages - 2,
            num_stages - 1,
            activation_shardings[num_stages - 1],
        )
        stage_inputs.append(hidden)
        loss, qb_betas_next[num_stages - 1], grads[num_stages - 1], d_hidden = mpmd.task(
            last_stage_loss_and_grads,
            name=f"grug_stage{num_stages - 1}_loss_backward",
            out_shardings=(
                last_loss_sharding,
                qb_shardings[num_stages - 1],
                param_shardings[num_stages - 1],
                activation_shardings[num_stages - 1],
            ),
        )(params[num_stages - 1], qb_betas[num_stages - 1], hidden, batches[num_stages - 1])

        for stage_index in range(num_stages - 2, 0, -1):
            d_hidden = transfer_between_stages(
                d_hidden,
                stage_index + 1,
                stage_index,
                activation_shardings[stage_index],
            )
            grads[stage_index], d_hidden = mpmd.task(
                stage_backward,
                name=f"grug_stage{stage_index}_backward",
                out_shardings=(param_shardings[stage_index], activation_shardings[stage_index]),
            )(params[stage_index], qb_betas[stage_index], stage_inputs[stage_index], batches[stage_index], d_hidden)

        d_hidden = transfer_between_stages(d_hidden, 1, 0, activation_shardings[0])
        grads[0] = mpmd.task(
            stage0_backward,
            name="grug_stage0_backward",
            out_shardings=param_shardings[0],
        )(params[0], qb_betas[0], batches[0], d_hidden)

        for stage_index in range(num_stages):
            params[stage_index], opt_state[stage_index] = mpmd.task(
                update_stage,
                name=f"grug_stage{stage_index}_update",
                out_shardings=(param_shardings[stage_index], opt_state_shardings[stage_index]),
            )(params[stage_index], opt_state[stage_index], grads[stage_index])
        step = mpmd.task(
            keep_step,
            name="grug_keep_step",
            out_shardings=stage0_step_sharding,
        )(state.step)
        loss_for_metrics = transfer_between_stages(loss, num_stages - 1, 0, stage0_loss_sharding)

        next_state = dataclasses.replace(
            state,
            step=step,
            params=tuple(params),
            opt_state=tuple(opt_state),
            pending_qb_betas=tuple(qb_betas_next),
        )
        metrics = {"train/loss": loss_for_metrics, "qb_beta_per_layer": tuple(qb_betas_next)}
        return next_state, metrics, None

    @mpmd.mpmd(mpmd_mesh, in_shardings=in_shardings, infer_donation=True)
    def explicit_gpipe_step(
        state: GrugPipelineTrainState,
        batches_by_microbatch,
    ) -> tuple[GrugPipelineTrainState, dict[str, Any], None]:
        params = list(state.params)
        opt_state = list(state.opt_state)
        qb_betas = state.pending_qb_betas
        qb_betas_next = [None] * num_stages
        grads = [None] * num_stages
        stage_inputs_by_microbatch = []
        loss_sum = None

        for microbatch_index in range(pipeline.microbatches):
            microbatches = batches_by_microbatch[microbatch_index]
            hidden, stage_qb_betas = mpmd.task(
                stage0_forward,
                name=f"grug_gpipe_mb{microbatch_index}_stage0_forward",
                out_shardings=(activation_shardings[0], qb_shardings[0]),
            )(params[0], qb_betas[0], microbatches[0])
            qb_betas_next[0] = accumulate_or_set(
                qb_betas_next[0],
                stage_qb_betas,
                name=f"grug_gpipe_mb{microbatch_index}_stage0_accumulate_qb",
                out_shardings=qb_shardings[0],
            )

            stage_inputs = [hidden]
            for stage_index in range(1, num_stages - 1):
                hidden = transfer_between_stages(
                    hidden,
                    stage_index - 1,
                    stage_index,
                    activation_shardings[stage_index],
                )
                stage_inputs.append(hidden)
                hidden, stage_qb_betas = mpmd.task(
                    stage_forward,
                    name=f"grug_gpipe_mb{microbatch_index}_stage{stage_index}_forward",
                    out_shardings=(activation_shardings[stage_index], qb_shardings[stage_index]),
                )(params[stage_index], qb_betas[stage_index], hidden, microbatches[stage_index])
                qb_betas_next[stage_index] = accumulate_or_set(
                    qb_betas_next[stage_index],
                    stage_qb_betas,
                    name=f"grug_gpipe_mb{microbatch_index}_stage{stage_index}_accumulate_qb",
                    out_shardings=qb_shardings[stage_index],
                )

            hidden = transfer_between_stages(
                hidden,
                num_stages - 2,
                num_stages - 1,
                activation_shardings[num_stages - 1],
            )
            stage_inputs.append(hidden)
            stage_inputs_by_microbatch.append(tuple(stage_inputs))

        for microbatch_index in range(pipeline.microbatches - 1, -1, -1):
            microbatches = batches_by_microbatch[microbatch_index]
            stage_inputs = stage_inputs_by_microbatch[microbatch_index]
            loss, stage_qb_betas, stage_grads, d_hidden = mpmd.task(
                last_stage_loss_and_grads,
                name=f"grug_gpipe_mb{microbatch_index}_stage{num_stages - 1}_loss_backward",
                out_shardings=(
                    last_loss_sharding,
                    qb_shardings[num_stages - 1],
                    param_shardings[num_stages - 1],
                    activation_shardings[num_stages - 1],
                ),
            )(
                params[num_stages - 1],
                qb_betas[num_stages - 1],
                stage_inputs[num_stages - 1],
                microbatches[num_stages - 1],
            )
            loss_sum = accumulate_or_set(
                loss_sum,
                loss,
                name=f"grug_gpipe_mb{microbatch_index}_accumulate_loss",
                out_shardings=last_loss_sharding,
            )
            qb_betas_next[num_stages - 1] = accumulate_or_set(
                qb_betas_next[num_stages - 1],
                stage_qb_betas,
                name=f"grug_gpipe_mb{microbatch_index}_stage{num_stages - 1}_accumulate_qb",
                out_shardings=qb_shardings[num_stages - 1],
            )
            grads[num_stages - 1] = accumulate_or_set(
                grads[num_stages - 1],
                stage_grads,
                name=f"grug_gpipe_mb{microbatch_index}_stage{num_stages - 1}_accumulate_grads",
                out_shardings=param_shardings[num_stages - 1],
            )

            for stage_index in range(num_stages - 2, 0, -1):
                d_hidden = transfer_between_stages(
                    d_hidden,
                    stage_index + 1,
                    stage_index,
                    activation_shardings[stage_index],
                )
                stage_grads, d_hidden = mpmd.task(
                    stage_backward,
                    name=f"grug_gpipe_mb{microbatch_index}_stage{stage_index}_backward",
                    out_shardings=(param_shardings[stage_index], activation_shardings[stage_index]),
                )(
                    params[stage_index],
                    qb_betas[stage_index],
                    stage_inputs[stage_index],
                    microbatches[stage_index],
                    d_hidden,
                )
                grads[stage_index] = accumulate_or_set(
                    grads[stage_index],
                    stage_grads,
                    name=f"grug_gpipe_mb{microbatch_index}_stage{stage_index}_accumulate_grads",
                    out_shardings=param_shardings[stage_index],
                )

            d_hidden = transfer_between_stages(d_hidden, 1, 0, activation_shardings[0])
            stage_grads = mpmd.task(
                stage0_backward,
                name=f"grug_gpipe_mb{microbatch_index}_stage0_backward",
                out_shardings=param_shardings[0],
            )(params[0], qb_betas[0], microbatches[0], d_hidden)
            grads[0] = accumulate_or_set(
                grads[0],
                stage_grads,
                name=f"grug_gpipe_mb{microbatch_index}_stage0_accumulate_grads",
                out_shardings=param_shardings[0],
            )

        if loss_sum is None:
            raise ValueError("explicit GPipe did not accumulate any microbatch losses")
        loss = mpmd.task(average_loss, name="grug_gpipe_average_loss", out_shardings=last_loss_sharding)(loss_sum)
        qb_betas_next = [
            mpmd.task(
                average_tree,
                name=f"grug_gpipe_stage{stage_index}_average_qb",
                out_shardings=qb_shardings[stage_index],
            )(stage_qb_betas)
            for stage_index, stage_qb_betas in enumerate(qb_betas_next)
        ]
        grads = [
            mpmd.task(
                average_tree,
                name=f"grug_gpipe_stage{stage_index}_average_grads",
                out_shardings=param_shardings[stage_index],
            )(stage_grads)
            for stage_index, stage_grads in enumerate(grads)
        ]

        for stage_index in range(num_stages):
            params[stage_index], opt_state[stage_index] = mpmd.task(
                update_stage,
                name=f"grug_gpipe_stage{stage_index}_update",
                out_shardings=(param_shardings[stage_index], opt_state_shardings[stage_index]),
            )(params[stage_index], opt_state[stage_index], grads[stage_index])
        step = mpmd.task(
            keep_step,
            name="grug_gpipe_keep_step",
            out_shardings=stage0_step_sharding,
        )(state.step)
        loss_for_metrics = transfer_between_stages(loss, num_stages - 1, 0, stage0_loss_sharding)

        next_state = dataclasses.replace(
            state,
            step=step,
            params=tuple(params),
            opt_state=tuple(opt_state),
            pending_qb_betas=tuple(qb_betas_next),
        )
        metrics = {"train/loss": loss_for_metrics, "qb_beta_per_layer": tuple(qb_betas_next)}
        return next_state, metrics, None

    interleaved_task_order = _interleaved_gpipe_task_order(pipeline) if pipeline.schedule == "interleaved_gpipe" else ()

    @mpmd.mpmd(mpmd_mesh, in_shardings=in_shardings, infer_donation=True)
    def explicit_interleaved_gpipe_step(
        state: GrugPipelineTrainState,
        batches_by_microbatch,
    ) -> tuple[GrugPipelineTrainState, dict[str, Any], None]:
        interleaved_batches = (batches_by_microbatch,) if pipeline.microbatches == 1 else batches_by_microbatch
        params = list(state.params)
        opt_state = list(state.opt_state)
        qb_betas = state.pending_qb_betas
        qb_betas_next = [None] * num_stages
        grads = [None] * num_stages
        stage_inputs = {}
        forward_edges = {}
        backward_edges = {}
        loss_sum = None

        for direction, stage_index, microbatch_index in interleaved_task_order:
            microbatches = interleaved_batches[microbatch_index]
            key = (stage_index, microbatch_index)
            if direction == "fwd":
                if stage_index == 0:
                    hidden, stage_qb_betas = mpmd.task(
                        stage0_forward,
                        name=f"grug_interleaved_mb{microbatch_index}_stage0_forward",
                        out_shardings=(activation_shardings[0], qb_shardings[0]),
                    )(params[0], qb_betas[0], microbatches[0])
                else:
                    hidden = receive_between_stages(forward_edges[key], stage_index - 1, stage_index)
                    stage_inputs[key] = hidden
                    if stage_index == num_stages - 1:
                        loss, stage_qb_betas = mpmd.task(
                            last_stage_loss,
                            name=f"grug_interleaved_mb{microbatch_index}_stage{stage_index}_loss_forward",
                            out_shardings=(last_loss_sharding, qb_shardings[stage_index]),
                        )(params[stage_index], qb_betas[stage_index], hidden, microbatches[stage_index])
                        loss_sum = accumulate_or_set(
                            loss_sum,
                            loss,
                            name=f"grug_interleaved_mb{microbatch_index}_accumulate_loss",
                            out_shardings=last_loss_sharding,
                        )
                    else:
                        hidden, stage_qb_betas = mpmd.task(
                            stage_forward,
                            name=f"grug_interleaved_mb{microbatch_index}_stage{stage_index}_forward",
                            out_shardings=(activation_shardings[stage_index], qb_shardings[stage_index]),
                        )(params[stage_index], qb_betas[stage_index], hidden, microbatches[stage_index])

                qb_betas_next[stage_index] = accumulate_or_set(
                    qb_betas_next[stage_index],
                    stage_qb_betas,
                    name=f"grug_interleaved_mb{microbatch_index}_stage{stage_index}_accumulate_qb",
                    out_shardings=qb_shardings[stage_index],
                )
                if stage_index < num_stages - 1:
                    forward_edges[(stage_index + 1, microbatch_index)] = send_between_stages(
                        hidden,
                        stage_index,
                        stage_index + 1,
                        activation_shardings[stage_index + 1],
                    )
                continue

            if stage_index == num_stages - 1:
                stage_grads, d_hidden = mpmd.task(
                    last_stage_backward,
                    name=f"grug_interleaved_mb{microbatch_index}_stage{stage_index}_backward",
                    out_shardings=(param_shardings[stage_index], activation_shardings[stage_index]),
                )(params[stage_index], qb_betas[stage_index], stage_inputs[key], microbatches[stage_index])
            else:
                d_hidden = receive_between_stages(backward_edges[key], stage_index + 1, stage_index)
                if stage_index == 0:
                    stage_grads = mpmd.task(
                        stage0_backward,
                        name=f"grug_interleaved_mb{microbatch_index}_stage0_backward",
                        out_shardings=param_shardings[0],
                    )(params[0], qb_betas[0], microbatches[0], d_hidden)
                else:
                    stage_grads, d_hidden = mpmd.task(
                        stage_backward,
                        name=f"grug_interleaved_mb{microbatch_index}_stage{stage_index}_backward",
                        out_shardings=(param_shardings[stage_index], activation_shardings[stage_index]),
                    )(
                        params[stage_index],
                        qb_betas[stage_index],
                        stage_inputs[key],
                        microbatches[stage_index],
                        d_hidden,
                    )

            grads[stage_index] = accumulate_or_set(
                grads[stage_index],
                stage_grads,
                name=f"grug_interleaved_mb{microbatch_index}_stage{stage_index}_accumulate_grads",
                out_shardings=param_shardings[stage_index],
            )
            if stage_index > 0:
                backward_edges[(stage_index - 1, microbatch_index)] = send_between_stages(
                    d_hidden,
                    stage_index,
                    stage_index - 1,
                    activation_shardings[stage_index - 1],
                )

        if loss_sum is None:
            raise ValueError("explicit interleaved GPipe did not accumulate any microbatch losses")
        loss = mpmd.task(
            average_loss,
            name="grug_interleaved_average_loss",
            out_shardings=last_loss_sharding,
        )(loss_sum)
        qb_betas_next = [
            mpmd.task(
                average_tree,
                name=f"grug_interleaved_stage{stage_index}_average_qb",
                out_shardings=qb_shardings[stage_index],
            )(stage_qb_betas)
            for stage_index, stage_qb_betas in enumerate(qb_betas_next)
        ]
        grads = [
            mpmd.task(
                average_tree,
                name=f"grug_interleaved_stage{stage_index}_average_grads",
                out_shardings=param_shardings[stage_index],
            )(stage_grads)
            for stage_index, stage_grads in enumerate(grads)
        ]

        for stage_index in range(num_stages):
            params[stage_index], opt_state[stage_index] = mpmd.task(
                update_stage,
                name=f"grug_interleaved_stage{stage_index}_update",
                out_shardings=(param_shardings[stage_index], opt_state_shardings[stage_index]),
            )(params[stage_index], opt_state[stage_index], grads[stage_index])
        step = mpmd.task(
            keep_step,
            name="grug_interleaved_keep_step",
            out_shardings=stage0_step_sharding,
        )(state.step)
        loss_for_metrics = transfer_between_stages(loss, num_stages - 1, 0, stage0_loss_sharding)

        next_state = dataclasses.replace(
            state,
            step=step,
            params=tuple(params),
            opt_state=tuple(opt_state),
            pending_qb_betas=tuple(qb_betas_next),
        )
        metrics = {"train/loss": loss_for_metrics, "qb_beta_per_layer": tuple(qb_betas_next)}
        return next_state, metrics, None

    def std_1f1b_stage_schedule(stage_index: int) -> tuple[tuple[str, int], ...]:
        warmup = min(num_stages - stage_index, pipeline.microbatches)
        tasks = [("fwd", microbatch_index) for microbatch_index in range(warmup)]
        for microbatch_index in range(warmup, pipeline.microbatches):
            tasks.append(("bwd", microbatch_index - warmup))
            tasks.append(("fwd", microbatch_index))
        tasks.extend(
            ("bwd", microbatch_index)
            for microbatch_index in range(pipeline.microbatches - warmup, pipeline.microbatches)
        )
        return tuple(tasks)

    @mpmd.mpmd(mpmd_mesh, in_shardings=in_shardings, infer_donation=True)
    def explicit_std_1f1b_step(
        state: GrugPipelineTrainState,
        batches_by_microbatch,
    ) -> tuple[GrugPipelineTrainState, dict[str, Any], None]:
        params = list(state.params)
        opt_state = list(state.opt_state)
        qb_betas = state.pending_qb_betas
        qb_betas_next = [None] * num_stages
        grads = [None] * num_stages
        loss_sum = None
        stage_inputs = {}
        forward_futures = {}
        d_hidden_futures = {}
        forward_done = set()
        backward_done = set()
        input_backward_done = set()
        weight_backward_done = set()
        backward_residuals = {}
        compute_params = (
            {}
            if pipeline.sonic_fsdp_materialization == "staged_per_step"
            else {stage_index: stage_params for stage_index, stage_params in enumerate(params)}
        )
        materialization_token_futures = {}
        prioritize_transfers = pipeline.explicit_mpmd_schedule_mode == "transfer_priority"

        def send_pipeline_wire_value(
            value,
            source_stage_index: int,
            target_stage_index: int,
            *,
            fp8_dtype: Fp8PipelineWireDtype,
            name: str,
        ):
            if stage_mpmd_indices[source_stage_index] == stage_mpmd_indices[target_stage_index]:
                return value
            if pipeline.explicit_mpmd_pipeline_wire_format == "bf16":
                return mpmd.transfer(value, out_shardings=activation_shardings[target_stage_index])
            packed = mpmd.task(
                functools.partial(pack_fp8_pipeline_wire, dtype=fp8_dtype),
                name=f"{name}_pack_fp8",
                out_shardings=activation_shardings[source_stage_index],
            )(value)
            return mpmd.transfer(packed, out_shardings=activation_shardings[target_stage_index])

        def receive_pipeline_wire_value(
            value_or_future,
            source_stage_index: int,
            target_stage_index: int,
            *,
            fp8_dtype: Fp8PipelineWireDtype,
            name: str,
        ):
            if stage_mpmd_indices[source_stage_index] == stage_mpmd_indices[target_stage_index]:
                return value_or_future
            transferred = value_or_future.done()
            if pipeline.explicit_mpmd_pipeline_wire_format == "bf16":
                return transferred
            return mpmd.task(
                functools.partial(unpack_fp8_pipeline_wire, dtype=fp8_dtype),
                name=f"{name}_unpack_fp8",
                out_shardings=activation_shardings[target_stage_index],
            )(transferred)

        def stage_compute_params(stage_index: int):
            if stage_index in compute_params:
                return compute_params[stage_index]
            if stage_index == 0:
                incoming_token = state.step
            else:
                incoming_token = materialization_token_futures[stage_index].done()
            stage_params = mpmd.task(
                materialize_compute_params,
                name=f"grug_1f1b_stage{stage_index}_materialize_sonic_weights",
                out_shardings=compute_param_shardings[stage_index],
            )(params[stage_index])
            compute_params[stage_index] = stage_params
            if stage_index + 1 < num_stages:
                completion_token = mpmd.task(
                    sonic_materialization_completion_token,
                    name=f"grug_1f1b_stage{stage_index}_sonic_materialization_completion_token",
                    out_shardings=stage_token_shardings[stage_index],
                )(stage_params, incoming_token)
                materialization_token_futures[stage_index + 1] = mpmd.transfer(
                    completion_token,
                    out_shardings=stage_token_shardings[stage_index + 1],
                )
            return stage_params

        def ensure_forward(stage_index: int, microbatch_index: int):
            key = (stage_index, microbatch_index)
            if key in forward_done:
                return
            microbatches = batches_by_microbatch[microbatch_index]
            stage_params = stage_compute_params(stage_index)

            if stage_index == 0:
                hidden, stage_qb_betas = mpmd.task(
                    stage0_forward,
                    name=f"grug_1f1b_mb{microbatch_index}_stage0_forward",
                    out_shardings=(activation_shardings[0], qb_shardings[0]),
                )(stage_params, qb_betas[0], microbatches[0])
                if prioritize_transfers:
                    forward_futures[(1, microbatch_index)] = send_pipeline_wire_value(
                        hidden,
                        0,
                        1,
                        fp8_dtype="e4m3",
                        name=f"grug_1f1b_mb{microbatch_index}_stage0_forward_wire",
                    )
                qb_betas_next[0] = accumulate_or_set(
                    qb_betas_next[0],
                    stage_qb_betas,
                    name=f"grug_1f1b_mb{microbatch_index}_stage0_accumulate_qb",
                    out_shardings=qb_shardings[0],
                )
                if not prioritize_transfers:
                    forward_futures[(1, microbatch_index)] = send_pipeline_wire_value(
                        hidden,
                        0,
                        1,
                        fp8_dtype="e4m3",
                        name=f"grug_1f1b_mb{microbatch_index}_stage0_forward_wire",
                    )
                forward_done.add(key)
                return

            ensure_forward(stage_index - 1, microbatch_index)
            hidden = receive_pipeline_wire_value(
                forward_futures[key],
                stage_index - 1,
                stage_index,
                fp8_dtype="e4m3",
                name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_forward_wire",
            )
            stage_inputs[key] = hidden
            if stage_index == num_stages - 1:
                forward_done.add(key)
                return

            hidden, stage_qb_betas = mpmd.task(
                stage_forward,
                name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_forward",
                out_shardings=(activation_shardings[stage_index], qb_shardings[stage_index]),
            )(stage_params, qb_betas[stage_index], hidden, microbatches[stage_index])
            if prioritize_transfers:
                forward_futures[(stage_index + 1, microbatch_index)] = send_pipeline_wire_value(
                    hidden,
                    stage_index,
                    stage_index + 1,
                    fp8_dtype="e4m3",
                    name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_forward_wire",
                )
            qb_betas_next[stage_index] = accumulate_or_set(
                qb_betas_next[stage_index],
                stage_qb_betas,
                name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_accumulate_qb",
                out_shardings=qb_shardings[stage_index],
            )
            if not prioritize_transfers:
                forward_futures[(stage_index + 1, microbatch_index)] = send_pipeline_wire_value(
                    hidden,
                    stage_index,
                    stage_index + 1,
                    fp8_dtype="e4m3",
                    name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_forward_wire",
                )
            forward_done.add(key)

        def ensure_backward(stage_index: int, microbatch_index: int):
            nonlocal loss_sum
            key = (stage_index, microbatch_index)
            if key in backward_done:
                return
            microbatches = batches_by_microbatch[microbatch_index]
            stage_params = stage_compute_params(stage_index)

            if stage_index == num_stages - 1:
                ensure_forward(stage_index, microbatch_index)
                loss, stage_qb_betas, stage_grads, d_hidden = mpmd.task(
                    last_stage_loss_and_grads,
                    name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_loss_backward",
                    out_shardings=(
                        last_loss_sharding,
                        qb_shardings[stage_index],
                        param_shardings[stage_index],
                        activation_shardings[stage_index],
                    ),
                )(
                    stage_params,
                    qb_betas[stage_index],
                    stage_inputs[key],
                    microbatches[stage_index],
                )
                if prioritize_transfers:
                    d_hidden_futures[(stage_index - 1, microbatch_index)] = send_pipeline_wire_value(
                        d_hidden,
                        stage_index,
                        stage_index - 1,
                        fp8_dtype="e5m2",
                        name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_backward_wire",
                    )
                loss_sum = accumulate_or_set(
                    loss_sum,
                    loss,
                    name=f"grug_1f1b_mb{microbatch_index}_accumulate_loss",
                    out_shardings=last_loss_sharding,
                )
                qb_betas_next[stage_index] = accumulate_or_set(
                    qb_betas_next[stage_index],
                    stage_qb_betas,
                    name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_accumulate_qb",
                    out_shardings=qb_shardings[stage_index],
                )
                grads[stage_index] = accumulate_or_set(
                    grads[stage_index],
                    stage_grads,
                    name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_accumulate_grads",
                    out_shardings=param_shardings[stage_index],
                )
                if not prioritize_transfers:
                    d_hidden_futures[(stage_index - 1, microbatch_index)] = send_pipeline_wire_value(
                        d_hidden,
                        stage_index,
                        stage_index - 1,
                        fp8_dtype="e5m2",
                        name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_backward_wire",
                    )
                backward_done.add(key)
                return

            ensure_backward(stage_index + 1, microbatch_index)
            d_hidden = receive_pipeline_wire_value(
                d_hidden_futures[key],
                stage_index + 1,
                stage_index,
                fp8_dtype="e5m2",
                name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_backward_wire",
            )
            if stage_index == 0:
                stage_grads = mpmd.task(
                    stage0_backward,
                    name=f"grug_1f1b_mb{microbatch_index}_stage0_backward",
                    out_shardings=param_shardings[0],
                )(stage_params, qb_betas[0], microbatches[0], d_hidden)
                grads[0] = accumulate_or_set(
                    grads[0],
                    stage_grads,
                    name=f"grug_1f1b_mb{microbatch_index}_stage0_accumulate_grads",
                    out_shardings=param_shardings[0],
                )
                backward_done.add(key)
                return

            ensure_forward(stage_index, microbatch_index)
            stage_grads, d_hidden = mpmd.task(
                stage_backward,
                name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_backward",
                out_shardings=(param_shardings[stage_index], activation_shardings[stage_index]),
            )(
                stage_params,
                qb_betas[stage_index],
                stage_inputs[key],
                microbatches[stage_index],
                d_hidden,
            )
            if prioritize_transfers:
                d_hidden_futures[(stage_index - 1, microbatch_index)] = send_pipeline_wire_value(
                    d_hidden,
                    stage_index,
                    stage_index - 1,
                    fp8_dtype="e5m2",
                    name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_backward_wire",
                )
            grads[stage_index] = accumulate_or_set(
                grads[stage_index],
                stage_grads,
                name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_accumulate_grads",
                out_shardings=param_shardings[stage_index],
            )
            if not prioritize_transfers:
                d_hidden_futures[(stage_index - 1, microbatch_index)] = send_pipeline_wire_value(
                    d_hidden,
                    stage_index,
                    stage_index - 1,
                    fp8_dtype="e5m2",
                    name=f"grug_1f1b_mb{microbatch_index}_stage{stage_index}_backward_wire",
                )
            backward_done.add(key)

        def ensure_input_backward(stage_index: int, microbatch_index: int):
            nonlocal loss_sum
            key = (stage_index, microbatch_index)
            if key in input_backward_done:
                return
            microbatches = batches_by_microbatch[microbatch_index]
            stage_params = stage_compute_params(stage_index)

            if stage_index == num_stages - 1:
                ensure_forward(stage_index, microbatch_index)
                loss, stage_qb_betas, d_hidden, residuals = mpmd.task(
                    last_stage_input_gradient_backward,
                    name=f"grug_zb_mb{microbatch_index}_stage{stage_index}_backward_input",
                    out_shardings=(
                        last_loss_sharding,
                        qb_shardings[stage_index],
                        activation_shardings[stage_index],
                        backward_residual_shardings[stage_index],
                    ),
                )(
                    stage_params,
                    qb_betas[stage_index],
                    stage_inputs[key],
                    microbatches[stage_index],
                )
                d_hidden_futures[(stage_index - 1, microbatch_index)] = send_pipeline_wire_value(
                    d_hidden,
                    stage_index,
                    stage_index - 1,
                    fp8_dtype="e5m2",
                    name=f"grug_zb_mb{microbatch_index}_stage{stage_index}_backward_wire",
                )
                backward_residuals[key] = residuals
                loss_sum = accumulate_or_set(
                    loss_sum,
                    loss,
                    name=f"grug_zb_mb{microbatch_index}_accumulate_loss",
                    out_shardings=last_loss_sharding,
                )
                qb_betas_next[stage_index] = accumulate_or_set(
                    qb_betas_next[stage_index],
                    stage_qb_betas,
                    name=f"grug_zb_mb{microbatch_index}_stage{stage_index}_accumulate_qb",
                    out_shardings=qb_shardings[stage_index],
                )
                input_backward_done.add(key)
                return

            ensure_input_backward(stage_index + 1, microbatch_index)
            d_hidden = receive_pipeline_wire_value(
                d_hidden_futures[key],
                stage_index + 1,
                stage_index,
                fp8_dtype="e5m2",
                name=f"grug_zb_mb{microbatch_index}_stage{stage_index}_backward_wire",
            )
            if stage_index == 0:
                stage_grads = mpmd.task(
                    stage0_backward,
                    name=f"grug_zb_mb{microbatch_index}_stage0_backward",
                    out_shardings=param_shardings[0],
                )(stage_params, qb_betas[0], microbatches[0], d_hidden)
                grads[0] = accumulate_or_set(
                    grads[0],
                    stage_grads,
                    name=f"grug_zb_mb{microbatch_index}_stage0_accumulate_grads",
                    out_shardings=param_shardings[0],
                )
                input_backward_done.add(key)
                weight_backward_done.add(key)
                return

            ensure_forward(stage_index, microbatch_index)
            d_hidden, residuals = mpmd.task(
                stage_input_gradient_backward,
                name=f"grug_zb_mb{microbatch_index}_stage{stage_index}_backward_input",
                out_shardings=(
                    activation_shardings[stage_index],
                    backward_residual_shardings[stage_index],
                ),
            )(
                stage_params,
                qb_betas[stage_index],
                stage_inputs[key],
                microbatches[stage_index],
                d_hidden,
            )
            d_hidden_futures[(stage_index - 1, microbatch_index)] = send_pipeline_wire_value(
                d_hidden,
                stage_index,
                stage_index - 1,
                fp8_dtype="e5m2",
                name=f"grug_zb_mb{microbatch_index}_stage{stage_index}_backward_wire",
            )
            backward_residuals[key] = residuals
            input_backward_done.add(key)

        def ensure_weight_backward(stage_index: int, microbatch_index: int):
            key = (stage_index, microbatch_index)
            if key in weight_backward_done:
                return
            ensure_input_backward(stage_index, microbatch_index)
            if stage_index == 0:
                return

            microbatches = batches_by_microbatch[microbatch_index]
            stage_params = stage_compute_params(stage_index)
            if stage_index == num_stages - 1:
                stage_grads = mpmd.task(
                    last_stage_weight_backward,
                    name=f"grug_zb_mb{microbatch_index}_stage{stage_index}_backward_weight",
                    out_shardings=param_shardings[stage_index],
                )(
                    stage_params,
                    qb_betas[stage_index],
                    backward_residuals[key],
                    microbatches[stage_index],
                )
            else:
                stage_grads = mpmd.task(
                    stage_weight_backward,
                    name=f"grug_zb_mb{microbatch_index}_stage{stage_index}_backward_weight",
                    out_shardings=param_shardings[stage_index],
                )(
                    stage_params,
                    qb_betas[stage_index],
                    backward_residuals[key],
                    microbatches[stage_index],
                )
            del backward_residuals[key]
            grads[stage_index] = accumulate_or_set(
                grads[stage_index],
                stage_grads,
                name=f"grug_zb_mb{microbatch_index}_stage{stage_index}_accumulate_grads",
                out_shardings=param_shardings[stage_index],
            )
            weight_backward_done.add(key)

        if pipeline.explicit_mpmd_schedule_mode == "input_gradient_first":
            if input_gradient_first_schedules is None:
                raise ValueError("input-gradient-first planner schedules were not initialized")
            planner_schedules = input_gradient_first_schedules
            for task_index in range(max(len(stage_schedule) for stage_schedule in planner_schedules)):
                for stage_index, stage_schedule in enumerate(planner_schedules):
                    if task_index >= len(stage_schedule):
                        continue
                    task_type, microbatch_index = stage_schedule[task_index]
                    if task_type == "FWD":
                        ensure_forward(stage_index, microbatch_index)
                    elif task_type == "BWD_I":
                        ensure_input_backward(stage_index, microbatch_index)
                    elif task_type == "BWD_W":
                        ensure_weight_backward(stage_index, microbatch_index)
                    else:
                        raise ValueError(f"unexpected ZeroBubble planner task type: {task_type}")
        else:
            stage_schedules = tuple(std_1f1b_stage_schedule(stage_index) for stage_index in range(num_stages))
            for task_index in range(2 * pipeline.microbatches):
                for stage_index, stage_schedule in enumerate(stage_schedules):
                    direction, microbatch_index = stage_schedule[task_index]
                    if direction == "fwd":
                        ensure_forward(stage_index, microbatch_index)
                    else:
                        ensure_backward(stage_index, microbatch_index)

        if loss_sum is None:
            raise ValueError("explicit 1F1B did not accumulate any microbatch losses")
        loss = mpmd.task(average_loss, name="grug_1f1b_average_loss", out_shardings=last_loss_sharding)(loss_sum)
        qb_betas_next = [
            mpmd.task(
                average_tree,
                name=f"grug_1f1b_stage{stage_index}_average_qb",
                out_shardings=qb_shardings[stage_index],
            )(stage_qb_betas)
            for stage_index, stage_qb_betas in enumerate(qb_betas_next)
        ]
        grads = [
            mpmd.task(
                average_tree,
                name=f"grug_1f1b_stage{stage_index}_average_grads",
                out_shardings=param_shardings[stage_index],
            )(stage_grads)
            for stage_index, stage_grads in enumerate(grads)
        ]

        for stage_index in range(num_stages):
            params[stage_index], opt_state[stage_index] = mpmd.task(
                update_stage,
                name=f"grug_1f1b_stage{stage_index}_update",
                out_shardings=(param_shardings[stage_index], opt_state_shardings[stage_index]),
            )(params[stage_index], opt_state[stage_index], grads[stage_index])
        step = mpmd.task(
            keep_step,
            name="grug_1f1b_keep_step",
            out_shardings=stage0_step_sharding,
        )(state.step)
        loss_for_metrics = mpmd.transfer(loss, out_shardings=stage0_loss_sharding).done()

        next_state = dataclasses.replace(
            state,
            step=step,
            params=tuple(params),
            opt_state=tuple(opt_state),
            pending_qb_betas=tuple(qb_betas_next),
        )
        metrics = {"train/loss": loss_for_metrics, "qb_beta_per_layer": tuple(qb_betas_next)}
        return next_state, metrics, None

    if pipeline.schedule == "interleaved_gpipe":
        return explicit_interleaved_gpipe_step
    if pipeline.microbatches == 1:
        return explicit_pipeline_step
    if pipeline.schedule == "gpipe":
        return explicit_gpipe_step
    if pipeline.schedule == "std_1f1b":
        return explicit_std_1f1b_step
    raise ValueError(f"Unsupported explicit MPMD schedule with microbatches > 1: {pipeline.schedule}")


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
) -> GrugTrainState:
    params = _cast_preserving_overwrites(Transformer.init(model_config, key=key), mp.cast_to_param)
    num_moe_layers = sum(1 for b in params.blocks if b.mlp is not None)
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=init_optimizer_for_trainables(optimizer, params),
        ema_params=params if ema_beta is not None else None,
        pending_qb_betas=jnp.zeros((num_moe_layers, model_config.num_experts)),
    )


def initial_pipeline_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    pipeline: GrugJaxPPConfig,
    mpmd_mesh,
) -> GrugPipelineTrainState:
    """Initialize explicit MPMD pipeline state without materializing full optimizer state."""
    params = _cast_preserving_overwrites(Transformer.init(model_config, key=key), mp.cast_to_param)
    stage_params = params.split_for_pipeline(pipeline.stages, pipeline.stage_layer_counts)
    stage_mpmd_indices = _pipeline_stage_mpmd_indices(pipeline)
    stage_params = _reshard_to_mpmd(
        mpmd_mesh,
        stage_params,
        tuple(
            _tree_mpmd_shardings_on_stage(mpmd_mesh, mpmd_index, params)
            for mpmd_index, params in zip(stage_mpmd_indices, stage_params, strict=True)
        ),
    )
    pending_qb_betas = tuple(
        jnp.zeros((stage.end_layer - stage.start_layer, model_config.num_experts), dtype=jnp.float32)
        for stage in stage_params
    )
    pending_qb_betas = _reshard_to_mpmd(
        mpmd_mesh,
        pending_qb_betas,
        tuple(
            _tree_mpmd_shardings_on_stage(mpmd_mesh, mpmd_index, qb_betas)
            for mpmd_index, qb_betas in zip(stage_mpmd_indices, pending_qb_betas, strict=True)
        ),
    )
    step = _stage_local_scalar(jnp.array(0, dtype=jnp.int32), NamedSharding(mpmd_mesh.unstack[0], P()))
    return GrugPipelineTrainState(
        step=step,
        params=stage_params,
        opt_state=tuple(
            _localize_stage_optimizer_state(mpmd_mesh, mpmd_index, init_optimizer_for_trainables(optimizer, params))
            for mpmd_index, params in zip(stage_mpmd_indices, stage_params, strict=True)
        ),
        ema_params=None,
        pending_qb_betas=pending_qb_betas,
    )


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
    pipeline: GrugJaxPPConfig | None = None,
    mpmd_mesh=None,
    in_shardings=None,
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

    if pipeline is None:
        train_step_jit = functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("compute_watch",))
        pipeline_schedule = None
    else:
        pp = _require_jaxpp()
        if mpmd_mesh is None:
            raise ValueError("mpmd_mesh is required when JaxPP pipeline training is enabled")
        train_step_jit = functools.partial(
            pp.mpmd_jit_with_loop,
            mpmd_mesh=mpmd_mesh,
            in_shardings=in_shardings,
            donate_argnums=(0,),
            static_argnames=("compute_watch",),
        )
        pipeline_schedule = _pipeline_schedule(pipeline)

    def apply_plain_updates(params, updates):
        if pipeline is None:
            return optax.apply_updates(params, updates)

        def apply_one(param, update):
            if param is None:
                return None
            return jnp.asarray(jax_ad.add_jaxvals_p.bind(param, update)).astype(jnp.asarray(param).dtype)

        return jax.tree.map(apply_one, params, updates, is_leaf=lambda x: x is None)

    @train_step_jit
    def train_step(state: GrugTrainState, batch, compute_watch: bool = False):
        # Apply pending QB betas to router biases inside JIT (avoids eager
        # host-side TPU kernel launches that can cause SPMD sync issues).
        qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)
        if ema_beta is not None:
            qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas)
        else:
            qb_ema_params = None

        def loss_fn(params):
            compute_params = _cast_preserving_overwrites(params, mp.cast_to_compute)
            return compute_params.next_token_loss(
                batch.tokens,
                batch.loss_weight,
                mask=batch.attn_mask,
                reduction="mean",
                logsumexp_weight=z_loss,
                return_router_metrics=True,
            )

        if pipeline is None:
            (loss, summarized_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(qb_params)
            metrics = {"train/loss": loss, **summarized_metrics}
        else:
            if pipeline_schedule is None:
                raise ValueError("pipeline_schedule must be initialized when JaxPP pipeline training is enabled")

            def microbatch_loss_fn(params, this_microbatch):
                compute_params = _cast_preserving_overwrites(params, mp.cast_to_compute)
                this_loss = compute_params.next_token_loss(
                    this_microbatch.tokens,
                    this_microbatch.loss_weight,
                    mask=this_microbatch.attn_mask,
                    reduction="mean",
                    logsumexp_weight=z_loss,
                    return_router_metrics=False,
                    pipeline_stages=pipeline.stages,
                    pipeline_stage_layer_counts=pipeline.stage_layer_counts,
                )
                return this_loss

            microbatch_grad = functools.partial(jax.value_and_grad(microbatch_loss_fn), qb_params)
            loss_sum, grads_sum = _require_jaxpp().treduce(
                microbatch_grad,
                batch,
                schedule=pipeline_schedule,
                operation=(_require_jaxpp().Add, _require_jaxpp().Add),
            )
            scale = jnp.asarray(1.0 / pipeline.microbatches, dtype=loss_sum.dtype)
            loss = loss_sum * scale
            grads = jax.tree.map(lambda grad: grad * scale, grads_sum)
            metrics = {
                "train/loss": loss,
                # Automatic JaxPP lowering currently rejects this per-layer,
                # per-expert aux tensor as an after-loop value. Keep the prior
                # QB feedback in place for schedule/MFU probes.
                "qb_beta_per_layer": state.pending_qb_betas,
            }
        overwrites, grads = partition_for_grad_overwrite(grads)
        _, ordinary_params = partition_for_grad_overwrite(qb_params)
        updates, opt_state = optimizer.update(grads, state.opt_state, ordinary_params)
        updated_params = apply_plain_updates(ordinary_params, updates)
        params = eqx.combine(overwrites, updated_params, is_leaf=_is_overwrite)

        if ema_beta is None:
            ema_params = None
        else:
            if qb_ema_params is None:
                raise ValueError("ema_params must be initialized when ema_beta is set.")
            ema_params = _ema_update(qb_ema_params, params, ema_beta)

        watch_stats = None
        if watch_config is not None and compute_watch:
            watch_stats = compute_watch_stats(
                watch_targets=watch_targets,
                include_norms=watch_config.include_norms,
                include_per_parameter_norms=watch_config.include_per_parameter_norms,
                include_histogram=watch_config.include_histograms,
                split_scan_layers=watch_config.split_scan_layers,
                params=ordinary_params,
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


def _bootstrap_nccl_ep(config: GrugRunConfig, mesh: Mesh, ep: Any) -> None:
    pipeline = config.trainer.pipeline
    if pipeline is None:
        raise ValueError("NCCL_EP bootstrap requires a pipeline configuration")
    if jax.local_device_count() != 1:
        raise ValueError(
            f"NCCL_EP requires one local GPU per process, got local_device_count={jax.local_device_count()}"
        )

    pipeline_groups = int(mesh.shape[pipeline.stage_axis_name])
    expert_size = int(mesh.shape["expert"])
    if jax.process_count() != pipeline_groups * expert_size:
        raise ValueError(
            "NCCL_EP requires the process world to be exactly pipeline x expert; "
            f"got process_count={jax.process_count()}, pipeline={pipeline_groups}, expert={expert_size}"
        )
    for axis in ("replica_dcn", "data", "model"):
        if int(mesh.shape[axis]) != 1:
            raise ValueError(f"NCCL_EP currently requires mesh axis {axis!r} to have size 1, got {mesh.shape[axis]}")

    max_batch_size = max(config.trainer.trainer.batch_schedule.unique_batch_sizes())
    if max_batch_size % pipeline.microbatches != 0:
        raise ValueError(
            f"NCCL_EP max batch size={max_batch_size} must be divisible by microbatches={pipeline.microbatches}"
        )
    global_microbatch_tokens = max_batch_size // pipeline.microbatches * config.model.max_seq_len
    if global_microbatch_tokens % expert_size != 0:
        raise ValueError(
            f"NCCL_EP microbatch tokens={global_microbatch_tokens} must be divisible by expert size={expert_size}"
        )
    max_tokens_per_rank = global_microbatch_tokens // expert_size

    nccl_ep_backend = importlib.import_module("levanter.grug._moe.ep_ncclep")
    recv_capacity_per_rank = nccl_ep_backend.ncclep_receive_capacity(
        global_tokens=global_microbatch_tokens,
        top_k=config.model.num_experts_per_token,
        ep_size=expert_size,
        capacity_factor=GRUG_MOE_NCCL_EP_CAPACITY_FACTOR,
    )
    ep.ep_bootstrap(
        world_size=jax.process_count(),
        rank=jax.process_index(),
        num_experts=config.model.num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity_per_rank,
        hidden_dim=config.model.hidden_dim,
    )
    te_cpp_ep = importlib.import_module("transformer_engine.jax.cpp_extensions.ep")
    physical_ep_config = te_cpp_ep.get_ep_config()
    if physical_ep_config.num_ep_groups != pipeline_groups:
        raise ValueError(
            "NCCL_EP bootstrap recorded an unexpected group count: "
            f"expected {pipeline_groups}, got {physical_ep_config.num_ep_groups}"
        )
    # JaxPP traces each localized stage with a size-1 pipeline axis. The C++
    # communicator is already one physical EP8 group; this Python snapshot only
    # controls TE's global abstract receive shape during stage-local tracing.
    te_cpp_ep.set_ep_config(
        dataclasses.replace(
            physical_ep_config,
            world_size=expert_size,
            rank=jax.process_index() % expert_size,
            num_ep_groups=1,
        )
    )
    logger.info(
        "NCCL_EP bootstrapped: world=%d groups=%d ep=%d max_tokens_per_rank=%d recv_capacity_per_rank=%d",
        jax.process_count(),
        pipeline_groups,
        expert_size,
        max_tokens_per_rank,
        recv_capacity_per_rank,
    )


def _run_grug_local(config: GrugRunConfig) -> None:
    """Entry point for the grug template training loop."""
    nccl_ep_modules = _load_nccl_ep_modules() if config.model.moe_implementation == "nccl_ep" else None
    trainer = config.trainer.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(config)

    run_id = trainer.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    optimizer = config.optimizer.build(trainer.num_train_steps)
    watch_config = trainer.watch

    # Grug uses raw PartitionSpecs rather than Trainer's logical axis mapping.
    # Keep the mesh compact so the batch pspec derived by `_batch_spec(mesh)` spans slices directly.
    # replica_axis_size=None lets compact_grug_mesh default to jax.process_count() (full
    # cross-slice replication); for JaxPP, it instead means one replica group per pipeline stage.
    mesh = _compact_or_pipeline_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
        pipeline=config.trainer.pipeline,
    )
    mpmd_mesh = None
    if config.trainer.pipeline is not None:
        mpmd_mesh = _require_jaxpp().MpmdMesh(mesh, config.trainer.pipeline.stage_axis_name)
        if config.trainer.pipeline.implementation == "auto" and os.environ.get(
            "GRUG_JAXPP_PATCH_CONST_SHARDINGS", "false"
        ).lower() in ("1", "true", "yes", "on"):
            _install_jaxpp_const_sharding_patch()

    explicit_mpmd = config.trainer.pipeline is not None and config.trainer.pipeline.implementation == "explicit_mpmd"
    if explicit_mpmd:
        if config.trainer.ema_beta is not None:
            raise ValueError("explicit_mpmd does not yet support EMA")
        train_step = None
    else:
        train_step = _make_train_step(
            optimizer,
            trainer.mp,
            z_loss_weight=config.trainer.z_loss_weight,
            ema_beta=config.trainer.ema_beta,
            pipeline=config.trainer.pipeline,
            mpmd_mesh=mpmd_mesh,
            watch_config=watch_config if watch_config.is_enabled else None,
        )

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)
    nccl_ep_sharding = None
    nccl_ep_mesh_resource = None
    if nccl_ep_modules is not None:
        if config.trainer.pipeline is None:
            raise ValueError("NCCL_EP requires a pipeline configuration")
        _, te_sharding = nccl_ep_modules
        nccl_ep_sharding = te_sharding
        nccl_ep_mesh_resource = te_sharding.MeshResource(
            dp_resource=config.trainer.pipeline.stage_axis_name,
            ep_resource="expert",
        )

    def nccl_ep_shard_guard():
        if nccl_ep_sharding is None:
            return contextlib.nullcontext()
        return nccl_ep_sharding.global_shard_guard(nccl_ep_mesh_resource)

    with set_mesh(mesh), nccl_ep_shard_guard():
        if nccl_ep_modules is not None:
            te_ep, _ = nccl_ep_modules
            _bootstrap_nccl_ep(config, mesh, te_ep)
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

        checkpointer = trainer.checkpointer.create(run_id)
        log_callbacks = not explicit_mpmd or (mpmd_mesh is not None and mpmd_mesh.my_mpmd_axis_index == 0)
        if explicit_mpmd:
            if mpmd_mesh is None or config.trainer.pipeline is None:
                raise ValueError("mpmd_mesh and pipeline must be initialized when explicit_mpmd is enabled")
            state = initial_pipeline_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=model_key,
                pipeline=config.trainer.pipeline,
                mpmd_mesh=mpmd_mesh,
            )
            logger.warning("explicit_mpmd uses stage-local JaxPP arrays; checkpoint writes are disabled for now")
            checkpointer = None
        else:

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
            state = restore_grug_state_from_checkpoint(
                state,
                checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
                load_checkpoint_setting=trainer.load_checkpoint,
                mesh=mesh,
                allow_partial=trainer.allow_partial_checkpoint,
            )
            dump_grug_state_sharding_run_artifact(
                state,
                log_dir=trainer.log_dir,
                run_id=run_id,
                path_override=config.trainer.sharding_dump_path,
            )

        if explicit_mpmd:
            parameter_count_value = _shape_parameter_count(state.params)
        else:
            _, ordinary_params = partition_for_grad_overwrite(state.params)
            parameter_count_value = parameter_count(ordinary_params)
        levanter.tracker.log_summary({"parameter_count": parameter_count_value})

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
        iterator_start_step = 0 if explicit_mpmd else int(state.step)
        iterator = LoadingTimeTrackerIterator(train_loader.iter_from_step(iterator_start_step))

        state_callbacks = StateCallbackRunner[Any](
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
                    sync_after_stop=not explicit_mpmd,
                ),
                every=1,
            )
        state_callbacks.add_hook(_make_mixture_stage_callback(train_dataset, batch_schedule), every=1)
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

        explicit_loop_step = 0 if explicit_mpmd else None

    last_loss: float | jax.Array = 0.0
    last_step_duration = 0.0

    # Main optimization loop. JaxPP enters a stage-local mesh while tracing, so
    # pipeline train steps must run outside the global Grug mesh context above.
    compiled_pipeline_train_step = None
    explicit_mpmd_train_step = None
    pipeline_state_sharding = None
    pipeline_batch_sharding = None
    try:
        while (explicit_loop_step if explicit_loop_step is not None else int(state.step)) < trainer.num_train_steps:
            with jax.profiler.TraceAnnotation("load_batch"):
                batch = next(iterator)
            step_start = time.perf_counter()
            current_step = explicit_loop_step if explicit_loop_step is not None else int(state.step)
            # grad_watch runs only on its configured interval.
            compute_watch = (
                watch_config.is_enabled and watch_config.interval > 0 and current_step % watch_config.interval == 0
            )
            if config.trainer.pipeline is None:
                with set_mesh(mesh):
                    if train_step is None:
                        raise ValueError("train_step must be initialized for non-pipeline training")
                    state, metrics, watch_stats = train_step(state, batch, compute_watch)
            elif explicit_mpmd:
                if mpmd_mesh is None or config.trainer.pipeline is None:
                    raise ValueError("mpmd_mesh and pipeline must be initialized when explicit_mpmd is enabled")
                if not isinstance(state, GrugPipelineTrainState):
                    raise TypeError("explicit_mpmd expects GrugPipelineTrainState")
                compute_watch = False
                stage_mpmd_indices = _pipeline_stage_mpmd_indices(config.trainer.pipeline)
                if config.trainer.pipeline.microbatches > 1:
                    with set_mesh(mesh):
                        explicit_batch = _reshape_batch_for_pipeline(batch, config.trainer.pipeline.microbatches)
                    host_stage_batches = tuple(
                        tuple(
                            (
                                _select_pipeline_microbatch(
                                    explicit_batch,
                                    microbatch_index,
                                    config.trainer.pipeline.microbatches,
                                )
                                if stage_index == 0
                                else _copy_shardable_tree(
                                    _select_pipeline_microbatch(
                                        explicit_batch,
                                        microbatch_index,
                                        config.trainer.pipeline.microbatches,
                                    )
                                )
                            )
                            for stage_index in range(config.trainer.pipeline.stages)
                        )
                        for microbatch_index in range(config.trainer.pipeline.microbatches)
                    )
                    stage_batches = tuple(
                        tuple(
                            _put_batch_on_stage(mpmd_mesh, mpmd_index, host_stage_batch)
                            for mpmd_index, host_stage_batch in zip(stage_mpmd_indices, microbatch_batches, strict=True)
                        )
                        for microbatch_batches in host_stage_batches
                    )
                else:
                    host_stage_batches = tuple(
                        batch if stage_index == 0 else _copy_shardable_tree(batch)
                        for stage_index in range(config.trainer.pipeline.stages)
                    )
                    stage_batches = tuple(
                        _put_batch_on_stage(mpmd_mesh, mpmd_index, host_stage_batch)
                        for mpmd_index, host_stage_batch in zip(stage_mpmd_indices, host_stage_batches, strict=True)
                    )
                with nccl_ep_shard_guard():
                    if explicit_mpmd_train_step is None:
                        explicit_mpmd_train_step = _make_explicit_mpmd_train_step(
                            optimizer,
                            trainer.mp,
                            z_loss_weight=config.trainer.z_loss_weight,
                            pipeline=config.trainer.pipeline,
                            mpmd_mesh=mpmd_mesh,
                            sample_state=state,
                            sample_batches=stage_batches,
                        )
                        lower_explicit_mpmd = os.environ.get("GRUG_JAXPP_LOWER_EXPLICIT", "true").lower() not in (
                            "0",
                            "false",
                            "no",
                        )
                        if mpmd_mesh.jax_mesh.is_multi_process and lower_explicit_mpmd:
                            explicit_mpmd_train_step = _LocalLoweredExplicitMpmdStep(
                                explicit_mpmd_train_step.lower(state, stage_batches)
                            )
                    state, metrics, watch_stats = explicit_mpmd_train_step(state, stage_batches)
            else:
                if mpmd_mesh is None:
                    raise ValueError("mpmd_mesh must be initialized when JaxPP pipeline training is enabled")
                pp = _require_jaxpp()
                compute_watch = False
                with set_mesh(mesh):
                    pipeline_batch = _reshape_batch_for_pipeline(batch, config.trainer.pipeline.microbatches)
                if compiled_pipeline_train_step is None:
                    if train_step is None:
                        raise ValueError("train_step must be initialized for automatic JaxPP training")
                    if isinstance(state, GrugTrainState) and state.pending_qb_betas is not None:
                        state = dataclasses.replace(state, pending_qb_betas=None)
                    explicit_auto_in_shardings = os.environ.get(
                        "GRUG_JAXPP_AUTO_EXPLICIT_IN_SHARDINGS", "false"
                    ).lower() in ("1", "true", "yes", "on")
                    if explicit_auto_in_shardings:
                        train_step = _make_train_step(
                            optimizer,
                            trainer.mp,
                            z_loss_weight=config.trainer.z_loss_weight,
                            ema_beta=config.trainer.ema_beta,
                            pipeline=config.trainer.pipeline,
                            mpmd_mesh=mpmd_mesh,
                            in_shardings=(
                                _tree_named_shardings_on_mesh(mesh, state),
                                _tree_named_shardings_on_mesh(mesh, pipeline_batch),
                            ),
                            watch_config=watch_config if watch_config.is_enabled else None,
                        )
                    compiled_pipeline_train_step = train_step.compile(
                        state,
                        pipeline_batch,
                    )
                    compiled_pipeline_train_step = _localize_automatic_jaxpp_input_shardings(
                        compiled_pipeline_train_step,
                        mpmd_mesh,
                    )
                    args_mpmd_shardings, kwargs_mpmd_shardings = compiled_pipeline_train_step.in_shardings
                    if kwargs_mpmd_shardings:
                        raise ValueError(f"Unexpected JaxPP keyword shardings: {kwargs_mpmd_shardings}")
                    pipeline_state_sharding, pipeline_batch_sharding = args_mpmd_shardings
                    state = pp.spmd_to_mpmd_reshard(mpmd_mesh, state, pipeline_state_sharding)
                if pipeline_batch_sharding is None:
                    raise ValueError("pipeline_batch_sharding must be initialized after compiling the JaxPP step")
                pipeline_batch = pp.spmd_to_mpmd_reshard(mpmd_mesh, pipeline_batch, pipeline_batch_sharding)
                state, metrics, watch_stats = compiled_pipeline_train_step(
                    state,
                    pipeline_batch,
                )
            if explicit_loop_step is not None:
                explicit_loop_step += 1
            step = current_step

            jax.block_until_ready(metrics["train/loss"])

            if math.isnan(float(metrics["train/loss"])):
                logger.error(f"NaN loss at step {current_step}. Stopping training.")
                break
            duration = time.perf_counter() - step_start
            hook_start = time.perf_counter()
            callback_state = state
            if log_callbacks:
                with jax.profiler.TraceAnnotation("callbacks"):
                    if isinstance(state, GrugPipelineTrainState) and not explicit_mpmd:
                        callback_state = _merge_pipeline_state(state)
                    else:
                        callback_state = state
                    if explicit_loop_step is not None:
                        callback_state = dataclasses.replace(
                            callback_state,
                            step=jnp.array(explicit_loop_step, dtype=jnp.int32),
                        )
                    state_callbacks.run(callback_state, loss=metrics["train/loss"], step_duration=duration)
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
                checkpoint_step = explicit_loop_step if explicit_loop_step is not None else int(state.step)
                checkpoint_tree = callback_state if explicit_loop_step is not None else state
                checkpointer.on_step(tree=checkpoint_tree, step=checkpoint_step)
    except BaseException:
        logger.exception("Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause")
        raise
    else:
        # Mirror classic trainer behavior: force callbacks on the last completed step.
        if log_callbacks:
            if isinstance(state, GrugPipelineTrainState) and not explicit_mpmd:
                final_state = _merge_pipeline_state(state)
            else:
                final_state = state
            if explicit_loop_step is not None:
                final_state = dataclasses.replace(final_state, step=jnp.array(explicit_loop_step, dtype=jnp.int32))
            state_callbacks.run(final_state, loss=last_loss, step_duration=last_step_duration, force=True)
        if checkpointer is not None:
            checkpoint_step = explicit_loop_step if explicit_loop_step is not None else int(state.step)
            checkpointer.on_step(tree=final_state, step=checkpoint_step, force=True)
            checkpointer.wait_until_finished()

    if profiler_enabled and log_callbacks:
        profile_dir = trainer.log_dir / run_id / "profiler"
        if profile_dir.exists():
            levanter.tracker.current_tracker().log_artifact(
                profile_dir,
                name=f"{run_id}-profiler",
                type="profiler",
            )

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
        processes_per_task=config.processes_per_task,
        post_setup_scripts=config.post_setup_scripts,
    )


__all__ = [
    "ExplicitMpmdPipelineWireFormat",
    "GrugEvalConfig",
    "GrugJaxPPConfig",
    "GrugPipelineTrainState",
    "GrugRunConfig",
    "GrugTrainState",
    "GrugTrainerConfig",
    "JaxPPExplicitMpmdScheduleMode",
    "JaxPPImplementation",
    "initial_state",
    "jaxpp_setup_scripts",
    "pack_fp8_pipeline_wire",
    "run_grug",
    "unpack_fp8_pipeline_wire",
]
