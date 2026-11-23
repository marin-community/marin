# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command line to ``GrugRunConfig`` for the AMD June scripts (no Fray or Iris).

The options cover the model size, the mesh, the batch and its microbatches, ZeRO-1, the attention kernel, profiling
and logging. The kernels and the loop are fixed in the model and trainer code, so there are no options for them. A
script adds its own options on top: the synthetic benchmark its step count and flat Adam.
"""

import argparse
import dataclasses
import json
import logging
import statistics
import time
from typing import get_args

import jax
import jmp
from levanter.callbacks._metrics import log_throughput_summary
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig, XprofUploadConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import DistributedConfig
from levanter.grug.attention import GrugAttentionImplementation
from levanter.tracker import TrackerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import multihost_broadcast_sync

from experiments.amd.june_67b_a2b.model import MOE_IMPLEMENTATION
from experiments.amd.june_67b_a2b.train import GrugTrainerConfig
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.synthetic_benchmark import FIRST_TIMED_STEP, snowball_model

logger = logging.getLogger("amd_june_run")

# ring_filtered's sender pool per destination shard, in units of the even share: the MI355X selected configuration.
POOLED_TRANSPORT_CAPACITY_FACTOR = 1.1


def preset(size: str, seq_len: int | None) -> tuple[GrugModelConfig, str]:
    """The model at ``size`` and its mixed-precision policy. ``full`` is June's production shape (``snowball_model``),
    ``medium`` that with 4 layers, ``tiny`` the smallest shape the kernels tile, for CPU smoke runs."""
    if size == "full":
        model, mp = snowball_model(seq_len or 4096), "p=f32,c=bfloat16"
    elif size == "medium":
        model, mp = dataclasses.replace(snowball_model(seq_len or 1024), num_layers=4), "p=f32,c=bfloat16"
    else:
        model = dataclasses.replace(
            snowball_model(seq_len or 32),
            vocab_size=128,
            hidden_dim=128,
            intermediate_dim=64,
            shared_expert_intermediate_dim=64,
            num_experts=16,
            num_layers=5,
            num_heads=2,
            num_kv_heads=1,
            head_dim=64,
            sliding_window=4,
        )
        mp = "f32"
    model = dataclasses.replace(
        model,
        moe_implementation=MOE_IMPLEMENTATION,
        pooled_transport_capacity_factor=POOLED_TRANSPORT_CAPACITY_FACTOR,
    )
    return model, mp


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """The model, mesh, batch, attention, profiling and logging options."""
    parser.add_argument("--size", choices=["tiny", "medium", "full"], required=True)
    parser.add_argument("--layers", type=int, help="Override the preset layer count.")
    parser.add_argument("--seq-len", type=int, help="Override the preset sequence length.")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Global batch of one step in sequences (default: one per device per microbatch).",
    )
    parser.add_argument(
        "--microbatches",
        type=int,
        default=1,
        help="Run each step's batch as this many microbatches, reducing the gradients once per step (1 = all at once).",
    )
    parser.add_argument(
        "--zero1",
        action="store_true",
        help="Shard the expert weights' parameters and optimizer state over the replicas (needs --microbatches >= 2).",
    )
    parser.add_argument("--mp", help="jmp policy override, e.g. 'p=f32,c=float16' (default: preset).")
    parser.add_argument("--expert-axis", type=int, default=1, help="Expert-parallel mesh axis size (1 = FSDP only).")
    parser.add_argument(
        "--replica-axis",
        type=int,
        default=1,
        help="Replica mesh axis size: full parameter copies that all-reduce their gradients (1 = no replicas).",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        help="Initialize jax.distributed from Slurm with one srun process per node owning this many GPUs "
        "(default: one process, no jax.distributed).",
    )
    parser.add_argument("--attention", choices=get_args(GrugAttentionImplementation), default="reference")
    parser.add_argument("--profile-steps", type=int, default=0, help="Profile this many steps (0 disables).")
    parser.add_argument("--profile-start", type=int, default=10, help="First profiled step.")
    parser.add_argument("--compilation-cache-dir", help="Persistent JAX compilation cache directory.")
    parser.add_argument(
        "--watch-interval", type=int, default=10, help="Steps between gradient watch steps (0 disables)."
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=300.0,
        help="Exit with status 1 if a step takes longer than this many seconds (0 disables). "
        "The first step and watch steps have no timeout.",
    )


def initialize_distributed(args: argparse.Namespace) -> None:
    """With --gpus-per-node, start jax.distributed from Slurm. Call before any JAX call that starts the backend."""
    if args.gpus_per_node is not None:
        # Without explicit device ids, JAX gives each Slurm process the single GPU numbered by its local rank.
        DistributedConfig(local_device_ids=list(range(args.gpus_per_node))).initialize()


def build_model(args: argparse.Namespace) -> tuple[GrugModelConfig, str]:
    """The preset model with the command line's options, and the preset's default mixed-precision policy."""
    model, default_mp = preset(args.size, args.seq_len)
    model = dataclasses.replace(model, attention_implementation=args.attention)
    if args.layers is not None:
        model = dataclasses.replace(model, num_layers=args.layers)
    return model, default_mp


def global_batch_size(args: argparse.Namespace) -> int:
    batch_size = jax.device_count() * args.microbatches if args.batch_size is None else args.batch_size
    if batch_size == 1:
        # jnp.roll in the next-token loss slices the (1, seq_len) token grid to (1, 1), and under an explicit mesh JAX
        # drops the sharding of that slice, then rejects concatenating it with the (1, seq_len - 1) remainder.
        raise ValueError("a global batch of 1 fails in the loss under explicit mesh axes; pass --batch-size 2 or more")
    return batch_size


def shared_run_id(args: argparse.Namespace, prefix: str) -> str:
    """--run-id, else ``<prefix>-<size>-<timestamp>``. Every process takes rank 0's id: each one's clock can give a
    different timestamp, which would split the run's logs and profiles across several ids."""
    return multihost_broadcast_sync(args.run_id or f"{prefix}-{args.size}-{time.strftime('%Y%m%d-%H%M%S')}")


def build_trainer(
    args: argparse.Namespace,
    *,
    batch_size: int,
    num_train_steps: int,
    run_id: str,
    default_mp: str,
    tracker: TrackerConfig,
    checkpointer: CheckpointerConfig,
) -> TrainerConfig:
    """The Levanter TrainerConfig of a run that always starts at step 0."""
    return TrainerConfig(
        id=run_id,
        log_dir=args.log_dir,
        use_explicit_mesh_axes=True,
        require_accelerator=False,
        mp=jmp.get_policy(args.mp or default_mp),
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        tracker=tracker,
        checkpointer=checkpointer,
        load_checkpoint=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        profiler=ProfilerConfig(
            enabled=args.profile_steps > 0,
            start_step=args.profile_start,
            num_steps=args.profile_steps,
            upload=XprofUploadConfig(enabled=False),
            # HLO protos add /host:metadata so trace tools can attribute kernels to HLO ops.
            profile_options=ProfileOptionsConfig(enable_hlo_proto=True),
        ),
        jax_compilation_cache_dir=args.compilation_cache_dir,
        log_jaxprs=False,
        log_xla_hlo=False,
        watch=WatchConfig(interval=args.watch_interval),
    )


def build_grug_trainer(args: argparse.Namespace, trainer: TrainerConfig, *, save_checkpoints: bool) -> GrugTrainerConfig:
    """The AMD Grug trainer config with the command line's mesh, microbatches and ZeRO-1."""
    return GrugTrainerConfig(
        trainer=trainer,
        expert_axis_size=args.expert_axis,
        replica_axis_size=args.replica_axis,
        microbatches=args.microbatches,
        zero1=args.zero1,
        save_checkpoints=save_checkpoints,
        step_timeout=args.step_timeout or None,
    )


def log_run_config(
    args: argparse.Namespace, model: GrugModelConfig, trainer: TrainerConfig, batch_size: int, flops_per_example: float
) -> None:
    logger.info(
        "size=%s layers=%d hidden=%d experts=%d topk=%d heads=%d kv=%d batch=%d microbatches=%d zero1=%s "
        "seq_len=%d window=%d mp=%s "
        "expert_axis=%d replica_axis=%d attention=%s moe=%s capacity=%s pooled_capacity=%s watch_interval=%d "
        "flops_per_example=%.4e",
        args.size,
        model.num_layers,
        model.hidden_dim,
        model.num_experts,
        model.num_experts_per_token,
        model.num_heads,
        model.num_kv_heads,
        batch_size,
        args.microbatches,
        args.zero1,
        model.max_seq_len,
        model.sliding_window,
        trainer.mp,
        args.expert_axis,
        args.replica_axis,
        model.attention_implementation,
        model.moe_implementation,
        model.capacity_factor,
        model.pooled_transport_capacity_factor,
        args.watch_interval,
        flops_per_example,
    )


class StepTimes(logging.Handler):
    """Collects each step's duration (``throughput/duration``) and wall period (``throughput/wall_step_s``) from the
    records the training loop logs through the JSON tracker (process 0 only), from step ``FIRST_TIMED_STEP`` on.

    The wall period is the time between consecutive steps' hooks, which includes the host work between steps that the
    duration leaves out. Attach it to ``logging.getLogger(JsonLoggerConfig.logger_name)`` around the run."""

    def __init__(self) -> None:
        super().__init__()
        self.durations: dict[int, float] = {}
        self.wall_steps: dict[int, float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        event = json.loads(record.getMessage())
        if event.get("event") != "log" or event["step"] < FIRST_TIMED_STEP:
            return
        metrics = event.get("metrics", {})
        # Keyed by step: the loop logs the last step a second time when it re-runs its hooks after training.
        for key, by_step in (("throughput/duration", self.durations), ("throughput/wall_step_s", self.wall_steps)):
            if key in metrics:
                by_step[event["step"]] = metrics[key]

    def log_summary(self, batch_size: int, seq_len: int, flops_per_example: float) -> None:
        """The headline: median step, tokens/s, model TFLOP/s and MFU, then the median wall step."""
        if not self.durations:
            return
        first, last = min(self.durations), max(self.durations)
        log_throughput_summary(
            list(self.durations.values()), batch_size, seq_len, flops_per_example, first_step=first, last_step=last
        )
        if self.wall_steps and jax.process_index() == 0:
            logger.info(
                "median wall step over steps %d-%d: %.3f s", first, last, statistics.median(self.wall_steps.values())
            )
