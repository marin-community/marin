# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Large sparse-MoE scale run for the CoreWeave cw-us-east-02a H100 cluster.

Launches a ~90B-total / ~5B-active Grug MoE (hidden 3072, 48 layers, 128 experts,
top-4 -> ~17x sparsity) across all 32 nodes / 256 H100s. Parameters are fully
sharded over the cross-node ``data`` axis (FSDP) while the 128 routed experts are
sharded 8-way over the intra-node NVLink ``expert`` axis (expert parallelism).

This is the size class the cluster can train as a *single* model. The canary
(``experiments/ferries/canary_ferry.py``) replicates parameters per node, so it
caps at the ~9.5B that fits on one node's 8 GPUs; here ``replica_axis_size=1``
shards one model across every device instead.

Env knobs (all optional; defaults give the full 90B run on 256 H100):

    SCALE_GPU_REPLICAS  number of 8xH100 nodes (default 32 -> 256 GPUs)
    SCALE_GPUS_PER_TASK number of GPUs assigned to each Iris task (default 8).
                        When SCALE_PROCESSES_PER_TASK > 1, each Python process
                        sees SCALE_GPUS_PER_TASK / SCALE_PROCESSES_PER_TASK GPUs.
    SCALE_EXPERT_AXIS   expert-parallel axis size, intra-node (default 8)
    SCALE_REPLICA_AXIS  cross-node replication; 1 = pure FSDP (default 1)
    SCALE_PROCESSES_PER_TASK  GPU processes per node: 1 = one process per node
                          (default), 8 = one JAX process per GPU (multi-controller)
    SCALE_BATCH         global batch in sequences (default 256)
    SCALE_SEQ_LEN       sequence length (default 2048)
    SCALE_STEPS         training steps (default 50)
    SCALE_HIDDEN_DIM / SCALE_NUM_LAYERS / SCALE_NUM_EXPERTS / SCALE_TOP_K
                        model-shape overrides (e.g. a smaller FSDP smoke test)
    SCALE_MOE_IMPLEMENTATION
                        MoE backend override, e.g. pallas_mgpu (default ring)
    SCALE_MOE_CAPACITY_FACTOR
                        MoE dispatch capacity factor (default 1.0; use 1.25 for
                        pallas_mgpu target smokes)
    SCALE_REMAT         recompute_all (default) | save_moe -- save_moe keeps the
                        tagged MoE dispatch tensors for backward so the EP
                        collectives are not re-run during recompute
    SCALE_MP            jmp policy (default params=float32,compute=bfloat16,
                        output=bfloat16); params=bfloat16 halves FSDP gather bytes
    SCALE_TRACKER       wandb | json_logger (default json_logger)
    SCALE_WATCH_TARGETS comma-separated Levanter watch targets. Empty disables
                        watch stats (default empty for scale throughput smokes)
    SCALE_WATCH_INTERVAL
                        watch interval when watch targets are set (default 10)
    SCALE_WATCH_PER_PARAMETER_NORMS
                        true | false (default false for scale runs)
    SCALE_PROFILER_STEPS  >0 enables a jax_profile capture window of N steps
                          (use SCALE_TRACKER=wandb so the artifact uploads)
    SCALE_PROFILER_START  profiler start step (default 8, past compile/warmup)
    SCALE_CHECKPOINTS   s3 (default) | local. local writes checkpoints to
                        node-local disk with no periodic saves -- for throughput
                        experiments where the checkpoint is disposable and a
                        slow S3 commit must not wedge the end-of-run barrier
    SCALE_TASK_CPU / SCALE_TASK_RAM / SCALE_TASK_DISK
                        per-process resources when SCALE_GPUS_PER_TASK != 8
    RUN_ID              unique run identifier
"""

import datetime
import os
from typing import cast

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import BlockShuffleConfig
from levanter.grug.grug_moe import MoeImplementation
from levanter.optim import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.launch import GrugMoeLaunchConfig, env_int, run_grug_moe_trial, slimpajama_6b_dataset
from experiments.grug.moe.model import GrugModelConfig, RematMode
from experiments.grug.moe.train import GrugTrainerConfig
from experiments.llama import llama3_tokenizer_vocab_size

# head_dim is fixed at 128; hidden_dim must be a multiple of it.
HEAD_DIM = 128
VOCAB_SIZE = llama3_tokenizer_vocab_size
GPUS_PER_NODE = 8  # H100s per gd-8xh100ib node; the batch-shard math and with_gpu(count=...) must track
# Default seq for the 90B run. FSDP reshards the [batch, seq, hidden] activation
# through a fully-replicated intermediate (an XLA SPMD limitation, pending Shardy),
# so peak memory scales with batch*seq; at the default 89.7B model, batch 256 x
# seq 2048 fits in 80GB while 512 x 4096 OOMs (~58GiB replicated tile).
DEFAULT_SEQ_LEN = 2048
DEFAULT_BATCH = 256

# Modest, schedule-stable Adam for a short scale/throughput run (not trained to
# convergence). expert weights share the schedule; the goal is to exercise the
# FSDP+expert-parallel mesh at scale, not to produce a checkpoint.
SCALE_OPTIMIZER = AdamConfig(
    learning_rate=6e-4,
    weight_decay=0.1,
    lr_schedule="cosine",
    warmup=10,
    min_lr_ratio=0.1,
)

SCALE_TRAINER_DEFAULTS = dict(z_loss_weight=1e-4, ema_beta=None, log_every=1)

# Subdirectory of MARIN_PREFIX these scale runs write their per-run output dirs
# into, so they stay grouped instead of cluttering the prefix root.
OUTPUT_SUBDIR = "experiments/grug-moe-cw"

# SlimPajama block-shuffle: a small, R2-local corpus for the scale/throughput run.
_SLIMPAJAMA_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")


def build_scale_model() -> GrugModelConfig:
    """~90B-total / ~5B-active sparse MoE (overridable via SCALE_* env vars)."""
    hidden_dim = env_int("SCALE_HIDDEN_DIM", 3072)
    if hidden_dim % HEAD_DIM != 0:
        raise ValueError(f"SCALE_HIDDEN_DIM={hidden_dim} must be a multiple of head_dim={HEAD_DIM}")
    num_heads = hidden_dim // HEAD_DIM
    # ~4:1 grouped-query attention; back off to the nearest divisor of num_heads.
    num_kv_heads = max(1, num_heads // 4)
    while num_heads % num_kv_heads != 0:
        num_kv_heads -= 1
    intermediate_dim = hidden_dim // 2  # expert FFN inner width (~d/2)
    seq_len = env_int("SCALE_SEQ_LEN", DEFAULT_SEQ_LEN)
    remat_mode = os.environ.get("SCALE_REMAT", "recompute_all")
    if remat_mode not in ("recompute_all", "save_moe"):
        raise ValueError(f"SCALE_REMAT={remat_mode!r} must be 'recompute_all' or 'save_moe'")
    moe_implementation = os.environ.get("SCALE_MOE_IMPLEMENTATION") or None
    moe_capacity_factor = float(os.environ.get("SCALE_MOE_CAPACITY_FACTOR") or "1.0")
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim,
        num_layers=env_int("SCALE_NUM_LAYERS", 48),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        intermediate_dim=intermediate_dim,
        shared_expert_intermediate_dim=intermediate_dim,
        num_experts=env_int("SCALE_NUM_EXPERTS", 128),
        num_experts_per_token=env_int("SCALE_TOP_K", 4),
        max_seq_len=seq_len,
        sliding_window=seq_len,
        moe_implementation=cast(MoeImplementation | None, moe_implementation),
        moe_capacity_factor=moe_capacity_factor,
        remat_mode=cast(RematMode, remat_mode),
    )


def env_bool(key: str, default: bool) -> bool:
    """Read a boolean from ``os.environ[key]``, falling back to ``default`` when unset/empty."""
    raw = os.environ.get(key, "")
    if not raw:
        return default
    normalized = raw.lower()
    if normalized in ("1", "true", "t", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"{key}={raw!r} must be a boolean")


def build_scale_watch_config() -> WatchConfig:
    """Build watch config for scale runs.

    Watch stats are useful diagnostics, but per-parameter norms are expensive
    enough to OOM the one-node target-shape smoke around the default interval.
    Keep scale runs throughput-oriented unless the caller opts in.
    """
    watch_targets = tuple(target.strip() for target in os.environ.get("SCALE_WATCH_TARGETS", "").split(","))
    watch_targets = tuple(target for target in watch_targets if target)
    return WatchConfig(
        watch_targets=list(watch_targets),
        include_norms=env_bool("SCALE_WATCH_NORMS", True),
        include_per_parameter_norms=env_bool("SCALE_WATCH_PER_PARAMETER_NORMS", False),
        include_histograms=env_bool("SCALE_WATCH_HISTOGRAMS", False),
        split_scan_layers=env_bool("SCALE_WATCH_SPLIT_SCAN_LAYERS", True),
        interval=env_int("SCALE_WATCH_INTERVAL", 10),
    )


def build_scale_checkpoint(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """Assemble the CoreWeave scale run as a lazy :class:`LevanterCheckpoint` from SCALE_* env."""
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")

    logical_replicas = env_int("SCALE_GPU_REPLICAS", 32)
    gpus_per_task = env_int("SCALE_GPUS_PER_TASK", GPUS_PER_NODE)
    expert_axis = env_int("SCALE_EXPERT_AXIS", 8)
    replica_axis = env_int("SCALE_REPLICA_AXIS", 1)
    batch_size = env_int("SCALE_BATCH", DEFAULT_BATCH)
    steps = env_int("SCALE_STEPS", 50)
    # 1 = one process per node (8 local GPUs). 8 = one JAX process per GPU
    # (multi-controller) via the iris.runtime.multigpu supervisor.
    processes_per_task = env_int("SCALE_PROCESSES_PER_TASK", 1)
    # SCALE_PROFILER_STEPS > 0 captures a jax_profile window of that many steps
    # (uploaded via the tracker, so pair with SCALE_TRACKER=wandb to retrieve it).
    profiler_steps = env_int("SCALE_PROFILER_STEPS", 0)
    profiler = ProfilerConfig(
        enabled=profiler_steps > 0,
        start_step=env_int("SCALE_PROFILER_START", 8),
        num_steps=profiler_steps,
    )

    checkpoint_mode = os.environ.get("SCALE_CHECKPOINTS", "s3").lower()
    if checkpoint_mode == "local":
        checkpointer = CheckpointerConfig(
            base_path=f"/tmp/grug-scale-ckpt/{run_id}",
            append_run_id_to_base_path=False,
            save_interval=None,
            keep=None,
        )
    elif checkpoint_mode == "s3":
        checkpointer = None
    else:
        raise ValueError(f"SCALE_CHECKPOINTS={checkpoint_mode!r} must be 's3' or 'local'")

    model = build_scale_model()
    if model.num_experts % expert_axis != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by SCALE_EXPERT_AXIS={expert_axis}")
    if gpus_per_task < 1 or gpus_per_task > GPUS_PER_NODE or GPUS_PER_NODE % gpus_per_task != 0:
        raise ValueError(f"SCALE_GPUS_PER_TASK={gpus_per_task} must divide the {GPUS_PER_NODE} GPUs in each H100 node")
    if gpus_per_task % processes_per_task != 0:
        raise ValueError(
            f"SCALE_PROCESSES_PER_TASK={processes_per_task} must divide SCALE_GPUS_PER_TASK={gpus_per_task}"
        )
    devices_per_process = gpus_per_task // processes_per_task
    if model.moe_implementation == "pallas_mgpu" and devices_per_process < expert_axis:
        raise ValueError(
            "SCALE_MOE_IMPLEMENTATION=pallas_mgpu requires each Python process "
            f"to see at least SCALE_EXPERT_AXIS={expert_axis} GPUs; got "
            f"SCALE_GPUS_PER_TASK/SCALE_PROCESSES_PER_TASK={devices_per_process}"
        )

    # Batch is sharded over the (replica_dcn, data, expert) axes; data absorbs the
    # rest of the 8*logical_replicas devices. SCALE_GPUS_PER_TASK only changes how
    # Iris decomposes that logical allocation into Python processes; it must not
    # change the global mesh size.
    total_gpus = logical_replicas * GPUS_PER_NODE
    fixed_axes = replica_axis * expert_axis
    if total_gpus % fixed_axes != 0:
        raise ValueError(
            f"total_gpus={total_gpus} must be divisible by SCALE_REPLICA_AXIS*SCALE_EXPERT_AXIS={fixed_axes}"
        )
    data_axis = total_gpus // fixed_axes
    batch_shards = replica_axis * data_axis * expert_axis
    if batch_size % batch_shards != 0:
        raise ValueError(f"SCALE_BATCH={batch_size} must be divisible by batch shards={batch_shards}")

    task_replicas = total_gpus // gpus_per_task
    task_cpu = env_int("SCALE_TASK_CPU", max(4, 32 * gpus_per_task // GPUS_PER_NODE))
    task_ram = os.environ.get("SCALE_TASK_RAM", f"{max(32, 256 * gpus_per_task // GPUS_PER_NODE)}g")
    task_disk = os.environ.get("SCALE_TASK_DISK", f"{max(32, 256 * gpus_per_task // GPUS_PER_NODE)}g")
    resources = ResourceConfig.with_gpu(
        "H100",
        count=gpus_per_task,
        cpu=task_cpu,
        ram=task_ram,
        disk=task_disk,
        replicas=task_replicas,
    )

    use_wandb = os.environ.get("SCALE_TRACKER", "json_logger").lower() == "wandb"
    json_logger_name = os.environ.get("SCALE_JSON_LOGGER", "grug_moe_scale.metrics")
    wandb_entity = os.environ.get("WANDB_ENTITY") or None
    wandb_project = os.environ.get("WANDB_PROJECT", "marin_moe")

    grug_trainer = GrugTrainerConfig(
        expert_axis_size=expert_axis,
        replica_axis_size=replica_axis,
        **SCALE_TRAINER_DEFAULTS,
    )

    name = (
        f"grug-moe-cw-d{model.hidden_dim}-L{model.num_layers}-e{model.num_experts}"
        f"-r{logical_replicas}-t{task_replicas}x{gpus_per_task}"
    )
    mp = os.environ.get("SCALE_MP", "params=float32,compute=bfloat16,output=bfloat16")
    slim = slimpajama_6b_dataset()

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        if use_wandb:
            tracker = WandbConfig(
                entity=wandb_entity,
                project=wandb_project,
                tags=["grug", "moe", "cw", "h100", "scale"],
                group="grug-moe-cw-scale",
                name=None,
                replicate_path=ctx.output_path,
            )
        else:
            tracker = JsonLoggerConfig(logger_name=json_logger_name)
        return GrugMoeLaunchConfig(
            model=model,
            data=mixture(ctx, {slim: 1.0}, shuffle=_SLIMPAJAMA_SHUFFLE),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp=mp,
            tracker=tracker,
            optimizer=SCALE_OPTIMIZER,
            watch=build_scale_watch_config(),
            grug_trainer=grug_trainer,
            processes_per_task=processes_per_task,
            eval=None,
            profiler=profiler,
            checkpointer=checkpointer,
        )

    return ArtifactStep(
        name=f"{OUTPUT_SUBDIR}/{name}-{run_id}",
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": resources},
    )


if __name__ == "__main__":
    StepRunner().run([build_scale_checkpoint().lower()])
