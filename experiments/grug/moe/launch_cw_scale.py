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
    SCALE_EXPERT_AXIS   expert-parallel axis size, intra-node (default 8)
    SCALE_REPLICA_AXIS  cross-node replication; 1 = pure FSDP (default 1)
    SCALE_MODEL_AXIS    tensor/model-parallel axis size (default 1)
    SCALE_BATCH         global batch in sequences (default 256)
    SCALE_SEQ_LEN       sequence length (default 2048)
    SCALE_STEPS         training steps (default 50)
    SCALE_HIDDEN_DIM / SCALE_NUM_LAYERS / SCALE_NUM_EXPERTS / SCALE_TOP_K
                        model-shape overrides (e.g. a smaller FSDP smoke test)
    SCALE_REMAT         none | recompute_all (default) | save_moe -- save_moe
                        keeps the tagged MoE dispatch tensors for backward so
                        the EP collectives are not re-run during recompute
    SCALE_CPU_PER_REPLICA
                        CPU request for each 8xH100 worker pod (default 32)
    SCALE_MP            jmp policy (default params=float32,compute=bfloat16,
                        output=bfloat16); params=bfloat16 halves FSDP gather bytes
    SCALE_CE_IMPLEMENTATION
                        empty = default; xla skips pallas CE autotune for quick
                        sharding/debug probes
    SCALE_MOE_IMPLEMENTATION
                        empty = default ring EP backend; ragged_all_to_all tests
                        the ragged all-to-all EP backend
    SCALE_TRACKER       wandb | json_logger (default json_logger)
    SCALE_DATA          slimpajama | synthetic (default slimpajama)
    SCALE_PROFILER_STEPS  >0 enables a jax_profile capture window of N steps
                          (use SCALE_TRACKER=wandb so the artifact uploads)
    SCALE_PROFILER_START  profiler start step (default 8, past compile/warmup)
    SCALE_WATCH_INTERVAL  grad/param watch interval; 0 disables (default 0)
    SCALE_LOG_EVERY     train progress/scalar logging cadence (default 1)
    SCALE_LOG_JAXPRS    true | false, JAXPR dump toggle (default false)
    SCALE_LOG_XLA_HLO   true | false, HLO dump toggle (default false)
    SCALE_CHECKPOINTS   none (default) | local | s3. none disables restore and
                        all saves for disposable throughput probes; local writes
                        only the final forced checkpoint to node-local disk; s3
                        uses the default output-path checkpointer.
    RUN_ID              unique run identifier
"""

import datetime
import os
from typing import cast

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.optim import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.launch import (
    GrugMoeLaunchConfig,
    env_bool,
    env_int,
    run_grug_moe_trial,
    slimpajama_6b_data,
    synthetic_grug_data,
    validate_local_expert_model_axes,
    validate_ring_expert_model_axes,
)
from experiments.grug.moe.model import VALID_REMAT_MODES, CrossEntropyImplementation, GrugModelConfig, RematMode
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

SCALE_TRAINER_DEFAULTS = dict(z_loss_weight=1e-4, ema_beta=None)


def build_data(model: GrugModelConfig):
    data = os.environ.get("SCALE_DATA", "slimpajama").lower()
    if data == "slimpajama":
        return slimpajama_6b_data()
    if data == "synthetic":
        return synthetic_grug_data(
            seq_len=model.max_seq_len,
            vocab_size=model.vocab_size,
            num_examples=env_int("SCALE_SYNTHETIC_EXAMPLES", 1 << 20),
            eos_id=env_int("SCALE_SYNTHETIC_EOS_ID", model.vocab_size - 1),
            eos_interval=env_int("SCALE_SYNTHETIC_EOS_INTERVAL", 0),
        )
    raise ValueError(f"SCALE_DATA={data!r} must be 'slimpajama' or 'synthetic'")


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
    if remat_mode not in VALID_REMAT_MODES:
        raise ValueError(f"SCALE_REMAT={remat_mode!r} must be one of {VALID_REMAT_MODES}")
    cross_entropy_implementation = os.environ.get("SCALE_CE_IMPLEMENTATION") or None
    moe_implementation = os.environ.get("SCALE_MOE_IMPLEMENTATION") or None
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
        cross_entropy_implementation=cast(CrossEntropyImplementation | None, cross_entropy_implementation),
        moe_implementation=cast(str | None, moe_implementation),
        remat_mode=cast(RematMode, remat_mode),
    )


def build_scale_step() -> ExecutorStep:
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

    replicas = env_int("SCALE_GPU_REPLICAS", 32)
    expert_axis = env_int("SCALE_EXPERT_AXIS", 8)
    replica_axis = env_int("SCALE_REPLICA_AXIS", 1)
    model_axis = env_int("SCALE_MODEL_AXIS", 1)
    batch_size = env_int("SCALE_BATCH", DEFAULT_BATCH)
    steps = env_int("SCALE_STEPS", 50)
    worker_cpu = env_int("SCALE_CPU_PER_REPLICA", 32)
    # SCALE_PROFILER_STEPS > 0 captures a jax_profile window of that many steps
    # (uploaded via the tracker, so pair with SCALE_TRACKER=wandb to retrieve it).
    profiler_steps = env_int("SCALE_PROFILER_STEPS", 0)
    profiler = ProfilerConfig(
        enabled=profiler_steps > 0,
        start_step=env_int("SCALE_PROFILER_START", 8),
        num_steps=profiler_steps,
    )

    checkpoint_mode = os.environ.get("SCALE_CHECKPOINTS", "none").lower()
    checkpointing_enabled = True
    if checkpoint_mode == "local":
        checkpointer = CheckpointerConfig(
            base_path=f"/tmp/grug-scale-ckpt/{run_id}",
            append_run_id_to_base_path=False,
            save_interval=None,
            keep=None,
        )
    elif checkpoint_mode == "s3":
        checkpointer = None
    elif checkpoint_mode in ("none", "off", "disabled"):
        checkpointer = None
        checkpointing_enabled = False
    else:
        raise ValueError(f"SCALE_CHECKPOINTS={checkpoint_mode!r} must be 'none', 'local', or 's3'")

    model = build_scale_model()
    if model.num_experts % expert_axis != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by SCALE_EXPERT_AXIS={expert_axis}")
    if model.num_heads % model_axis != 0:
        raise ValueError(f"num_heads={model.num_heads} must be divisible by SCALE_MODEL_AXIS={model_axis}")
    validate_local_expert_model_axes(
        expert_axis=expert_axis,
        model_axis=model_axis,
        local_device_count=GPUS_PER_NODE,
        env_prefix="SCALE",
    )
    validate_ring_expert_model_axes(
        expert_axis=expert_axis,
        model_axis=model_axis,
        moe_implementation=model.moe_implementation,
        env_prefix="SCALE",
    )

    global_devices = replicas * GPUS_PER_NODE
    fixed_axes = replica_axis * expert_axis * model_axis
    if global_devices % fixed_axes != 0:
        raise ValueError(
            f"global devices={global_devices} must be divisible by "
            f"SCALE_REPLICA_AXIS={replica_axis} * SCALE_EXPERT_AXIS={expert_axis} * "
            f"SCALE_MODEL_AXIS={model_axis}"
        )

    # Batch is sharded over the (replica_dcn, data, expert) axes. The model axis
    # does tensor/model parallel work, so it reduces the residual data axis but
    # does not itself increase the number of batch shards.
    data_axis = global_devices // fixed_axes
    batch_shards = replica_axis * data_axis * expert_axis
    if batch_size % batch_shards != 0:
        raise ValueError(f"SCALE_BATCH={batch_size} must be divisible by batch shards={batch_shards}")

    resources = ResourceConfig.with_gpu(
        "H100",
        count=GPUS_PER_NODE,
        cpu=worker_cpu,
        ram="256g",
        disk="256g",
        replicas=replicas,
    )

    if os.environ.get("SCALE_TRACKER", "json_logger").lower() == "wandb":
        tracker = WandbConfig(
            entity=os.environ.get("WANDB_ENTITY") or None,
            project=os.environ.get("WANDB_PROJECT", "marin_moe"),
            tags=["grug", "moe", "cw", "h100", "scale"],
            group="grug-moe-cw-scale",
            name=None,
            replicate_path=this_output_path(),
        )
    else:
        tracker = JsonLoggerConfig(logger_name=os.environ.get("SCALE_JSON_LOGGER", "grug_moe_scale.metrics"))

    grug_trainer = GrugTrainerConfig(
        expert_axis_size=expert_axis,
        replica_axis_size=replica_axis,
        model_axis_size=model_axis,
        log_every=env_int("SCALE_LOG_EVERY", 1),
        **SCALE_TRAINER_DEFAULTS,
    )

    name = f"grug-moe-cw-d{model.hidden_dim}-L{model.num_layers}-e{model.num_experts}-r{replicas}-cpu{worker_cpu}"
    return ExecutorStep(
        name=f"{name}-{run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=build_data(model),
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(resources),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned(os.environ.get("SCALE_MP", "params=float32,compute=bfloat16,output=bfloat16")),
            tracker=tracker,
            optimizer=versioned(SCALE_OPTIMIZER),
            grug_trainer=versioned(grug_trainer),
            eval=None,
            profiler=profiler,
            watch=WatchConfig(interval=env_int("SCALE_WATCH_INTERVAL", 0)),
            checkpointing_enabled=checkpointing_enabled,
            checkpointer=checkpointer,
            log_jaxprs=env_bool("SCALE_LOG_JAXPRS", False),
            log_xla_hlo=env_bool("SCALE_LOG_XLA_HLO", False),
        ),
    )


scale_moe_step = build_scale_step()


def main():
    executor_main(steps=[scale_moe_step])


if __name__ == "__main__":
    main()
