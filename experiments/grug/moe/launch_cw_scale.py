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
    SCALE_BATCH         global batch in sequences (default 256)
    SCALE_SEQ_LEN       sequence length (default 2048)
    SCALE_STEPS         training steps (default 50)
    SCALE_HIDDEN_DIM / SCALE_NUM_LAYERS / SCALE_NUM_EXPERTS / SCALE_TOP_K
                        model-shape overrides (e.g. a smaller FSDP smoke test)
    SCALE_TRACKER       wandb | json_logger (default json_logger)
    RUN_ID              unique run identifier
"""

import dataclasses
import datetime
import os

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.data.text import BlockShuffleConfig, TextLmDatasetFormat
from levanter.optim import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize.data_configs import lm_data_config

from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugTrainerConfig
from experiments.llama import llama3_tokenizer
from experiments.tokenization import default_tokenize

# head_dim is fixed at 128; hidden_dim must be a multiple of it.
HEAD_DIM = 128
VOCAB_SIZE = 128_256
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


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def build_scale_model() -> GrugModelConfig:
    """~90B-total / ~5B-active sparse MoE (overridable via SCALE_* env vars)."""
    hidden_dim = _env_int("SCALE_HIDDEN_DIM", 3072)
    if hidden_dim % HEAD_DIM != 0:
        raise ValueError(f"SCALE_HIDDEN_DIM={hidden_dim} must be a multiple of head_dim={HEAD_DIM}")
    num_heads = hidden_dim // HEAD_DIM
    # ~4:1 grouped-query attention; back off to the nearest divisor of num_heads.
    num_kv_heads = max(1, num_heads // 4)
    while num_heads % num_kv_heads != 0:
        num_kv_heads -= 1
    intermediate_dim = hidden_dim // 2  # expert FFN inner width (~d/2)
    seq_len = _env_int("SCALE_SEQ_LEN", DEFAULT_SEQ_LEN)
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim,
        num_layers=_env_int("SCALE_NUM_LAYERS", 48),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        intermediate_dim=intermediate_dim,
        shared_expert_intermediate_dim=intermediate_dim,
        num_experts=_env_int("SCALE_NUM_EXPERTS", 128),
        num_experts_per_token=_env_int("SCALE_TOP_K", 4),
        max_seq_len=seq_len,
        sliding_window=seq_len,
    )


def _slimpajama_data():
    # SlimPajama-6B with block-shuffle, re-tokenized on first run. Small and
    # R2-local; a real pretraining mixture (Nemotron) would require its tokenized
    # cache to already exist on marin-na to avoid a cross-region tokenize.
    tokenize_step = default_tokenize(
        name="slimpajama-6b-cw",
        dataset="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        format=TextLmDatasetFormat(),
    )
    tokenize_step = dataclasses.replace(
        tokenize_step,
        config=dataclasses.replace(
            tokenize_step.config,
            worker_resources=ResourceConfig(ram="64g", disk="64g"),
        ),
    )
    return lm_data_config(
        training_set=tokenize_step,
        shuffle=BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel"),
    )


def build_scale_step() -> ExecutorStep:
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

    replicas = _env_int("SCALE_GPU_REPLICAS", 32)
    expert_axis = _env_int("SCALE_EXPERT_AXIS", 8)
    replica_axis = _env_int("SCALE_REPLICA_AXIS", 1)
    batch_size = _env_int("SCALE_BATCH", DEFAULT_BATCH)
    steps = _env_int("SCALE_STEPS", 50)

    model = build_scale_model()
    if model.num_experts % expert_axis != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by SCALE_EXPERT_AXIS={expert_axis}")

    # Batch is sharded over the (replica_dcn, data, expert) axes; data absorbs the
    # rest of the 8*replicas devices. Require the global batch to cover every shard.
    data_axis = (replicas * 8) // (replica_axis * expert_axis)
    batch_shards = replica_axis * data_axis * expert_axis
    if batch_size % batch_shards != 0:
        raise ValueError(f"SCALE_BATCH={batch_size} must be divisible by batch shards={batch_shards}")

    resources = ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="256g", disk="256g", replicas=replicas)

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
        **SCALE_TRAINER_DEFAULTS,
    )

    name = f"grug-moe-cw-d{model.hidden_dim}-L{model.num_layers}-e{model.num_experts}-r{replicas}"
    return ExecutorStep(
        name=f"{name}-{run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=_slimpajama_data(),
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(resources),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=tracker,
            optimizer=versioned(SCALE_OPTIMIZER),
            grug_trainer=versioned(grug_trainer),
            eval=None,
            profiler=ProfilerConfig(enabled=False),
        ),
    )


scale_moe_step = build_scale_step()


def main():
    executor_main(steps=[scale_moe_step])


if __name__ == "__main__":
    main()
