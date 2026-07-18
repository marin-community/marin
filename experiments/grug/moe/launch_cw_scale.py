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
    SCALE_PROCESSES_PER_TASK  GPU processes per node: 1 = one process per node
                          (default), 8 = one JAX process per GPU (multi-controller)
    SCALE_BATCH         global batch in sequences (default 256)
    SCALE_SEQ_LEN       sequence length (default 2048)
    SCALE_STEPS         training steps (default 50)
    SCALE_HIDDEN_DIM / SCALE_NUM_LAYERS / SCALE_NUM_EXPERTS / SCALE_TOP_K
                        model-shape overrides (e.g. a smaller FSDP smoke test)
    SCALE_REMAT         recompute_all (default) | save_moe -- save_moe keeps the
                        tagged MoE dispatch tensors for backward so the EP
                        collectives are not re-run during recompute
    SCALE_SCAN_LAYERS   1 stacks all blocks into one lax.scan body (one compiled
                        layer subgraph instead of num_layers of them); default off
    SCALE_MP            jmp policy (default params=float32,compute=bfloat16,
                        output=bfloat16); params=bfloat16 halves FSDP gather bytes
    SCALE_TRACKER       wandb | json_logger (default json_logger)
    SCALE_PROFILER_STEPS  >0 enables a jax_profile capture window of N steps
                          (use SCALE_TRACKER=wandb so the artifact uploads)
    SCALE_PROFILER_START  profiler start step (default 8, past compile/warmup)
    SCALE_CHECKPOINTS   s3 (default) | local. local writes checkpoints to
                        node-local disk with no periodic saves -- for throughput
                        experiments where the checkpoint is disposable and a
                        slow S3 commit must not wedge the end-of-run barrier
    SCALE_DATA          slimpajama (default) | datakit. slimpajama is the fast MFU/
                        throughput dataset; datakit uses the two-phase datakit store
                        mixture (marin_prefix-rooted, phase 1 at 80% of steps).
    SCALE_OPTIMIZER     adam (default) | adamh | muonh. muonh runs Newton-Schulz on
                        2D/3D/4D weight matrices; all use linear LR decay to 5% of
                        peak with 1% warmup.
    RUN_ID              unique run identifier
"""

import dataclasses
import datetime
import os
from typing import cast

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.grug._moe.common import resolve_moe_implementation
from levanter.grug.attention import GrugAttentionImplementation
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.heuristic import MoeHeuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, env_int, run_grug_moe_trial, slimpajama_6b_dataset
from experiments.grug.moe.launch_datakit_moe_mix import datakit_data_config
from experiments.grug.moe.model import GrugModelConfig, RematMode
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig
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
    """Model config from the May-Recipe heuristic, with explicit architecture + backend overrides.

    The heuristic (``MoeHeuristic.build_model_config``) sizes hidden/layers/heads/intermediate and
    sets ``initializer_std = 0.5 / sqrt(hidden_dim)``. We override the routed-expert count and top-k
    (the heuristic leaves the ``GrugModelConfig`` default 256/4) plus the runtime/backend knobs the
    heuristic does not set (attention/MoE impl, scan, remat, head_dim, sliding window).
    """
    hidden_dim = env_int("SCALE_HIDDEN_DIM", 3072)
    if hidden_dim % HEAD_DIM != 0:
        raise ValueError(f"SCALE_HIDDEN_DIM={hidden_dim} must be a multiple of head_dim={HEAD_DIM}")
    num_heads = hidden_dim // HEAD_DIM
    # SCALE_MLA=1 switches the block to Multi-head Latent Attention with its own qk/v head dims
    # (128 nope + 64 rope, v=128); the dense head_dim is unused. SCALE_MLA_HEAD_MULT scales the head
    # count = mult * d/128 (default 2x, the DeepSeek-style wide-head count; 1x = dense head count).
    use_mla = os.environ.get("SCALE_MLA") == "1"
    if use_mla:
        num_heads = env_int("SCALE_MLA_HEAD_MULT", 2) * hidden_dim // HEAD_DIM
    # Q latent rank. SCALE_Q_LORA_RANK=0 uses a direct Q projection; >0 routes Q through a
    # compressed latent. Defaults to d/2 (the grug MLA prototype value) when unset.
    q_lora_rank = env_int("SCALE_Q_LORA_RANK", hidden_dim // 2)
    # MLA qk head dim = SCALE_QK_NOPE + SCALE_QK_ROPE (default 128 + 64 = 192). 64 + 64 = 128 makes
    # qk symmetric with v=128 (no asymmetric-192 -> avoids the widest coarse-tile forward).
    qk_nope = env_int("SCALE_QK_NOPE", 128)
    qk_rope = env_int("SCALE_QK_ROPE", 64)
    # SCALE_MLA_SCALE_Q_LORA / SCALE_MLA_SCALE_KV_LORA=1 rescale the post-RMSNorm Q/KV latents
    # by sqrt(hidden_dim / latent_dim) before the up-projection.
    mla_scale_q_lora = os.environ.get("SCALE_MLA_SCALE_Q_LORA") == "1"
    mla_scale_kv_lora = os.environ.get("SCALE_MLA_SCALE_KV_LORA") == "1"
    # SCALE_NUM_KV_HEADS overrides the global KV-head count (== num_heads for full MHA). Default ~4:1 GQA.
    kv_env = os.environ.get("SCALE_NUM_KV_HEADS")
    if kv_env is not None:
        num_kv_heads = int(kv_env)
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError(f"SCALE_NUM_KV_HEADS={num_kv_heads} must be a positive divisor of num_heads={num_heads}")
    else:
        num_kv_heads = max(1, num_heads // 4)
        while num_heads % num_kv_heads != 0:
            num_kv_heads -= 1
    seq_len = env_int("SCALE_SEQ_LEN", DEFAULT_SEQ_LEN)
    remat_mode = os.environ.get("SCALE_REMAT", "recompute_all")
    if remat_mode not in ("recompute_all", "save_moe"):
        raise ValueError(f"SCALE_REMAT={remat_mode!r} must be 'recompute_all' or 'save_moe'")
    # SCALE_MOE_IMPL selects the expert-GEMM backend (e.g. "sonic_cute" = QuACK SM100 on B200);
    # None keeps the config default. SCALE_ATTN_IMPL likewise overrides the attention backend.
    moe_impl_env = os.environ.get("SCALE_MOE_IMPL")
    moe_implementation = resolve_moe_implementation(moe_impl_env) if moe_impl_env else None
    attn_impl_env = os.environ.get("SCALE_ATTN_IMPL")
    attention_implementation = cast("GrugAttentionImplementation | None", attn_impl_env or None)
    base = MoeHeuristic().build_model_config(hidden_dim, seq_len=seq_len)
    return dataclasses.replace(
        base,
        vocab_size=VOCAB_SIZE,
        head_dim=HEAD_DIM,
        num_layers=env_int("SCALE_NUM_LAYERS", 48),
        num_experts=env_int("SCALE_NUM_EXPERTS", 64),
        use_mla=use_mla,
        q_lora_rank=q_lora_rank,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mla_scale_q_lora=mla_scale_q_lora,
        mla_scale_kv_lora=mla_scale_kv_lora,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        num_experts_per_token=env_int("SCALE_TOP_K", 4),
        # Routed-expert MLP width; default keeps the heuristic value (hidden/2 at hidden=5120).
        intermediate_dim=env_int("SCALE_INTERMEDIATE", base.intermediate_dim),
        shared_expert_intermediate_dim=env_int("SCALE_SHARED_INTERMEDIATE", hidden_dim),
        # Attention sliding-window span; default keeps the heuristic value (2048).
        sliding_window=env_int("SCALE_SLIDING_WINDOW", base.sliding_window),
        remat_mode=cast(RematMode, remat_mode),
        moe_implementation=moe_implementation,
        attention_implementation=attention_implementation,
        use_array_stacked_blocks=os.environ.get("SCALE_SCAN_LAYERS") == "1",
    )


def build_scale_checkpoint(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """Assemble the CoreWeave scale run as a lazy :class:`LevanterCheckpoint` from SCALE_* env."""
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")

    replicas = env_int("SCALE_GPU_REPLICAS", 32)
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

    # GPUs per node: 8 for the H100 nodes, 4 for a GB200 node (SCALE_GPUS_PER_NODE=4).
    gpus_per_node = env_int("SCALE_GPUS_PER_NODE", GPUS_PER_NODE)
    gpu_type = os.environ.get("SCALE_GPU_TYPE", "H100")
    # Batch is sharded over the (replica_dcn, data, expert) axes; data absorbs the
    # rest of the gpus_per_node*replicas devices. Require the global batch to cover every shard.
    data_axis = (replicas * gpus_per_node) // (replica_axis * expert_axis)
    batch_shards = replica_axis * data_axis * expert_axis
    if batch_size % batch_shards != 0:
        raise ValueError(f"SCALE_BATCH={batch_size} must be divisible by batch shards={batch_shards}")

    resources = ResourceConfig.with_gpu(
        gpu_type, count=gpus_per_node, cpu=32, ram="256g", disk="256g", replicas=replicas
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

    mp = os.environ.get("SCALE_MP", "params=float32,compute=bfloat16,output=bfloat16")

    # LR from the May-Recipe heuristic: it scales the peak with (tokens x hidden_dim) and sets
    # muonh_lr = 13/3 * adam_lr, linear decay to 5% of peak, 1% warmup. SCALE_OPTIMIZER picks the
    # family ("muonh" uses the heuristic config directly; "adamh"/"adam" use its adam_lr). SCALE_LR
    # overrides the peak. SCALE_MAX_LR caps the heuristic peak (heuristic default cap 0.05).
    total_tokens = float(steps * batch_size * model.max_seq_len)
    max_lr = float(os.environ.get("SCALE_MAX_LR", "0.05"))
    heuristic = MoeHeuristic(min_lr_ratio=0.05, max_learning_rate=max_lr).build_optimizer_config(
        batch_size=batch_size, tokens=total_tokens, hidden_dim=model.hidden_dim, seq_len=model.max_seq_len
    )
    schedule = dict(lr_schedule="linear", min_lr_ratio=0.05, warmup=0.01)
    lr_override = os.environ.get("SCALE_LR")
    opt_name = os.environ.get("SCALE_OPTIMIZER", "muonh").lower()
    optimizer: OptimizerConfig
    if opt_name in ("muonh", "grug_moe_muonh"):
        optimizer = (
            dataclasses.replace(heuristic, learning_rate=float(lr_override), adam_lr=float(lr_override))
            if lr_override
            else heuristic
        )
    elif opt_name in ("adamh", "grug_moe_adamh"):
        lr = float(lr_override) if lr_override else heuristic.adam_lr
        optimizer = GrugMoeAdamHConfig(learning_rate=lr, adam_lr=lr, **schedule)
    else:
        lr = float(lr_override) if lr_override else heuristic.adam_lr
        optimizer = dataclasses.replace(SCALE_OPTIMIZER, learning_rate=lr, **schedule)
    print(
        f"[scale] optimizer={opt_name} muonh_lr={heuristic.learning_rate:.5f} adam_lr={heuristic.adam_lr:.5f} "
        f"(heuristic: {total_tokens / 1e9:.1f}B tokens, dim={model.hidden_dim}); "
        f"peak override SCALE_LR={lr_override or 'none'}",
        flush=True,
    )

    name = f"grug-moe-cw-d{model.hidden_dim}-L{model.num_layers}-e{model.num_experts}-r{replicas}"
    # SCALE_DATA=datakit uses the two-phase datakit store mixture; default is SlimPajama.
    use_datakit = os.environ.get("SCALE_DATA", "slimpajama").lower() == "datakit"
    slim = None if use_datakit else slimpajama_6b_dataset()

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
        if use_datakit:
            # Two-phase datakit store mixture (phase 1 begins at 80% of steps). Bucket cache dirs
            # are relative and rooted at marin_prefix() -> the local CoreWeave bucket, so there is
            # no cross-region I/O and no hardcoded bucket names.
            data = datakit_data_config(
                total_steps=steps,
                batch_size=batch_size,
                max_seq_len=model.max_seq_len,
                enable_simulated_epoching=False,
                val_components={},
            )
        else:
            data = mixture(ctx, {slim: 1.0}, shuffle=_SLIMPAJAMA_SHUFFLE)
        return GrugMoeLaunchConfig(
            model=model,
            data=data,
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp=mp,
            tracker=tracker,
            optimizer=optimizer,
            grug_trainer=grug_trainer,
            processes_per_task=processes_per_task,
            eval=None,
            profiler=profiler,
            checkpointer=checkpointer,
        )

    return ArtifactStep(
        name=user_namespaced_name(f"{OUTPUT_SUBDIR}/{name}-{run_id}", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=() if use_datakit else (slim,),
        runtime_args={"train_resources": resources},
    )


if __name__ == "__main__":
    StepRunner().run([build_scale_checkpoint().lower()])
