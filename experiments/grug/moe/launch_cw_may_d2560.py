# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave H100 speed/profiling launcher for the May d=2560 Grug MoE recipe.

This matches the model shape from issue #6044 while using the current
CoreWeave/R2 launch path. Defaults are for a fast profiling run, not a full
10T-token training run:

    MAY_GPU_REPLICAS=32      32 gd-8xh100ib nodes, 256 H100s
    MAY_EXPERT_AXIS=8        expert parallelism inside each NVLink node
    MAY_REPLICA_AXIS=1       FSDP over the whole data axis
    MAY_MODEL_AXIS=1         tensor/model-parallel axis size
    MAY_BATCH=256            seq=4096 context; raise only after profiling memory
    MAY_SLIDING_WINDOW=2048  short-layer attention window; long layers remain full causal
    MAY_NUM_LAYERS=26        override layer count for narrow diagnostics
    MAY_STEPS=50             throughput/profiling length
    MAY_CPU_PER_REPLICA=32   CPU request for each 8xH100 worker pod
    MAY_CHECKPOINTS=none     disable checkpoint restore/saves for throughput probes
    MAY_MP=params=float32,compute=bfloat16,output=bfloat16
    MAY_CE_IMPLEMENTATION=   empty = default; xla forces streaming XLA CE
    MAY_WATCH_INTERVAL=0     grad/param watch interval; 0 disables
    MAY_LOG_EVERY=1          train progress/scalar logging cadence
    MAY_LOG_JAXPRS=false     disable JAXPR dumps for throughput probes
    MAY_LOG_XLA_HLO=false    disable HLO dumps for throughput probes
    MAY_SAVE_XLA_DUMPS=false upload XLA_FLAGS dump directory to W&B
    MAY_REMAT=save_moe       none | recompute_all | save_moe
    MAY_USE_PKO=true         enable PKO/doc-start mask path on long layers
    MAY_PKO_ON_LAST_LAYER=true
    MAY_BLOCK_CROSS_DOCUMENT_ATTENTION=true  synthetic data segment-id diagnostic
    MAY_INPUT_EMBED_SHARDING=hidden_batch  hidden_batch | replicated diagnostic
    MAY_OUTPUT_PROJ_SHARDING=lm_head  lm_head | replicated diagnostic
    MAY_OPTIMIZER=muonh     muonh | sgd diagnostic for optimizer overhead
    MAY_MUON_BACKEND_STEPS=5  Newton-Schulz steps for MuonH when MAY_OPTIMIZER=muonh
    MAY_MUON_ORTHOGONALIZATION_LAYOUT=stack_batch_sharded  stack_batch_sharded | vmap_replicated
    MAY_MUON_MAX_GROUPED_STACK_SIZE=256  Maximum grouped Muon stack size
    MAY_MUON_NS_COMPUTE_DTYPE=input  input | bf16 | fp32 | fp16 Newton-Schulz compute dtype
    MAY_EXPERT_3D_OPTIMIZER=muonh  muonh | adamh | grouped_muonh for routed expert weights
    MAY_ORDINARY_2D_OPTIMIZER=muonh  muonh | adamh | adam for ordinary non-expert 2D weights
    MAY_EXPERT_GROUPED_MUONH_GROUP_SIZE=  optional grouped_muonh stack group size

The default parameter policy keeps one sharded fp32 parameter tree plus sharded
optimizer state. Set ``MAY_LIVE_PARAM_MODE=compute_with_master`` to keep a
live bf16 parameter tree for forward/backward plus a sharded fp32 master tree
for optimizer state and updates.
"""

import dataclasses
import datetime
import os
from typing import cast

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.optim import OptimizerConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.cw_storage import set_default_cw_grug_moe_prefix
from experiments.grug.moe.heuristic import MoeAdamHHeuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    env_int,
    run_grug_moe_trial,
    slimpajama_6b_data,
    synthetic_grug_data,
    trainer_mesh_expert_axis_size,
    validate_local_expert_model_axes,
    validate_ring_expert_model_axes,
)
from experiments.grug.moe.model import (
    VALID_INPUT_EMBED_SHARDINGS,
    VALID_OUTPUT_PROJ_SHARDINGS,
    VALID_REMAT_MODES,
    CrossEntropyImplementation,
    GrugModelConfig,
    InputEmbedSharding,
    OutputProjSharding,
    RematMode,
)
from experiments.grug.moe.optimizer import (
    VALID_EXPERT_3D_OPTIMIZERS,
    VALID_MAY_OPTIMIZERS,
    VALID_ORDINARY_2D_OPTIMIZERS,
    Expert3DOptimizer,
    GrugMoeMuonHConfig,
    GrugMoeSgdConfig,
    MayOptimizer,
    Ordinary2DOptimizer,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig, LiveParamMode

GPUS_PER_NODE = 8
DEFAULT_HIDDEN_DIM = 2560
DEFAULT_SEQ_LEN = 4096
DEFAULT_SLIDING_WINDOW = 2048
DEFAULT_BATCH = 256
DEFAULT_STEPS = 50
DEFAULT_TOTAL_TOKENS = 1.0e13
DEFAULT_WARMUP_FRACTION = 0.01

# Subdirectory of MARIN_PREFIX these May d=2560 runs write their per-run output
# dirs into, so they stay grouped instead of cluttering the prefix root.
OUTPUT_SUBDIR = "experiments/grug-moe-cw"

set_default_cw_grug_moe_prefix()

MAY_HEURISTIC = MoeAdamHHeuristic(
    lr_coeff=0.06602,
    lr_tokens_exp=-0.395,
    lr_dim_exp=-0.150,
)


def env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    return float(raw) if raw else default


def env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    normalized = raw.lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{key}={raw!r} must be a boolean")


def env_optional_int(key: str) -> int | None:
    raw = os.environ.get(key, "")
    return int(raw) if raw else None


def build_may_model() -> GrugModelConfig:
    hidden_dim = env_int("MAY_HIDDEN_DIM", DEFAULT_HIDDEN_DIM)
    seq_len = env_int("MAY_SEQ_LEN", DEFAULT_SEQ_LEN)
    sliding_window = env_int("MAY_SLIDING_WINDOW", DEFAULT_SLIDING_WINDOW)
    remat_mode = os.environ.get("MAY_REMAT", "save_moe")
    if remat_mode not in VALID_REMAT_MODES:
        raise ValueError(f"MAY_REMAT={remat_mode!r} must be one of {VALID_REMAT_MODES}")
    output_proj_sharding = os.environ.get("MAY_OUTPUT_PROJ_SHARDING", "lm_head")
    if output_proj_sharding not in VALID_OUTPUT_PROJ_SHARDINGS:
        valid = ", ".join(VALID_OUTPUT_PROJ_SHARDINGS)
        raise ValueError(f"MAY_OUTPUT_PROJ_SHARDING={output_proj_sharding!r} must be one of {valid}")
    input_embed_sharding = os.environ.get("MAY_INPUT_EMBED_SHARDING", "hidden_batch")
    if input_embed_sharding not in VALID_INPUT_EMBED_SHARDINGS:
        valid = ", ".join(VALID_INPUT_EMBED_SHARDINGS)
        raise ValueError(f"MAY_INPUT_EMBED_SHARDING={input_embed_sharding!r} must be one of {valid}")
    attention_implementation = os.environ.get("MAY_ATTENTION_IMPLEMENTATION", "gpu_fa4_cute")
    cross_entropy_implementation = os.environ.get("MAY_CE_IMPLEMENTATION") or None

    model = MAY_HEURISTIC.build_model_config(hidden_dim, seq_len=seq_len)
    return dataclasses.replace(
        model,
        num_layers=env_int("MAY_NUM_LAYERS", model.num_layers),
        sliding_window=sliding_window,
        num_experts=env_int("MAY_NUM_EXPERTS", 256),
        num_experts_per_token=env_int("MAY_TOP_K", 4),
        router_z_loss_coef=0.0,
        routing_renorm_sum=env_float("MAY_ROUTING_RENORM_SUM", 2.5),
        use_half_rope=True,
        use_pko=env_bool("MAY_USE_PKO", True),
        pko_on_last_layer=env_bool("MAY_PKO_ON_LAST_LAYER", True),
        moe_implementation=cast(str, os.environ.get("MAY_MOE_IMPLEMENTATION", "ring")),
        attention_implementation=cast(str, attention_implementation or None),
        cross_entropy_implementation=cast(CrossEntropyImplementation | None, cross_entropy_implementation),
        input_embed_sharding=cast(InputEmbedSharding, input_embed_sharding),
        output_proj_sharding=cast(OutputProjSharding, output_proj_sharding),
        remat_mode=cast(RematMode, remat_mode),
    )


def build_may_optimizer(*, batch_size: int, seq_len: int) -> OptimizerConfig:
    total_tokens = env_float("MAY_TOTAL_TOKENS", DEFAULT_TOTAL_TOKENS)
    hidden_dim = env_int("MAY_HIDDEN_DIM", DEFAULT_HIDDEN_DIM)
    optimizer = os.environ.get("MAY_OPTIMIZER", "muonh")
    if optimizer not in VALID_MAY_OPTIMIZERS:
        valid = ", ".join(VALID_MAY_OPTIMIZERS)
        raise ValueError(f"MAY_OPTIMIZER={optimizer!r} must be one of {valid}")
    expert_3d_optimizer = os.environ.get("MAY_EXPERT_3D_OPTIMIZER", "muonh")
    if expert_3d_optimizer not in VALID_EXPERT_3D_OPTIMIZERS:
        valid = ", ".join(VALID_EXPERT_3D_OPTIMIZERS)
        raise ValueError(f"MAY_EXPERT_3D_OPTIMIZER={expert_3d_optimizer!r} must be one of {valid}")
    ordinary_2d_optimizer = os.environ.get("MAY_ORDINARY_2D_OPTIMIZER", "muonh")
    if ordinary_2d_optimizer not in VALID_ORDINARY_2D_OPTIMIZERS:
        valid = ", ".join(VALID_ORDINARY_2D_OPTIMIZERS)
        raise ValueError(f"MAY_ORDINARY_2D_OPTIMIZER={ordinary_2d_optimizer!r} must be one of {valid}")
    base_optimizer = MAY_HEURISTIC.build_optimizer_config(batch_size, total_tokens, hidden_dim, seq_len=seq_len)
    if cast(MayOptimizer, optimizer) == "sgd":
        return GrugMoeSgdConfig(
            learning_rate=base_optimizer.learning_rate,
            min_lr_ratio=base_optimizer.min_lr_ratio,
            warmup=env_float("MAY_WARMUP_FRACTION", DEFAULT_WARMUP_FRACTION),
            lr_schedule=base_optimizer.lr_schedule,
            decay=base_optimizer.decay,
        )
    return GrugMoeMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=env_float("MAY_WARMUP_FRACTION", DEFAULT_WARMUP_FRACTION),
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        backend_steps=env_int("MAY_MUON_BACKEND_STEPS", GrugMoeMuonHConfig.backend_steps),
        orthogonalization_layout=os.environ.get(
            "MAY_MUON_ORTHOGONALIZATION_LAYOUT", GrugMoeMuonHConfig.orthogonalization_layout
        ),
        max_grouped_stack_size=env_int("MAY_MUON_MAX_GROUPED_STACK_SIZE", GrugMoeMuonHConfig.max_grouped_stack_size),
        ns_compute_dtype=os.environ.get("MAY_MUON_NS_COMPUTE_DTYPE", GrugMoeMuonHConfig.ns_compute_dtype),
        expert_grouped_muonh_group_size=env_optional_int("MAY_EXPERT_GROUPED_MUONH_GROUP_SIZE"),
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
        expert_3d_optimizer=cast(Expert3DOptimizer, expert_3d_optimizer),
        ordinary_2d_optimizer=cast(Ordinary2DOptimizer, ordinary_2d_optimizer),
    )


def build_data(model: GrugModelConfig):
    data = os.environ.get("MAY_DATA", "slimpajama").lower()
    if data == "slimpajama":
        return slimpajama_6b_data()
    if data == "nemotron":
        return NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
    if data == "synthetic":
        return synthetic_grug_data(
            seq_len=model.max_seq_len,
            vocab_size=model.vocab_size,
            num_examples=env_int("MAY_SYNTHETIC_EXAMPLES", 1 << 20),
            eos_id=env_int("MAY_SYNTHETIC_EOS_ID", model.vocab_size - 1),
            eos_interval=env_int("MAY_SYNTHETIC_EOS_INTERVAL", 0),
            block_cross_document_attention=env_bool("MAY_BLOCK_CROSS_DOCUMENT_ATTENTION", True),
        )
    raise ValueError(f"MAY_DATA={data!r} must be 'slimpajama', 'nemotron', or 'synthetic'")


def build_tracker(run_id: str):
    if os.environ.get("MAY_TRACKER", "json_logger").lower() == "wandb":
        return WandbConfig(
            entity=os.environ.get("WANDB_ENTITY") or "marin-community",
            project=os.environ.get("WANDB_PROJECT", "marin_moe"),
            tags=["grug", "moe", "may", "cw", "h100", f"d{env_int('MAY_HIDDEN_DIM', DEFAULT_HIDDEN_DIM)}"],
            group=os.environ.get("MAY_WANDB_GROUP", "grug-moe-cw-may-d2560"),
            name=run_id,
            replicate_path=this_output_path(),
            save_xla_dumps=env_bool("MAY_SAVE_XLA_DUMPS", False),
        )
    return JsonLoggerConfig(logger_name=os.environ.get("MAY_JSON_LOGGER", "grug_moe_cw_may.metrics"))


def build_checkpointer(run_id: str) -> tuple[CheckpointerConfig | None, bool]:
    checkpoint_mode = os.environ.get("MAY_CHECKPOINTS", "none").lower()
    if checkpoint_mode == "local":
        return (
            CheckpointerConfig(
                base_path=f"/tmp/grug-may-d2560-ckpt/{run_id}",
                append_run_id_to_base_path=False,
                save_interval=None,
                keep=None,
            ),
            True,
        )
    if checkpoint_mode == "s3":
        return None, True
    if checkpoint_mode in ("none", "off", "disabled"):
        return None, False
    raise ValueError(f"MAY_CHECKPOINTS={checkpoint_mode!r} must be 'none', 'local', or 's3'")


def build_eval() -> GrugEvalConfig | None:
    if os.environ.get("MAY_EVAL", "").lower() not in ("1", "true", "yes"):
        return None
    return GrugEvalConfig(
        eval_batch_size=env_int("MAY_EVAL_BATCH", 512),
        steps_per_eval=env_int("MAY_EVAL_INTERVAL", 1000),
        max_eval_batches=env_int("MAY_MAX_EVAL_BATCHES", 8),
        eval_current=True,
        eval_ema=False,
    )


def build_may_step() -> ExecutorStep:
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.timezone.utc).strftime(
        "cw-may-d2560-%Y%m%d-%H%M%S"
    )

    replicas = env_int("MAY_GPU_REPLICAS", 32)
    expert_axis = env_int("MAY_EXPERT_AXIS", 8)
    replica_axis = env_int("MAY_REPLICA_AXIS", 1)
    model_axis = env_int("MAY_MODEL_AXIS", 1)
    batch_size = env_int("MAY_BATCH", DEFAULT_BATCH)
    steps = env_int("MAY_STEPS", DEFAULT_STEPS)
    worker_cpu = env_int("MAY_CPU_PER_REPLICA", 32)
    profiler_steps = env_int("MAY_PROFILER_STEPS", 0)

    model = build_may_model()
    if model.num_experts % expert_axis != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by MAY_EXPERT_AXIS={expert_axis}")
    if model.num_heads % model_axis != 0:
        raise ValueError(f"num_heads={model.num_heads} must be divisible by MAY_MODEL_AXIS={model_axis}")
    allow_cross_node_expert_axis = env_bool("MAY_ALLOW_CROSS_NODE_EXPERT_AXIS", False)
    validate_local_expert_model_axes(
        expert_axis=expert_axis,
        model_axis=model_axis,
        local_device_count=GPUS_PER_NODE,
        env_prefix="MAY",
        allow_cross_node_expert_axis=allow_cross_node_expert_axis,
    )
    validate_ring_expert_model_axes(
        expert_axis=expert_axis,
        model_axis=model_axis,
        moe_implementation=model.moe_implementation,
        env_prefix="MAY",
    )

    global_devices = replicas * GPUS_PER_NODE
    fixed_axes = replica_axis * expert_axis * model_axis
    if global_devices % fixed_axes != 0:
        raise ValueError(
            f"global devices={global_devices} must be divisible by "
            f"MAY_REPLICA_AXIS={replica_axis} * MAY_EXPERT_AXIS={expert_axis} * "
            f"MAY_MODEL_AXIS={model_axis}"
        )

    data_axis = global_devices // fixed_axes
    batch_shards = replica_axis * data_axis * expert_axis
    if batch_size % batch_shards != 0:
        raise ValueError(f"MAY_BATCH={batch_size} must be divisible by batch shards={batch_shards}")

    resources = ResourceConfig.with_gpu(
        "H100",
        count=GPUS_PER_NODE,
        cpu=worker_cpu,
        ram="256g",
        disk="256g",
        replicas=replicas,
        image=os.environ.get("MAY_TASK_IMAGE") or None,
    )
    grug_trainer = GrugTrainerConfig(
        expert_axis_size=expert_axis,
        replica_axis_size=replica_axis,
        model_axis_size=model_axis,
        live_param_mode=cast(LiveParamMode, os.environ.get("MAY_LIVE_PARAM_MODE", "param")),
        z_loss_weight=0.0,
        ema_beta=None,
        log_every=env_int("MAY_LOG_EVERY", 1),
    )
    profiler = ProfilerConfig(
        enabled=profiler_steps > 0,
        start_step=env_int("MAY_PROFILER_START", 8),
        num_steps=profiler_steps,
        perfetto_link=env_bool("MAY_PROFILER_PERFETTO_LINK", False),
        profile_options=ProfileOptionsConfig(
            host_tracer_level=env_optional_int("MAY_PROFILER_HOST_TRACER_LEVEL"),
            python_tracer_level=env_optional_int("MAY_PROFILER_PYTHON_TRACER_LEVEL"),
            device_tracer_level=env_optional_int("MAY_PROFILER_DEVICE_TRACER_LEVEL"),
            enable_hlo_proto=env_bool("MAY_PROFILER_ENABLE_HLO_PROTO", False),
            include_dataset_ops=env_bool("MAY_PROFILER_INCLUDE_DATASET_OPS", False),
        ),
    )
    eval_cfg = build_eval()
    checkpointer, checkpointing_enabled = build_checkpointer(run_id)

    name = f"grug-moe-cw-may-d{model.hidden_dim}-L{model.num_layers}-e{model.num_experts}-r{replicas}-cpu{worker_cpu}"
    return ExecutorStep(
        name=f"{OUTPUT_SUBDIR}/{name}-{run_id}",
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
            mp=versioned(os.environ.get("MAY_MP", "params=float32,compute=bfloat16,output=bfloat16")),
            tracker=build_tracker(run_id),
            optimizer=versioned(build_may_optimizer(batch_size=batch_size, seq_len=model.max_seq_len)),
            grug_trainer=versioned(grug_trainer),
            trainer_mesh_expert_axis_size=versioned(
                trainer_mesh_expert_axis_size(
                    expert_axis=expert_axis,
                    model_axis=model_axis,
                    local_device_count=GPUS_PER_NODE,
                    allow_cross_node_expert_axis=allow_cross_node_expert_axis,
                )
            ),
            eval=versioned(eval_cfg) if eval_cfg is not None else None,
            profiler=profiler,
            watch=WatchConfig(interval=env_int("MAY_WATCH_INTERVAL", 0)),
            checkpointing_enabled=checkpointing_enabled,
            checkpointer=checkpointer,
            log_jaxprs=env_bool("MAY_LOG_JAXPRS", False),
            log_xla_hlo=env_bool("MAY_LOG_XLA_HLO", False),
        ),
    )


may_d2560_step = build_may_step()


def main() -> None:
    executor_main(steps=[may_d2560_step])


if __name__ == "__main__":
    main()
