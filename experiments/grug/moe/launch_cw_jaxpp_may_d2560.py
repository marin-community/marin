# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave H100 JaxPP launcher for the May d=2560 Grug MoE recipe."""

import dataclasses
import datetime
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
import numpy as np
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.heuristic import MoeHeuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, env_bool, env_int, run_grug_moe_trial
from experiments.grug.moe.model import GrugModelConfig, RematMode
from experiments.grug.moe.train import (
    GrugJaxPPConfig,
    GrugTrainerConfig,
    JaxPPExplicitMpmdScheduleMode,
    JaxPPImplementation,
    JaxPPSchedule,
    SonicFsdpMaterialization,
    jaxpp_setup_scripts,
)

GPUS_PER_NODE = 8
DEFAULT_HIDDEN_DIM = 2560
DEFAULT_SEQ_LEN = 4096
DEFAULT_NUM_LAYERS = 24
DEFAULT_NUM_EXPERTS = 256
DEFAULT_TOP_K = 4
DEFAULT_BATCH = 256
DEFAULT_STEPS = 20
DEFAULT_TOTAL_TOKENS = 1.0e13
DEFAULT_JAXPP_REVISION = "7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
DEFAULT_DEEPEP_REVISION = "7febc6e25660af0f54d95dd781ecdcd62265ecca"
DEEPEP_CUDA_TOOLCHAIN_VERSION = "13.2.78"
DEEPEP_CUDA_CCCL_VERSION = "13.3.3.4.1"
DEEPEP_CUDA_RUNTIME_VERSION = "13.2.75"


def env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    return float(raw) if raw else default


def env_optional_int(key: str) -> int | None:
    raw = os.environ.get(key, "")
    return int(raw) if raw else None


def env_optional_int_tuple(key: str) -> tuple[int, ...] | None:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    parts = tuple(part.strip() for part in raw.split(","))
    if any(not part for part in parts):
        raise ValueError(f"{key} must be a comma-separated list of integers, got {raw!r}")
    return tuple(int(part) for part in parts)


def deepep_setup_scripts(*, source_root: str, revision: str) -> tuple[str, ...]:
    """Check out the DeepEP sources required by Levanter's JAX FFI backend."""
    return (
        "\n".join(
            [
                'cd "$IRIS_WORKDIR"',
                "echo 'installing DeepEP source runtime'",
                "uv pip install --link-mode symlink "
                f"nvidia-cuda-nvcc=={DEEPEP_CUDA_TOOLCHAIN_VERSION} "
                f"nvidia-nvvm=={DEEPEP_CUDA_TOOLCHAIN_VERSION} "
                f"nvidia-cuda-cccl=={DEEPEP_CUDA_CCCL_VERSION} "
                f"nvidia-cuda-runtime=={DEEPEP_CUDA_RUNTIME_VERSION}",
                'cuda_bin="$(find "$IRIS_VENV"/lib/python*/site-packages/nvidia/cu*/bin ' '-name nvcc -print -quit)"',
                'test -n "$cuda_bin" || { echo "nvcc not found after CUDA toolchain install" >&2; exit 1; }',
                'cuda_root="$(dirname "$(dirname "$cuda_bin")")"',
                'ln -sf libcudart.so.13 "$cuda_root/lib/libcudart.so"',
                'ln -sf "$(dirname "$cuda_bin")"/* "$IRIS_VENV/bin/"',
                'rm -f "$IRIS_VENV/bin/nvcc"',
                "printf '%s\\n' '#!/usr/bin/env bash' "
                '"export LIBRARY_PATH=\\"$cuda_root/lib:\\${LIBRARY_PATH:-}\\"" '
                '"export LD_LIBRARY_PATH=\\"$cuda_root/lib:\\${LD_LIBRARY_PATH:-}\\"" '
                '"exec \\"$cuda_bin\\" \\"\\$@\\"" > "$IRIS_VENV/bin/nvcc"',
                'chmod +x "$IRIS_VENV/bin/nvcc"',
                f"test -d {source_root!r}/.git || git clone --filter=blob:none --no-checkout "
                f"https://github.com/deepseek-ai/DeepEP.git {source_root!r}",
                f"git -C {source_root!r} fetch --depth 1 origin {revision!r}",
                f"git -C {source_root!r} checkout --detach FETCH_HEAD",
                "uv run python -m levanter.kernels.deepep.preflight --component transport",
            ]
        )
        + "\n",
    )


@dataclass(frozen=True)
class SyntheticGrugDataset(AsyncDataset[GrugLmExample]):
    """Deterministic in-memory token stream for distributed systems probes."""

    seq_len: int
    vocab_size: int
    num_examples: int

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.vocab_size <= 1:
            raise ValueError(f"vocab_size must be greater than 1, got {self.vocab_size}")
        if self.num_examples <= 0:
            raise ValueError(f"num_examples must be positive, got {self.num_examples}")
        object.__setattr__(self, "_positions", np.arange(self.seq_len, dtype=np.int64))
        loss_weight = (np.arange(self.seq_len) < (self.seq_len - 1)).astype(np.float32)
        object.__setattr__(self, "_loss_weight", loss_weight)
        object.__setattr__(self, "_attn_mask", AttentionMask.causal())

    async def async_len(self) -> int:
        return self.num_examples

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[GrugLmExample]:
        if not indices:
            return []
        positions = cast(np.ndarray, self.__dict__["_positions"])
        offsets = np.asarray(indices, dtype=np.int64)[:, None] * 9973
        tokens = ((positions[None, :] + offsets) % self.vocab_size).astype(np.int32, copy=False)
        loss_weight = cast(np.ndarray, self.__dict__["_loss_weight"])
        attn_mask = cast(AttentionMask, self.__dict__["_attn_mask"])
        return [
            GrugLmExample(
                tokens=jnp.asarray(row, dtype=jnp.int32),
                loss_weight=jnp.asarray(loss_weight),
                attn_mask=attn_mask,
            )
            for row in tokens
        ]


def synthetic_grug_data(*, seq_len: int, vocab_size: int, num_examples: int) -> LmDataConfig:
    dataset = SyntheticGrugDataset(seq_len=seq_len, vocab_size=vocab_size, num_examples=num_examples)
    return LmDataConfig(
        tokenizer="passthrough",
        vocab_size=vocab_size,
        cache_dir=None,
        auto_build_caches=False,
        shuffle=False,
        block_cross_document_attention=True,
        components={"synthetic": DirectDatasetComponent(datasets={"train": dataset, "validation": dataset})},
        train_weights={"synthetic": 1.0},
    )


def build_model() -> GrugModelConfig:
    heuristic = MoeHeuristic()
    hidden_dim = env_int("MAY_HIDDEN_DIM", DEFAULT_HIDDEN_DIM)
    seq_len = env_int("MAY_SEQ_LEN", DEFAULT_SEQ_LEN)
    model = heuristic.build_model_config(hidden_dim, seq_len=seq_len)
    remat_mode = os.environ.get("MAY_REMAT", "save_moe")
    if remat_mode not in ("recompute_all", "save_moe"):
        raise ValueError(f"MAY_REMAT={remat_mode!r} must be 'recompute_all' or 'save_moe'")
    return dataclasses.replace(
        model,
        vocab_size=env_int("MAY_VOCAB_SIZE", model.vocab_size),
        num_layers=env_int("MAY_NUM_LAYERS", DEFAULT_NUM_LAYERS),
        num_experts=env_int("MAY_NUM_EXPERTS", DEFAULT_NUM_EXPERTS),
        num_experts_per_token=env_int("MAY_TOP_K", DEFAULT_TOP_K),
        router_z_loss_coef=0.0,
        attention_implementation=cast(str | None, os.environ.get("MAY_ATTENTION_IMPLEMENTATION") or None),
        moe_implementation=cast(str | None, os.environ.get("MAY_MOE_IMPLEMENTATION", "ring")),
        loss_implementation=cast(str | None, os.environ.get("MAY_LOSS_IMPLEMENTATION") or None),
        remat_mode=cast(RematMode, remat_mode),
    )


def build_data(model: GrugModelConfig) -> LmDataConfig:
    data = os.environ.get("MAY_DATA", "synthetic").lower()
    if data == "synthetic":
        return synthetic_grug_data(
            seq_len=model.max_seq_len,
            vocab_size=model.vocab_size,
            num_examples=env_int("MAY_SYNTHETIC_EXAMPLES", 1 << 20),
        )
    raise ValueError(f"MAY_DATA={data!r} must be 'synthetic'")


def build_tracker(run_id: str, output_path: str):
    tracker = os.environ.get("MAY_TRACKER", "wandb").lower()
    schedule = os.environ.get("PP_SCHEDULE", "std_1f1b")
    if tracker == "wandb":
        return WandbConfig(
            entity=os.environ.get("WANDB_ENTITY") or "marin-community",
            project=os.environ.get("WANDB_PROJECT", "marin_moe"),
            tags=["grug", "moe", "jaxpp", "cw", "h100", "may-d2560", schedule],
            group=os.environ.get("MAY_WANDB_GROUP", "grug-moe-jaxpp-may-d2560"),
            name=run_id,
            replicate_path=output_path,
        )
    if tracker == "json_logger":
        return JsonLoggerConfig(logger_name=os.environ.get("MAY_JSON_LOGGER", "grug_moe_jaxpp_may.metrics"))
    raise ValueError(f"MAY_TRACKER={tracker!r} must be 'wandb' or 'json_logger'")


def build_pipeline_config() -> GrugJaxPPConfig:
    schedule = cast(JaxPPSchedule, os.environ.get("PP_SCHEDULE", "std_1f1b"))
    implementation = cast(JaxPPImplementation, os.environ.get("PP_IMPLEMENTATION", "auto"))
    interleaved_schedule = schedule in ("interleaved_gpipe", "interleaved_1f1b", "dualpipe_v", "kimi_k2")
    explicit_stages = env_optional_int("PP_STAGES")
    explicit_mpmd_dim = env_optional_int("PP_MPMD_DIM")
    if explicit_mpmd_dim is not None:
        mpmd_dim = explicit_mpmd_dim
    elif explicit_stages is not None and interleaved_schedule:
        mpmd_dim = explicit_stages // 2
    elif explicit_stages is not None:
        mpmd_dim = explicit_stages
    else:
        mpmd_dim = 4
    stages_default = 2 * mpmd_dim if interleaved_schedule else mpmd_dim
    return GrugJaxPPConfig(
        stages=env_int("PP_STAGES", stages_default),
        microbatches=env_int("PP_MICROBATCHES", 8),
        schedule=schedule,
        implementation=implementation,
        mpmd_dim=mpmd_dim,
        stage_layer_counts=env_optional_int_tuple("PP_STAGE_LAYER_COUNTS"),
        explicit_mpmd_schedule_mode=cast(
            JaxPPExplicitMpmdScheduleMode,
            os.environ.get("PP_EXPLICIT_MPMD_SCHEDULE_MODE", "default"),
        ),
        sonic_fsdp_materialization=cast(
            SonicFsdpMaterialization,
            os.environ.get("PP_SONIC_FSDP_MATERIALIZATION", "per_task"),
        ),
    )


def build_jaxpp_may_checkpoint(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.UTC).strftime("jaxpp-may-d2560-%Y%m%d-%H%M%S")
    replicas = env_int("MAY_GPU_REPLICAS", 4)
    gpus_per_replica = env_int("MAY_GPUS_PER_REPLICA", GPUS_PER_NODE)
    expert_axis = env_int("MAY_EXPERT_AXIS", 8)
    replica_axis = env_int("MAY_REPLICA_AXIS", 1)
    batch_size = env_int("MAY_BATCH", DEFAULT_BATCH)
    steps = env_int("MAY_STEPS", DEFAULT_STEPS)
    model = build_model()
    pipeline = build_pipeline_config() if env_bool("MAY_PIPELINE", True) else None
    if pipeline is not None and pipeline.sonic_fsdp_materialization == "staged_per_step":
        if model.moe_implementation != "sonic":
            raise ValueError("staged_per_step Sonic FSDP materialization requires MAY_MOE_IMPLEMENTATION=sonic")
        if expert_axis != 1:
            raise ValueError(
                "staged_per_step Sonic FSDP materialization requires MAY_EXPERT_AXIS=1 because Sonic does not "
                "support expert parallelism"
            )
    if (
        pipeline is not None
        and pipeline.stage_layer_counts is not None
        and sum(pipeline.stage_layer_counts) != model.num_layers
    ):
        raise ValueError(
            f"PP_STAGE_LAYER_COUNTS must sum to MAY_NUM_LAYERS={model.num_layers}; "
            f"got {pipeline.stage_layer_counts} (sum={sum(pipeline.stage_layer_counts)})"
        )
    post_setup_scripts = ()
    if pipeline is not None:
        post_setup_scripts = jaxpp_setup_scripts(revision=os.environ.get("JAXPP_REVISION", DEFAULT_JAXPP_REVISION))
    if model.moe_implementation == "deepep":
        source_root = os.environ.get("DEEPEP_SRC_ROOT")
        if not source_root:
            raise ValueError("DEEPEP_SRC_ROOT must be set when MAY_MOE_IMPLEMENTATION=deepep")
        post_setup_scripts += deepep_setup_scripts(
            source_root=source_root,
            revision=os.environ.get("DEEPEP_REVISION", DEFAULT_DEEPEP_REVISION),
        )

    mpmd_dim = 1 if pipeline is None else pipeline.mpmd_dim or pipeline.stages
    global_devices = replicas * gpus_per_replica
    fixed_axes = mpmd_dim * replica_axis * expert_axis
    if global_devices % fixed_axes != 0:
        raise ValueError(
            f"global devices={global_devices} must be divisible by "
            f"PP_MPMD_DIM={mpmd_dim} * MAY_REPLICA_AXIS={replica_axis} * MAY_EXPERT_AXIS={expert_axis}"
        )
    data_axis = global_devices // fixed_axes
    batch_shards = replica_axis * data_axis * expert_axis
    if batch_size % batch_shards != 0:
        raise ValueError(f"MAY_BATCH={batch_size} must be divisible by batch shards={batch_shards}")
    if pipeline is not None:
        if batch_size % pipeline.microbatches != 0:
            raise ValueError(f"MAY_BATCH={batch_size} must be divisible by PP_MICROBATCHES={pipeline.microbatches}")
        microbatch_size = batch_size // pipeline.microbatches
        if microbatch_size % batch_shards != 0:
            raise ValueError(
                f"microbatch size={microbatch_size} must be divisible by batch shards={batch_shards}; "
                f"got MAY_BATCH={batch_size} and PP_MICROBATCHES={pipeline.microbatches}"
            )
    if model.num_experts % expert_axis != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by MAY_EXPERT_AXIS={expert_axis}")

    resources = ResourceConfig.with_gpu(
        "H100",
        count=gpus_per_replica,
        cpu=env_int("MAY_CPU_PER_REPLICA", 32),
        ram=os.environ.get("MAY_WORKER_RAM", "256g"),
        disk=os.environ.get("MAY_WORKER_DISK", "256g"),
        replicas=replicas,
    )
    optimizer = MoeHeuristic().build_optimizer_config(
        batch_size,
        env_float("MAY_TOTAL_TOKENS", DEFAULT_TOTAL_TOKENS),
        model.hidden_dim,
        seq_len=model.max_seq_len,
    )
    grug_trainer = GrugTrainerConfig(
        expert_axis_size=expert_axis,
        replica_axis_size=replica_axis,
        pipeline=pipeline,
        z_loss_weight=0.0,
        ema_beta=None,
        log_every=env_int("MAY_LOG_EVERY", 1),
    )
    if pipeline is None:
        name = f"grug-moe-may-d{model.hidden_dim}-L{model.num_layers}-e{model.num_experts}-n{replicas}-no-pipeline"
    else:
        name = (
            f"grug-moe-jaxpp-may-d{model.hidden_dim}-L{model.num_layers}-e{model.num_experts}-"
            f"n{replicas}-{pipeline.schedule}-s{pipeline.stages}-m{pipeline.microbatches}"
        )

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        return GrugMoeLaunchConfig(
            model=model,
            data=build_data(model),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp=os.environ.get("MAY_MP", "params=float32,compute=bfloat16,output=bfloat16"),
            tracker=build_tracker(run_id, ctx.output_path),
            optimizer=optimizer,
            grug_trainer=grug_trainer,
            eval=None,
            profiler=ProfilerConfig(
                enabled=env_int("MAY_PROFILER_STEPS", 0) > 0,
                start_step=env_int("MAY_PROFILER_START", 8),
                num_steps=env_int("MAY_PROFILER_STEPS", 0),
            ),
            processes_per_task=env_int("MAY_PROCESSES_PER_TASK", 1),
            post_setup_scripts=post_setup_scripts,
            checkpointer=CheckpointerConfig(
                base_path=f"/tmp/grug-jaxpp-may-d2560-ckpt/{run_id}",
                append_run_id_to_base_path=False,
                save_interval=None,
                keep=None,
            ),
            init_from=None,
        )

    return ArtifactStep(
        name=user_namespaced_name(f"experiments/grug-moe-jaxpp/{name}-{run_id}", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(),
        runtime_args={"train_resources": resources},
    )


jaxpp_may_step = build_jaxpp_may_checkpoint()


def build_direct_config() -> GrugMoeLaunchConfig:
    step = build_jaxpp_may_checkpoint()
    ctx = StepContext.for_run(
        output_path=f"/tmp/grug-moe-jaxpp/{os.environ.get('RUN_ID', 'dev')}",
        prefix="/tmp/grug-moe-jaxpp",
        runtime_args=step.runtime_args,
        deps=(),
    )
    return step.build_config(ctx)


def main() -> None:
    if os.environ.get("MAY_DIRECT", "true").lower() in ("1", "true", "yes", "on"):
        run_grug_moe_trial(build_direct_config())
    else:
        StepRunner().run([jaxpp_may_step.lower()])


if __name__ == "__main__":
    main()
