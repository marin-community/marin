# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct Iris worker entrypoint for May d=2560 Grug MoE runs.

This bypasses the usual Executor/Fray parent job and runs the final Grug train
entrypoint directly inside a replicated GPU Iris job. It is intended for
CoreWeave diagnostics when the tiny CPU parent launcher is not scheduling.
"""

from __future__ import annotations

import dataclasses
import os
from typing import cast

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.training.training import temporary_checkpoint_base_path

from experiments.grug.moe.launch import (
    DisabledCheckpointerConfig,
    trainer_mesh_expert_axis_size,
    validate_local_expert_model_axes,
    validate_ring_expert_model_axes,
)
from experiments.grug.moe.launch_cw_may_d2560 import (
    DEFAULT_BATCH,
    DEFAULT_STEPS,
    GPUS_PER_NODE,
    OUTPUT_SUBDIR,
    build_checkpointer,
    build_data,
    build_eval,
    build_may_model,
    build_may_optimizer,
    build_tracker,
    env_bool,
    env_int,
)
from experiments.grug.moe.train import GrugRunConfig, GrugTrainerConfig, LiveParamMode, _run_grug_local


def _output_path(run_id: str) -> str:
    prefix = os.environ.get("MARIN_PREFIX", "s3://marin-na/tmp/ttl=7d").rstrip("/")
    return f"{prefix}/{OUTPUT_SUBDIR}/direct-{run_id}"


def _build_direct_config() -> GrugRunConfig:
    run_id = os.environ["RUN_ID"]
    replicas = env_int("MAY_GPU_REPLICAS", 2)
    expert_axis = env_int("MAY_EXPERT_AXIS", 16)
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
        ram=os.environ.get("MAY_WORKER_RAM", "256g"),
        disk=os.environ.get("MAY_WORKER_DISK", "256g"),
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

    checkpointer, checkpointing_enabled = build_checkpointer(run_id)
    output_path = _output_path(run_id)
    if checkpointer is None:
        checkpointer = CheckpointerConfig(
            base_path=os.path.join(output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(output_path),
            append_run_id_to_base_path=False,
            save_interval=None,
            keep=None,
        )
    load_checkpoint = None
    if not checkpointing_enabled:
        checkpointer = DisabledCheckpointerConfig(base_path="/tmp/grug-disabled-checkpoints")
        load_checkpoint = False

    trainer_mesh_expert_axis = trainer_mesh_expert_axis_size(
        expert_axis=expert_axis,
        model_axis=model_axis,
        local_device_count=GPUS_PER_NODE,
        allow_cross_node_expert_axis=allow_cross_node_expert_axis,
    )
    trainer = TrainerConfig(
        id=run_id,
        seed=0,
        train_batch_size=batch_size,
        num_train_steps=steps,
        mesh=MeshConfig(
            axes={
                "data": -1,
                "expert": trainer_mesh_expert_axis,
                "model": model_axis,
            },
            dcn_axes={"replica_dcn": -1},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
        ),
        profiler=dataclasses.replace(
            build_may_step_profiler(),
            enabled=profiler_steps > 0,
        ),
        watch=WatchConfig(interval=env_int("MAY_WATCH_INTERVAL", 0)),
        mp=jmp.get_policy(os.environ.get("MAY_MP", "params=float32,compute=bfloat16,output=bfloat16")),
        tracker=build_tracker(run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        load_checkpoint=load_checkpoint,
        checkpointer=checkpointer,
        log_jaxprs=env_bool("MAY_LOG_JAXPRS", False),
        log_xla_hlo=env_bool("MAY_LOG_XLA_HLO", False),
    )

    return GrugRunConfig(
        model=model,
        data=build_data(model),
        resources=resources,
        optimizer=build_may_optimizer(batch_size=batch_size, seq_len=model.max_seq_len),
        trainer=dataclasses.replace(grug_trainer, trainer=trainer),
        eval=build_eval(),
    )


def build_may_step_profiler():
    return ProfilerConfig(
        enabled=env_int("MAY_PROFILER_STEPS", 0) > 0,
        start_step=env_int("MAY_PROFILER_START", 8),
        num_steps=env_int("MAY_PROFILER_STEPS", 0),
        perfetto_link=env_bool("MAY_PROFILER_PERFETTO_LINK", False),
        profile_options=ProfileOptionsConfig(
            host_tracer_level=env_optional_int("MAY_PROFILER_HOST_TRACER_LEVEL"),
            python_tracer_level=env_optional_int("MAY_PROFILER_PYTHON_TRACER_LEVEL"),
            device_tracer_level=env_optional_int("MAY_PROFILER_DEVICE_TRACER_LEVEL"),
            enable_hlo_proto=env_bool("MAY_PROFILER_ENABLE_HLO_PROTO", False),
            include_dataset_ops=env_bool("MAY_PROFILER_INCLUDE_DATASET_OPS", False),
        ),
    )


def env_optional_int(key: str) -> int | None:
    raw = os.environ.get(key, "")
    return int(raw) if raw else None


def main() -> None:
    _run_grug_local(_build_direct_config())


if __name__ == "__main__":
    main()
