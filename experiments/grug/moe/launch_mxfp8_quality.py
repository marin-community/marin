# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-token BF16/MXFP8 quality comparison for issue #7271.

Each arm trains the same d2560/L26/E128/top-4 model for 31,474 steps at
batch 512 and sequence length 4096. Set ``MXFP8_QUALITY_ARM`` to ``bf16`` or
``mxfp8`` and provide an experiment-specific ``MXFP8_QUALITY_PAIR_ID``.
``MXFP8_QUALITY_STEPS`` can shorten the run for an exact-shape smoke. The
durable run identity includes the pair, arm, and exact step count.
"""

import dataclasses
import os
from datetime import timedelta

from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint, resolve_checkpointer_output_path

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, env_int, run_grug_moe_trial
from experiments.grug.moe.launch_datakit_moe_mix import _val_component, datakit_data_config
from experiments.grug.moe.model import GrugFp8Config, GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer

_BUDGET = 1e21
_HIDDEN_DIM = 2560
_TARGET_STEPS = 32_768
_SEQ_LEN = 4096
_NUM_EXPERTS = 128
_BATCH_SIZE = 512
_GPU_REPLICAS = 8
_GPUS_PER_REPLICA = 4
_EXPERT_AXIS_SIZE = 8
_REPLICA_AXIS_SIZE = 2
_WANDB_GROUP = "mxfp8-quality-7271"
_OUTPUT_SUBDIR = "experiments/grug-moe-mxfp8-quality"
_TRAIN_ENV = {
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
    "SCALE_MUON_INTRA_RACK": "1",
    "SCALE_MUON_DIST_NONEXPERT": "1",
    "SCALE_MUON_PAD_NONEXPERT": "1",
    "NCCL_SOCKET_IFNAME": "^ibs,ibp,lo,docker,veth,cilium,lxc",
    "CE_IMPL": "liger",
}
_CLOSED_TRAIN_ENV_PREFIXES = ("XLA_", "NCCL_", "JAX_", "CE_", "SCALE_MUON_")
_CLOSED_TRAIN_ENV_KEYS = ("LIBTPU_INIT_ARGS",)
_IGNORED_TRAIN_ENV_KEYS = ("JAX_PLATFORMS",)

_VALIDATION = [
    *paloma_datasets(tokenizer=marin_tokenizer).values(),
    *uncheatable_datasets(tokenizer=marin_tokenizer).values(),
]


def quality_cell() -> tuple[GrugModelConfig, GrugMoeMuonHConfig, int, int]:
    """Return the fixed model, full MuonH optimizer, batch size, and step count."""
    model, optimizer, batch_size, steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
        seq_len=_SEQ_LEN,
    )
    model = dataclasses.replace(
        model,
        num_experts=_NUM_EXPERTS,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="ring",
        use_array_stacked_blocks=True,
    )
    return model, optimizer, batch_size, steps


def quality_model(arm: str) -> GrugModelConfig:
    """Return the matched BF16 or hybrid MXFP8 quality model."""
    model, _, _, _ = quality_cell()
    if arm == "bf16":
        return model
    if arm == "mxfp8":
        return dataclasses.replace(
            model,
            fp8=GrugFp8Config(
                dense=True,
                grouped=True,
                recipe="mxfp8",
                mxfp8_producer="xla",
            ),
        )
    raise ValueError(f"unknown quality arm {arm!r}; expected 'bf16' or 'mxfp8'")


def _apply_train_env() -> None:
    conflicting = {}
    for key, value in os.environ.items():
        dispatcher_forwards_key = key.startswith(_CLOSED_TRAIN_ENV_PREFIXES) or key in _CLOSED_TRAIN_ENV_KEYS
        unsupported_forwarded_key = (
            dispatcher_forwards_key and key not in _TRAIN_ENV and key not in _IGNORED_TRAIN_ENV_KEYS
        )
        conflicting_pinned_value = key in _TRAIN_ENV and value != _TRAIN_ENV[key]
        if unsupported_forwarded_key or conflicting_pinned_value:
            conflicting[key] = value
    if conflicting:
        details = ", ".join(f"{key}={value!r}" for key, value in sorted(conflicting.items()))
        raise ValueError(f"unsupported quality runtime environment: {details}")
    os.environ.update(_TRAIN_ENV)


def build_quality_checkpoint(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """Build one durable BF16/MXFP8 quality arm from ``MXFP8_QUALITY_*`` env."""
    arm = os.environ.get("MXFP8_QUALITY_ARM", "")
    model = quality_model(arm)
    _, optimizer, batch_size, full_steps = quality_cell()
    assert batch_size == _BATCH_SIZE

    steps = env_int("MXFP8_QUALITY_STEPS", full_steps)
    pair_id = os.environ.get("MXFP8_QUALITY_PAIR_ID", "")
    if not pair_id:
        raise ValueError("MXFP8_QUALITY_PAIR_ID must identify this quality comparison")
    run_id = f"{pair_id}-{arm}-s{steps}"

    # Grug forwards XLA_*, SCALE_MUON_*, CE_*, and NCCL_* from the dispatcher to the GPU tasks.
    _apply_train_env()

    resources = ResourceConfig.with_gpu(
        "GB200",
        count=_GPUS_PER_REPLICA,
        cpu=128,
        ram="800g",
        disk="256g",
        replicas=_GPU_REPLICAS,
        preemptible=False,
    )
    name = f"{_OUTPUT_SUBDIR}/{run_id}"

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        if ctx.is_fingerprint:
            val_components = {dataset.name: _val_component(ctx.artifact_path(dataset)) for dataset in _VALIDATION}
        else:
            val_components = {dataset.name: ctx.resolved(dataset).as_component() for dataset in _VALIDATION}
        data = datakit_data_config(
            total_steps=steps,
            batch_size=batch_size,
            max_seq_len=model.max_seq_len,
            # Short shape smokes cannot downsample every rare finite component to at least one sequence.
            enable_simulated_epoching=steps == full_steps,
            val_components=val_components,
        )
        return GrugMoeLaunchConfig(
            model=model,
            data=data,
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin_moe",
                tags=["mxfp8-quality", "7271", arm, "grug", "moe", "gb200"],
                group=_WANDB_GROUP,
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            optimizer=optimizer,
            grug_trainer=GrugTrainerConfig(
                expert_axis_size=_EXPERT_AXIS_SIZE,
                replica_axis_size=_REPLICA_AXIS_SIZE,
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            ),
            eval=GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            ),
            checkpointer=resolve_checkpointer_output_path(
                CheckpointerConfig(save_interval=timedelta(hours=1), keep=None),
                ctx.output_path,
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=tuple(_VALIDATION),
        runtime_args={"train_resources": resources},
    )


if __name__ == "__main__":
    StepRunner().run([build_quality_checkpoint().lower()])
