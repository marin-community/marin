# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short Grug MoE perf run on a Qwen3-inspired 32B-A4B shape using v5p-64."""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.data import qwen3_moe_perf_mix_block_shuffle
from experiments.grug.moe.launch import _resolve_run_id, _resolve_tracker
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug


@dataclass(frozen=True)
class GrugMoeV5pLaunchConfig:
    """Last-mile config for the short v5p-64 perf bring-up."""

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = None


QWEN3_32B_A4B_V5P_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=2048,
    intermediate_dim=768,
    shared_expert_intermediate_dim=2048,
    num_experts=128,
    num_experts_per_token=8,
    num_layers=48,
    num_heads=32,
    num_kv_heads=4,
    head_dim=128,
    max_seq_len=4096,
    layer_norm_eps=1e-6,
    router_z_loss_coef=0.001,
)


def run_grug_moe_v5p(config: GrugMoeV5pLaunchConfig) -> None:
    """Map v5p launch knobs onto the standard grug train loop."""
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=8, num_steps=15, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1, "expert": 8}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=30),
            keep=[{"every": 1000}],
        ),
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


RESOLVED_RUN_ID = _resolve_run_id("grug-moe-qwen3-32b-a4b-v5p64-bs320-block-shuffle")


grug_moe_qwen3_32b_a4b_v5p64_bs320_block_shuffle = ExecutorStep(
    name="grug/moe-qwen3-32b-a4b-v5p64-bs320-block-shuffle",
    fn=run_grug_moe_v5p,
    config=GrugMoeV5pLaunchConfig(
        model=versioned(QWEN3_32B_A4B_V5P_MODEL),
        data=qwen3_moe_perf_mix_block_shuffle(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-64")),
        steps=versioned(30),
        batch_size=versioned(320),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "moe", "perf", "qwen3-shape", "v5p-64", "bs320", "block-shuffle", "no-profile"],
            group="grug-moe-qwen3-32b-a4b-v5p64-bs320-block-shuffle",
            name=None,
        ),
        optimizer=versioned(
            AdamConfig(
                learning_rate=1e-4,
                weight_decay=0.1,
                lr_schedule="constant",
                warmup=0,
            )
        ),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=None,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[grug_moe_qwen3_32b_a4b_v5p64_bs320_block_shuffle],
        description=(
            "Short Grug MoE perf run on a Qwen3-inspired ~32B-A4B shape using v5p-64 at batch size 320 "
            "with block-shuffle defaults."
        ),
    )
