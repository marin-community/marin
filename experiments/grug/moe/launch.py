# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Template: grug-moe trial run.

This keeps model, train loop, and launch wiring in `experiments/grug/moe` so
the MoE variant can be iterated independently from the dense base template.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.pretraining_datasets import nemotron_mix_block_shuffle


@dataclass(frozen=True)
class GrugMoeLaunchConfig:
    """Last-mile run config for the MoE grug template.

    Keep this as the main entry point for day-to-day edits (model/data/optimizer/trainer/eval knobs).
    """

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    mesh: MeshConfig = field(default_factory=lambda: MeshConfig(axes={"expert": 1}))
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id and append `FERRY_DATE` when launching from ferry workflows."""
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    # Map template launch knobs onto full Levanter TrainerConfig.
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=config.mesh,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
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


RESOLVED_RUN_ID = _resolve_run_id("4_10_test_moe")


# Baseline: 1e18 compute budget, d1024. Model + optimizer + batch + steps are
# all derived from `MoeAdamHHeuristic`. To override any of these, swap in
# an explicit `GrugModelConfig` / `GrugMoeAdamHConfig` below.
_BASELINE_BUDGET: float = 1e18
_BASELINE_HIDDEN_DIM: int = 1024
_BASELINE_TARGET_STEPS: int = 2**14
_baseline_model, _baseline_optimizer, _baseline_batch, _baseline_steps = build_from_heuristic(
    budget=_BASELINE_BUDGET,
    hidden_dim=_BASELINE_HIDDEN_DIM,
    target_steps=_BASELINE_TARGET_STEPS,
)

# Public alias for the heuristic-derived baseline GrugModelConfig. Kept
# because consumers (e.g. experiments/ferries/canary_ferry.py) import it by
# name.
GRUG_MOE_TRIAL_MODEL: GrugModelConfig = _baseline_model


baseline_moe = ExecutorStep(
    name="grug/4_10_baseline_moe",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(_baseline_model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        # this_output_path() resolves to this step's output root (e.g. gs://.../grug/moe-trial-<version>).
        output_path=this_output_path(),
        # Keep run id out of versioning so changing job metadata doesn't create a new output path.
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(_baseline_steps),
        batch_size=versioned(_baseline_batch),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=["moe"],
            group="moe-iter04",
            name=None,
        ),
        optimizer=versioned(_baseline_optimizer),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


# ---------------------------------------------------------------------------
# ~120B-A12B bring-up on v4-1024 (issue #4301)
# Two shared-expert widths: 1x (sx3072) and 2x (sx6144).
# ---------------------------------------------------------------------------

GRUG_MOE_120B_SX3072 = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=4096,
    intermediate_dim=3072,
    shared_expert_intermediate_dim=3072,
    dense_intermediate_dim=12288,
    num_experts=64,
    num_experts_per_token=4,
    num_layers=48,
    num_heads=64,
    num_kv_heads=4,
    head_dim=128,
    max_seq_len=4096,
    initializer_std=0.5 / 4096**0.5,
    qk_mult=1.3,
)

GRUG_MOE_120B_SX6144 = dataclasses.replace(
    GRUG_MOE_120B_SX3072,
    shared_expert_intermediate_dim=6144,
)

_V4_1024_MESH = MeshConfig(
    axes={"expert": 8, "data": -1},
    dcn_axes={"data": -1},
)

_120B_STEPS = 200
_120B_BATCH_SIZE = 512
_120B_MP = "params=float32,compute=bfloat16,output=bfloat16"

_120B_OPTIMIZER = GrugMoeAdamHConfig(
    learning_rate=0.003,
    adam_lr=0.003,
    beta1=0.96,
    beta2=0.995,
    epsilon=1e-15,
    lr_schedule="linear",
    decay=0.2,
    min_lr_ratio=0.0,
    warmup=0.1,
    max_grad_norm=1,
)

_120B_TRAINER = GrugTrainerConfig(
    z_loss_weight=1e-4,
    ema_beta=None,
    log_every=1,
)

_120B_EVAL = GrugEvalConfig(
    eval_batch_size=512,
    steps_per_eval=100,
    max_eval_batches=4,
    eval_current=True,
    eval_ema=False,
)

_120B_PROFILER = ProfilerConfig(
    enabled=True,
    start_step=10,
    num_steps=5,
)

RESOLVED_120B_SX3072_RUN_ID = _resolve_run_id("03_31_120b_sx3072")
RESOLVED_120B_SX6144_RUN_ID = _resolve_run_id("03_31_120b_sx6144")

moe_120b_sx3072 = ExecutorStep(
    name="grug/03_31_120b_sx3072",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(GRUG_MOE_120B_SX3072),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_120B_SX3072_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v4-1024")),
        steps=versioned(_120B_STEPS),
        batch_size=versioned(_120B_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned(_120B_MP),
        mesh=versioned(_V4_1024_MESH),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["adamh", "qb", "sharded-qb", "gatednorm", "xsa", "zloss", "120b", "v4-1024"],
            group="moe-120b-bringup",
            name=None,
        ),
        optimizer=versioned(_120B_OPTIMIZER),
        profiler=versioned(_120B_PROFILER),
        grug_trainer=versioned(_120B_TRAINER),
        eval=versioned(_120B_EVAL),
    ),
)

moe_120b_sx6144 = ExecutorStep(
    name="grug/03_31_120b_sx6144",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(GRUG_MOE_120B_SX6144),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_120B_SX6144_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v4-1024")),
        steps=versioned(_120B_STEPS),
        batch_size=versioned(_120B_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned(_120B_MP),
        mesh=versioned(_V4_1024_MESH),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["adamh", "qb", "sharded-qb", "gatednorm", "xsa", "zloss", "120b", "v4-1024"],
            group="moe-120b-bringup",
            name=None,
        ),
        optimizer=versioned(_120B_OPTIMIZER),
        profiler=versioned(_120B_PROFILER),
        grug_trainer=versioned(_120B_TRAINER),
        eval=versioned(_120B_EVAL),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[baseline_moe, moe_120b_sx3072, moe_120b_sx6144],
        description="Grug MoE runs: baseline trial + 120B-A12B bring-up on v4-1024.",
    )
