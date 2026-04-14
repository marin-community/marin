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
from fray.v2.types import PriorityBand
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
from experiments.grug.moe.heuristic import (
    MoeAdamHHeuristic,
    build_from_heuristic,
    compute_flops_per_token,
)
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
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    expert_parallel: int = 1
    priority_band: PriorityBand | None = None


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
        mesh=MeshConfig(axes={"expert": config.expert_parallel}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=30),
            keep=[{"every": 10_000}],
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
        priority_band=config.priority_band,
    )
    run_grug(run_config)


RESOLVED_RUN_ID = _resolve_run_id("moe_1e23_d5120_bs2048_ep8_ragged")


# 1e23 compute budget, d5120. Model +
# optimizer + batch + steps are all derived from `MoeAdamHHeuristic`. To
# override any of these, swap in an explicit `GrugModelConfig` /
# `GrugMoeAdamHConfig` below.
_BASELINE_BUDGET: float = 1e23
_BASELINE_HIDDEN_DIM: int = 5120
_BASELINE_TARGET_STEPS: int = 120_000
_baseline_model, _baseline_optimizer, _baseline_batch, _baseline_steps = build_from_heuristic(
    budget=_BASELINE_BUDGET,
    hidden_dim=_BASELINE_HIDDEN_DIM,
    target_steps=_BASELINE_TARGET_STEPS,
)
# Stack MoE blocks via jax.lax.scan to keep XLA compile + peak HBM tractable at
# the heuristic-derived depth, and force ragged dispatch so the smoke exercises
# the high-EP path from #4697.
_baseline_model = dataclasses.replace(
    _baseline_model,
    moe_implementation="ragged_all_to_all",
    use_array_stacked_blocks=True,
)

# Override the heuristic-derived batch_size (round_up_pow2 only produces powers
# of two; we want something in between). Recompute the optimizer at the new
# batch + matching tokens so the LR formula stays consistent.
_BASELINE_BATCH_OVERRIDE: int | None = 2048
if _BASELINE_BATCH_OVERRIDE is not None:
    _heuristic = MoeAdamHHeuristic()
    _fpt = compute_flops_per_token(_baseline_model)
    _tokens = _BASELINE_BUDGET / (3 * _fpt)
    _baseline_batch = _BASELINE_BATCH_OVERRIDE
    _baseline_steps = max(1, round(_tokens / (_baseline_batch * 4096)))
    _baseline_optimizer = _heuristic.build_optimizer_config(_baseline_batch, _tokens, _BASELINE_HIDDEN_DIM)


baseline_moe = ExecutorStep(
    name="grug/moe_1e23_d5120_bs2048_ep8_ragged",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(_baseline_model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        # this_output_path() resolves to this step's output root (e.g. gs://.../grug/moe-trial-<version>).
        output_path=this_output_path(),
        # Keep run id out of versioning so changing job metadata doesn't create a new output path.
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v4-2048", regions=["us-central2"])),
        steps=versioned(_baseline_steps),
        batch_size=versioned(_baseline_batch),
        expert_parallel=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["adamh", "qb", "sharded-qb", "gatednorm", "xsa", "zloss", "eq3e3"],
            group="moe-iter04",
            name=None,
        ),
        optimizer=versioned(_baseline_optimizer),
        priority_band="production",
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=1024,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[baseline_moe],
        description="Baseline grug MoE (QB+GN+XSA+zloss) on Nemotron mix.",
    )
