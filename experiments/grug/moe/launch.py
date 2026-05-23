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
from marin.training.training import temporary_checkpoint_base_path

from experiments.defaults import default_validation_sets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.pretraining_datasets import nemotron_mix


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
    # Step interval for permanent checkpoints written to ``base_path``. When
    # ``None``, no step-interval permanent checkpoints are written — only
    # temporary checkpoints (every ``save_interval``) plus the final
    # permanent checkpoint at the end of training.
    checkpoint_keep_every: int | None = 1000


NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
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
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[] if config.checkpoint_keep_every is None else [{"every": config.checkpoint_keep_every}],
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


RESOLVED_RUN_ID = _resolve_run_id("5_22_test_moe")


# Baseline: d512 at its compute-optimal budget (2.19e17). Model + optimizer +
# batch + steps are all derived from `MoeAdamHHeuristic`. To override any of
# these, swap in an explicit `GrugModelConfig` / `GrugMoeMuonHConfig` below.
_BASELINE_BUDGET: float = 2.19e17
_BASELINE_HIDDEN_DIM: int = 512
_BASELINE_TARGET_STEPS: int = 2**14
_baseline_model, _heuristic_optimizer, _baseline_batch, _baseline_steps = build_from_heuristic(
    budget=_BASELINE_BUDGET,
    hidden_dim=_BASELINE_HIDDEN_DIM,
    target_steps=_BASELINE_TARGET_STEPS,
)
# May Recipe optimizer: MuonH on matrices + AdamH on lm_head + plain Adam on
# router/embed/1-D leaves. ``max_grad_norm=None`` (no clipping); ``warmup=0.01``
# (1pct-noclip schedule).
_baseline_optimizer = GrugMoeMuonHConfig(
    learning_rate=_heuristic_optimizer.learning_rate,
    adam_lr=_heuristic_optimizer.adam_lr,
    min_lr_ratio=_heuristic_optimizer.min_lr_ratio,
    warmup=0.01,
    beta1=_heuristic_optimizer.beta1,
    beta2=_heuristic_optimizer.beta2,
    epsilon=_heuristic_optimizer.epsilon,
    max_grad_norm=None,
    lr_schedule=_heuristic_optimizer.lr_schedule,
    decay=_heuristic_optimizer.decay,
)

# Public alias for the heuristic-derived baseline GrugModelConfig. Kept
# because consumers (e.g. experiments/ferries/canary_ferry.py) import it by
# name.
GRUG_MOE_TRIAL_MODEL: GrugModelConfig = _baseline_model


baseline_moe = ExecutorStep(
    name="grug/5_22_baseline_moe",
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
                z_loss_weight=0.0,
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


if __name__ == "__main__":
    executor_main(
        steps=[baseline_moe],
        description="Baseline grug MoE on Nemotron mix.",
    )
