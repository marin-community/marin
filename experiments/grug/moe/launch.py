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
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.training.training import temporary_checkpoint_base_path

from experiments.defaults import default_validation_sets
from experiments.grug.moe.heuristic_v2 import MoeMuonHHeuristic
from experiments.grug.moe.model import GrugModelConfig
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
    # Mesh size along the "expert" axis (expert-parallelism). 1 = no EP.
    expert_parallel: int = 1


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
        mesh=MeshConfig(axes={"expert": config.expert_parallel}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            # Checkpoint interval (minutes), env-configurable. Default 10 (baseline) for clean throughput
            # on reserved TPUs; set GRUG_CKPT_MIN=3 on heavily-preempted preemptible TPUs so a run reaches
            # its first checkpoint before the next preemption (frequent saves otherwise drag mean tok/s).
            save_interval=timedelta(minutes=int(os.environ.get("GRUG_CKPT_MIN", "10"))),
            keep=None,
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


# May Recipe compute-optimal cells from the drop-1e18 isoflop fit
# (issue #6074). ``MoeMuonHHeuristic`` (heuristic_v2) supplies LR / beta2 /
# epsilon; (bs, steps) hardcoded so callers don't depend on
# ``compute_tokens_and_batch`` heuristics for cell selection.
#
#   dim   budget     bs    steps   tokens     muonh_lr   tpu
#   512   3.82e17    32    10_980  1.44e9     0.00980    v4-32 (EP=1)
#   768   2.81e18    64    16_875  4.42e9     0.00837    v4-32 (EP=1)
#   1024  1.16e19    128   16_080  8.43e9     0.00879    v4-32 (EP=1)
#   1280  3.46e19    256   14_325  1.50e10    0.00957    v4-32 (EP=1)
_SEQ_LEN: int = 4096
_COMPUTE_OPT_CELLS: tuple[tuple[int, int, int], ...] = (
    (512, 32, 10_980),
    # (768, 64, 16_875),
    # (1024, 128, 16_080),
    # (1280, 256, 14_325),
)

_heuristic = MoeMuonHHeuristic()

# Public alias for the d=512 baseline GrugModelConfig. Kept because
# consumers (e.g. experiments/ferries/canary_ferry.py) import it by name.
GRUG_MOE_TRIAL_MODEL: GrugModelConfig = _heuristic.build_model_config(512, seq_len=_SEQ_LEN)

compute_opt_steps: list[ExecutorStep] = []
for _dim, _bs, _steps in _COMPUTE_OPT_CELLS:
    _model = _heuristic.build_model_config(_dim, seq_len=_SEQ_LEN)
    _tokens = float(_steps * _bs * _SEQ_LEN)
    _optimizer = _heuristic.build_muonh_config(_bs, _tokens, _dim, seq_len=_SEQ_LEN)
    _run_id = f"moe_may_compute_opt_d{_dim}_demo"
    compute_opt_steps.append(
        ExecutorStep(
            name=f"grug/{_run_id}",
            fn=run_grug_moe_trial,
            config=GrugMoeLaunchConfig(
                model=versioned(_model),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=_run_id,
                resources=versioned(ResourceConfig.with_tpu("v4-32")),
                steps=versioned(_steps),
                batch_size=versioned(_bs),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                tracker=WandbConfig(
                    project="marin_moe",
                    tags=["moe", "moe_may_compute_opt", f"d{_dim}"],
                    group="moe-may-compute-opt",
                    name=None,
                ),
                optimizer=versioned(_optimizer),
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
    )


if __name__ == "__main__":
    executor_main(
        steps=compute_opt_steps,
        description="May Recipe compute-optimal cells at d ∈ {512, 768, 1024, 1280} on v4-32 (EP=1).",
    )
