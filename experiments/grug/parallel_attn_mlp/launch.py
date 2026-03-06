# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Variant: grug parallel-attn-mlp trial run.

This variant keeps attention and MLP branches parallel inside each block:
`x + attn(ln(x)) + mlp(ln(x))`.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.parallel_attn_mlp.model import GrugModelConfig
from experiments.grug.parallel_attn_mlp.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.tootsie.exp1295_32b import nemotron_mix


@dataclass(frozen=True)
class GrugBaseLaunchConfig:
    """Last-mile run config for the base grug template.

    Keep this as the main entry point for day-to-day edits (model/data/optimizer/trainer/eval knobs).
    """

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


GRUG_130M_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=1792,
    num_layers=6,
    num_heads=8,
    num_kv_heads=8,
    max_seq_len=4096,
    head_dim=None,
)

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id and append `FERRY_DATE` when launching from ferry workflows."""
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_wandb_group(default_group: str) -> str:
    return os.environ.get("GRUG_WANDB_GROUP", default_group)


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_base_trial(config: GrugBaseLaunchConfig) -> None:
    # Map template launch knobs onto full Levanter TrainerConfig.
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
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
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


RESOLVED_RUN_ID = _resolve_run_id("grug-parallel-attn-mlp-trial")
RESOLVED_WANDB_GROUP = _resolve_wandb_group("grug-parallel-attn-mlp-trial")
RESOLVED_DISABLE_EVAL = _env_bool("GRUG_DISABLE_EVAL", False)
RESOLVED_EVAL_CONFIG = (
    None
    if RESOLVED_DISABLE_EVAL
    else GrugEvalConfig(
        eval_batch_size=512,
        steps_per_eval=_env_int("GRUG_STEPS_PER_EVAL", 200),
        max_eval_batches=8,
        eval_current=True,
        eval_ema=False,
    )
)


grug_parallel_attn_mlp_trial = ExecutorStep(
    name="grug/parallel-attn-mlp-trial",
    fn=run_grug_base_trial,
    config=GrugBaseLaunchConfig(
        model=versioned(GRUG_130M_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        # this_output_path() resolves to this step's output root.
        output_path=this_output_path(),
        # Keep run id out of versioning so changing job metadata doesn't create a new output path.
        run_id=RESOLVED_RUN_ID,
        steps=versioned(_env_int("GRUG_STEPS", 2_000)),
        batch_size=versioned(_env_int("GRUG_BATCH_SIZE", 512)),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "parallel-attn-mlp"],
            group=RESOLVED_WANDB_GROUP,
            name=None,  # filled from run_id in _resolve_tracker
        ),
        optimizer=versioned(
            AdamConfig(
                learning_rate=_env_float("GRUG_LR", 3e-3),
                weight_decay=_env_float("GRUG_WEIGHT_DECAY", 0.1),
                lr_schedule="cosine",
                decay=_env_float("GRUG_DECAY", 0.2),
                min_lr_ratio=_env_float("GRUG_MIN_LR_RATIO", 0.1),
                warmup=_env_int("GRUG_WARMUP", 1000),
                max_grad_norm=_env_float("GRUG_MAX_GRAD_NORM", 1.0),
            )
        ),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=_env_float("GRUG_Z_LOSS_WEIGHT", 1e-4),
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(RESOLVED_EVAL_CONFIG),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[grug_parallel_attn_mlp_trial],
        description="Grug 130M parallel-attn-mlp trial run (~2000 steps) on Nemotron mix.",
    )
