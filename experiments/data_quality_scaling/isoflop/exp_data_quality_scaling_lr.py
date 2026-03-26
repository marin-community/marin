# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data quality scaling laws (isoflop LR sweep): train models on varying mixes
of low-quality and high-quality Nemotron CC data, sweeping learning rate around
the isoflop recipe's formula LR.

Grid: len(HIDDEN_SIZES) model sizes × len(STEP_COUNTS) step counts
      × len(LR_MULTIPLIERS) LR multipliers.
"""

import argparse
import sys
from dataclasses import dataclass, field

from experiments.defaults import default_train
from experiments.scaling_law_sweeps.c_adamc import c_adamc_heuristic
from experiments.pretraining_datasets.nemotron import tokenize_nemotron
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from levanter.models.lm_model import LmConfig
from levanter.optim.cautious import CautiousConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of
from marin.processing.tokenize import lm_mixture_data_config

RECIPE = c_adamc_heuristic
SEQ_LEN = 1024
BATCH_SIZE = 128

# --- Data components ---
nemotron_steps = tokenize_nemotron()
low_q = nemotron_steps["nemotron_cc/low_actual"]
high_q = nemotron_steps["nemotron_cc/hq_actual"]

# --- Sweep axes ---
HIDDEN_SIZES = [128, 256, 512, 768]
STEP_COUNTS = [1172, 2344, 4688]  # 150M, 300M, 600M tokens
HQ_TOTAL_EFFECTIVE_STEPS = [500]
LR_MULTIPLIERS = [0.2, 0.5, 1.0, 2.0, 5.0]


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    model_config: LmConfig
    hidden_dim: int
    num_steps: int
    hq_fraction: float
    lr: float
    batch_size: int = BATCH_SIZE
    seq_len: int = SEQ_LEN
    phase2_num_steps: int = 500
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_tpu("v4-8"))


def _make_optimizer_config(lr: float, batch_size: int, warmup: float | None = None) -> CautiousConfig:
    """Build a CautiousConfig with hyperparameters from the isoflop recipe."""
    beta2 = RECIPE._compute_beta2(batch_size)
    return CautiousConfig(
        learning_rate=lr,
        weight_decay=RECIPE.weight_decay,
        min_lr_ratio=RECIPE.min_lr_ratio,
        warmup=warmup if warmup is not None else RECIPE.warmup,
        beta1=RECIPE.beta1,
        beta2=beta2,
        epsilon=RECIPE.epsilon,
        max_grad_norm=RECIPE.max_grad_norm,
        lr_schedule=RECIPE.lr_schedule,
        decay=RECIPE.decay,
        adamc_weight_decay=True,
    )


def make_nemotron_train_step(cfg: RunConfig) -> ExecutorStep:
    optimizer_config = _make_optimizer_config(cfg.lr, cfg.batch_size)
    train_config = SimpleTrainConfig(
        resources=cfg.resources,
        train_batch_size=cfg.batch_size,
        train_seq_len=cfg.seq_len,
        num_train_steps=cfg.num_steps,
        learning_rate=cfg.lr,
        steps_per_eval=250,
        optimizer_config=optimizer_config,
    )

    weights = {"low": 1.0 - cfg.hq_fraction, "high": cfg.hq_fraction}
    mixture = lm_mixture_data_config(
        components={"low": low_q, "high": high_q},
        weights=weights,
        num_validation_sequences={"low": 1024, "high": 1024},
    )

    run_name = f"dq-iso-lr-{cfg.model_name}-{cfg.hq_fraction:.4f}-lr{cfg.lr:.0e}"
    return default_train(
        name=run_name,
        tokenized=mixture,
        model_config=cfg.model_config,
        train_config=train_config,
        tags=[
            "data-quality-scaling-iso-lr",
            f"type={cfg.model_name}",
            f"hq_frac={cfg.hq_fraction:.4f}",
            f"lr={cfg.lr:.0e}",
            f"hidden={cfg.hidden_dim}",
        ],
        wandb_group="data-quality-scaling-iso-lr",
        eval_harness_tasks=[],
        use_default_validation=False,
    )


def make_nemotron_phase2_train_step(
    cfg: RunConfig,
    phase1_step: ExecutorStep,
) -> ExecutorStep:
    optimizer_config = _make_optimizer_config(cfg.lr, cfg.batch_size, warmup=0.05)
    train_config = SimpleTrainConfig(
        resources=cfg.resources,
        train_batch_size=cfg.batch_size,
        train_seq_len=cfg.seq_len,
        num_train_steps=cfg.phase2_num_steps,
        learning_rate=cfg.lr,
        initialize_from_checkpoint_path=output_path_of(phase1_step, "checkpoints"),
        data_seed=42,
        steps_per_eval=250,
        optimizer_config=optimizer_config,
    )

    hq_only_mixture = lm_mixture_data_config(
        components={"high": high_q},
        weights={"high": 1.0},
        num_validation_sequences={"high": 1024},
    )

    run_name = f"dq-iso-lr-phase2-{cfg.model_name}-{cfg.hq_fraction:.4f}-lr{cfg.lr:.0e}"
    return default_train(
        name=run_name,
        tokenized=hq_only_mixture,
        model_config=cfg.model_config,
        train_config=train_config,
        tags=[
            "data-quality-scaling-iso-lr",
            "phase2",
            f"type={cfg.model_name}",
            f"hq_frac={cfg.hq_fraction:.4f}",
            f"lr={cfg.lr:.0e}",
            f"hidden={cfg.hidden_dim}",
        ],
        wandb_group="data-quality-scaling-iso-lr",
        eval_harness_tasks=[],
        use_default_validation=False,
    )


DEFAULT_TPU_TYPE = "v4-8"


def build_steps(tpu_type: str = DEFAULT_TPU_TYPE) -> list[ExecutorStep]:
    all_steps: list[ExecutorStep] = []

    run_cfgs = []
    for hidden in HIDDEN_SIZES:
        model_config = RECIPE._build_model_config(hidden, SEQ_LEN)
        base_lr = RECIPE._compute_learning_rate(BATCH_SIZE, hidden)
        for num_steps in STEP_COUNTS:
            for hq_steps in HQ_TOTAL_EFFECTIVE_STEPS:
                for mult in LR_MULTIPLIERS:
                    run_cfgs.append(RunConfig(
                        model_name=f"{hidden}h",
                        model_config=model_config,
                        hidden_dim=hidden,
                        num_steps=num_steps,
                        hq_fraction=float(hq_steps / num_steps),
                        lr=base_lr * mult,
                        resources=ResourceConfig.with_tpu(tpu_type),
                    ))

    for cfg in run_cfgs:
        train_step = make_nemotron_train_step(cfg)
        phase2_step = make_nemotron_phase2_train_step(cfg, phase1_step=train_step)
        all_steps.append(train_step)
        all_steps.append(phase2_step)

    return all_steps


parser = argparse.ArgumentParser(description="Data quality scaling laws (isoflop LR sweep).")
parser.add_argument(
    "--tpu-type",
    type=str,
    default=DEFAULT_TPU_TYPE,
    help=f"TPU type for ResourceConfig.with_tpu (default {DEFAULT_TPU_TYPE}).",
)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]

all_steps = build_steps(tpu_type=args.tpu_type)

if __name__ == "__main__":
    executor_main(steps=all_steps)
