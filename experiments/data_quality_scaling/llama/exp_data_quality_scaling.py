# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data quality scaling laws: train fixed-architecture models on varying mixes
of low-quality and high-quality Nemotron CC data, sweeping learning rate and
weight decay.

Grid: 4 model sizes × 5 mix ratios × 6 HP configs = 120 training runs.
"""

import argparse
import sys
from dataclasses import dataclass, field

from experiments.defaults import default_train
from experiments.llama import llama_30m, llama_75m, llama_150m, llama_300m
from levanter.models.lm_model import LmConfig
from experiments.pretraining_datasets.nemotron import tokenize_nemotron
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of
from marin.processing.tokenize import lm_mixture_data_config

@dataclass(frozen=True)
class ModelConfig:
    name: str
    config: LmConfig

@dataclass(frozen=True)
class RunConfig:
    model_name: str
    model_config: LmConfig
    num_steps: int
    hq_fraction: float
    batch_size: int = 128
    seq_len: int = 1024
    lr: float = 3e-3
    wd: float = 0.1
    phase2_lr: float = 3e-3
    phase2_wd: float = 0.1
    phase2_num_steps: int = 500
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_tpu("v4-8"))

# --- Data components ---
nemotron_steps = tokenize_nemotron()
low_q = nemotron_steps["nemotron_cc/low_actual"]
high_q = nemotron_steps["nemotron_cc/hq_actual"]

MODELS = [
    ModelConfig(name="30m", config=llama_30m),
    ModelConfig(name="75m", config=llama_75m),
    ModelConfig(name="150m", config=llama_150m),
    ModelConfig(name="300m", config=llama_300m),
]

# --- Sweep axes ---
STEP_COUNTS = [1172, 2344, 4688] # 150M, 300M, 600M tokens
HQ_TOTAL_EFFECTIVE_STEPS =  [500]

def make_nemotron_train_step(
    cfg: RunConfig,
) -> ExecutorStep:
    train_config = SimpleTrainConfig(
        resources=cfg.resources,
        train_batch_size=cfg.batch_size,
        train_seq_len=cfg.seq_len,
        num_train_steps=cfg.num_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.wd,
        steps_per_eval=250,
    )

    weights = {"low": 1.0 - cfg.hq_fraction, "high": cfg.hq_fraction}
    mixture = lm_mixture_data_config(
        components={"low": low_q, "high": high_q},
        weights=weights,
        num_validation_sequences={"low": 1024, "high": 1024},
    )

    run_name = f"dq-{cfg.model_name}-{cfg.hq_fraction:.4f}-lr{cfg.lr:.0e}-wd{cfg.wd}"
    return default_train(
        name=run_name,
        tokenized=mixture,
        model_config=cfg.model_config,
        train_config=train_config,
        tags=[
            "data-quality-scaling",
            f"type={cfg.model_name}",
            f"hq_frac={cfg.hq_fraction:.4f}",
            f"lr={cfg.lr:.0e}",
            f"wd={cfg.wd:.0e}",
        ],
        wandb_group="data-quality-scaling",
        eval_harness_tasks=[],
        use_default_validation=False,
    )


def make_nemotron_phase2_train_step(
    cfg: RunConfig,
    phase1_step: ExecutorStep,
) -> ExecutorStep:
    train_config = SimpleTrainConfig(
        resources=cfg.resources,
        train_batch_size=cfg.batch_size,
        train_seq_len=cfg.seq_len,
        num_train_steps=cfg.phase2_num_steps,
        learning_rate=cfg.phase2_lr,
        weight_decay=cfg.phase2_wd,
        initialize_from_checkpoint_path=output_path_of(phase1_step, "checkpoints"),
        data_seed=42,
        steps_per_eval=250,
        warmup=0.05,
    )

    hq_only_mixture = lm_mixture_data_config(
        components={"high": high_q},
        weights={"high": 1.0},
        num_validation_sequences={"high": 1024},
    )

    run_name = f"dq-phase2-{cfg.model_name}-{cfg.hq_fraction:.4f}-lr{cfg.phase2_lr:.0e}-wd{cfg.phase2_wd}"
    return default_train(
        name=run_name,
        tokenized=hq_only_mixture,
        model_config=cfg.model_config,
        train_config=train_config,
        tags=[
            "data-quality-scaling",
            "phase2",
            f"type={cfg.model_name}",
            f"hq_frac={cfg.hq_fraction:.4f}",
            f"lr={cfg.phase2_lr:.0e}",
            f"wd={cfg.phase2_wd:.0e}",
        ],
        wandb_group="data-quality-scaling",
        eval_harness_tasks=[],
        use_default_validation=False,
    )


DEFAULT_TPU_TYPE = "v4-8"


def build_steps(tpu_type: str = DEFAULT_TPU_TYPE) -> list[ExecutorStep]:
    all_steps: list[ExecutorStep] = []

    run_cfgs = [
        RunConfig(
            model_name=model.name,
            model_config=model.config,
            num_steps=num_steps,
            hq_fraction=float(hq_steps / num_steps),
            resources=ResourceConfig.with_tpu(tpu_type),
        )
        for model in MODELS
        for num_steps in STEP_COUNTS
        for hq_steps in HQ_TOTAL_EFFECTIVE_STEPS
    ]

    for cfg in run_cfgs:
        train_step = make_nemotron_train_step(cfg)
        phase2_step = make_nemotron_phase2_train_step(cfg, phase1_step=train_step)
        all_steps.append(train_step)
        all_steps.append(phase2_step)

    return all_steps


parser = argparse.ArgumentParser(description="Data quality scaling laws.")
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
