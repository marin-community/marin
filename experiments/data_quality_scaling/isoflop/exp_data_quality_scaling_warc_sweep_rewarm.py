# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data quality scaling laws (isoflop, raw HTML baseline, sweep, LR rewarmup):
train models on varying mixes of raw Common Crawl HTML and high-quality
Nemotron CC data, using model architectures and hyperparameters from the Marin
2025 isoflop scaling recipe.

Identical to exp_data_quality_scaling_random_sweep_rewarm.py except the
low-quality data is raw HTML from Common Crawl WARCs instead of word-shuffled
Nemotron CC text.

Pipeline:
  1. Sample files from raw Common Crawl WARC HTML data, then tokenize.
  2. LR tuning: len(HIDDEN_SIZES) model sizes × len(LR_MULTIPLIERS) LRs
     × LR_TUNING_STEPS steps (single-cycle, no rewarmup)
  3. LR selection: len(HIDDEN_SIZES) steps, picks best LR by eval/high/loss
  4. Full training: len(HIDDEN_SIZES) model sizes × len(STEP_COUNTS) step counts
     × len(HQ_ANNEAL_STEPS) anneal configs (with LR rewarmup at the HQ
     transition boundary)
"""

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field

import fsspec

from experiments.defaults import default_train
from experiments.isoflop_sweep import MARIN_2025_RECIPE
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.nemotron import tokenize_nemotron
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.models.lm_model import LmConfig
from levanter.optim.cautious import CautiousConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, lm_mixture_data_config, tokenize
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.utils import fsspec_glob

from experiments.data_quality_scaling.utils import SelectBestLRConfig, select_best_lr

logger = logging.getLogger("ray")

RECIPE = MARIN_2025_RECIPE
SEQ_LEN = 1024
BATCH_SIZE = 128
RAMP_BEFORE = 100
REWARMUP = 0.1

# --- Raw HTML Common Crawl data ---
WARC_DATA_PATH = "gs://marin-us-central2/raw/commoncrawl/rephraser_sweep_batch0-231d96"
WARC_NUM_FILES = 10
WARC_SEED = 42


# ---------------------------------------------------------------------------
# Low-quality data: sampled raw HTML from Common Crawl WARCs
# ---------------------------------------------------------------------------

nemotron_steps = tokenize_nemotron()

_warc_rng = random.Random(WARC_SEED)
_all_warc_files = sorted(fsspec_glob(os.path.join(WARC_DATA_PATH, "**/*.jsonl.gz")))
WARC_SAMPLE_PATHS = _warc_rng.sample(_all_warc_files, k=min(WARC_NUM_FILES, len(_all_warc_files)))

tokenize_warc_step = ExecutorStep(
    name="tokenized/commoncrawl_warc_html",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[InputName(step=None, name=p) for p in WARC_SAMPLE_PATHS],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
        format=TextLmDatasetFormat(text_key="html"),
    ),
)

low_q = tokenize_warc_step
high_q = nemotron_steps["nemotron_cc/hq_actual"]


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    name: str
    config: LmConfig
    hidden_dim: int


HIDDEN_SIZES = [128, 256, 512, 768]

MODELS = [
    ModelConfig(
        name=f"{h}h",
        config=RECIPE._build_model_config_from_hidden_size(h, SEQ_LEN),
        hidden_dim=h,
    )
    for h in HIDDEN_SIZES
]


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    model_config: LmConfig
    hidden_dim: int
    num_steps: int
    batch_size: int = BATCH_SIZE
    seq_len: int = SEQ_LEN
    hq_anneal_steps: int = 500
    lr: float = 1e-3  # placeholder, overridden by formula or best-LR selection
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_tpu("v4-8"))


# --- Sweep axes ---
STEP_COUNTS = [100, 1172, 2344, 4688, 9375]  # ~154M, ~307M, ~614M, ~1.2B tokens
HQ_ANNEAL_STEPS = [500, 1000, 2000]
LR_MULTIPLIERS = [0.2, 0.5, 1.0, 2.0, 5.0]
LR_TUNING_STEPS = 1000


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


def _make_rewarm_optimizer_config(
    lr: float,
    batch_size: int,
    num_steps: int,
    hq_anneal_steps: int,
    ramp_before: int = RAMP_BEFORE,
) -> CautiousConfig:
    """Build a CautiousConfig with two-cycle LR schedule that rewarms at the HQ transition.

    Cycle 1: steps 0 to (num_steps - ramp_before) — low-quality phase.
    Cycle 2: steps (num_steps - ramp_before) to (num_steps + hq_anneal_steps) — HQ transition + anneal.
    """
    beta2 = RECIPE._compute_beta2(batch_size)
    low_q_phase_steps = num_steps - ramp_before
    hq_phase_steps = ramp_before + hq_anneal_steps
    return CautiousConfig(
        learning_rate=lr,
        weight_decay=RECIPE.weight_decay,
        min_lr_ratio=RECIPE.min_lr_ratio,
        warmup=RECIPE.warmup,
        rewarmup=REWARMUP,
        beta1=RECIPE.beta1,
        beta2=beta2,
        epsilon=RECIPE.epsilon,
        max_grad_norm=RECIPE.max_grad_norm,
        lr_schedule=RECIPE.lr_schedule,
        decay=RECIPE.decay,
        adamc_weight_decay=True,
        cycle_length=[low_q_phase_steps, hq_phase_steps],
    )


def make_anneal_schedule(
    num_steps: int,
    ramp_before: int = RAMP_BEFORE,
    n_stages: int = 10,
) -> list[tuple[int, dict[str, float]]]:
    """Generate a mixture weight schedule that linearly anneals to 100% HQ.

    The ramp starts at step (num_steps - ramp_before) and ends at
    step num_steps.
    """
    ramp_start = num_steps - ramp_before
    ramp_end = num_steps

    schedule = [(0, {"low": 1.0, "high": 0.0})]
    for i in range(1, n_stages + 1):
        t = i / n_stages
        step = ramp_start + int(t * (ramp_end - ramp_start))
        low_w = 1.0 - t
        schedule.append((step, {"low": low_w, "high": 1.0 - low_w}))

    return schedule


def make_train_step(cfg: RunConfig) -> ExecutorStep:
    total_steps = cfg.num_steps + cfg.hq_anneal_steps
    optimizer_config = _make_rewarm_optimizer_config(
        cfg.lr, cfg.batch_size, cfg.num_steps, cfg.hq_anneal_steps,
    )
    train_config = SimpleTrainConfig(
        resources=cfg.resources,
        train_batch_size=cfg.batch_size,
        train_seq_len=cfg.seq_len,
        num_train_steps=versioned(total_steps),
        learning_rate=versioned(cfg.lr),
        steps_per_eval=250,
        optimizer_config=optimizer_config,
    )

    mixture = lm_varying_mixture_data_config(
        components={"low": low_q, "high": high_q},
        weights_list=make_anneal_schedule(cfg.num_steps),
        mixture_block_size=cfg.batch_size,
        num_validation_sequences={"low": 1024, "high": 1024},
    )

    run_name = f"dq-iso-warc-rewarm-train-{cfg.model_name}-steps{cfg.num_steps}"
    return default_train(
        name=run_name,
        tokenized=mixture,
        model_config=cfg.model_config,
        train_config=train_config,
        tags=[
            "data-quality-scaling-iso-warc-rewarm",
            f"type={cfg.model_name}",
            f"steps={cfg.num_steps}",
            f"hidden={cfg.hidden_dim}",
        ],
        wandb_group="data-quality-scaling-iso-warc-rewarm",
        eval_harness_tasks=[],
        use_default_validation=False,
    )


def make_tuning_step(cfg: RunConfig) -> ExecutorStep:
    """Create a short tuning run to evaluate a candidate learning rate."""
    optimizer_config = _make_optimizer_config(cfg.lr, cfg.batch_size)
    train_config = SimpleTrainConfig(
        resources=cfg.resources,
        train_batch_size=cfg.batch_size,
        train_seq_len=cfg.seq_len,
        num_train_steps=versioned(cfg.num_steps),
        learning_rate=versioned(cfg.lr),
        steps_per_eval=250,
        optimizer_config=optimizer_config,
    )

    mixture = lm_mixture_data_config(
        components={"low": low_q, "high": high_q},
        weights={"low": 1e-10, "high": 1.0},
        num_validation_sequences={"low": 1024, "high": 1024},
    )

    run_name = f"dq-iso-warc-tune-{cfg.model_name}-steps{cfg.num_steps}-lr{cfg.lr:.0e}"
    return default_train(
        name=run_name,
        tokenized=mixture,
        model_config=cfg.model_config,
        train_config=train_config,
        tags=[
            "data-quality-scaling-iso-warc",
            "lr-tuning",
            f"type={cfg.model_name}",
            f"steps={cfg.num_steps}",
            f"lr={cfg.lr:.0e}",
            f"hidden={cfg.hidden_dim}",
        ],
        wandb_group="data-quality-scaling-iso-warc",
        eval_harness_tasks=[],
        use_default_validation=False,
    )


@dataclass(frozen=True)
class TrainWithBestLRConfig:
    """Config for a training run that reads the best LR from a selection step at runtime."""
    best_lr_selection_path: str
    train_pod_config: TrainLmOnPodConfig
    train_weights_version: object = None


def run_train_with_best_lr(config: TrainWithBestLRConfig):
    """Read best LR from selection step, patch the training config, and run training."""
    best_lr_file = os.path.join(config.best_lr_selection_path, "best_lr.json")
    with fsspec.open(best_lr_file, "rt") as f:
        best = json.load(f)

    lr = best["best_lr"]
    logger.info(f"Using best LR: {lr}")

    inner = config.train_pod_config.train_config
    optimizer = dataclasses.replace(inner.optimizer, learning_rate=lr)
    inner = dataclasses.replace(inner, optimizer=optimizer)
    train_pod_config = dataclasses.replace(config.train_pod_config, train_config=inner)

    run_levanter_train_lm(train_pod_config)


DEFAULT_TPU_TYPE = "v4-8"


def build_steps(tpu_type: str = DEFAULT_TPU_TYPE) -> list[ExecutorStep]:
    all_steps: list[ExecutorStep] = [tokenize_warc_step]

    # --- LR tuning: short runs per model × LR multiplier, then select best ---
    best_lr_steps: dict[str, ExecutorStep] = {}
    for model in MODELS:
        base_lr = RECIPE._compute_learning_rate(BATCH_SIZE, model.hidden_dim)
        tuning_steps: list[ExecutorStep] = []
        for mult in LR_MULTIPLIERS:
            cfg = RunConfig(
                model_name=model.name,
                model_config=model.config,
                hidden_dim=model.hidden_dim,
                num_steps=LR_TUNING_STEPS,
                lr=base_lr * mult,
                resources=ResourceConfig.with_tpu(tpu_type),
            )
            step = make_tuning_step(cfg)
            tuning_steps.append(step)
            all_steps.append(step)

        select_step = ExecutorStep(
            name=f"select-best-lr/dq-iso-warc-{model.name}",
            fn=select_best_lr,
            config=SelectBestLRConfig(
                tuning_run_paths=[output_path_of(step) for step in tuning_steps],
                tuning_run_configs=[step.config for step in tuning_steps],
                output_path=this_output_path(),
            ),
        )
        all_steps.append(select_step)
        best_lr_steps[model.name] = select_step

    # --- Full training runs, using best LR per model ---
    for model in MODELS:
        select_step = best_lr_steps[model.name]
        base_lr = RECIPE._compute_learning_rate(BATCH_SIZE, model.hidden_dim)
        for num_steps in STEP_COUNTS:
            for hq_anneal_steps in HQ_ANNEAL_STEPS:
                cfg = RunConfig(
                    model_name=model.name,
                    model_config=model.config,
                    hidden_dim=model.hidden_dim,
                    num_steps=num_steps,
                    hq_anneal_steps=hq_anneal_steps,
                    lr=base_lr,  # placeholder, patched at runtime by run_train_with_best_lr
                    resources=ResourceConfig.with_tpu(tpu_type),
                )

                # Build template step to get the TrainLmOnPodConfig; the LR
                # inside is a placeholder — run_train_with_best_lr patches it.
                template_step = make_train_step(cfg)
                train_step = ExecutorStep(
                    name=template_step.name,
                    fn=run_train_with_best_lr,
                    config=TrainWithBestLRConfig(
                        best_lr_selection_path=output_path_of(select_step),
                        train_pod_config=template_step.config,
                        train_weights_version=versioned(template_step.config.train_config.data.train_weights),
                    ),
                )
                all_steps.append(train_step)

    return all_steps


parser = argparse.ArgumentParser(description="Data quality scaling laws (isoflop, raw HTML baseline, sweep, LR rewarmup).")
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
