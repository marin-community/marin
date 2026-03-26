# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data quality scaling laws (isoflop, shuffled baseline, sweep): train models on
varying mixes of word-shuffled Nemotron CC text and high-quality Nemotron CC
data, using model architectures and hyperparameters from the Marin 2025
isoflop scaling recipe.

Same structure as exp_data_quality_scaling_random.py but with the mixture
schedule versioned in TrainWithBestLRConfig so that changes to the anneal
schedule (e.g. more HQ steps) produce distinct version hashes.

Pipeline:
  1. Shuffle generation: reuses the same shuffled data steps from the random
     baseline (deduplication handles this).
  2. LR tuning: len(HIDDEN_SIZES) model sizes × len(LR_MULTIPLIERS) LRs
     × LR_TUNING_STEPS steps
  3. LR selection: len(HIDDEN_SIZES) steps, picks best LR by eval/high/loss
  4. Full training: len(HIDDEN_SIZES) model sizes × len(STEP_COUNTS) step counts
     (each run uses a mixture schedule that anneals to 100% HQ for the last
     hq_anneal_steps steps)
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field

import fsspec

from experiments.defaults import default_train
from experiments.scaling_law_sweeps.c_adamc import c_adamc_heuristic
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.nemotron import tokenize_nemotron
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from levanter.models.lm_model import LmConfig
from levanter.optim.cautious import CautiousConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, lm_mixture_data_config, tokenize
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.utils import fsspec_glob
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_jsonl

from experiments.data_quality_scaling.utils import SelectBestLRConfig, select_best_lr

logger = logging.getLogger("ray")

RECIPE = c_adamc_heuristic
SEQ_LEN = 1024
BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Shuffled Nemotron data generation (reuses the same steps as random baseline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerateShuffledConfig:
    """Config for generating word-shuffled documents from Nemotron CC low-quality data."""
    input_paths: list[str]
    output_path: str
    num_files: int = 10
    seed: int = 42


def _shuffle_words(doc: dict, rng: random.Random) -> dict:
    """Shuffle words within a document."""
    words = doc["text"].split()
    rng.shuffle(words)
    return {"text": " ".join(words)}


def generate_shuffled(config: GenerateShuffledConfig):
    """Sample files from Nemotron CC low-quality, shuffle words within each document."""
    rng = random.Random(config.seed)

    all_files = []
    for path in config.input_paths:
        all_files.extend(fsspec_glob(path))
    sampled_files = rng.sample(all_files, k=min(config.num_files, len(all_files)))
    logger.info(f"Sampled {len(sampled_files)} files from {len(all_files)} total")

    ctx = ZephyrContext(name="generate-shuffled-nemotron")
    ctx.execute(
        Dataset.from_list(sampled_files)
        .flat_map(load_jsonl)
        .map(lambda doc: _shuffle_words(doc, random.Random(int(hashlib.md5(doc["text"][:100].encode()).hexdigest(), 16))))
        .write_jsonl(os.path.join(config.output_path, "shard-{shard:04d}.jsonl.gz"))
    )
    logger.info("Shuffled Nemotron generation complete.")


# --- Shuffled data steps ---
nemotron_steps = tokenize_nemotron()

generate_shuffled_step = ExecutorStep(
    name="raw/shuffled_nemotron",
    description="Generate word-shuffled Nemotron CC low-quality JSONL for noise baseline.",
    fn=generate_shuffled,
    config=GenerateShuffledConfig(
        input_paths=nemotron_steps["nemotron_cc/low_actual"].config.train_paths,
        output_path=this_output_path(),
        num_files=versioned(50),
    ),
)

tokenize_shuffled_step = ExecutorStep(
    name="tokenized/shuffled_nemotron",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[output_path_of(generate_shuffled_step)],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
    ),
)

low_q = tokenize_shuffled_step
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
        config=RECIPE._build_model_config(h, SEQ_LEN),
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


def make_anneal_schedule(
    num_steps: int,
    ramp_before: int = 100,
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
    optimizer_config = _make_optimizer_config(cfg.lr, cfg.batch_size)
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

    run_name = f"dq-iso-rand-train-{cfg.model_name}-steps{cfg.num_steps}"
    return default_train(
        name=run_name,
        tokenized=mixture,
        model_config=cfg.model_config,
        train_config=train_config,
        tags=[
            "data-quality-scaling-iso-random",
            f"type={cfg.model_name}",
            f"steps={cfg.num_steps}",
            f"hidden={cfg.hidden_dim}",
        ],
        wandb_group="data-quality-scaling-iso-random",
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

    run_name = f"dq-iso-rand-tune-{cfg.model_name}-steps{cfg.num_steps}-lr{cfg.lr:.0e}"
    return default_train(
        name=run_name,
        tokenized=mixture,
        model_config=cfg.model_config,
        train_config=train_config,
        tags=[
            "data-quality-scaling-iso-random",
            "lr-tuning",
            f"type={cfg.model_name}",
            f"steps={cfg.num_steps}",
            f"lr={cfg.lr:.0e}",
            f"hidden={cfg.hidden_dim}",
        ],
        wandb_group="data-quality-scaling-iso-random",
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
    all_steps: list[ExecutorStep] = [generate_shuffled_step, tokenize_shuffled_step]

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
            name=f"select-best-lr/dq-iso-rand-{model.name}",
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


parser = argparse.ArgumentParser(description="Data quality scaling laws (isoflop, random baseline, sweep).")
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
