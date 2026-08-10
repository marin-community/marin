# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daily ferry template for the 125M-ish integration run.

This script is intentionally simple and serves as a living ferry template that
agents can update on a daily cadence with small, reviewable changes.

Expected workflow:
1. Propose at least one bounded config update in the run issue.
2. If no obvious change emerges from recent commits/ferries, use judgment to pick a low-risk tweak
   (for example data-mix or hyperparameter) that could improve loss at the same FLOPs budget.
3. Get human approval.
4. Push the launch commit, then launch per the `run-ferries` skill
   (`uv run iris --cluster=marin job run ...`) and monitor to completion.
"""

import datetime as dt
import os
from dataclasses import replace

from fray.cluster import ResourceConfig
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.execution.step_runner import StepRunner
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint

from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.llama import (
    compute_num_parameters,
    llama3_tokenizer,
    llama3_tokenizer_vocab_size,
    llama_150m,
)

# ---------------------------
# Daily ferry policy defaults
# ---------------------------
DEFAULT_MODEL_FLOPS_TARGET = int(1e19)
TRAIN_BATCH_SIZE = 512
TRAIN_SEQ_LEN = 4096

# Nemotron CC mixture weights: the corpus's TiB proportions, plus starcoder and
# proof-pile at their published weights. Policy lives here, in the experiment.
_NEMOTRON_WEIGHTS = {
    "hq_actual": 0.91351,
    "hq_synth": 2.72,
    "medium_high": 0.82471,
    "medium": 3.38,
    "medium_low": 1.54,
    "low_actual": 0.70123,
    "low_synth": 0.62771,
}
_STARCODER_WEIGHT = 0.25
_PROOFPILE_WEIGHT = 0.055


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


# Approximate FLOPs model used for scaling guidance:
# total_flops ~= 6 * num_params * num_tokens
MODEL_FLOPS_TARGET = _int_env("FERRY_MODEL_FLOPS_TARGET", DEFAULT_MODEL_FLOPS_TARGET)
NUM_MODEL_PARAMS = compute_num_parameters(llama_150m, llama3_tokenizer_vocab_size)
NUM_TRAIN_TOKENS = MODEL_FLOPS_TARGET // (6 * NUM_MODEL_PARAMS)
NUM_TRAIN_STEPS = _int_env(
    "FERRY_NUM_TRAIN_STEPS",
    max(1, NUM_TRAIN_TOKENS // (TRAIN_BATCH_SIZE * TRAIN_SEQ_LEN)),
)

# Agents can override date from the launch environment to force deterministic naming.
FERRY_DATE = os.environ.get("FERRY_DATE", dt.date.today().isoformat())
RUN_NAME = f"ferry_daily_125m_{FERRY_DATE}"


def build() -> ArtifactStep[LevanterCheckpoint]:
    """The daily 125M-ish nemotron integration run as a lazy checkpoint."""
    nem = nemotron_datasets(tokenizer=llama3_tokenizer)
    train = {nem[split]: weight for split, weight in _NEMOTRON_WEIGHTS.items()}
    train[starcoder_dataset(tokenizer=llama3_tokenizer)] = _STARCODER_WEIGHT
    train[proofpile_dataset(tokenizer=llama3_tokenizer)] = _PROOFPILE_WEIGHT
    validation = [
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    ]

    return train_lm(
        name=RUN_NAME,
        version="2026.06.28",
        model=replace(llama_150m, max_seq_len=TRAIN_SEQ_LEN),
        # Agent edit surface: keep daily changes small (usually 1-2 knobs).
        optimizer=AdamConfig(
            learning_rate=3e-3,
            lr_schedule="linear",
            decay=0.2,
            weight_decay=0.1,
            min_lr_ratio=0.1,
            warmup=1000,
        ),
        datasets=train,
        validation=validation,
        batch_size=TRAIN_BATCH_SIZE,
        seq_len=TRAIN_SEQ_LEN,
        num_train_steps=NUM_TRAIN_STEPS,
        z_loss_weight=1e-4,
        evals=None,
        resources=ResourceConfig.with_tpu("v5p-8"),
        run_id=RUN_NAME,
        tags=["ferry", "daily", "integration", "125m", "nemotron", "seq4096"],
    )


if __name__ == "__main__":
    StepRunner().run([build().lower()])
