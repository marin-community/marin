# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train a 1B Llama model on Nemotron CC high-quality data (real + synthetic).
"""

import dataclasses
import math
import os

from experiments.defaults import default_train
from experiments.evals.evals import default_base_eval
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from experiments.llama import llama3_tokenizer, llama_3_2_1b
from experiments.pretraining_datasets.nemotron import tokenize_nemotron
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from levanter.utils.flop_utils import lm_flops_per_token
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config


################################################################
# Compute Budget Configuration
# Following heuristics from "Relative Scaling Laws for LLMs"
# (arXiv:2510.24626, Appendix A)
################################################################

# Llama 3.2 1B model configuration
LLAMA_1B_CONFIG = {
    "hidden_dim": 2048,
    "intermediate_dim": 8192,
    "num_layers": 16,
    "num_kv_heads": 8,
    "num_heads": 32,
    "vocab_size": 128256,
}

TRAIN_SEQ_LEN = 4096  # From llama_3_2_1b.max_seq_len

# Can be overridden via COMPUTE_BUDGET environment variable (e.g., -e COMPUTE_BUDGET 1e21)
COMPUTE_BUDGET_FLOPS = float(os.environ.get("COMPUTE_BUDGET", "1e20"))


def compute_flops_per_token(
    hidden_dim: int,
    intermediate_dim: int,
    num_layers: int,
    num_kv_heads: int,
    num_heads: int,
    seq_len: int,
    vocab_size: int,
    glu: bool = True,
) -> float:
    """Compute FLOPs per token for forward pass."""
    return lm_flops_per_token(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        seq_len=seq_len,
        vocab_size=vocab_size,
        glu=glu,
    )


def compute_total_tokens(total_flops: float, flops_per_token: float) -> int:
    """Compute total training tokens from FLOPs budget.

    FLOPs = 6 * N * T approximately, where N is params and T is tokens.
    More precisely: total_flops = 3 * flops_per_token * total_tokens
    (3x for forward + backward)
    """
    return int(total_flops / (3 * flops_per_token))


def nearest_power_of_two(x: int) -> int:
    """Round to nearest power of two."""
    if x <= 0:
        return 1
    log2 = math.log2(x)
    lower = 2 ** int(log2)
    upper = 2 ** (int(log2) + 1)
    return lower if (x - lower) < (upper - x) else upper


# Compute FLOPs per token
flops_per_token = compute_flops_per_token(seq_len=TRAIN_SEQ_LEN, **LLAMA_1B_CONFIG)

# Compute total tokens from compute budget
total_tokens = compute_total_tokens(COMPUTE_BUDGET_FLOPS, flops_per_token)

# Heuristic 1: Batch size B = T / 2^16, rounded to nearest power of 2
# This targets 2^16 = 65536 training steps
TARGET_STEPS = 2 ** 16  # 65536 steps
tokens_per_batch = total_tokens // TARGET_STEPS
TRAIN_BATCH_SIZE = nearest_power_of_two(tokens_per_batch // TRAIN_SEQ_LEN)

# Actual number of training steps
num_train_steps = total_tokens // (TRAIN_BATCH_SIZE * TRAIN_SEQ_LEN)

# Heuristic 2: Learning rate η = η_base × (√B / d)
# η_base calibrated so that η < 0.01 to avoid loss spikes
HIDDEN_DIM = LLAMA_1B_CONFIG["hidden_dim"]  # 2048
LR_BASE = 0.1  # Base learning rate (calibrated for stability)
learning_rate = LR_BASE * (math.sqrt(TRAIN_BATCH_SIZE) / HIDDEN_DIM)
# Clamp to avoid instability (paper notes η ≥ 0.01 causes loss spikes)
learning_rate = min(learning_rate, 0.01)

# Heuristic 3: Warmup = 5% of total steps
warmup_ratio = 0.05
warmup_steps = int(num_train_steps * warmup_ratio)

# Heuristic 4: Decay = 20% linear decay phase
decay_ratio = 0.2


################################################################
# Nemotron HQ Data Configuration
################################################################

# Get all Nemotron tokenized datasets
nemotron_tokenized = tokenize_nemotron(tokenizer=llama3_tokenizer)

# Only use high-quality data (real + synthetic)
nemotron_hq_components = {
    "nemotron_cc/hq_actual": nemotron_tokenized["nemotron_cc/hq_actual"],
    "nemotron_cc/hq_synth": nemotron_tokenized["nemotron_cc/hq_synth"],
}

# Weights based on dataset sizes (in TiB)
nemotron_hq_weights = {
    "nemotron_cc/hq_actual": 0.91351,
    "nemotron_cc/hq_synth": 2.72,
}

# Normalize weights to sum to 1
total_weight = sum(nemotron_hq_weights.values())
nemotron_hq_weights_normalized = {k: v / total_weight for k, v in nemotron_hq_weights.items()}

nemotron_hq_data_config = lm_mixture_data_config(
    components=nemotron_hq_components,
    weights=nemotron_hq_weights_normalized,
)


################################################################
# 1B Model Training Configuration
################################################################

# Can be overridden via TPU_TYPE environment variable (e.g., -e TPU_TYPE v4-128)
TPU_TYPE = os.environ.get("TPU_TYPE", "v4-64")

# Can be overridden via EXP_NAME environment variable (e.g., -e EXP_NAME llama-1b-tootsie-run1)
EXP_NAME = os.environ.get("EXP_NAME", "llama-1b-tootsie")

tootsie_1b_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE),
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=num_train_steps,
    # Heuristic from arXiv:2510.24626: η = η_base × (√B / d)
    learning_rate=learning_rate,
    # Paper uses weight_decay=0.1
    weight_decay=0.1,
    # Heuristic: 5% warmup
    warmup=warmup_steps,
    # Heuristic: 20% linear decay phase
    decay=decay_ratio,
    lr_schedule="linear",
    steps_per_eval=5000,
    steps_per_export=10000,
)

llama_1b_tootsie = dataclasses.replace(
    default_train(
        name=EXP_NAME,
        tokenized=nemotron_hq_data_config,
        model_config=llama_3_2_1b,
        train_config=tootsie_1b_train_config,
        tags=["llama", "1b", "nemotron-hq", "exp600"],
        eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
    ),
    override_output_path=f"checkpoints/{EXP_NAME}",
)


if __name__ == "__main__":
    executor_main(
        steps=[
            llama_1b_tootsie,
            *default_base_eval(llama_1b_tootsie),
        ],
        description="Train 1B Llama model on Nemotron CC high-quality data (real + synthetic).",
    )
