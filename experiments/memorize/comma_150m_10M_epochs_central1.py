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

"""Train 150M Llama model on ~10M token COMMA seed set with varying epoch counts on v5p-8.

This experiment creates multiple training runs on the same ~10M token seed set:
- 10 epochs: 150 steps (~78M tokens)
- 20 epochs: 300 steps (~157M tokens)
- 50 epochs: 750 steps (~393M tokens)
- 100 epochs: 1500 steps (~786M tokens)
- 200 epochs: 3000 steps (~1.57B tokens)

The goal is to measure how memorization (P(z)) scales with epoch count on a fixed dataset.

Hardware: v5p-8 (8 chips x 96GB = 768GB total HBM)
Based on levanter config: 1 batch per dataset x 15 datasets = 15 batches per epoch = ~7.8M tokens
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

from experiments.common_pile.tokenize_common_pile import (
    common_pile_tokenized,
    COMMA_MAIN_MIXTURE_WEIGHTS,
)
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from experiments.defaults import default_train
from experiments.llama import llama_150m, llama3_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from levanter.eval_pz_innerloop import PzInnerLoopConfig

# ============================================================================
# Seed Set Configuration (~10M tokens)
# ============================================================================
# Following levanter/config/memorize/comma_150m_10M_100epoch.yaml:
# - 1 batch per dataset x 15 datasets = 15 batches per epoch
# - batch_size=128, seq_len=4096
# - Tokens per batch = 128 * 4096 = 524,288 tokens
# - Total per epoch = 15 * 524,288 = ~7.8M tokens
#
# This creates a ~10M token seed set that the model will see repeatedly.
# To epoch over this seed set, multiply num_train_steps by epoch count.
# ============================================================================

TPU_TYPE = "v5p-8"
BATCH_SIZE = 128  # Match levanter config batch size
BATCHES_PER_DATASET = 1  # 1 batch per dataset (10M seed set)
SEED_SET_BATCHES = BATCHES_PER_DATASET * 15  # 15 batches total
SEED_SET_TOKENS = SEED_SET_BATCHES * BATCH_SIZE * 4096  # ~7.8M tokens

# Model: 150M parameters with seq_len=4096 for memorization studies
# Override seq_len from default 1024 to 4096 to match memorization configs
model_config = dataclasses.replace(llama_150m, seq_len=4096)

# Build the COMMA mixture with max_train_batches to create ~100M seed set
_tokenized = common_pile_tokenized(tokenizer=llama3_tokenizer)

# Create max_train_batches dict for all datasets
max_train_batches_dict = {dataset: BATCHES_PER_DATASET for dataset in _tokenized}

comma_mixture = lm_mixture_data_config(
    components=_tokenized,
    weights=COMMA_MAIN_MIXTURE_WEIGHTS,
    max_train_batches=max_train_batches_dict,
    shuffle=False,  # Disable shuffle for reproducible memorization experiments
)

# Generate timestamp for unique run names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# P(z) Evaluation Configuration
# ============================================================================
# Inner-loop P(z) evaluation to measure memorization during training
# Optimized: single dataset + limited doc length for 150x faster evaluation
# CRITICAL: Dataset names must match the keys from common_pile_tokenized() which are "common_pile/{name}"
pz_config = PzInnerLoopConfig(
    datasets=["common_pile/wikimedia"],  # MUST use full key including "common_pile/" prefix!
    mode="first",
    num_documents=1,
    doc_tokens=1024,  # Limit document length to 1024 tokens (was: None = full doc, some >5k tokens)
    chunk_size=256,
    prompt_tokens=200,
    cursor_inc_tokens=16,
    histogram=False,
    pz_npz=False,
    decode_preview=None,
    verify_treecache=False,
)

# ============================================================================
# Epoch Configurations
# ============================================================================
# Hyperparameters for memorization experiments (from levanter config):
#   - LR: 3e-3 for 130M model (size-tied LR)
#   - Weight decay: 0.0 for memorization experiments (no regularization)
#   - Warmup: 0.01 (1% of training)
#   - LR schedule: Cosine with anneal to 0
#   - Batch size: 128
#   - P(z) eval frequency: ~1% of training steps
# ============================================================================


def _mk_run(name_suffix: str, epochs: int):
    """Helper to create a training step for a given epoch count.

    Mirrors the per-epoch configs above. We set z_loss_weight=0 for <=10 epochs,
    and 1e-4 beyond that to match the original file.
    """
    z_loss = 0 if epochs <= 10 else 1e-4
    # Approx. 1% of training steps for P(z) evals, rounded to nearest, min 1
    pz_steps = max(1, int((SEED_SET_BATCHES * epochs) / 100 + 0.5))
    return default_train(
        name=f"memorize/comma_150m_10M_{name_suffix}_central1_{timestamp}",
        tokenized=comma_mixture,
        model_config=model_config,
        train_config=SimpleTrainConfig(
            resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
            train_batch_size=BATCH_SIZE,
            num_train_steps=SEED_SET_BATCHES * epochs,
            learning_rate=0.003,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.95,
            lr_schedule="cosine",
            warmup=0.01,
            min_lr_ratio=0.0,
            z_loss_weight=z_loss,
            steps_per_eval=1000,
            max_eval_batches=10,
            steps_per_task_eval=None,
            seed=0,
        ),
        tags=["memorize", "comma", "150m", "10M", name_suffix, "central1", TPU_TYPE],
        eval_harness_tasks=(),
        pz_eval_config=pz_config,
        pz_eval_steps=pz_steps,
    )


# Define runs succinctly
train_1epoch = _mk_run("1epoch", 1)
train_10epoch = _mk_run("10epoch", 10)
train_20epoch = _mk_run("20epoch", 20)
train_50epoch = _mk_run("50epoch", 50)
train_75epoch = _mk_run("75epoch", 75)
train_100epoch = _mk_run("100epoch", 100)
train_200epoch = _mk_run("200epoch", 200)


if __name__ == "__main__":
    # executor_main will discover and run all dependencies (tokenization → mixture → training)
    # Each training step is independent and can be run separately
    executor_main(
        steps=[train_1epoch, train_10epoch, train_20epoch, train_50epoch, train_75epoch, train_100epoch, train_200epoch],
        description="Train 150M Llama on ~10M COMMA seed set across epoch counts",
    )
