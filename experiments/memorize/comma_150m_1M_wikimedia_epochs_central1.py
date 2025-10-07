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

"""Train 150M Llama model on ~1M token Wikimedia-only seed set with varying epoch counts on v5p-8.

This experiment creates multiple training runs on the same ~1M token seed set from Wikimedia only:
- 1 epoch: 2 steps (~1M tokens)
- 10 epochs: 20 steps (~10M tokens)
- 20 epochs: 40 steps (~20M tokens)
- 50 epochs: 100 steps (~52M tokens)
- 75 epochs: 150 steps (~78M tokens)
- 100 epochs: 200 steps (~104M tokens)
- 200 epochs: 400 steps (~209M tokens)

The goal is to measure how memorization (P(z)) scales with epoch count on a fixed, small dataset.

Hardware: v5p-8 (8 chips x 96GB = 768GB total HBM)
Based on levanter config: 2 batches from Wikimedia only = ~1M tokens
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
# Seed Set Configuration (~1M tokens, Wikimedia only)
# ============================================================================
# Following the same pattern as comma_150m_10M_epochs_central1.py, but:
# - 2 batches from Wikimedia only
# - batch_size=128, seq_len=4096
# - Tokens per batch = 128 * 4096 = 524,288 tokens
# - Total for seed set = 2 * 524,288 = ~1,048,576 tokens (~1M)
#
# This creates a ~1M token seed set that the model will see repeatedly.
# To epoch over this seed set, multiply num_train_steps by epoch count.
# ============================================================================

TPU_TYPE = "v5p-8"
BATCH_SIZE = 128  # Match levanter config batch size
BATCHES_PER_DATASET = 2  # 2 batches from Wikimedia (1M seed set)
SEED_SET_BATCHES = BATCHES_PER_DATASET  # Only 1 dataset
SEED_SET_TOKENS = SEED_SET_BATCHES * BATCH_SIZE * 4096  # ~1M tokens

# Model: 150M parameters with seq_len=4096 for memorization studies
# Override seq_len from default 1024 to 4096 to match memorization configs
model_config = dataclasses.replace(llama_150m, seq_len=4096)

# Build the COMMA mixture with max_train_batches to create ~1M seed set (Wikimedia only)
_tokenized = common_pile_tokenized(tokenizer=llama3_tokenizer)

# Create max_train_batches dict: 2 batches for Wikimedia, 0 for all others
max_train_batches_dict = {dataset: BATCHES_PER_DATASET if "wikimedia" in dataset else 0 for dataset in _tokenized}

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
# Using Wikimedia for evaluation since that's our training dataset
pz_config = PzInnerLoopConfig(
    datasets=["common_pile/wikimedia"],  # MUST use full key including "common_pile/" prefix!
    mode="first",
    num_documents=1,
    doc_tokens=1024,  # Limit document length to 1024 tokens
    chunk_size=100,
    prompt_tokens=50,
    cursor_inc_tokens=16,
    histogram=False,
    pz_npz=False,
    decode_preview=1,
    verify_treecache=False,
)


# Helper constructors to simplify per-run definitions
def _mk_run_epochs(name_suffix: str, epochs: int, pz_steps: int):
    return default_train(
        name=f"memorize/comma_150m_1M_wikimedia_{name_suffix}_central1_{timestamp}",
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
            z_loss_weight=0,
            steps_per_eval=1000,
            max_eval_batches=10,
            steps_per_task_eval=None,
            seed=0,
        ),
        tags=["memorize", "comma", "150m", "1M", "wikimedia", name_suffix, "central1", TPU_TYPE],
        eval_harness_tasks=(),
        pz_eval_config=pz_config,
        pz_eval_steps=pz_steps,
    )


def _mk_run_steps(name_suffix: str, steps: int, pz_steps: int):
    return default_train(
        name=f"memorize/comma_150m_1M_wikimedia_{name_suffix}_central1_{timestamp}",
        tokenized=comma_mixture,
        model_config=model_config,
        train_config=SimpleTrainConfig(
            resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
            train_batch_size=BATCH_SIZE,
            num_train_steps=steps,
            learning_rate=0.003,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.95,
            lr_schedule="cosine",
            warmup=0.01,
            min_lr_ratio=0.0,
            z_loss_weight=0,
            steps_per_eval=1000,
            max_eval_batches=10,
            steps_per_task_eval=None,
            seed=0,
        ),
        tags=["memorize", "comma", "150m", "1M", "wikimedia", name_suffix, "central1", TPU_TYPE],
        eval_harness_tasks=(),
        pz_eval_config=pz_config,
        pz_eval_steps=pz_steps,
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

# 1 Epoch: ~1M tokens (2 steps)
train_1epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_1epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=SEED_SET_BATCHES * 1,  # 2 steps
        learning_rate=0.003,  # 3e-3 for 130M
        weight_decay=0.0,  # No weight decay for memorization
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,  # 1% warmup
        min_lr_ratio=0.0,  # Anneal to 0
        z_loss_weight=0,  # Disable z-loss for memorization experiments
        steps_per_eval=1000,  # Match levanter config
        max_eval_batches=10,  # Match levanter config
        steps_per_task_eval=None,  # Disable eval harness completely
        seed=0,  # Fixed seed for reproducible model init and data ordering
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "1epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),  # Use empty tuple instead of list - ABSOLUTELY NO EVAL HARNESS
    pz_eval_config=pz_config,
    pz_eval_steps=1,  # Evaluate at least once
)

# 10 Epochs: ~10M tokens (20 steps)
train_10epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_10epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=SEED_SET_BATCHES * 10,  # 20 steps
        learning_rate=0.003,
        weight_decay=0.0,  # No weight decay for memorization
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "10epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=2,  # ~10% of training (20 steps)
)

# 20 Epochs: ~20M tokens (40 steps)
train_20epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_20epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=SEED_SET_BATCHES * 20,  # 40 steps
        learning_rate=0.003,
        weight_decay=0.0,  # No weight decay for memorization
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "20epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=4,  # ~10% of training (40 steps)
)

# 50 Epochs: ~52M tokens (100 steps)
train_50epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_50epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=SEED_SET_BATCHES * 50,  # 100 steps
        learning_rate=0.003,
        weight_decay=0.0,  # No weight decay for memorization
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "50epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=10,  # ~10% of training (100 steps)
)

# 75 Epochs: ~78M tokens (150 steps)
train_75epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_75epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=SEED_SET_BATCHES * 75,  # 150 steps
        learning_rate=0.003,
        weight_decay=0.0,  # No weight decay for memorization
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "75epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=15,  # ~10% of training (150 steps)
)

# 100 Epochs: ~104M tokens (200 steps)
train_100epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_100epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=SEED_SET_BATCHES * 100,  # 200 steps
        learning_rate=0.003,
        weight_decay=0.0,  # No weight decay for memorization
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "100epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=20,  # ~10% of training (200 steps)
)

# 200 Epochs: ~209M tokens (400 steps)
train_200epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_200epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=SEED_SET_BATCHES * 200,  # 400 steps
        learning_rate=0.003,
        weight_decay=0.0,  # No weight decay for memorization
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "200epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=40,  # ~10% of training (400 steps)
)

# ============================================================================
# Parity Experiments: Fixed Steps with 1M Seed Set
# ============================================================================
# These experiments use the SAME number of training steps as the 10M seed set
# experiments but with only 1M unique tokens. This tests the effect of
# reducing dataset diversity while keeping total training tokens constant.
#
# Comparison:
# - 10M seed: 150 steps = 10 epochs over 15 batches (diverse)
# - 1M seed: 150 steps = 75 epochs over 2 batches (repetitive)
# ============================================================================

# 150 steps (same as 10M 10epoch): 75 epochs over 1M seed
train_75epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_75epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=150,  # Same as 10M 10epoch
        learning_rate=0.003,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "75epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=2,  # ~1% of training
)

# 300 steps (same as 10M 20epoch): 150 epochs over 1M seed
train_150epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_150epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=300,  # Same as 10M 20epoch
        learning_rate=0.003,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "150epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=3,  # ~1% of training
)

# 750 steps (same as 10M 50epoch): 375 epochs over 1M seed
train_375epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_375epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=750,  # Same as 10M 50epoch
        learning_rate=0.003,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "375epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=8,  # ~1% of training
)

# 1125 steps (same as 10M 75epoch): 562 epochs over 1M seed
train_562epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_562epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=1125,  # Same as 10M 75epoch
        learning_rate=0.003,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "562epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=11,  # ~1% of training
)

# 1500 steps (same as 10M 100epoch): 750 epochs over 1M seed
train_750epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_750epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=1500,  # Same as 10M 100epoch
        learning_rate=0.003,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "750epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=15,  # ~1% of training
)

# 3000 steps (same as 10M 200epoch): 1500 epochs over 1M seed
train_1500epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_1500epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=3000,  # Same as 10M 200epoch
        learning_rate=0.003,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "1500epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=30,  # ~1% of training
)

# 6000 steps: 3000 epochs over 1M seed
train_3000epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_3000epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=6000,  # 3000 epochs over 1M seed (2-step epoch)
        learning_rate=0.003,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "3000epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=30,  # ~1% of training
)

# 12000 steps: 6000 epochs over 1M seed
train_6000epoch = default_train(
    name=f"memorize/comma_150m_1M_wikimedia_6000epoch_central1_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=12000,  # 6000 epochs over 1M seed (2-step epoch)
        learning_rate=0.003,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=0,
        steps_per_eval=1000,
        max_eval_batches=10,
        steps_per_task_eval=None,
        seed=0,
    ),
    tags=["memorize", "comma", "150m", "1M", "wikimedia", "6000epoch", "central1", TPU_TYPE],
    eval_harness_tasks=(),
    pz_eval_config=pz_config,
    pz_eval_steps=30,  # ~1% of training
)


# Refactored run definitions (mirror the blocks above, but shorter)
# Baseline epoch runs
train_1epoch = _mk_run_epochs("1epoch", 1, 1)
train_10epoch = _mk_run_epochs("10epoch", 10, 2)
train_20epoch = _mk_run_epochs("20epoch", 20, 4)
train_50epoch = _mk_run_epochs("50epoch", 50, 10)
train_75epoch = _mk_run_epochs("75epoch", 75, 15)
train_100epoch = _mk_run_epochs("100epoch", 100, 20)
train_200epoch = _mk_run_epochs("200epoch", 200, 40)

# Parity (fixed-steps) runs; intentionally reuse some variable names to preserve
# the file's existing behavior (e.g., train_75epoch).
train_75epoch = _mk_run_steps("75epoch", 150, 2)
train_150epoch = _mk_run_steps("150epoch", 300, 3)
train_375epoch = _mk_run_steps("375epoch", 750, 8)
train_562epoch = _mk_run_steps("562epoch", 1125, 11)
train_750epoch = _mk_run_steps("750epoch", 1500, 15)
train_1500epoch = _mk_run_steps("1500epoch", 3000, 30)
train_3000epoch = _mk_run_steps("3000epoch", 6000, 30)
train_6000epoch = _mk_run_steps("6000epoch", 12000, 30)

if __name__ == "__main__":
    # executor_main will discover and run all dependencies (tokenization → mixture → training)
    # Each training step is independent and can be run separately
    executor_main(
        steps=[
            train_1epoch,
            train_10epoch,
            train_20epoch,
            train_50epoch,
            train_75epoch,
            train_100epoch,
            train_200epoch,
            train_150epoch,
            train_375epoch,
            train_562epoch,
            train_750epoch,
            train_1500epoch,
            train_3000epoch,
            train_6000epoch,
        ],
        description="Train 150M Llama on ~1M Wikimedia-only seed set across epoch counts",
    )
