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

"""Train 150M Llama model on ~100M token COMMA seed set with varying epoch counts.

This experiment creates multiple training runs on the same ~100M token seed set:
- 1 epoch: 195 steps (~102M tokens)
- 2 epochs: 390 steps (~204M tokens)
- 5 epochs: 975 steps (~510M tokens)
- 10 epochs: 1950 steps (~1.02B tokens)

The goal is to measure how memorization (P(z)) scales with epoch count on a fixed dataset.
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
# Seed Set Configuration
# ============================================================================
# With batch_size=128 and seq_len=4096:
#   - Tokens per batch = 128 * 4096 = 524,288 tokens
#   - 15 datasets in COMMA mixture
#   - Tokens per "round" = 15 * 524,288 = 7,864,320 tokens
#   - 13 rounds = 13 * 15 = 195 batches = ~102M tokens
#
# To epoch over this seed set, multiply num_train_steps by epoch count.
# ============================================================================

TPU_TYPE = "v4-64"
BATCH_SIZE = 128
BATCHES_PER_DATASET = 13  # 13 batches x 15 datasets = 195 total batches
SEED_SET_BATCHES = BATCHES_PER_DATASET * 15  # 195 batches
SEED_SET_TOKENS = SEED_SET_BATCHES * BATCH_SIZE * 4096  # ~102.2M tokens

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
)

# Generate timestamp for unique run names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# P(z) Evaluation Configuration
# ============================================================================
# Inner-loop P(z) evaluation to measure memorization during training
# Optimized: single dataset + limited doc length for 150x faster evaluation
pz_config = PzInnerLoopConfig(
    datasets=["wikimedia"],  # Evaluate single dataset for speed (was: None = all 15 datasets)
    mode="first",
    num_documents=1,
    doc_tokens=1024,  # Limit document length to 1024 tokens (was: None = full doc, some >5k tokens)
    chunk_size=512,
    prompt_tokens=256,
    cursor_inc_tokens=16,
    histogram=False,
    pz_npz=False,
    decode_preview=None,
    verify_treecache=False,
)

# ============================================================================
# Epoch Configurations
# ============================================================================
# Hyperparameters for memorization experiments:
#   - LR: 3e-3 for 130M model (size-tied LR)
#   - Weight decay: 0.0 for memorization experiments (no regularization)
#   - Warmup: 0.01 (1% of training)
#   - LR schedule: Cosine with anneal to 0
#   - Batch size: 128
#   - P(z) eval frequency: ~1% of training steps
# ============================================================================

# 1 Epoch: ~102M tokens (195 steps)
train_1epoch = default_train(
    name=f"memorize/comma_150m_100M_1epoch_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=SEED_SET_BATCHES * 1,  # 195 steps
        learning_rate=0.003,  # 3e-3 for 130M
        weight_decay=0.0,  # No weight decay for memorization
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,  # 1% warmup
        min_lr_ratio=0.0,  # Anneal to 0
        z_loss_weight=1e-4,
    ),
    tags=["memorize", "comma", "150m", "100M", "1epoch", "central2", TPU_TYPE],
    eval_harness_tasks=[],  # Disable eval harness, using P(z) instead
    pz_eval_config=pz_config,
    pz_eval_steps=2,  # ~1% of training (195 steps)
)

# 2 Epochs: ~204M tokens (390 steps)
train_2epoch = default_train(
    name=f"memorize/comma_150m_100M_2epoch_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=SEED_SET_BATCHES * 2,  # 390 steps
        learning_rate=0.003,
        weight_decay=0.0,  # No weight decay for memorization
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=1e-4,
    ),
    tags=["memorize", "comma", "150m", "100M", "2epoch", "central2", TPU_TYPE],
    eval_harness_tasks=[],
    pz_eval_config=pz_config,
    pz_eval_steps=4,  # ~1% of training (390 steps)
)

# 5 Epochs: ~510M tokens (975 steps)
train_5epoch = default_train(
    name=f"memorize/comma_150m_100M_5epoch_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=SEED_SET_BATCHES * 5,  # 975 steps
        learning_rate=0.003,
        weight_decay=0.0,  # No weight decay for memorization
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=1e-4,
    ),
    tags=["memorize", "comma", "150m", "100M", "5epoch", "central2", TPU_TYPE],
    eval_harness_tasks=[],
    pz_eval_config=pz_config,
    pz_eval_steps=10,  # ~1% of training (975 steps)
)

# 10 Epochs: ~1.02B tokens (1950 steps)
train_10epoch = default_train(
    name=f"memorize/comma_150m_100M_10epoch_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=SEED_SET_BATCHES * 10,  # 1950 steps
        learning_rate=0.003,
        weight_decay=0.0,  # No weight decay for memorization
        beta1=0.9,
        beta2=0.95,
        lr_schedule="cosine",
        warmup=0.01,
        min_lr_ratio=0.0,
        z_loss_weight=1e-4,
    ),
    tags=["memorize", "comma", "150m", "100M", "10epoch", "central2", TPU_TYPE],
    eval_harness_tasks=[],
    pz_eval_config=pz_config,
    pz_eval_steps=20,  # ~1% of training (1950 steps)
)


if __name__ == "__main__":
    # executor_main will discover and run all dependencies (tokenization → mixture → training)
    # Each training step is independent and can be run separately
    executor_main(
        steps=[train_1epoch],  # , train_2epoch, train_5epoch, train_10epoch],
        description="Train 150M Llama on ~100M COMMA seed set with epochs to measure memorization scaling.",
    )
