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

"""Train 150M Llama model on ~1M token Wikimedia-only seed set with varying epoch counts on v6e-8.

This mirrors the central1 experiment but targets us-east1-d where v6e-8 is available.

Seed set (Wikimedia only):
- 2 batches per epoch, batch_size=128, seq_len=4096 → ~1,048,576 tokens per epoch (~1M)
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
TPU_TYPE = "v6e-8"  # us-east1-d availability
BATCH_SIZE = 128
BATCHES_PER_DATASET = 2  # 2 batches from Wikimedia (1M seed set)
SEED_SET_BATCHES = BATCHES_PER_DATASET  # Only 1 dataset

# Model: 150M parameters with seq_len=4096 for memorization studies
model_config = dataclasses.replace(llama_150m, seq_len=4096)

# Build the COMMA mixture with max_train_batches to create ~1M seed set (Wikimedia only)
_tokenized = common_pile_tokenized(tokenizer=llama3_tokenizer)
max_train_batches_dict = {dataset: BATCHES_PER_DATASET if "wikimedia" in dataset else 0 for dataset in _tokenized}
comma_mixture = lm_mixture_data_config(
    components=_tokenized,
    weights=COMMA_MAIN_MIXTURE_WEIGHTS,
    max_train_batches=max_train_batches_dict,
    shuffle=False,
)

# Generate timestamp for unique run names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# P(z) Evaluation Configuration
# ============================================================================
pz_config = PzInnerLoopConfig(
    datasets=["common_pile/wikimedia"],
    mode="first",
    num_documents=1,
    doc_tokens=1024,
    chunk_size=100,
    prompt_tokens=50,
    cursor_inc_tokens=16,
    histogram=False,
    pz_npz=False,
    decode_preview=1,
    verify_treecache=False,
)

# ============================================================================
# Epoch Configurations
# ============================================================================


def _mk_run(name_suffix: str, epochs: int, steps_per_eval: int, pz_steps: int):
    return default_train(
        name=f"memorize/comma_150m_1M_wikimedia_{name_suffix}_{timestamp}",
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
            steps_per_eval=steps_per_eval,
            max_eval_batches=10,
            steps_per_task_eval=None,
            seed=0,
        ),
        tags=["memorize", "comma", "150m", "1M", "wikimedia", name_suffix, "east1d", TPU_TYPE],
        eval_harness_tasks=(),
        pz_eval_config=pz_config,
        pz_eval_steps=pz_steps,
    )


# Baseline epochs over 1M seed
train_1epoch = _mk_run("1epoch", 1, 1000, 1)
train_10epoch = _mk_run("10epoch", 10, 1000, 2)
train_20epoch = _mk_run("20epoch", 20, 1000, 4)
train_50epoch = _mk_run("50epoch", 50, 1000, 10)
train_75epoch = _mk_run("75epoch", 75, 1000, 15)
train_100epoch = _mk_run("100epoch", 100, 1000, 20)
train_200epoch = _mk_run("200epoch", 200, 1000, 40)

# Additional epoch counts derived from fixed step parity with 10M runs
train_150epoch = _mk_run("150epoch", 150, 1000, 3)
train_375epoch = _mk_run("375epoch", 375, 1000, 8)
train_562epoch = _mk_run("562epoch", 562, 1000, 11)
train_750epoch = _mk_run("750epoch", 750, 1000, 15)
train_1500epoch = _mk_run("1500epoch", 1500, 1000, 30)
train_3000epoch = _mk_run("3000epoch", 3000, 1000, 30)
train_6000epoch = _mk_run("6000epoch", 6000, 1000, 30)


if __name__ == "__main__":
    executor_main(
        steps=[
            train_1epoch,
            train_10epoch,
            # train_20epoch,
            # train_50epoch,
            # train_75epoch,
            # train_100epoch,
            # train_200epoch,
            # train_150epoch,
            # train_375epoch,
            # train_562epoch,
            # train_750epoch,
            # train_1500epoch,
            # train_3000epoch,
            # train_6000epoch,
        ],
        description=(
            "Train 150M Llama on ~1M Wikimedia-only seed set across epochs "
            "1, 10, 20, 50, 75, 100, 200, 150, 375, 562, 750, 1500, 3000, 6000 "
            "on v6e-8 (us-east1-d) to measure memorization scaling."
        ),
    )
