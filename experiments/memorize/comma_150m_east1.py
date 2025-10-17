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
#  us-east1 job helper


"""Train an ~150M Llama model on the Common Pile (comma) mixture in us-east1.

This script mirrors submodules/levanter/config/comma_150m_mixture_east1d.yaml
but runs through the Marin executor so tokenization and training can be
orchestrated together.

We request a v6e-64 slice (64 chips, 32 GB each) for this region.
"""

from __future__ import annotations

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

# Model: ~150M parameters (see experiments/llama.py: llama_150m)
model_config = llama_150m

# Select TPU type. us-east1 with v6e-64 slices (akin to a v4-128 footprint).
TPU_TYPE = "v6e-64"

# Training configuration (matches the Levanter YAML)
training_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
    train_batch_size=1024,
    # With seq_len=1024 and B=1024, 600k steps ≈ 629B tokens.
    num_train_steps=600_000,
    learning_rate=0.0073,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    lr_schedule="linear",
    min_lr_ratio=0.0,
    warmup=5_000,
    decay=0.15,
    z_loss_weight=1e-4,
    steps_per_eval=1_000,
    max_eval_batches=10,
)


COMMA_150M_MIXTURE_WEIGHTS = {
    "common_pile/arxiv_abstracts": 0.03458550185249307,
    "common_pile/arxiv_papers": 0.1037565055574792,
    "common_pile/caselaw_access_project": 0.017292750926246536,
    "common_pile/doab": 0.008646375463123268,
    "common_pile/github_archive": 0.04323187731561634,
    "common_pile/libretexts": 0.017292750926246536,
    "common_pile/news": 0.03458550185249307,
    "common_pile/peS2o": 0.1556522341405229,
    "common_pile/project_gutenberg": 0.008646375463123268,
    "common_pile/pubmed": 0.06917100370498614,
    "common_pile/stackexchange": 0.10799027340696585,
    "common_pile/stackv2_edu": 0.10799027340696585,
    "common_pile/stackv2_html": 0.05183012707641196,
    "common_pile/wikimedia": 0.08323088002604166,
    "common_pile/youtube": 0.20718425463647342,
}

if not set(COMMA_150M_MIXTURE_WEIGHTS).issubset(COMMA_MAIN_MIXTURE_WEIGHTS):
    raise ValueError("150M mixture weights must map to known Common Pile datasets")


# Build the comma mixture with component names that match the provided weights
_tokenized = common_pile_tokenized(tokenizer=llama3_tokenizer)
comma_mixture = lm_mixture_data_config(
    components={dataset: _tokenized[dataset] for dataset in COMMA_150M_MIXTURE_WEIGHTS},
    weights=COMMA_150M_MIXTURE_WEIGHTS,
)

# Generate timestamp for unique run names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the training step
comma_150m_train = default_train(
    name=f"memorize/comma_150m_east1_{TPU_TYPE}_{timestamp}",
    tokenized=comma_mixture,
    model_config=model_config,
    train_config=training_config,
    tags=["memorize", "comma", "150m", "east1", TPU_TYPE, "isoflop-aligned", "levanter-yaml"],
    eval_harness_tasks=[],  # Disable evaluation tasks for faster training
)


if __name__ == "__main__":
    # executor_main will discover and run all dependencies (tokenization -> mixture -> training)
    executor_main(steps=[comma_150m_train])
